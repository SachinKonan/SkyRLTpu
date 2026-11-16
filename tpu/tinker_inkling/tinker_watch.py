"""Watchdog for Inkling Tinker runs on della-vis1 (runs as the transient user unit tinker-watch, Restart=always).

Every CHECK_S seconds:
- refreshes the grading registry timestamp (atomic) so clients trust it; restarts the tunnel unit if it is not active;
- grading capacity on the 6 gradefarm hosts: emails once per episode when fewer than MIN_READY hosts are ready;
- each run unit tinker-run-<name>:
  * hang: no progress for HANG_MIN minutes -> restart (the client resumes from its last checkpoint) and email. Not while
    grading capacity is low, and not while the run has grades in flight that were submitted less than PENDING_MAX_H
    hours ago (a long grading queue is not a hang; restarting would throw that work away);
  * crash loop: LOOP_MAX restarts (systemd or watchdog) without a new committed step -> stop the unit, write
    RUNS/<name>/BLOCKED with an expiry (BLOCK_H, doubling per further loop up to BLOCK_MAX_H; grid.py starts it again
    after that, run_service.sh refuses while the file exists) and email;
  * emails when a unit fails, crashes, finishes or disappears, and when its log reports "Timeout grading";
- grid units tinker-grid-<G>: emails when one stops, and on new "GRID ERROR" / "reconcile error" lines in grid.log;
- Tinker spend (`tinker billing usage` since COST_SINCE, hourly): emails at each COST_ALERTS threshold; at COST_STOP it
  writes RUNS/COST_STOP (nothing starts), stops every grid and run unit, and emails. Emails if the check keeps failing.
Progress = newest of: the run's group_timeline.jsonl / metrics.jsonl / global_step.json mtimes and its newest grading
event other than registry-staleness and no-pool heartbeats (a hung or waiting client still writes those).
"""
import json, os, re, subprocess, sys, time, urllib.request
from pathlib import Path

RUNS = Path(os.environ.get('WATCH_RUNS', '/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling-runs'))
OPS = Path('/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling/ops')
REG = Path(os.environ.get('WATCH_REGISTRY', str(OPS / 'registry.json')))
TO = 'sk7524@princeton.edu'
CHECK_S = int(os.environ.get('WATCH_CHECK_S', '300'))
HANG_MIN = float(os.environ.get('WATCH_HANG_MIN', '120'))
PENDING_MAX_H = float(os.environ.get('WATCH_PENDING_MAX_H', '6'))
LOOP_MAX = int(os.environ.get('WATCH_LOOP_MAX', '4'))
BLOCK_H = float(os.environ.get('WATCH_BLOCK_H', '1'))         # first crash-loop pause; doubles per further loop
BLOCK_MAX_H = float(os.environ.get('WATCH_BLOCK_MAX_H', '8'))
MIN_READY = int(os.environ.get('WATCH_MIN_READY', '3'))
STATE = RUNS / 'watch_state.json'
TINKER_CLI = '/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling/venv/bin/tinker'
COST_SINCE = os.environ.get('WATCH_COST_SINCE', '')            # RFC 3339 hour, e.g. 2026-11-16T06:00:00Z ('' = off)
COST_ALERTS = [float(x) for x in os.environ.get('WATCH_COST_ALERTS', '250,500,1000,1500,2000,2500,3000,4000').split(',') if x]
COST_STOP = float(os.environ.get('WATCH_COST_STOP', '5000') or 0)   # 0 = no hard stop
COST_STOP_FILE = RUNS / 'COST_STOP'
COST_EVERY_S = 3600
MAIL_EVERY_S = 3600          # repeated-problem emails (grid errors, grading timeouts) at most hourly per source
LOG = RUNS / 'watch.log'


def log(msg):
    with open(LOG, 'a') as f:
        f.write(f'{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} {msg}\n')


def mail(subject, body):
    msg = f'To: {TO}\nSubject: [tinker-inkling] {subject}\n\n{body}\n'
    subprocess.run(['/usr/sbin/sendmail', '-t'], input=msg, text=True, check=False)
    log(f'emailed: {subject}')


def sh(*args):
    return subprocess.run(list(args), capture_output=True, text=True).stdout.strip()


def unit_state(unit):
    out = sh('systemctl', '--user', 'show', unit, '-p', 'ActiveState', '-p', 'SubState', '-p', 'Result',
             '-p', 'ActiveEnterTimestampMonotonic', '-p', 'NRestarts')
    return dict(l.split('=', 1) for l in out.splitlines() if '=' in l)


def list_units(pattern):
    out = sh('systemctl', '--user', 'list-units', '--all', '--plain', '--no-legend', pattern)
    return [l.split()[0] for l in out.splitlines() if l.strip()]


def tinker_spend(since):
    """Total estimated Tinker cost (USD) from `since` to the next hour (billing usage allows <= 14 days)."""
    import csv, datetime, tempfile
    key = subprocess.run(['bash', '-ic', 'printf %s "$TINKER_API_KEY"'], capture_output=True, text=True).stdout
    end = (datetime.datetime.utcnow() + datetime.timedelta(hours=1)).strftime('%Y-%m-%dT%H:00:00Z')
    with tempfile.NamedTemporaryFile(suffix='.csv') as f:
        r = subprocess.run([TINKER_CLI, 'billing', 'usage', since, end, '--csv', f.name], capture_output=True,
                           text=True, env=dict(os.environ, TINKER_API_KEY=key), timeout=300)
        if r.returncode != 0:
            raise RuntimeError(f'billing usage failed: {r.stderr.strip()[-200:]}')
        rows = list(csv.DictReader(open(f.name)))
    return sum(float(x['estimated_cost_usd'] or 0) for x in rows)


def refresh_registry():
    farms = [dict(name=f'gradefarm-harish-{i}', url=f'http://127.0.0.1:2786{i}', ok=True, tunnel_ok=True,
                  health='healthy') for i in range(1, 7)]
    tmp = REG.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(time=time.time(), farms=farms)))
    os.replace(tmp, REG)


def capacity():
    ready, slots = 0, 0
    for i in range(1, 7):
        try:
            req = urllib.request.Request(f'http://127.0.0.1:2786{i}/skyrl/v1/grading/capacity',
                                         headers={'X-Grading-Run': 'watch', 'X-Grading-Token': 'watch'})
            c = json.load(urllib.request.urlopen(req, timeout=8))
            if c.get('ready'):
                ready += 1
                slots += c['families']['math']['total']
        except Exception:
            pass
    return ready, slots


HEARTBEATS = ('"grading_registry_stale"', '"grading_no_pool"')


def last_grading_event(path, tail_bytes=200_000):
    """Time of the newest grading event other than heartbeats (a hung or waiting client still writes those)."""
    try:
        with open(path, 'rb') as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - tail_bytes))
            lines = f.read().decode(errors='ignore').splitlines()
    except OSError:
        return 0.0
    for line in reversed(lines):
        if any(h in line for h in HEARTBEATS) or '"time"' not in line:
            continue
        try:
            return float(json.loads(line)['time'])
        except Exception:
            continue
    return 0.0


def progress_mtime(run):
    newest = last_grading_event(run / 'grading-events.jsonl')
    for pat in ('tinker_log/*/group_timeline.jsonl', 'tinker_log/*/metrics.jsonl', 'tinker_log/*/global_step.json'):
        for p in run.glob(pat):
            newest = max(newest, p.stat().st_mtime)
    return newest


def committed_step(run):
    """Highest checkpoint batch in the run's ledger (what a restart resumes from)."""
    best = 0
    for p in run.glob('tinker_log/*/member_inkling/checkpoints.jsonl'):
        try:
            for l in open(p):
                m = re.search(r'"batch": (\d+)', l)
                if m:
                    best = max(best, int(m.group(1)))
        except OSError:
            pass
    return best


class Tail:
    """Incremental reader: new complete lines of a file since the last call (in memory; a restart re-reads from 0)."""
    def __init__(self):
        self.off = {}

    def new_lines(self, path):
        path = str(path)
        try:
            with open(path, 'rb') as f:
                f.seek(0, 2)
                size = f.tell()
                off = self.off.get(path, 0)
                if size < off:      # truncated / replaced
                    off = 0
                f.seek(off)
                data = f.read()
        except OSError:
            return []
        end = data.rfind(b'\n') + 1
        self.off[path] = off + end
        return data[:end].decode(errors='ignore').splitlines()


TAIL = Tail()
JOURNAL = {}   # run -> {request_id: submit time} still waiting for a result
SUBMIT_RE = re.compile(r'^\{"event": "(submitted|result)", "request_id": "([^"]+)"')
TIME_RE = re.compile(r'"time": ([0-9.]+)\}?\s*$')


def pending_grades(run, since):
    """(count, oldest submit time) of grades submitted at/after `since` (this unit start) without a result yet."""
    pend = JOURNAL.setdefault(str(run), {})
    for line in TAIL.new_lines(run / 'grading-journal.jsonl'):
        m = SUBMIT_RE.match(line)
        if not m:
            continue
        if m.group(1) == 'submitted':
            t = TIME_RE.search(line)
            pend.setdefault(m.group(2), float(t.group(1)) if t else time.time())
        else:
            pend.pop(m.group(2), None)
    times = [t for t in pend.values() if t >= since]
    return len(times), (min(times) if times else 0.0)


def count_new(path, needle):
    return sum(1 for l in TAIL.new_lines(path) if needle in l)


def last_exit(name):
    try:
        lines = [l for l in open(RUNS / f'{name}.log', errors='ignore') if 'client exited rc=' in l]
        return lines[-1].strip() if lines else ''
    except OSError:
        return ''


def throttled_mail(state, key, subject, body):
    if time.time() - state.get(f'mailed:{key}', 0) >= MAIL_EVERY_S:
        mail(subject, body)
        state[f'mailed:{key}'] = time.time()
    else:
        log(f'(not emailed, sent < {MAIL_EVERY_S}s ago) {subject}')


def check_cost(state):
    if not COST_SINCE or time.time() - state.get('cost_checked', 0) <= COST_EVERY_S:
        return
    state['cost_checked'] = time.time()
    try:
        spend = tinker_spend(COST_SINCE)
    except Exception as e:
        state['cost_fail'] = state.get('cost_fail', 0) + 1
        log(f'cost check failed ({state["cost_fail"]} in a row): {e!r}')
        if state['cost_fail'] == 3:
            mail('Tinker cost check FAILING', f'3 hourly `tinker billing usage` checks in a row failed; spend is not '
                                              f'being watched. Last error: {e!r}')
        return
    state['cost_fail'], state['spend'] = 0, spend
    log(f'tinker spend since {COST_SINCE}: ${spend:.2f}')
    alerted = state.get('cost_alerted')
    if not isinstance(alerted, dict):
        alerted = state['cost_alerted'] = {}
    done = alerted.setdefault(COST_SINCE, [])
    for t in COST_ALERTS:
        if spend >= t and t not in done:
            mail(f'Tinker spend passed ${t:.0f}', f'${spend:.2f} since {COST_SINCE} (tinker billing usage). '
                 f'Alerts at {COST_ALERTS}; hard stop at ${COST_STOP:.0f}.')
            done.append(t)
    if COST_STOP and spend >= COST_STOP and not COST_STOP_FILE.exists():
        COST_STOP_FILE.write_text(f'{time.time()} spend ${spend:.2f} since {COST_SINCE} >= ${COST_STOP:.0f}\n')
        stopped = []
        for unit in list_units('tinker-grid-*') + list_units('tinker-run-*'):
            subprocess.run(['systemctl', '--user', 'stop', unit], check=False)
            stopped.append(unit)
        mail(f'COST STOP at ${spend:.2f}', f'Spend ${spend:.2f} since {COST_SINCE} reached the hard stop '
             f'${COST_STOP:.0f}. Wrote {COST_STOP_FILE} (nothing starts while it exists) and stopped: {stopped}. '
             f'Runs resume from their last checkpoint after you delete the file and restart the grid.')


def check_grids(state):
    present = set()
    for unit in list_units('tinker-grid-*'):
        g = unit[len('tinker-grid-'):-len('.service')]
        present.add(g)
        active = unit_state(unit).get('ActiveState')
        prev = state.get(f'grid:{g}')
        if prev == 'active' and active not in ('active', 'activating', 'reloading'):
            mail(f'grid {g} scheduler {active}', f'Unit {unit} is {active}; no new runs start until it is back. '
                 f'Log: {RUNS}/grid-{g}/grid.log')
        state[f'grid:{g}'] = active
    for key in [k for k in state if k.startswith('grid:')]:
        g = key[len('grid:'):]
        if g not in present and state[key] == 'active':
            mail(f'grid {g} scheduler GONE', f'Unit tinker-grid-{g} disappeared; no new runs start. '
                 f'Restart: run_service.sh grid {g}')
            state[key] = 'gone'
        errs = [l for l in TAIL.new_lines(RUNS / f'grid-{g}' / 'grid.log') if 'GRID ERROR' in l or 'reconcile error' in l]
        if errs:
            throttled_mail(state, f'griderr:{g}', f'grid {g}: {len(errs)} scheduler error(s)',
                           '\n'.join(errs[-10:]) + f'\nLog: {RUNS}/grid-{g}/grid.log')


def note_restart(state, key, name, run, why):
    """Count a restart toward the crash-loop cap; stop + block the run when it keeps restarting without progress."""
    s = state[key]
    step = committed_step(run)
    if step > s.get('loop_step', -1):
        s['loop_step'], s['loop_n'] = step, 1
    else:
        s['loop_n'] = s.get('loop_n', 0) + 1
    log(f'{name}: restart ({why}); {s["loop_n"]} restart(s) at committed step {step}')
    if s['loop_n'] >= LOOP_MAX:
        hours = min(BLOCK_MAX_H, BLOCK_H * 2 ** (s['loop_n'] - LOOP_MAX))
        until = time.time() + hours * 3600
        (run / 'BLOCKED').write_text(f'{until} {s["loop_n"]} restarts at step {step}: {why}\n')
        subprocess.run(['systemctl', '--user', 'stop', f'tinker-run-{name}.service'], check=False)
        mail(f'{name} PAUSED {hours:g} h: crash loop', f'{s["loop_n"]} restarts without a new committed step '
             f'(step {step}); last: {why}. Wrote {run}/BLOCKED: the grid restarts it after '
             f'{time.strftime("%Y-%m-%d %H:%MZ", time.gmtime(until))} (each further loop doubles the pause, max '
             f'{BLOCK_MAX_H:g} h). Delete the file to restart sooner. Log: {RUNS}/{name}.log')
        return True
    return False


def check_runs(state, low):
    units = list_units('tinker-run-*')
    present = {u[len('tinker-run-'):-len('.service')] for u in units}
    for key in [k for k in state if k.startswith('unit:')]:
        gone_name = key[len('unit:'):]
        if gone_name not in present and state[key].get('active') in ('active', 'activating'):
            ex = last_exit(gone_name)
            ok = 'rc=0 ' in ex
            mail(f'{gone_name} {"finished" if ok else "STOPPED"}', f'{ex or "no exit line"}\nLog: {RUNS}/{gone_name}.log')
            state[key] = dict(state[key], active='gone', restarts=None)
    for unit in units:
        name = unit[len('tinker-run-'):-len('.service')]
        run = RUNS / name
        st = unit_state(unit)
        key = f'unit:{name}'
        prev = dict(state.get(key, {}))
        state[key] = s = dict(prev, active=st.get('ActiveState'), restarts=st.get('NRestarts'))
        active = s['active']
        n = count_new(RUNS / f'{name}.log', 'Timeout grading')
        if n:
            throttled_mail(state, f'timeout:{name}', f'{name}: {n} grading timeout(s)',
                           f'"Timeout grading" x{n} in {RUNS}/{name}.log (the run fails and resumes rather than '
                           f'training on zero rewards). Check the gradefarm hosts.')
        if prev.get('restarts') is not None and int(st.get('NRestarts') or 0) > int(prev.get('restarts') or 0):
            ex = last_exit(name)
            mail(f'{name} crashed and was restarted', f'NRestarts {st.get("NRestarts")}; it resumes from its last '
                 f'checkpoint. {ex}\nLog: {RUNS}/{name}.log')
            if note_restart(state, key, name, run, f'crash: {ex or "no exit line"}'):
                continue
        if active == 'active':
            started_mono = int(st.get('ActiveEnterTimestampMonotonic', '0') or 0) / 1e6
            up_s = time.monotonic() - started_mono if started_mono else 0
            started_wall = time.time() - up_s
            npend, oldest = pending_grades(run, started_wall)
            idle_s = time.time() - max(progress_mtime(run), started_wall)
            if idle_s > HANG_MIN * 60:
                if low:
                    why = 'grading capacity low'
                elif npend and time.time() - oldest < PENDING_MAX_H * 3600:
                    why = f'{npend} grades in flight (oldest {(time.time() - oldest) / 3600:.1f} h)'
                else:
                    why = ''
                if why:
                    if time.time() - s.get('skip_logged', 0) > 3600:
                        log(f'{name}: no progress for {idle_s/60:.0f} min but not restarting: {why}')
                        s['skip_logged'] = time.time()
                    continue
                log(f'{name}: no progress for {idle_s/60:.0f} min; restarting')
                if note_restart(state, key, name, run, f'no progress {idle_s/60:.0f} min'):
                    continue
                subprocess.run(['systemctl', '--user', 'restart', unit], check=False)
                s['restarts'] = unit_state(unit).get('NRestarts')   # a manual restart is not a crash
                mail(f'{name} restarted (no progress {idle_s/60:.0f} min)',
                     f'Unit {unit} restarted by the watchdog; the client resumes from its last checkpoint.')
        elif active == 'failed' and prev.get('active') != 'failed':
            mail(f'{name} FAILED', f'Unit {unit} failed ({st.get("Result")}, restarts {st.get("NRestarts")}). '
                                   f'Log: {RUNS}/{name}.log')
        elif active == 'inactive' and prev.get('active') == 'active':
            mail(f'{name} finished', f'Unit {unit} exited ({st.get("Result")}).')


def check_once(state):
    refresh_registry()
    if unit_state('tinker-tunnel.service').get('ActiveState') != 'active':
        log('tunnel unit not active; starting it')
        subprocess.run(['bash', str(OPS / 'run_service.sh'), 'tunnel'], check=False)
    check_cost(state)
    ready, slots = capacity()
    low = ready < MIN_READY
    if low and not state.get('capacity_low'):
        mail('grading capacity LOW', f'{ready}/6 gradefarm hosts ready ({slots} Erdős slots). Runs wait; '
                                     'no other graders are used. Check Harish\'s machines / the tunnel.')
    if not low and state.get('capacity_low'):
        mail('grading capacity back', f'{ready}/6 hosts ready ({slots} slots).')
    state['capacity_low'] = low
    check_grids(state)
    check_runs(state, low)


def main():
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    log(f'watch start (check {CHECK_S}s, hang {HANG_MIN} min, pending max {PENDING_MAX_H} h, loop max {LOOP_MAX}, '
        f'min ready {MIN_READY}, cost since {COST_SINCE or "OFF"} alerts {COST_ALERTS} stop {COST_STOP})')
    while True:
        try:
            check_once(state)
            tmp = STATE.with_suffix('.tmp')
            tmp.write_text(json.dumps(state))
            os.replace(tmp, STATE)
        except Exception as e:  # never die; log and continue
            import traceback
            log(f'watch error {e!r}\n{traceback.format_exc()}')
        time.sleep(CHECK_S)


if __name__ == '__main__':
    main()
