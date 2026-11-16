"""Watchdog for Inkling Tinker runs on della-vis1 (runs as the transient user unit tinker-watch, Restart=always).

Every CHECK_S seconds:
- refreshes the grading registry timestamp (atomic) so clients trust it;
- for each run unit tinker-run-<name>: if active and no progress for HANG_MIN minutes, restarts it (the client resumes
  from its last checkpoint) and emails; emails once when a unit fails or finishes;
- checks grading capacity on the 6 gradefarm hosts through the local forwards; emails once per episode when fewer than
  MIN_READY hosts are ready (runs then wait; no other graders are mixed in);
- restarts the tunnel unit if it is not active.
Progress = newest mtime of the run's group_timeline.jsonl, metrics.jsonl, global_step.json (heartbeats and registry
events do not count: a hung client still writes those).
"""
import json, os, subprocess, sys, time, urllib.request
from pathlib import Path

RUNS = Path('/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling-runs')
REG = Path('/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling/ops/registry.json')
TO = 'sk7524@princeton.edu'
CHECK_S = int(os.environ.get('WATCH_CHECK_S', '300'))
HANG_MIN = float(os.environ.get('WATCH_HANG_MIN', '60'))
MIN_READY = int(os.environ.get('WATCH_MIN_READY', '3'))
STATE = RUNS / 'watch_state.json'
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


def last_grading_event(path, tail_bytes=200_000):
    """Time of the newest grading event other than registry-staleness heartbeats (a hung client still writes those)."""
    try:
        with open(path, 'rb') as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - tail_bytes))
            lines = f.read().decode(errors='ignore').splitlines()
    except OSError:
        return 0.0
    for line in reversed(lines):
        if '"grading_registry_stale"' in line or '"time"' not in line:
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


def last_exit(name):
    try:
        lines = [l for l in open(RUNS / f'{name}.log', errors='ignore') if 'client exited rc=' in l]
        return lines[-1].strip() if lines else ''
    except OSError:
        return ''


def main():
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    log(f'watch start (check {CHECK_S}s, hang {HANG_MIN} min, min ready {MIN_READY})')
    while True:
        try:
            refresh_registry()
            if unit_state('tinker-tunnel.service').get('ActiveState') != 'active':
                log('tunnel unit not active; starting it')
                subprocess.run(['bash', '/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling/ops/run_service.sh', 'tunnel'],
                               check=False)
            ready, slots = capacity()
            low = ready < MIN_READY
            if low and not state.get('capacity_low'):
                mail('grading capacity LOW', f'{ready}/6 gradefarm hosts ready ({slots} Erdős slots). Runs wait; '
                                             'no other graders are used. Check Harish\'s machines / the tunnel.')
            if not low and state.get('capacity_low'):
                mail('grading capacity back', f'{ready}/6 hosts ready ({slots} slots).')
            state['capacity_low'] = low
            units = sh('systemctl', '--user', 'list-units', '--all', '--plain', '--no-legend', 'tinker-run-*')
            present = {l.split()[0][len('tinker-run-'):-len('.service')] for l in units.splitlines() if l.strip()}
            for key in [k for k in state if k.startswith('unit:')]:
                gone_name = key[len('unit:'):]
                if gone_name not in present and state[key].get('active') in ('active', 'activating'):
                    ex = last_exit(gone_name)
                    ok = 'rc=0 ' in ex
                    mail(f'{gone_name} {"finished" if ok else "STOPPED"}', f'{ex or "no exit line"}\nLog: {RUNS}/{gone_name}.log')
                    state[key] = dict(active='gone', restarts=None)
            for line in units.splitlines():
                unit = line.split()[0]
                name = unit[len('tinker-run-'):-len('.service')]
                st = unit_state(unit)
                key = f'unit:{name}'
                prev = state.get(key, {})
                active = st.get('ActiveState')
                if active == 'active':
                    run = RUNS / name
                    started_mono = int(st.get('ActiveEnterTimestampMonotonic', '0') or 0) / 1e6
                    up_s = time.monotonic() - started_mono if started_mono else 0
                    idle_s = time.time() - max(progress_mtime(run), time.time() - up_s)
                    if idle_s > HANG_MIN * 60:
                        log(f'{name}: no progress for {idle_s/60:.0f} min; restarting')
                        subprocess.run(['systemctl', '--user', 'restart', unit], check=False)
                        mail(f'{name} restarted (no progress {idle_s/60:.0f} min)',
                             f'Unit {unit} restarted by the watchdog; the client resumes from its last checkpoint.')
                elif active == 'failed' and prev.get('active') != 'failed':
                    mail(f'{name} FAILED', f'Unit {unit} failed ({st.get("Result")}, restarts {st.get("NRestarts")}). '
                                           f'Log: {RUNS}/{name}.log')
                elif active == 'inactive' and prev.get('active') == 'active':
                    mail(f'{name} finished', f'Unit {unit} exited ({st.get("Result")}).')
                state[key] = dict(active=active, restarts=st.get('NRestarts'))
                if prev.get('restarts') is not None and int(st.get('NRestarts') or 0) > int(prev.get('restarts') or 0):
                    log(f'{name}: systemd restarted it (NRestarts {prev.get("restarts")} -> {st.get("NRestarts")})')
                    mail(f'{name} crashed and was restarted', f'NRestarts now {st.get("NRestarts")}; it resumes from '
                                                              f'its last checkpoint. Log: {RUNS}/{name}.log')
            STATE.write_text(json.dumps(state))
        except Exception as e:  # never die; log and continue
            log(f'watch error {e!r}')
        time.sleep(CHECK_S)


if __name__ == '__main__':
    main()
