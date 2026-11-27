"""Stall watchdog for the live Tinker runs (user 2026-11-26, after a 2.5 h silent circuit stall).

For every active tinker-run-<run> unit on this host:
  * grading stall: the run is grading (grading events in the last 5 min) but no grade has completed for GRADE_STALL_MIN
    -> email, and restart the run's unit (it resumes from its last checkpoint) at most once per RESTART_GAP_H per run;
  * step stall: no new step for max(2.5 x its median step time, STEP_FLOOR_MIN) -> email only.
Also emails when a tunnel log gains new 'Too many open files' lines (the 2026-11-26 cause).
State (last alert/restart per run, tunnel log offsets) in stall_watch_state.json. usage: stall_watch.py [--dry]"""
import glob, json, os, statistics as S, subprocess, sys, time

R = '/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling-runs'
O = os.path.dirname(os.path.abspath(__file__))
GRADE_STALL_MIN, STEP_FLOOR_MIN, RESTART_GAP_H, REALERT_H = 40, 120, 2.0, 2.0
TAIL_BYTES = 4 << 20
DRY = '--dry' in sys.argv
now = time.time()
st_path = f'{O}/stall_watch_state.json'
try:
    state = json.load(open(st_path))
except (OSError, ValueError):
    state = {}


def active_runs():
    out = subprocess.run(['systemctl', '--user', 'list-units', '--state=active', '--no-legend', 'tinker-run-*.service'],
                         capture_output=True, text=True).stdout
    return [l.split()[0][len('tinker-run-'):-len('.service')] for l in out.splitlines() if l.strip()]


def grading_status(run):
    """(last completion time or None, last grading activity time or None), reading the events file backwards."""
    path = f'{R}/{run}/grading-events.jsonl'
    try:
        size = os.path.getsize(path)
    except OSError:
        return None, None
    last_any = None
    with open(path, 'rb') as f:
        pos, carry = size, b''
        while pos > 0:
            start = max(0, pos - TAIL_BYTES)
            f.seek(start)
            lines = (f.read(pos - start) + carry).split(b'\n')
            carry, lines = (lines[0], lines[1:]) if start > 0 else (b'', lines)
            pos = start
            for raw in reversed(lines):
                try:
                    d = json.loads(raw)
                except ValueError:
                    continue
                t = float(d.get('time') or 0)
                if last_any is None and d.get('request_id'):
                    last_any = t
                if d.get('event') == 'grading_completed':
                    return t, last_any
                if t and now - t > GRADE_STALL_MIN * 60 * 1.5:
                    return None, last_any              # nothing completed in the whole window
    return None, last_any


def steps(run):
    return sorted(os.path.getmtime(f) for f in glob.glob(f'{R}/{run}/tinker_log/{run}/member_*/trajectories/*_step_*.jsonl.gz'))


def started(run):
    """Unit start time (epoch) so a freshly (re)started run is not flagged before it can complete anything."""
    out = subprocess.run(['systemctl', '--user', 'show', f'tinker-run-{run}', '-p', 'ActiveEnterTimestampMonotonic'],
                         capture_output=True, text=True).stdout.strip().split('=')[-1]
    try:
        return now - (time.monotonic() - int(out) / 1e6)
    except ValueError:
        return now


alerts, restarts = [], []
for run in active_runs():
    rs = state.setdefault(run, {})
    up = started(run)
    done, active = grading_status(run)
    if active and now - active < 300 and now - up > GRADE_STALL_MIN * 60:
        since = max(done or 0, up)
        if now - since > GRADE_STALL_MIN * 60:
            msg = f'{run}: grading but no grade completed for {(now - since) / 60:.0f} min'
            if now - rs.get('restart', 0) > RESTART_GAP_H * 3600:
                rs['restart'] = now
                restarts.append(run)
                msg += ' -> restarted its unit (resumes from last checkpoint)'
            else:
                msg += ' (restarted < 2 h ago; NOT restarting again -- needs a look)'
            alerts.append(msg)
    ss = steps(run)
    if len(ss) >= 3:
        med = S.median(b - a for a, b in zip(ss, ss[1:]))
        lim = max(2.5 * med, STEP_FLOOR_MIN * 60)
        age = now - max(ss[-1], up)
        if age > lim and now - rs.get('step_alert', 0) > REALERT_H * 3600:
            rs['step_alert'] = now
            alerts.append(f'{run}: no new step for {age / 60:.0f} min (median step {med / 60:.0f} min, {len(ss)} steps done)')

offs = state.setdefault('_tunnel_offsets', {})
for log in glob.glob(f'{O}/tunnel*.log'):
    size = os.path.getsize(log)
    off = offs.get(log, size)                          # first sight: start from the end
    if size < off:
        off = 0
    if size > off:
        with open(log, 'rb') as f:
            f.seek(off)
            n = f.read(size - off).count(b'Too many open files')
        if n:
            alerts.append(f'{os.path.basename(log)}: {n} new "Too many open files" lines -- tunnel out of fds')
    offs[log] = size

# MOMO Inkling shims (user-approved MOMO prep 2026-11-26): must answer a malformed POST with a JSON error within 30 s.
# Alert only (a restart would drop MOMO's in-flight calls); re-alert at most every REALERT_H.
import urllib.request, urllib.error
for name, port in (('erdos', 18790), ('circuit', 18791)):
    ok = False
    try:
        req = urllib.request.Request(f'http://127.0.0.1:{port}/sample', data=b'{}', headers={'Content-Type': 'application/json'})
        urllib.request.urlopen(req, timeout=30)
    except urllib.error.HTTPError as e:
        ok = b'error' in e.read()
    except Exception:  # noqa: BLE001 -- refused / timeout / reset
        ok = False
    ss_ = state.setdefault(f'_shim_{name}', {})
    if not ok and now - ss_.get('alert', 0) > REALERT_H * 3600:
        ss_['alert'] = now
        alerts.append(f'MOMO Inkling shim {name} (vis1 127.0.0.1:{port}) not answering -- check unit tinker-shim-{name}')

for run in restarts:
    if not DRY:
        subprocess.run(['systemctl', '--user', 'restart', f'tinker-run-{run}'])
if alerts:
    print('\n'.join(alerts))
    if not DRY:
        msg = (f'To: sk7524@princeton.edu\nSubject: [tinker-inkling] stall: {len(alerts)} issue(s)\n\n' + '\n'.join(alerts)
               + f'\n\n(stall_watch.py on della-vis1, {time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime(now))})\n')
        subprocess.run(['/usr/sbin/sendmail', '-t'], input=msg, text=True)
if not DRY:
    json.dump(state, open(st_path, 'w'))
