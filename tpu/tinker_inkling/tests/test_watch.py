"""Offline tests for tinker_watch.py (systemctl, sendmail, billing and capacity are faked).

  python3 tests/test_watch.py tinker_watch.py grid.py
Covers: heartbeats are not progress; no hang restart while grades are in flight (until PENDING_MAX_H) or capacity is
low; crash-loop pause with expiring BLOCKED file (and grid.py honoring it); cost alerts keyed by COST_SINCE, hard stop,
failing-check email; grid unit / grid.log error emails (throttled); grading-timeout emails."""
import importlib.util, json, os, shutil, sys, tempfile, time, types
from pathlib import Path

WATCH_PY, GRID_PY = sys.argv[1], sys.argv[2]
tmp = Path(tempfile.mkdtemp(prefix='watchtest-', dir=os.environ.get('TMPDIR', '/tmp')))
os.environ.update(WATCH_RUNS=str(tmp), WATCH_REGISTRY=str(tmp / 'registry.json'), WATCH_COST_SINCE='2026-11-16T00:00:00Z',
                  WATCH_COST_ALERTS='250,500', WATCH_COST_STOP='1000', WATCH_HANG_MIN='120', WATCH_LOOP_MAX='3')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


w = load(WATCH_PY, 'watch')
mails, calls = [], []
w.mail = lambda subj, body: mails.append(subj)
units = {}     # unit -> dict(ActiveState, NRestarts, ActiveEnterTimestampMonotonic)


def fake_run(cmd, **k):
    calls.append(cmd)
    if cmd[:3] == ['systemctl', '--user', 'stop']:
        units.pop(cmd[-1] if cmd[-1].endswith('.service') else cmd[-1] + '.service', None)
    return types.SimpleNamespace(stdout='', returncode=0)


w.subprocess = types.SimpleNamespace(run=fake_run)
w.list_units = lambda pat: sorted(u for u in units if u.startswith(pat.rstrip('*')))
w.unit_state = lambda u: {k: str(v) for k, v in units.get(u, {'ActiveState': 'inactive'}).items()}


def make_run(name, step, idle_min, grades=()):
    run = tmp / name
    (run / 'tinker_log' / name / 'member_inkling').mkdir(parents=True, exist_ok=True)
    (run / 'tinker_log' / name / 'member_inkling' / 'checkpoints.jsonl').write_text(
        ''.join(json.dumps(dict(name=f'{b:06d}', batch=b)) + '\n' for b in range(1, step + 1)))
    old = time.time() - idle_min * 60
    p = run / 'tinker_log' / name / 'metrics.jsonl'
    p.write_text('{}\n')
    os.utime(p, (old, old))
    ev = [dict(event='grading_completed', time=old)] + [dict(event='grading_no_pool', time=time.time())]
    (run / 'grading-events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in ev))
    (run / 'grading-journal.jsonl').write_text(''.join(
        json.dumps(dict(event=e, request_id=rid, time=t)) + '\n' for e, rid, t in grades))
    (tmp / f'{name}.log').write_text('')
    # unit started 3 h ago
    units[f'tinker-run-{name}.service'] = dict(ActiveState='active', NRestarts=0,
                                               ActiveEnterTimestampMonotonic=int((time.monotonic() - 3 * 3600) * 1e6))
    return run


def restarted(name):
    return any(c[:4] == ['systemctl', '--user', 'restart', f'tinker-run-{name}.service'] for c in calls)


state = {}
# 1. heartbeats (no_pool) are not progress -> idle 150 min with nothing in flight -> restart
make_run('a', 2, 150)
assert abs(w.progress_mtime(tmp / 'a') - (time.time() - 150 * 60)) < 5, 'grading_no_pool counted as progress'
w.check_runs(state, low=False)
assert restarted('a') and any('a restarted' in m for m in mails), (calls, mails)
print('1 OK: heartbeats are not progress; hung run restarted')

# 2. grades in flight -> no restart; 3. capacity low -> no restart; 4. in flight > PENDING_MAX_H -> restart
now = time.time()
make_run('b', 2, 150, grades=[('submitted', 'r1', now - 3600), ('submitted', 'r2', now - 3500), ('result', 'r1', now - 100)])
calls.clear()
w.check_runs(state, low=False)
assert not restarted('b'), 'restarted with a grade in flight'
make_run('c', 2, 150)
w.check_runs(state, low=True)
assert not restarted('c'), 'restarted while capacity low'
make_run('d', 2, 150, grades=[('submitted', 'q1', now - 7 * 3600)])
units['tinker-run-d.service']['ActiveEnterTimestampMonotonic'] = int((time.monotonic() - 8 * 3600) * 1e6)
w.check_runs(state, low=False)
assert restarted('d'), 'grade pending 7 h should not block the hang restart'
print('2-4 OK: no restart while grades in flight (<6 h) or capacity low; restart once pending is older than 6 h')

# 5. crash loop: LOOP_MAX=3 systemd restarts at the same committed step -> stop + BLOCKED with expiry
make_run('e', 3, 1)
mails.clear()
calls.clear()
for n in range(1, 4):
    w.check_runs(state, low=False)
    if 'tinker-run-e.service' in units:
        units['tinker-run-e.service']['NRestarts'] = n
w.check_runs(state, low=False)
blk = tmp / 'e' / 'BLOCKED'
assert blk.exists() and any('e PAUSED 1 h' in m for m in mails), (mails, calls)
until = float(blk.read_text().split()[0])
assert 3500 < until - time.time() < 3700
os.environ.update(GRID='t', GRID_RUNS=str(tmp), GRID_HOLD_AFTER='0')
g = load(GRID_PY, 'grid_t')
assert g.blocked('e')
blk.write_text(f'{time.time() - 1} expired\n')
assert not g.blocked('e') and not blk.exists()
# progress resets the counter
make_run('f', 3, 1)
for n, step in ((1, 3), (2, 4), (3, 5)):
    make_run('f', step, 1)
    units['tinker-run-f.service']['NRestarts'] = n - 1
    w.check_runs(state, low=False)
    units['tinker-run-f.service']['NRestarts'] = n
    w.check_runs(state, low=False)
assert not (tmp / 'f' / 'BLOCKED').exists(), 'restarts with progress must not pause'
print('5 OK: crash loop pauses with an expiring BLOCKED file; grid honors and clears it; progress resets the count')

# 6. cost: alerts per COST_SINCE, hard stop, failing check
mails.clear()
spends = iter([300.0, 600.0, 1200.0])
w.tinker_spend = lambda since: next(spends)
st = {'cost_alerted': [250.0]}       # old list format from the previous watcher
for _ in range(3):
    st['cost_checked'] = 0
    w.check_cost(st)
assert mails[:2] == ['Tinker spend passed $250', 'Tinker spend passed $500'] and mails[2].startswith('COST STOP'), mails
assert (tmp / 'COST_STOP').exists() and 'tinker-run-b.service' not in units
assert g.reconcile() is None
units['tinker-run-late.service'] = dict(ActiveState='active')      # slipped through: stopped on the next pass
w.enforce_cost_stop(st)
assert 'tinker-run-late.service' not in units and sum(m.startswith('COST STOP') for m in mails) == 1, mails
w.tinker_spend = lambda since: (_ for _ in ()).throw(RuntimeError('down'))
mails.clear()
for _ in range(3):
    st['cost_checked'] = 0
    w.check_cost(st)
assert mails == ['Tinker cost check FAILING'], mails
make_run('cs', 2, 150)                 # hung run while the cost stop is in force: must NOT be restarted
calls.clear()
w.check_runs(state, low=False)
assert not restarted('cs'), 'restarted a run under COST_STOP'
(tmp / 'COST_STOP').unlink()
print('6 OK: cost alerts once each, hard stop writes COST_STOP + stops units (grid starts nothing), failing check emails')

# 7. grid unit stop + grid.log errors (throttled); 8. grading timeouts
mails.clear()
gs = {}
units['tinker-grid-t.service'] = dict(ActiveState='active')
(tmp / 'grid-t').mkdir(exist_ok=True)
(tmp / 'grid-t' / 'grid.log').write_text('x start\nx GRID ERROR old (before the watchdog started)\n')
w.check_grids(gs)            # first sight: start at the end (a watchdog restart must not re-email old errors)
assert mails == [], mails
with open(tmp / 'grid-t' / 'grid.log', 'a') as f:
    f.write('x GRID ERROR t-s1: boom\n')
units['tinker-grid-t.service'] = dict(ActiveState='failed')
w.check_grids(gs)
with open(tmp / 'grid-t' / 'grid.log', 'a') as f:
    f.write('x reconcile error again\n')
del units['tinker-grid-t.service']
gs['grid:t'] = 'active'
w.check_grids(gs)
assert mails == ['grid t scheduler failed', 'grid t: 1 scheduler error(s)', 'grid t scheduler GONE'], mails
make_run('h', 1, 1)
w.check_runs(state, low=False)
mails.clear()
with open(tmp / 'h.log', 'a') as f:
    f.write('WARNING Timeout grading: took 172800.0s\n' * 2)
w.check_runs(state, low=False)
w.check_runs(state, low=False)
assert [m for m in mails if 'timeout' in m] == ['h: 2 grading timeout(s)'], mails
print('7-8 OK: grid stop/gone and grid.log errors email (throttled); grading timeouts email')

# 9. blank billing rows are priced, not dropped
rows = [dict(base_model='m', type='sampling_sample', cached='', estimated_cost_usd='2.0', token_count='2000000',
             effective_rate_usd_per_million_tokens='1.0'),
        dict(base_model='m', type='sampling_sample', cached='', estimated_cost_usd='', token_count='3000000'),
        dict(base_model='m', type='training', cached='', estimated_cost_usd='', token_count='1000000'),
        dict(base_model='m', type='mystery', cached='', estimated_cost_usd='', token_count='1000000'),
        dict(base_model='', type='storage', cached='', estimated_cost_usd='', gigabyte_hours='720')]
total, filled = w.price_rows(rows)
assert abs(total - (2 + 3 + 1.5 + 1.5 + 0.1)) < 1e-9 and abs(filled - 6.1) < 1e-9, (total, filled)
print('9 OK: blank billing rows priced (seen rate, fallback rate, highest rate for unknown kinds, storage)')

# 10. a journal 'failed' event clears the pending grade
make_run('k', 1, 150, grades=[('submitted', 'z1', time.time() - 3600), ('failed', 'z1', time.time() - 60)])
calls.clear()
w.check_runs(state, low=False)
assert restarted('k'), 'a failed (dropped) grade must not count as in flight'
print('10 OK: grades dropped by the transport (failed) do not block a hang restart')
shutil.rmtree(tmp)
print('WATCH TESTS PASSED')
