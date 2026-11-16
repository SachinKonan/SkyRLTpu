"""grid.py: frozen controls (GRID_FROZEN), step count without checkpoint rows, clone inherits its source's code pin;
tinker_watch.committed_step falls back to global_step for frozen runs.
  python3 tests/test_grid_frozen.py grid.py tinker_watch.py"""
import importlib.util, json, os, shutil, sys, tempfile, types
from pathlib import Path

tmp = Path(tempfile.mkdtemp(prefix='gridfrz-', dir=os.environ.get('TMPDIR', '/tmp')))
os.environ.update(GRID='fz', GRID_RUNS=str(tmp), GRID_HOLD_AFTER='1', GRID_TARGETS='1:10,2:10,3:10',
                  GRID_FROZEN='1:10', WATCH_RUNS=str(tmp), WATCH_REGISTRY=str(tmp / 'reg.json'))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


g = load(sys.argv[1], 'grid_fz')
w = load(sys.argv[2], 'watch_fz')
p, d = g.plan()
assert sorted(p) == ['fz-f1', 'fz-s1', 'fz-s2', 'fz-s3'] and d is None, p
assert p['fz-f1'] == (10, 1, ['TTD_FROZEN_POLICY=1', 'LEARNING_RATE=0']) and p['fz-s1'] == (10, 1, []), p
print('plan OK: pilot seeds 1-3 to 10 + frozen control fz-f1 (seed 1, lr 0, frozen) to 10; no decision with 3 seeds')

cmds = []
g.unit_running = lambda n: False
g.subprocess = types.SimpleNamespace(run=lambda cmd, **k: cmds.append(cmd))
g.reconcile()
started = {c[3]: c for c in cmds}
assert sorted(started) == ['fz-f1', 'fz-s1', 'fz-s2', 'fz-s3'], started
assert started['fz-f1'][4:] == ['1', '1', 'TTD_FROZEN_POLICY=1', 'LEARNING_RATE=0'], started['fz-f1']
print('start OK: held at step 1, the control starts with its seed batch and the frozen env')

ld = g.logdir('fz-f1')
ld.mkdir(parents=True)
(ld / 'global_step.json').write_text(json.dumps(dict(global_step=4)))
assert g.gstep('fz-f1') == 4 and w.committed_step(tmp / 'fz-f1') == 4
(ld / 'member_inkling').mkdir()
(ld / 'member_inkling' / 'checkpoints.jsonl').write_text(json.dumps(dict(batch=2)) + '\n')
assert g.gstep('fz-f1') == 2 and w.committed_step(tmp / 'fz-f1') == 2, 'ledger wins when present'
print('step OK: no checkpoint rows -> global_step (grid and watchdog); rows present -> ledger')

# clone inherits the source's code pin
src = g.logdir('fz-s2')
(src / 'member_inkling').mkdir(parents=True)
(src / 'member_inkling' / 'checkpoints.jsonl').write_text(''.join(
    json.dumps(dict(name=f'{b:06d}', batch=b, state_path=f'tinker://x/{b}')) + '\n' for b in range(1, 6)))
(src / 'puct_sampler_step_000005.json').write_text(json.dumps(dict(step=5, puct_n={'a': 1}, puct_T=1, puct_m={},
                                                                   states=[dict(id='a', value=-0.3809)])))
(tmp / 'fz-s2' / 'code.pin').write_text('PIN_SKYRL=/x/skyrl-old\nPIN_DISCOVER=/x/discover-old\n')
g.prepare_clone('fz-c1-s2', 2)
assert (tmp / 'fz-c1-s2' / 'code.pin').read_text() == (tmp / 'fz-s2' / 'code.pin').read_text()
print('clone OK: a clone runs its source seed\'s pinned code')
shutil.rmtree(tmp)
print('FROZEN/PIN TESTS PASSED')
