"""grid.py: step-1 start in batches of 3 while held, and per-line ledger parsing.
  python3 tests/test_grid_stagger.py grid.py"""
import importlib.util, json, os, shutil, sys, tempfile, types
from pathlib import Path

tmp = Path(tempfile.mkdtemp(prefix='gridstag-', dir=os.environ.get('TMPDIR', '/tmp')))
os.environ.update(GRID='st', GRID_RUNS=str(tmp), GRID_HOLD_AFTER='1')
spec = importlib.util.spec_from_file_location('grid_st', sys.argv[1])
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
running = set()
g.unit_running = lambda name: name in running


def started():
    cmds = []
    g.subprocess = types.SimpleNamespace(run=lambda cmd, **k: cmds.append(cmd[3]))
    g.reconcile()
    running.update(cmds)
    return sorted(cmds)


def submit(i, n):
    d = tmp / g.seed_name(i)
    d.mkdir(parents=True, exist_ok=True)
    (d / 'grading-journal.jsonl').write_text(''.join(
        json.dumps(dict(event='submitted', request_id=f'r{k}', time=0)) + '\n' for k in range(n)))


def commit_step1(i):
    ld = g.logdir(g.seed_name(i)) / 'member_inkling'
    ld.mkdir(parents=True, exist_ok=True)
    (ld / 'checkpoints.jsonl').write_text(json.dumps(dict(name='000001', batch=1)) + '\n')


assert started() == ['st-s1', 'st-s2', 'st-s3'], 'first batch'
assert started() == [], 'nothing more until batch 1 has sampled'
submit(1, 400); submit(2, 400); submit(3, 100)
assert started() == [], 'seed 3 has only 100 grades submitted'
commit_step1(3)
assert started() == ['st-s4', 'st-s5', 'st-s6'], 'second batch once batch 1 sampled or reached step 1'
for i in (4, 5, 6):
    submit(i, 384)
assert started() == ['st-s7', 'st-s8', 'st-s9']
print('stagger OK: 1-3, then 4-6 once 1-3 submitted >= 384 grades or reached step 1, then 7-9')

(tmp / 'grid-st').mkdir(exist_ok=True)
(tmp / 'grid-st' / 'RELEASE').write_text('')
running.clear()
for i in range(1, 10):
    commit_step1(i)
assert len(started()) == 9, 'after RELEASE all seeds continue at once'
print('release OK: no stagger after RELEASE')

ld = g.logdir(g.seed_name(2)) / 'member_inkling' / 'checkpoints.jsonl'
ld.write_text(json.dumps(dict(name='000001', batch=1)) + '\n{"name": "0000' + '\n' +
              json.dumps(dict(name='000002', batch=2)) + '\n')
assert g.gstep(g.seed_name(2)) == 2 and g.has_checkpoint(g.seed_name(2), 1), g.ledger(g.seed_name(2))
print('ledger OK: a torn line is skipped, the other rows still count')
shutil.rmtree(tmp)
print('STAGGER TESTS PASSED')
