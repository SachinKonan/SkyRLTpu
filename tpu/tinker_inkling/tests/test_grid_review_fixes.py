"""Tests for the pre-launch review fixes in grid.py:
- the step-5 decision waits until EVERY seed has a committed step-5 checkpoint (tree alone is not enough);
- the step count comes from the checkpoint ledger, not global_step.json;
- unit_running treats anything but explicit inactive/failed as running;
- one failing run does not block reconcile for the others;
- clone names are matched exactly (a grid name containing '-c' does not turn seeds into clones)."""
import importlib.util, json, os, shutil, sys, tempfile, types
from pathlib import Path

GRID_PY = sys.argv[1]


def load(grid, runs):
    os.environ.update(GRID=grid, GRID_RUNS=str(runs), GRID_HOLD_AFTER='0')
    spec = importlib.util.spec_from_file_location(f'grid_{grid}', GRID_PY)
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    return g


def make_seed(g, i, c5_gap, rows_upto):
    ld = g.logdir(g.seed_name(i))
    (ld / 'member_inkling').mkdir(parents=True, exist_ok=True)
    val = -(0.38085 + c5_gap * 1e-6)
    tree = dict(step=5, initial_states=[], puct_T=9, puct_m={'a': val}, puct_n={'a': 3},
                states=[dict(id='a', value=val, code='x')])
    (ld / 'puct_sampler_step_000005.json').write_text(json.dumps(tree))
    rows = [dict(name=f'{k:06d}', batch=k, state_path=f'tinker://s{i}/weights/{k:06d}') for k in range(1, rows_upto + 1)]
    (ld / 'member_inkling' / 'checkpoints.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (ld / 'global_step.json').write_text(json.dumps(dict(global_step=max(0, rows_upto - 1))))  # lags by one


tmp = Path(tempfile.mkdtemp(prefix='gridfix-', dir=os.environ.get('TMPDIR', '/tmp')))

# 1. decision waits for committed step-5 checkpoints; 2. gstep from the ledger
g = load('r1', tmp / 'a')
for i in range(1, 10):
    make_seed(g, i, c5_gap=i * 3, rows_upto=5 if i != 6 else 4)   # seed 6: step-5 tree written, not committed
assert g.decide() is None and not g.DECISION.exists(), 'decided before seed 6 committed step 5'
assert g.gstep(g.seed_name(1)) == 5 and g.gstep(g.seed_name(6)) == 4, 'gstep must follow the ledger'
make_seed(g, 6, c5_gap=18, rows_upto=5)
d = g.decide()
assert d and d['top3'] == [1, 2, 3], d
print('1-2 OK: decision waits for all committed step-5 checkpoints; step count from ledger (global_step lagged)')

# 3. unit_running semantics
for out, expect in (('active', True), ('activating', True), ('deactivating', True), ('', True),
                    ('inactive', False), ('failed', False)):
    g.subprocess = types.SimpleNamespace(run=lambda *a, out=out, **k: types.SimpleNamespace(stdout=out + '\n'))
    assert g.unit_running('x') is expect, (out, expect)
print('3 OK: only explicit inactive/failed counts as not running')

# 4. one failing run does not block the others
started = []
g.unit_running = lambda name: False
real_prepare = g.prepare_clone
g.prepare_clone = lambda name, src: (_ for _ in ()).throw(RuntimeError('boom')) if name.endswith('c1-s1') else real_prepare(name, src)
g.subprocess = types.SimpleNamespace(run=lambda cmd, **k: started.append(cmd))
g.reconcile()
names = [c[3] for c in started]
assert 'r1-c1-s1' not in names and any(n.startswith('r1-c') for n in names) and 'r1-s9' in names, names
assert 'GRID ERROR r1-c1-s1' in g.LOG.read_text()
print(f'4 OK: failing clone skipped and logged; {len(names)} other runs started')

# 5. exact clone matching with a grid name containing '-c'
g2 = load('x-c1', tmp / 'b')
assert not g2.is_clone(g2.seed_name(3)) and g2.is_clone('x-c1-c2-s4') and not g2.is_clone('x-c1-s4')
print("5 OK: a grid name containing '-c' does not make seeds look like clones")
shutil.rmtree(tmp)
print('REVIEW-FIX TESTS PASSED')
