"""Offline test of grid.py: fake 9 seed runs at step 5 (C5 known), check decision, prune/clone splits, clone dirs,
and the plan's targets. No Tinker, no systemd (unit_running stubbed)."""
import importlib.util, json, math, os, shutil, sys, tempfile
from pathlib import Path

tmp = Path(tempfile.mkdtemp(prefix='gridtest-', dir=os.environ.get('TMPDIR', '/tmp')))
os.environ.update(GRID='t1', GRID_RUNS=str(tmp), GRID_HOLD_AFTER='0')
spec = importlib.util.spec_from_file_location('grid', sys.argv[1])
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
g.unit_running = lambda name: False
started = []
g.subprocess = type('S', (), {'run': staticmethod(lambda cmd, **k: started.append(cmd)),
                              'PIPE': None})

# C5 at step 5 (x1e-6 above 0.38085): seed 4 best, seed 7 5 behind, seed 2 10 behind, others far
c5 = {1: 40, 2: 10, 3: 60, 4: 0, 5: 80, 6: 55, 7: 5, 8: 70, 9: 90}
for i, d in c5.items():
    ld = g.logdir(g.seed_name(i))
    (ld / 'member_inkling').mkdir(parents=True)
    val = -(0.38085 + d * 1e-6)
    tree = dict(step=5, initial_states=[], puct_T=123, puct_m={'a': val}, puct_n={'a': 7},
                states=[dict(id='a', value=val, code='x'), dict(id='b', value=val - 1e-4, code='y')])
    (ld / 'puct_sampler_step_000005.json').write_text(json.dumps(tree))
    rows = [dict(name=f'{k:06d}', batch=k, state_path=f'tinker://seed{i}/weights/{k:06d}') for k in range(1, 6)]
    (ld / 'member_inkling' / 'checkpoints.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (ld / 'global_step.json').write_text(json.dumps(dict(global_step=5)))

d = g.decide()
print('top3', d['top3'], 'weights', [round(x, 3) for x in d['weights']])
print('prune', d['prune_extra_steps'], 'sum', sum(d['prune_extra_steps'].values()))
print('clones', d['clone_copies'], 'sum', sum(d['clone_copies'].values()))
assert d['top3'] == [4, 7, 2] and sum(d['prune_extra_steps'].values()) == 45 and sum(d['clone_copies'].values()) == 6
assert d['weights_rule'] == 'one_bit_entropic' and abs(sum(d['weights']) - 1) < 1e-9
assert [round(x, 2) for x in d['weights']] == [0.88, 0.11, 0.01], d['weights']
assert d['prune_extra_steps'] == {'4': 25, '7': 20, '2': 0}, d['prune_extra_steps']
assert d['clone_copies'] == {'4': 5, '7': 1, '2': 0}, d['clone_copies']
assert all(5 + n <= 30 for n in d['prune_extra_steps'].values())
assert g.capped_split(45, [1.0, 0, 0], 25) == [25, 20, 0] and g.capped_split(45, [.34, .33, .33], 25) == [15, 15, 15]
again = g.decide()
assert again == d, 'decision must be written once and reused'

g.reconcile()
plan, _ = g.plan()
for name in sorted(plan):
    print(f'  {name:10s} target {plan[name][0]:2d} seed {plan[name][1]} env {plan[name][2]}')
assert plan['t1-s4'][0] == 30 and plan['t1-s7'][0] == 25 and plan['t1-s2'][0] == 30
assert plan['t1-s1'][0] == 30 and plan['t1-s9'][0] == 10
clones = [n for n in plan if '-c' in n]
assert len(clones) == 6 and all(plan[n][0] == 10 and plan[n][2] == ['TTD_FRESH_OPTIMIZER_AT_BATCH=5'] for n in clones)
c = clones[0]
cd = g.logdir(c)
tree = json.loads((cd / 'puct_sampler_step_000005.json').read_text())
row = json.loads((cd / 'member_inkling' / 'checkpoints.jsonl').read_text())
src = int(c.rsplit('-s', 1)[1])
assert tree['puct_n'] == {} and tree['puct_T'] == 0 and tree['puct_m'] and len(tree['states']) == 2
assert row['batch'] == 5 and row['state_path'] == f'tinker://seed{src}/weights/000005'
assert json.loads((cd / 'global_step.json').read_text())['global_step'] == 5
print('clone dir ok:', c, '->', row['state_path'])
print('start commands issued:', len(started), '| first:', ' '.join(started[0][2:]))
assert len(started) == 15
shutil.rmtree(tmp)
print('ALL GRID TESTS PASSED')
