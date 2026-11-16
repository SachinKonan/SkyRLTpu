"""Launch gate: with GRID_HOLD_AFTER=1 and no RELEASE file, all 9 seeds start with target 1; after RELEASE, full targets."""
import importlib.util, os, shutil, sys, tempfile
from pathlib import Path

tmp = Path(tempfile.mkdtemp(prefix='gridgate-', dir=os.environ.get('TMPDIR', '/tmp')))
os.environ.update(GRID='t2', GRID_RUNS=str(tmp), GRID_HOLD_AFTER='1', GRID_STAGGER='0')  # batching: test_grid_stagger.py
spec = importlib.util.spec_from_file_location('grid2', sys.argv[1])
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
g.unit_running = lambda name: False
cmds = []
g.subprocess = type('S', (), {'run': staticmethod(lambda cmd, **k: cmds.append(cmd)), 'PIPE': None})
g.reconcile()
targets = {c[3]: int(c[4]) for c in cmds}
print('held start targets:', sorted(set(targets.values())), 'for', len(targets), 'seeds')
assert len(targets) == 9 and set(targets.values()) == {1}
cmds.clear()
g.RELEASE.parent.mkdir(parents=True, exist_ok=True)
g.RELEASE.write_text('go')
g.reconcile()
targets = {c[3]: int(c[4]) for c in cmds}
print('released start targets:', dict(sorted(targets.items())))
assert targets['t2-s1'] == 30 and targets['t2-s5'] == 15 and targets['t2-s9'] == 10
shutil.rmtree(tmp)
print('GATE TEST PASSED')
