# Like dryrun.py, but runs the flow through floorplan (make ... floorplan) for one design, to check
# that the macro TCL places every macro and PDN generation succeeds.
import sys, json, shutil, subprocess, os
from pathlib import Path
import numpy as np, torch
O = Path('/scratch/gpfs/ZHUANGL/sk7524/placement-fair-eval/orfs')
design, method = sys.argv[1], sys.argv[2]
os.chdir(O / 'challenge'); sys.path.insert(0, str(O / 'challenge/scripts'))
import evaluate_with_orfs as E
root = O / 'fptest' / f'{method}-{design}'
def floorplan_only(design_dir, orfs_root, *a, **k):
    cmd = ['make', f'DESIGN_CONFIG=./designs/nangate45/{design_dir.name}/config.mk', 'OPENROAD_ARGS=-threads 4', 'floorplan']
    r = subprocess.run(cmd, cwd=orfs_root / 'flow', capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-1500:])
    return {'rc': r.returncode}
E.run_orfs_flow = floorplan_only
spec = json.load(open(f'/scratch/gpfs/ZHUANGL/sk7524/placement-fair-eval/pli-queue/done/eval-{method}-{design}-r1.json'))
att = spec['out'] + '/' + spec['attempts'][-1]['attempt']
shutil.rmtree(root, ignore_errors=True); (root / 'out').mkdir(parents=True); (root / 'ORFS').mkdir()
subprocess.run(['rsync', '-a', '--include', 'platforms/nangate45/***', '--exclude', 'platforms/*',
                '--include', 'designs/nangate45/***', '--exclude', 'designs/*',
                f'{O}/ORFS-src/flow/', f'{root}/ORFS/flow/'], check=True)
os.symlink('/OpenROAD-flow-scripts/tools', root / 'ORFS/tools')
torch.save(torch.from_numpy(np.load(att + '/positions.npy')), root / 'placement.pt')
print(E.evaluate_benchmark(f'{design}_ng45', root / 'ORFS', root / 'out', use_docker=False, placement_path=root / 'placement.pt'))
