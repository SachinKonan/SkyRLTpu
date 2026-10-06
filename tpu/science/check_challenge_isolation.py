"""Exercise the exact seed/prompt interface through the CPU namespace runner."""
import json
from pathlib import Path
import sys

import numpy as np
from isolation import Limits, python_mounts, run

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root/'.science/challenge-probe'))
import torch
from macro_place.loader import load_benchmark_from_dir
from macro_place.utils import validate_placement

torch.set_num_threads(4)
out = root/'tpu/science/results/placement/scaffold-isolated'
out.mkdir(parents=True, exist_ok=True)
reports = []
for name in ['ibm01', 'ibm04', 'ibm08', 'ibm18']:
    dest = out/name
    dest.mkdir(exist_ok=True)
    elapsed = run([sys.executable, '/runner.py', '175'],
        limits=Limits(180, 16, 4), log=dest/'candidate.log',
        readonly=python_mounts(sys.executable)+[
            (Path(__file__).with_name('challenge_seed.py'), '/candidate.py'),
            (Path(__file__).with_name('challenge_candidate_child.py'), '/runner.py'),
            (root/'tpu/science/results/placement/scaffold-pilot'/f'{name}-problem.npz', '/problem.npz')],
        writable=[(dest, '/output')],
        env={'OPENBLAS_NUM_THREADS':'4', 'OMP_NUM_THREADS':'4', 'MKL_NUM_THREADS':'4'})
    positions = np.load(dest/'positions.npy', allow_pickle=False)
    expected = np.load(root/'tpu/science/results/placement/scaffold-pilot'/f'{name}-seed.npy', allow_pickle=False)
    np.testing.assert_array_equal(positions, expected)
    b, _ = load_benchmark_from_dir(str(root/'.science/challenge-probe/external/MacroPlacement/Testcases/ICCAD04'/name))
    valid, errors = validate_placement(torch.from_numpy(positions), b)
    assert valid, errors
    reports.append({'case':name,'valid':True,'identical_to_direct_seed':True,'isolated_seconds':elapsed})
    print(json.dumps(reports[-1]), flush=True)
    (out/'report.json').write_text(json.dumps(reports,indent=2)+'\n')
