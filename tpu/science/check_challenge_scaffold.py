"""Real-netlist checks of numeric schema, seed legality, and HPWL parity."""
import contextlib
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
from challenge_contract import problem_from_native, reward, wirelength
from challenge_seed import place

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root/'.science/challenge-probe'))
import torch
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
out = root/'tpu/science/results/placement/scaffold-pilot'
out.mkdir(parents=True, exist_ok=True)
report = {'host': platform.node(), 'cases': [], 'cpu_cores': 4, 'job_memory_gib': 16}
for name in ['ibm01', 'ibm04', 'ibm08', 'ibm18']:
    row = {'name': name}
    try:
        folder = root/'.science/challenge-probe/external/MacroPlacement/Testcases/ICCAD04'/name
        row['data_sha256'] = {f: hashlib.sha256((folder/f).read_bytes()).hexdigest()
                              for f in ['netlist.pb.txt', 'initial.plc']}
        with (out/(name+'.log')).open('w') as log, contextlib.redirect_stdout(log):
            b, plc = load_benchmark_from_dir(str(folder))
            p = problem_from_native(b, plc)
            row.update(hard=b.num_hard_macros, soft=b.num_soft_macros,
                       nets=len(p['net_weights']), endpoints=len(p['pin_owner']),
                       nonunit_weights=int(np.count_nonzero(p['net_weights'] != 1)),
                       input_bytes=sum(v.nbytes for v in p.values() if isinstance(v, np.ndarray)))
            start = time.monotonic()
            positions = place(p, 42, time_budget_s=180)['positions']
            row['solve_seconds'] = time.monotonic()-start
            tensor = torch.from_numpy(positions)
            valid, errors = validate_placement(tensor, b)
            assert valid, errors
            start = time.monotonic()
            costs = compute_proxy_cost(tensor, b, plc)
            row['grading_seconds'] = time.monotonic()-start
            assert costs['overlap_count'] == 0, costs
            hpwl = wirelength(p, positions)
            np.testing.assert_allclose(hpwl, costs['wirelength_cost'], rtol=2e-6, atol=1e-8)
            row.update(valid=True, metrics={k: float(v) for k,v in costs.items()}, adapter_wirelength=hpwl,
                       reward=reward([costs['proxy_cost']], all_valid=True))
            # Real validator must reject deliberately corrupted outputs.
            broken = tensor.clone(); broken[0, 0] = float('nan')
            assert not validate_placement(broken, b)[0]
            broken = tensor.clone(); broken[0, 0] = -1
            assert not validate_placement(broken, b)[0]
            broken = tensor.clone(); broken[1] = broken[0]
            assert not validate_placement(broken, b)[0]
            assert reward([costs['proxy_cost']], all_valid=False) == 0
            row['invalid_output_checks_passed'] = True
            np.savez(out/(name+'-problem.npz'), **p)
            np.save(out/(name+'-seed.npy'), positions, allow_pickle=False)
    except Exception as exc:
        row.update(valid=False, error=f'{type(exc).__name__}: {exc}')
    report['cases'].append(row)
    report['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(row), flush=True)
assert all(row['valid'] for row in report['cases']), 'scaffold checks failed; see report'
