"""Run one placer on one benchmark the way macro_place.evaluate does, then score it officially.

Placement runs in the method's own repository and environment (cwd = that repo), using its
loader and its placer file exactly as `evaluate <placer> -b <case>` would. The returned
positions are saved and re-scored by the pinned official challenge repository
(validate_placement + compute_proxy_cost), so every method is graded by identical code.

Usage: eval_one.py --placer PATH --case ibm01 --out DIR --official-python PY --official-repo DIR
(run with cwd = the method's repository)
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

SCORE = r'''
import json, sys, numpy as np, torch
from pathlib import Path
repo, case, positions, out = sys.argv[1:5]
sys.path.insert(0, repo)
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement
b, plc = load_benchmark_from_dir(str(Path(repo) / 'external/MacroPlacement/Testcases/ICCAD04' / case))
pos = np.load(positions)
assert pos.shape == tuple(b.macro_positions.shape), pos.shape
t = torch.from_numpy(pos)   # validate the placement in the dtype the placer returned, as evaluate.py does
fixed = b.macro_fixed.numpy()
ok, errors = validate_placement(t, b)
fixed_ok = bool(np.array_equal(pos[fixed], b.macro_positions.numpy()[fixed]))
m = {k: float(v) for k, v in compute_proxy_cost(t, b, plc).items()}
Path(out).write_text(json.dumps(dict(valid=bool(ok) and fixed_ok, fixed_unchanged=fixed_ok,
                                     errors=[str(e) for e in errors][:10], metrics=m), indent=1))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--placer', required=True)
    ap.add_argument('--case', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--official-python', required=True)
    ap.add_argument('--official-repo', required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path.cwd()))
    from macro_place.evaluate import _load_placer
    from macro_place.loader import load_benchmark_from_dir
    report = dict(case=a.case, placer=str(Path(a.placer).resolve()), repo=str(Path.cwd()))
    benchmark, _ = load_benchmark_from_dir(f'external/MacroPlacement/Testcases/ICCAD04/{a.case}')
    placer = _load_placer(Path(a.placer))
    start = time.time()
    placement = placer.place(benchmark)
    report['placer_seconds'] = time.time() - start
    # Keep the returned dtype: macro_place.evaluate validates the tensor as returned, and casting
    # float64 output to float32 can push an edge one float32 step outside the canvas.
    pos = placement.detach().cpu().numpy() if hasattr(placement, 'detach') else np.asarray(placement)
    report['returned_dtype'] = str(pos.dtype)
    np.save(out / 'positions.npy', pos)
    (out / 'score_official.py').write_text(SCORE)
    subprocess.run([a.official_python, str(out / 'score_official.py'), a.official_repo, a.case,
                    str(out / 'positions.npy'), str(out / 'official.json')], check=True, timeout=600)
    report.update(json.loads((out / 'official.json').read_text()))
    (out / 'report.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(dict(case=a.case, valid=report['valid'], proxy=report['metrics']['proxy_cost'],
                          placer_seconds=round(report['placer_seconds'], 1))))


if __name__ == '__main__':
    main()
