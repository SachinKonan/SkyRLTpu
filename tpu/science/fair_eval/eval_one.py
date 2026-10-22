"""Run one placer on one benchmark the way macro_place.evaluate does, then score it officially.

Placement runs in the method's own repository and environment (cwd = that repo), using its
loader and its placer file exactly as `evaluate <placer> -b <case>` would; only placer.place()
is timed, as in evaluate.py. The returned positions are saved unchanged (dtype included) and
re-scored by the pinned official challenge repository, so every method is graded by identical code.

Validity is recorded two ways, both on exactly what the placer returned:
  valid_official  the criterion the official evaluate.py reports and the leaderboard ranked by:
                  finite, right shape, fixed macros unchanged (allclose atol=1e-3, as
                  validate_placement), and zero hard-macro overlaps (compute_overlap_metrics);
  valid_strict    validate_placement() in full, which also rejects a macro edge any amount past
                  the canvas (ArchGen's official 0.9507 includes edges 1-2 float32 steps past it).
max_bound_excess records how far (microns) any macro edge lies outside the canvas.

NG45 designs are given to the placer under the Tier-2 names '<design>_ng45' (as the challenge's
benchmarks/processed/public/*.pt and scripts/evaluate_with_orfs.py name them); the official
rescoring loads them from the same files.

Outputs go to <out>/<TASK_ATTEMPT>/ so a retried task never overwrites an earlier attempt.

Usage: eval_one.py --placer PATH --case ibm01 --out DIR --official-python PY --official-repo DIR
(run with cwd = the method's repository)
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np

NG45 = {'ariane133', 'ariane136', 'mempool_tile', 'nvdla'}

SCORE = r'''
import json, sys, numpy as np, torch
from pathlib import Path
repo, case, positions, out = sys.argv[1:5]
sys.path.insert(0, repo)
from macro_place.loader import load_benchmark_from_dir, load_benchmark
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement
NG45 = {'ariane133', 'ariane136', 'mempool_tile', 'nvdla'}
if case in NG45:
    d = Path(repo) / 'external/MacroPlacement/Flows/NanGate45' / case / 'netlist/output_CT_Grouping'
    b, plc = load_benchmark(str(d / 'netlist.pb.txt'), str(d / 'initial.plc'), name=case)
else:
    b, plc = load_benchmark_from_dir(str(Path(repo) / 'external/MacroPlacement/Testcases/ICCAD04' / case))
pos = np.load(positions)
shape_ok = pos.shape == tuple(b.macro_positions.shape)
finite = bool(np.isfinite(pos).all())
t = torch.from_numpy(pos)   # exactly what the placer returned
fixed = b.macro_fixed
fixed_ok = bool(not fixed.any() or torch.allclose(b.macro_positions[fixed].to(t.dtype), t[fixed], atol=1e-3))
strict_ok, errors = validate_placement(t, b)
m = {k: float(v) for k, v in compute_proxy_cost(t, b, plc).items()}
half = b.macro_sizes.numpy().astype(np.float64) / 2
p64 = pos.astype(np.float64)
excess = max(0.0, float(np.max(half - p64)), float(np.max(p64 + half - np.array([b.canvas_width, b.canvas_height]))))
Path(out).write_text(json.dumps(dict(
    valid_official=bool(shape_ok and finite and fixed_ok and m['overlap_count'] == 0),
    valid_strict=bool(strict_ok and shape_ok and finite),
    fixed_unchanged=fixed_ok, max_bound_excess=excess, errors=[str(e) for e in errors][:10], metrics=m), indent=1))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--placer', required=True)
    ap.add_argument('--case', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--official-python', required=True)
    ap.add_argument('--official-repo', required=True)
    a = ap.parse_args()
    attempt = os.environ.get('TASK_ATTEMPT', 'single')
    out = Path(a.out) / attempt
    out.mkdir(parents=True, exist_ok=True)
    os.environ['MOMO_LOG'] = str(out / 'momo_log.json')   # read by the MOMO placer only
    sys.path.insert(0, str(Path.cwd()))
    from macro_place.evaluate import _load_placer
    from macro_place.loader import load_benchmark_from_dir, load_benchmark
    report = dict(case=a.case, attempt=attempt, job=os.environ.get('SLURM_JOB_ID'), host=socket.gethostname(),
                  started=time.strftime('%F %T'), placer=str(Path(a.placer).resolve()), repo=str(Path.cwd()))
    if a.case in NG45:   # same files as macro_place.evaluate's NG45_BENCHMARKS, Tier-2 name for the placer
        d = f'external/MacroPlacement/Flows/NanGate45/{a.case}/netlist/output_CT_Grouping'
        benchmark, _ = load_benchmark(f'{d}/netlist.pb.txt', f'{d}/initial.plc', name=f'{a.case}_ng45')
    else:
        benchmark, _ = load_benchmark_from_dir(f'external/MacroPlacement/Testcases/ICCAD04/{a.case}')
    report['benchmark_name'] = benchmark.name
    placer = _load_placer(Path(a.placer))
    start = time.time()
    placement = placer.place(benchmark)
    report['placer_seconds'] = time.time() - start
    report['within_1h'] = report['placer_seconds'] <= 3600
    pos = placement.detach().cpu().numpy() if hasattr(placement, 'detach') else np.asarray(placement)
    report['returned_dtype'] = str(pos.dtype)
    np.save(out / 'positions.npy', pos)
    (out / 'score_official.py').write_text(SCORE)
    subprocess.run([a.official_python, str(out / 'score_official.py'), a.official_repo, a.case,
                    str(out / 'positions.npy'), str(out / 'official.json')], check=True, timeout=900)
    report.update(json.loads((out / 'official.json').read_text()))
    (out / 'report.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(dict(case=a.case, attempt=attempt, valid_official=report['valid_official'],
                          valid_strict=report['valid_strict'], proxy=report['metrics']['proxy_cost'],
                          placer_seconds=round(report['placer_seconds'], 1))))


if __name__ == '__main__':
    main()
