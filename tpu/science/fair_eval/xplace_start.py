"""One legal starting layout from upstream Xplace, using the submission's own pipeline code.

Runs submission/placer.py's xplace_gp (Bookshelf export -> upstream cuhk-eda Xplace global
placement -> unit calibration) and legalize for one case and one setting, then scores the raw
and legal layouts with the official validate_placement + compute_proxy_cost.
Writes raw_positions.npy, positions.npy and report.json into --out.

Usage: xplace_start.py --case ibm01 --config td075 --out DIR --xplace DIR --probe DIR
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--xplace', required=True)
    ap.add_argument('--probe', required=True)
    ap.add_argument('--legalize-seconds', type=float, default=120)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    os.environ['MOMO_XPLACE_ROOT'] = a.xplace
    sys.path.insert(0, a.probe); sys.path.insert(0, str(HERE / 'submission'))
    import torch
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost
    from macro_place.utils import validate_placement
    import placer as P
    report = dict(case=a.case, config=a.config, settings=P.XPLACE_CONFIGS[a.config], valid=False)
    t0 = time.time()
    b, plc = load_benchmark_from_dir(str(Path(a.probe) / 'external/MacroPlacement/Testcases/ICCAD04' / a.case))
    p = P.build_problem(b, plc)
    raw = P.xplace_gp(p, a.case, a.config, out, timeout_s=1500)
    report['xplace_seconds'] = time.time() - t0
    np.save(out / 'raw_positions.npy', raw)
    t1 = time.time()
    legal = P.legalize(dict(p, initial_positions=raw), a.legalize_seconds)
    report['legalize_seconds'] = time.time() - t1
    np.save(out / 'positions.npy', legal)
    t = torch.from_numpy(legal)
    ok, errors = validate_placement(t, b)
    report.update(valid=bool(ok), errors=[str(e) for e in errors][:5],
                  metrics={k: float(v) for k, v in compute_proxy_cost(t, b, plc).items()},
                  raw_metrics={k: float(v) for k, v in compute_proxy_cost(torch.from_numpy(raw), b, plc).items()},
                  total_seconds=time.time() - t0)
    (out / 'report.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(dict(case=a.case, config=a.config, valid=report['valid'],
                          proxy=report['metrics']['proxy_cost'], seconds=round(report['total_seconds']))))


if __name__ == '__main__':
    main()
