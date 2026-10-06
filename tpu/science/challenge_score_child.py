"""Trusted CPU scoring process; candidate source is never imported here."""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True); p.add_argument('--case', required=True)
    p.add_argument('--positions', required=True); p.add_argument('--result', required=True)
    args = p.parse_args(); root = Path(args.root)
    started = time.monotonic()
    import numpy as np
    import torch
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    sys.path.insert(0, str(root/'.science/challenge-probe'))
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost
    from macro_place.utils import validate_placement
    folder = root/'.science/challenge-probe/external/MacroPlacement/Testcases/ICCAD04'/args.case
    b, plc = load_benchmark_from_dir(str(folder))
    f = Path(args.positions)
    if f.stat().st_size > 2*1024**2: raise ValueError('oversized candidate output')
    positions = np.load(f, allow_pickle=False)
    if positions.dtype.kind not in 'fiu': raise ValueError('nonnumeric output')
    if positions.shape != tuple(b.macro_positions.shape): raise ValueError('wrong placement shape')
    if not np.isfinite(positions).all(): raise ValueError('nonfinite placement')
    fixed = b.macro_fixed.numpy()
    if not np.array_equal(positions[fixed], b.macro_positions.numpy()[fixed]):
        raise ValueError('fixed positions changed')
    tensor = torch.from_numpy(positions.astype(np.float32))
    valid, errors = validate_placement(tensor, b)
    if not valid: raise ValueError('illegal placement: '+str(errors))
    metrics = {k:float(v) for k,v in compute_proxy_cost(tensor,b,plc).items()}
    if not all(np.isfinite(v) for v in metrics.values()) or metrics['overlap_count'] != 0:
        raise ValueError('nonfinite proxy or hard overlap')
    metrics['grading_seconds'] = time.monotonic()-started
    Path(args.result).write_text(json.dumps(metrics, allow_nan=False))


if __name__ == '__main__': main()
