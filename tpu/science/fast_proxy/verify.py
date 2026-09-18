"""Read real case inputs; compare helper with the independent pinned scorer.

Run from repository root: .science/venv/bin/python -m tpu.science.fast_proxy.verify
"""
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from . import Evaluator
from ..challenge_contract import problem_from_native, CASES


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / '.science/challenge-probe'))
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost
    keys = ('proxy_cost', 'wirelength_cost', 'density_cost', 'congestion_cost')
    failures = []
    for case in CASES:
        b, plc = load_benchmark_from_dir(str(root / '.science/challenge-probe/external/MacroPlacement/Testcases/ICCAD04' / case))
        p = problem_from_native(b, plc)
        pos = np.load(root / 'tpu/science/results/placement/gpu-suite-20260916' / ('xplace-' + case) / 'positions.npy')
        start = time.perf_counter()
        with Evaluator(p, pos) as ev:
            setup = time.perf_counter() - start
            errors = []
            layouts = [pos.copy()]
            movable = np.flatnonzero(~p['fixed'] & (np.arange(len(pos)) >= p['num_hard']))
            rng = np.random.default_rng(123)
            # Multiple changed soft blocks exercise shared-net accounting too.
            for _ in range(2):
                altered = pos.copy()
                ids = rng.choice(movable, min(8, len(movable)), replace=False)
                altered[ids] = np.clip(altered[ids] + rng.normal(size=(len(ids), 2)) * p['canvas'] * .002,
                                       p['sizes'][ids]/2, p['canvas']-p['sizes'][ids]/2)
                layouts.append(altered)
            for layout in layouts:
                actual = ev.evaluate(layout)
                expected = compute_proxy_cost(torch.from_numpy(layout.astype(np.float32)), b, plc)
                errors.append({k: abs(actual[k] - float(expected[k])) for k in keys})
            base = ev.score()
            mid = int(movable[0])
            new = np.clip(pos[mid] + .1, p['sizes'][mid]/2, p['canvas']-p['sizes'][mid]/2)
            altered = pos.copy(); altered[mid] = new
            full = ev.evaluate(altered)
            probe_start = time.perf_counter()
            drift = 0.
            for _ in range(200):
                got = ev.apply(mid, new)
                drift = max(drift, *(abs(got[k] - full[k]) for k in keys))
                restored = ev.revert()
                drift = max(drift, *(abs(restored[k] - base[k]) for k in keys))
            probe_seconds = (time.perf_counter() - probe_start) / 200
            start = time.perf_counter()
            for _ in range(10): ev.evaluate(pos)
            full_seconds = (time.perf_counter()-start)/10
            row = dict(case=case, setup_seconds=setup, full_evaluate_restore_seconds=full_seconds,
                       apply_revert_seconds=probe_seconds, max_incremental_error=drift,
                       parity_errors=errors)
            print(json.dumps(row), flush=True)
            if max(e[k] for e in errors for k in keys) >= 1e-5 or drift >= 1e-7:
                failures.append(case)
    if failures:
        raise AssertionError('Parity failed: ' + ', '.join(failures))


if __name__ == '__main__':
    main()
