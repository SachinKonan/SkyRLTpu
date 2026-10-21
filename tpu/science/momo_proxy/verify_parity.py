"""Parity of momo_proxy against the official scorer (macro_place.objective.compute_proxy_cost).

For each IBM case, scores several layouts with both and reports the largest absolute
difference per cost term:
  initial   the frozen start the candidate receives
  starts    each extra starting layout in the problem
  outputs   finished placements given with --outputs (e.g. ablation positions.npy)
  random    random macro moves inside the canvas (overlaps allowed; the cost is still defined)
  walk      a random apply/commit/revert walk, checked at the end against a rebuild and the grader

Usage: python verify_parity.py --problems DIR --probe CHALLENGE_PROBE [--outputs DIR] [--cases ...]
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

KEYS = ('proxy_cost', 'wirelength_cost', 'density_cost', 'congestion_cost')


def official(b, plc, positions, compute_proxy_cost):
    import torch
    m = compute_proxy_cost(torch.from_numpy(np.asarray(positions, dtype=np.float32)), b, plc)
    return {k: float(m[k]) for k in KEYS}


def random_layout(problem, rng, frac):
    pos = np.array(problem['initial_positions'], dtype=np.float64)
    movable = np.flatnonzero(~np.asarray(problem['fixed'], dtype=bool))
    pick = rng.choice(movable, size=max(1, int(frac * len(movable))), replace=False)
    half = np.asarray(problem['sizes'])[pick] / 2
    canvas = np.asarray(problem['canvas'])
    pos[pick] = rng.uniform(half, canvas - half)
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--problems', required=True)
    ap.add_argument('--probe', required=True)
    ap.add_argument('--outputs')
    ap.add_argument('--cases', nargs='*')
    ap.add_argument('--random', type=int, default=4)
    ap.add_argument('--walk', type=int, default=3000)
    ap.add_argument('--report')
    ap.add_argument('--compare', help='directory of another Evaluator package (e.g. fast_proxy) to time on the same walk')
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.probe)))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost
    from momo_proxy import Evaluator
    import torch
    torch.set_num_threads(4)
    cases = a.cases or sorted(p.name.split('-')[0] for p in Path(a.problems).glob('ibm*-problem.npz'))
    rng = np.random.default_rng(0)
    report, worst = {}, {k: 0.0 for k in KEYS}
    for case in cases:
        with np.load(Path(a.problems) / f'{case}-problem.npz', allow_pickle=False) as d:
            problem = {k: d[k].copy() for k in d.files}
        for k in ('num_hard', 'wirelength_normalizer', 'congestion_smoothing_range'):
            problem[k] = problem[k].item()
        b, plc = load_benchmark_from_dir(str(Path(a.probe) / 'external/MacroPlacement/Testcases/ICCAD04' / case))
        layouts = [('initial', problem['initial_positions'])]
        for i, s in enumerate(problem.get('starting_layouts', [])):
            layouts.append((f'start{i}', s))
        if a.outputs:
            for f in sorted(Path(a.outputs).glob(f'*/{case}/positions.npy'))[:3]:
                layouts.append((f'output:{f.parent.parent.name}', np.load(f)))
        for i in range(a.random):
            layouts.append((f'random{i}', random_layout(problem, rng, [0.05, 0.3, 1.0, 1.0][i % 4])))
        rows = []
        for name, pos in layouts:
            ref = official(b, plc, pos, compute_proxy_cost)
            with Evaluator(problem, pos) as ev:
                got = ev.score()
            diff = {k: abs(got[k] - ref[k]) for k in KEYS}
            rows.append(dict(layout=name, official=ref, ours=got, abs_diff=diff))
            for k in KEYS:
                worst[k] = max(worst[k], diff[k])
        # random incremental walk from the initial layout
        ev = Evaluator(problem)
        pos = ev._positions.copy()
        movable = np.flatnonzero(~np.asarray(problem['fixed'], dtype=bool))
        half = np.asarray(problem['sizes']) / 2
        canvas = np.asarray(problem['canvas'])
        moves = [(int(i), rng.uniform(half[i], canvas - half[i]), rng.random() < 0.3)
                 for i in rng.choice(movable, size=a.walk)]
        def run_walk(e):
            t = time.perf_counter()
            for i, xy, keep in moves:
                e.apply(i, xy)
                e.commit() if keep else e.revert()
            return 1000 * (time.perf_counter() - t) / len(moves)
        per_probe_ms = run_walk(ev)
        inc = ev.score()
        reb = ev.rebuild()
        ref = official(b, plc, ev._positions, compute_proxy_cost)
        ev.close()
        other_ms = other_cost = None
        if a.compare:
            import importlib.util
            spec = importlib.util.spec_from_file_location('other_eval', Path(a.compare) / '__init__.py')
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            with mod.Evaluator(problem) as other:
                other_ms = run_walk(other)
                other_cost = other.rebuild()['proxy_cost']
        walk = dict(per_probe_ms=per_probe_ms, compare_per_probe_ms=other_ms, compare_final_proxy=other_cost,
                    incremental_vs_rebuild={k: abs(inc[k] - reb[k]) for k in KEYS},
                    rebuild_vs_official={k: abs(reb[k] - ref[k]) for k in KEYS})
        for k in KEYS:
            worst[k] = max(worst[k], walk['incremental_vs_rebuild'][k], walk['rebuild_vs_official'][k])
        report[case] = dict(layouts=rows, walk=walk)
        mx = max(max(r['abs_diff'].values()) for r in rows)
        print(f"{case}: {len(rows)} layouts, max |diff| {mx:.2e}; walk {per_probe_ms:.3f} ms/probe, "
              f"inc-vs-rebuild {max(walk['incremental_vs_rebuild'].values()):.2e}, "
              f"rebuild-vs-official {max(walk['rebuild_vs_official'].values()):.2e}"
              + (f"; compare {other_ms:.3f} ms/probe" if other_ms else ''), flush=True)
    print('worst abs diff per term:', json.dumps(worst))
    if a.report:
        Path(a.report).write_text(json.dumps(dict(worst=worst, cases=report), indent=1))


if __name__ == '__main__':
    main()
