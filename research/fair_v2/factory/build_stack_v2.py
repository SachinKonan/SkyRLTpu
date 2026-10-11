"""Stack v2 = stack v1 (v3-12 without Oh) + dirpower at its best full-mode configuration.

  python factory/build_stack_v2.py [--run fair-data/factory/runs/gh_dirpower_full]

Adds dirpower's per-fold misfit member ('Xdp', weight = its best misfit weight) and its theta vote
(ridge or profile, as chosen) as an extra consRplus view ('dirpower'), from the della full-mode grade.
Writes fair-data/factory/cache_v2/fold{0,1,2}.pkl + stack.json and prints the v2 baseline LCO.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grade as G  # noqa: E402

V1 = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/factory/cache')
V2 = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/factory/cache_v2')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, default=Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/factory/runs/gh_dirpower_full'))
    args = ap.parse_args()
    r = json.loads((args.run / 'result.json').read_text())
    w, vote = r['config']['misfit_weight'], r['config']['vote']
    stack = dict(G.STACK)
    views = list(G.VIEWS0)
    if w > 0:
        stack['Xdp'] = w
    if vote != 'none':
        views.append('dirpower')
    V2.mkdir(parents=True, exist_ok=True)
    allb, allv1 = [], []
    for k in G.FOLDS:
        with open(V1 / f'fold{k}.pkl', 'rb') as fh:
            C = pickle.load(fh)
        with open(args.run / f'fold{k}_full.pkl', 'rb') as fh:
            P = pickle.load(fh)
        assert P['take'].all(), 'dirpower full run must cover every held shift row'
        if w > 0:
            C['comp']['Xdp'] = P['mis']
        if vote != 'none':
            C['TH']['dirpower'] = P[vote]
        with open(V2 / f'fold{k}.tmp', 'wb') as fh:
            pickle.dump(C, fh)
        (V2 / f'fold{k}.tmp').replace(V2 / f'fold{k}.pkl')
        comp = dict(C['comp'])
        comp['consRplus'] = G.cons(C['TH'], views, G.NAMES)
        sh, exc = G.metrics(G.mean_of(*[comp[m] for m in stack], w=list(stack.values())))
        allb.append(sh)
        comp1 = dict(C['comp'])
        comp1['consRplus'] = G.cons(C['TH'], G.VIEWS0, G.NAMES)
        allv1.append(G.metrics(G.mean_of(*[comp1[m] for m in G.STACK], w=list(G.STACK.values())))[0])
    agg = lambda shs, which: float(np.mean([np.mean([s[x] for x in which]) for s in shs]))
    out = dict(stack=stack, views=views, source=str(args.run), dirpower_config=r['config'],
               base_all19=round(agg(allb, G.lib.SHIFTS), 4), base_visible=round(agg(allb, G.VISIBLE), 4),
               base_hidden=round(agg(allb, G.HIDDEN), 4), v1_all19=round(agg(allv1, G.lib.SHIFTS), 4),
               per_shift={s: round(float(np.mean([a[s] for a in allb])), 4) for s in G.lib.SHIFTS})
    (V2 / 'stack.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != 'per_shift'}, indent=1))


if __name__ == '__main__':
    main()
