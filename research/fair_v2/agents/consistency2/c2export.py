"""consistency2 stage 3: full-data components + consRplus, export recipes to consistency/export4/.

Fitting: likelihood members / theta views from consistency/fam (fit3), probes fit on fit3 (alpha by
cosmology-grouped CV on fit3), standardization + Mahalanobis covariance on calib3. Test only scored.
--recipe 'tag={"Dr":1,...}' (';'-separated) --views pden_b,pdif (consRplus extra views)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'consistency'))
import lib  # noqa: E402
from analyze import ev, mean_of, fam  # noqa: E402
from export3 import components, cons  # noqa: E402

PDIR = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/consistency2/probes')
RB = ['b0', 'b1', 'b2', 'b3', 'cnn', 'cnn1', 'pred0', 'pred1', 'dog', 'den_r', 'den_sd']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--recipe', required=True)
    ap.add_argument('--views', default='')
    ap.add_argument('--export', action='store_true')
    args = ap.parse_args()
    sets = lib.SETS + ['test']
    comp = components({})
    TH = {}
    for f in [x for x in RB if not x.startswith('pred')]:
        F = fam(f)
        TH[f] = {s: F[f'{s}/th'] for s in sets}
    TH['pred0'] = {s: lib.load('cnn0', s)[..., -5:].reshape(-1, 5) for s in sets}
    TH['pred1'] = {s: lib.load('cnn1', s)[..., -5:].reshape(-1, 5) for s in sets}
    views = [v for v in args.views.split(',') if v]
    for v in views:
        if v.startswith('p'):
            with np.load(PDIR / f'{v[1:]}_full.npz') as H:
                TH[v] = {s: H[s] for s in sets}
        elif v.startswith('L'):
            F = fam(v[1:])
            TH[v] = {s: F[f'{s}/th'] for s in sets}
    for v in TH:
        for s in sets:
            assert TH[v][s].shape == TH['dog'][s].shape, (v, s, TH[v][s].shape)
    if views:
        comp['consRplus'] = cons(TH, RB + views, sets)
    r = ev(comp['consR']); print(f'consR audit m15={r["shift_mean"]:.4f} new4={r.get("new_mean", 0):.3f} q99.9={r["nq"][4]:.2f}')
    if views:
        r = ev(comp['consRplus']); print(f'consRplus audit m15={r["shift_mean"]:.4f} new4={r.get("new_mean", 0):.3f} q99.9={r["nq"][4]:.2f}')
    d = lib.OUT / 'export4'
    d.mkdir(exist_ok=True)
    for spec in filter(None, args.recipe.split(';')):
        tag, _, js = spec.partition('=')
        w = json.loads(js)
        z = mean_of(*[comp[c] for c in w], w=list(w.values()))
        r = ev(z)
        print(f'{tag:30s} audit m15={r["shift_mean"]:.4f} new4={r.get("new_mean", float("nan")):.3f} '
              f'q99/99.5/99.9={r["nq"][2]:.2f}/{r["nq"][3]:.2f}/{r["nq"][4]:.2f} w={w}', flush=True)
        if args.export:
            assert z['test'].shape == (10000,) and np.isfinite(z['test']).all()
            np.save(d / f'test_z_{tag}.npy', z['test'].astype(np.float64))
            np.savez(d / f'z_{tag}.npz', **z)
            np.savez(d / f'components_{tag}.npz', **{f'{c}/{k}': v for c in w for k, v in comp[c].items()})
            (d / f'recipe_{tag}.txt').write_text(json.dumps(dict(weights=w, consRplus_views=RB + views if 'consRplus' in w else None)) + '\n')
            print('  exported', d / f'test_z_{tag}.npy', flush=True)


if __name__ == '__main__':
    main()
