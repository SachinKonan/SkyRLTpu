"""Per-band profiling of the field-level features + theta-consistency score.

For each family (feature source/subset) fit the conditional likelihood on fit3 (as fit_v3),
profile every map -> theta_hat and copula energy. Then
  cons = log(1 + Mahalanobis^2 of the per-family theta deviations from their mean),
covariance from calib3 normal maps, standardized on calib3 by fit_v3's Standardizer
(location/scale quadratic in the mean theta). Everything per map; fitting on fit3/calib3 only.
Saves per-family theta/copula (fam/<tag>.npz) and z-scores (OUT/z_*.npz incl. test).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from groups import groups  # noqa: E402
from model import Likelihood, Standardizer, phase2_score  # noqa: E402

ROOT = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/diffusion/bandcons'))
    ap.add_argument('--fams', default='lo=difb_lo:raw,mid=difb_mid:raw,hi=difb_hi:raw,FA=difa:raw,C=cnn0:raw,Sd=stats:dog')
    ap.add_argument('--cons', default='lo+mid+hi;lo+mid+hi+C;lo+mid+hi+FA+C+Sd')
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    (args.out / 'fam').mkdir(parents=True, exist_ok=True)
    G, _ = groups()
    shifts = sorted(p.stem for p in (ROOT / 'stats').glob('shift_*.npz'))
    sets = ['calib', 'audit'] + shifts + ['test']
    lab = {n: np.load(ROOT / 'stats' / f'{n}.npz')['labels'] for n in ('fit', 'calib')}

    def columns(src, sub, name):
        f = np.load(ROOT / src / f'{name}.npz')['feats']
        return f if sub == 'raw' else f[..., np.concatenate([G[g] for g in sub.split('+')])]

    TH, COP = {}, {}
    for item in args.fams.split(','):
        tag, spec = item.split('=')
        src, sub = spec.split(':')
        fp = args.out / 'fam' / f'{tag}.npz'
        if fp.exists():
            d = np.load(fp)
            TH[tag] = {s: d[f'{s}/th'] for s in sets}; COP[tag] = {s: d[f'{s}/cop'] for s in sets}
            continue
        lk = Likelihood(dev, 0.02).fit(columns(src, sub, 'fit'), lab['fit'], steps=3000, width=256,
                                       log=lambda m: None, jitter=0.1)
        TH[tag], COP[tag] = {}, {}
        for s in sets:
            f = columns(src, sub, s)
            n, v, d = f.shape
            _, th = lk.profile(f.reshape(n * v, d))
            TH[tag][s] = th; COP[tag][s] = lk.last_copula
        np.savez(fp, **{f'{s}/th': TH[tag][s] for s in sets}, **{f'{s}/cop': COP[tag][s] for s in sets})
        print('fitted', tag, flush=True)
    tm, ts = lab['fit'].mean(0), lab['fit'].std(0)

    def std_z(raw, th):
        S = Standardizer().fit(raw['calib'], th['calib'], tm, ts)
        return {s: S(raw[s], th[s]) for s in sets}

    Z = {t: std_z(COP[t], TH[t]) for t in TH}
    for spec in args.cons.split(';'):
        fs = spec.split('+')
        vec, cen = {}, {}
        for s in sets:
            A = np.stack([(TH[f][s] - tm) / ts for f in fs], 1)            # (n, K, 5)
            vec[s] = (A - A.mean(1, keepdims=True))[:, 1:].reshape(len(A), -1)
            cen[s] = A.mean(1) * ts + tm
        C = np.cov(vec['calib'].T)
        C = 0.95 * C + 0.05 * np.diag(np.diag(C))
        Ci = np.linalg.inv(C)
        raw = {s: np.log1p(np.einsum('ni,ij,nj->n', vec[s], Ci, vec[s])) for s in sets}
        Z['cons:' + spec] = std_z(raw, cen)

    def ev(z):
        sh = {s.removeprefix('shift_'): round(phase2_score(z['audit'], z[s]), 4) for s in shifts}
        return dict(nq=np.quantile(z['audit'], [.99, .995, .999]).round(3).tolist(),
                    shift_mean=round(float(np.mean(list(sh.values()))), 4), shifts=sh)

    def mean_of(keys, w=None):
        w = w or [1.0] * len(keys)
        return {s: sum(wi * Z[k][s] for k, wi in zip(keys, w)) / sum(w) for s in sets}

    recipes = {
        'BASE Sd+C': (['Sd', 'C'], None),
        'Sd+C+FA': (['Sd', 'C', 'FA'], None),
    }
    for spec in args.cons.split(';'):
        recipes[f'Sd+C+.5cons({spec})'] = (['Sd', 'C', 'cons:' + spec], [1, 1, .5])
        recipes[f'Sd+C+FA+.5cons({spec})'] = (['Sd', 'C', 'FA', 'cons:' + spec], [1, 1, 1, .5])
        recipes[f'Sd+C+FA+1cons({spec})'] = (['Sd', 'C', 'FA', 'cons:' + spec], [1, 1, 1, 1])
    rep = {}
    for k in Z:
        rep[k] = ev(Z[k])
    for name, (keys, w) in recipes.items():
        Z['recipe:' + name] = mean_of(keys, w)
        rep['recipe:' + name] = ev(Z['recipe:' + name])
    for k, r in rep.items():
        print(f'{k:48s} m={r["shift_mean"]:.4f} q99/99.5/99.9={r["nq"]} ' +
              ' '.join(f'{a[:7]}={b:.3f}' for a, b in r['shifts'].items()), flush=True)
    (args.out / 'report.json').write_text(json.dumps(rep, indent=1))
    for k, z in Z.items():
        np.savez(args.out / f'z_{k.replace(":", "_").replace("+", "-").replace(" ", "_").replace("(", "").replace(")", "")}.npz', **z)


if __name__ == '__main__':
    main()
