"""Directional (targeted) detector from synthetic training negatives.

Omnibus likelihood scores pay a dimension penalty: a shift of Mahalanobis size delta moves a chi2_p
score by ~delta^2 against a normal spread of ~sqrt(2p). A test along K << p physically motivated
directions does not. Directions:  w_k = (Sigma + lam I)^-1 delta_k  where Sigma is the covariance of
standardized NORMAL fit3 features (cosmology/baryon/Delta_z variation included, so w_k avoids
directions explainable by theta) and delta_k is the paired mean feature change of synthetic shift k
applied to fit3 maps (tshift/). Score: T = z^T C^-1 z over the K projections (C from calib3 normal maps),
mapped to a normal score and standardized on calib3. Everything per map; only fit3/calib3 used to fit.

Honest evaluation: leave-family-out -- when scoring shift family F, directions trained on F are dropped.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy import stats as sst

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from model import phase2_score  # noqa: E402

V3 = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3')
TS = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/statistics/tshift')
DS = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/statistics/dshift')
OUT = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/statistics/direct')
FAMS = {'baryS': ['bary9.0', 'supp0.15'], 'baryW': ['baryDMO', 'bary6.5', 'boost0.15'],
        'spread': ['spread0.6_t2', 'spread0.3_t2'], 'concen': ['concen0.6_t3', 'concen0.3_t2'],
        'ia': ['ia0.2'], 'peakcut': ['peakcut0.3'], 'psf': ['psfpat'], 'srcclust': ['srcclust0.15']}
DROP = {'stx_nois': [7, 8, 25, 26, 53]}


def load_src(src, name, root=V3):
    p = root / src / f'{name}.npz'
    if not p.exists() and root == V3:
        p = DS / src / f'{name}.npz'
    f = np.load(p)['feats']
    if f.ndim == 2:
        f = f[:, None]
    if src in DROP:
        f = np.delete(f, DROP[src], axis=-1)
    return f.astype(np.float64)


def load(srcs, name, root=V3):
    return np.concatenate([load_src(s, name, root) for s in srcs], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--srcs', default='stats,cnn0,stx_bisp,stx_prof,stx_pkcl,stx_ipow,stx_nois')
    ap.add_argument('--lam', type=float, default=0.05)
    ap.add_argument('--tag', default='d1')
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--combine', default='Sd,C')        # r2 models to average with
    args = ap.parse_args()
    srcs = args.srcs.split(',')
    out = OUT / args.tag; out.mkdir(parents=True, exist_ok=True)
    fit = load(srcs, 'fit'); n, v, d = fit.shape
    flat = fit.reshape(n * v, d)
    mu, sd = flat.mean(0), flat.std(0)
    keep = sd > 1e-6
    S = lambda x: ((x - mu) / np.where(keep, sd, 1))[..., keep]
    Z = S(flat)
    Sig = np.cov(Z.T)
    p = Sig.shape[0]
    Sinv = np.linalg.inv(Sig + args.lam * np.eye(p))
    normal_t = S(load(srcs, 'normal', TS)[:, 0])
    shifts_avail = [s for f in FAMS.values() for s in f if (TS / 'stats' / f'{s}.npz').exists()]
    W, names = [], []
    for s in shifts_avail:
        dlt = (S(load(srcs, s, TS)[:, 0]) - normal_t).mean(0)
        W.append(Sinv @ dlt); names.append(s)
    W = np.stack(W, 1)                                    # (p, K)
    print('directions', names, 'p', p, flush=True)
    cal = S(load(srcs, 'calib')).reshape(-1, p)
    pc = cal @ W
    pm, ps = pc.mean(0), pc.std(0)

    def T(x, cols):
        z = ((x @ W - pm) / ps)[:, cols]
        zc = ((pc - pm) / ps)[:, cols]
        C = np.cov(zc.T) + 1e-6 * np.eye(len(cols))
        t = np.einsum('ni,ij,nj->n', z, np.linalg.inv(C), z)
        tc = np.einsum('ni,ij,nj->n', zc, np.linalg.inv(C), zc)
        g = sst.norm.ppf(np.clip(sst.chi2.cdf(t, len(cols)), 1e-300, 1 - 1e-16))
        gc = sst.norm.ppf(np.clip(sst.chi2.cdf(tc, len(cols)), 1e-300, 1 - 1e-16))
        g = np.where(np.isfinite(g), g, 8.3)
        return (g - gc.mean()) / gc.std()

    audit = S(load(srcs, 'audit')[:, 0])
    eval_shifts = sorted({p.stem for p in (V3 / 'stats').glob('shift_*.npz')} |
                         {p.stem for p in (DS / 'stats').glob('shift_*.npz')})
    fam_of = {s: f for f, ss in FAMS.items() for s in ss}
    zcomb = {}
    for t in filter(None, args.combine.split(',')):
        zcomb[t] = dict(np.load(f'/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/statistics/runs/r2/z_{t}.npz'))
    allcols = list(range(len(names)))
    za_all = T(audit, allcols)
    res = {}
    for sh in eval_shifts:
        s = sh.removeprefix('shift_')
        fam = fam_of.get(s)
        cols = [i for i, nm in enumerate(names) if fam is None or fam_of[nm] != fam]
        if not cols:
            continue
        xa = S(load(srcs, sh)[:, 0])
        za = T(audit, cols); zs = T(xa, cols)
        r = dict(lofo=round(phase2_score(za, zs), 4), infam=round(phase2_score(za_all, T(xa, allcols)), 4))
        if zcomb and all(sh in zcomb[t] for t in zcomb):
            ca = np.mean([za] + [zcomb[t]['audit'] for t in zcomb], 0)
            cs = np.mean([zs] + [zcomb[t][sh] for t in zcomb], 0)
            r['lofo+combo'] = round(phase2_score(ca, cs), 4)
            ba = np.mean([zcomb[t]['audit'] for t in zcomb], 0); bs = np.mean([zcomb[t][sh] for t in zcomb], 0)
            r['combo_only'] = round(phase2_score(ba, bs), 4)
        res[s] = r
        print(f'{s:14s} fam={str(fam):9s} ' + ' '.join(f'{k}={v:.3f}' for k, v in r.items()), flush=True)
    print('normal audit quantiles (all dirs):', np.quantile(za_all, [.5, .9, .99, .995, .999]).round(3).tolist())
    keys = ['lofo', 'lofo+combo', 'combo_only']
    std_shifts = [s for s in res if (V3 / 'stats' / f'shift_{s}.npz').exists()]
    for k in keys:
        vals = [res[s][k] for s in std_shifts if k in res[s]]
        if vals:
            print(f'mean over {len(vals)} standard shifts {k}: {np.mean(vals):.4f}')
    (out / 'report.json').write_text(json.dumps(dict(directions=names, lam=args.lam, srcs=srcs, shifts=res), indent=1))
    np.savez(out / 'z_D.npz', audit=za_all, calib=T(cal, allcols),
             **{sh: T(S(load(srcs, sh)[:, 0]), allcols) for sh in eval_shifts},
             **({'test': T(S(load(srcs, 'test')[:, 0]), allcols)} if args.test else {}))


if __name__ == '__main__':
    main()
