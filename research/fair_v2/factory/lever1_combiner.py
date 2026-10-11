"""Lever 1: learned, regularized linear combiner over the v3-16b member scores (CPU, LCO fold caches).

  python factory/lever1_combiner.py [--lam 0.002]

Data: fair-data/factory/lever1_folds.npz (factory/lever1_data.py). Normals = each fold's held-out-universe normal
maps ('out'); positives = that fold's 19 synthetic shifts of the held-out audit maps. The score is model.py's
combine (sum w z / sum w; ranking is scale-free). Metric = the official 100-threshold mean TPR (G.phase2_score).

Two learners, both shrunk toward v3-16b's weights w0:
  (b) coordinate search over per-member multipliers of w0 on the exact metric, penalty lam * sum log2(mult)^2;
  (a) ridge logistic regression of shifted vs normal on top of the v3-16b score (bounded so weights stay >= 0).
Validation is nested: leave-one-universe-fold-out x leave-one-shift-family-out (30 cells), plus fold-only and
family-only CV; bootstrap over cosmologies (thresholds recomputed) for the per-fold difference vs v3-16b.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'agents' / 'consistency'))
import grade as G  # noqa: E402
import lib  # noqa: E402

DATA = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/factory')
MEMBERS = ['Sd', 'Dr', 'xSL', 'consP8', 'consRplus', 'N2', 'PsBh', 'Oh', 'Xdp', 'Ol']
W0 = {'Sd': .5, 'Dr': 2.5, 'xSL': 1., 'consP8': .5, 'consRplus': 3., 'N2': .5, 'PsBh': .5, 'Oh': 1.5, 'Xdp': 2.,
      'Ol': 3.}
FAMILIES = {
    'baryon': ['bary6.5', 'bary9.0', 'baryDMO'], 'power': ['supp0.15', 'boost0.15'], 'peaks': ['peakcut0.3'],
    'shear': ['mult+0.05', 'mult-0.05'], 'noise_amp': ['noise+0.05', 'noise-0.05'],
    'noise_shape': ['noise_t5', 'poisson120'], 'noise_spatial': ['noisegrad0.15', 'srcclust0.15'],
    'psf': ['smooth0.5', 'psfpat', 'aniso0.7'], 'ia': ['ia0.2'], 'gradient': ['gradient']}
assert sorted(sum(FAMILIES.values(), [])) == sorted(lib.SHIFTS)
MULTS = [0.0, 0.25, 0.5, 0.71, 1.0, 1.41, 2.0, 4.0, 8.0]
FOLDS = (0, 1, 2)


def load():
    z = np.load(DATA / 'lever1_folds.npz')
    F = {}
    for k in FOLDS:
        F[k] = dict(Z={n: np.stack([z[f'{k}/{m}/{n}'] for m in MEMBERS], 1)
                       for n in ['out'] + ['sh_' + s for s in lib.SHIFTS]},
                    cos_out=z[f'{k}/cos_out'], cos_sh=z[f'{k}/cos_sh'])
    return F


def wvec(w):
    return np.array([w[m] for m in MEMBERS], np.float64)


def metric_cells(F, w, cells):
    """Mean over (fold, shift) cells of the official metric for weights w (vector)."""
    vals = []
    for k, s in cells:
        Z = F[k]['Z']
        vals.append(G.phase2_score(Z['out'] @ w, Z['sh_' + s] @ w))
    return float(np.mean(vals)) if vals else float('nan')


def penalty(mult, lam):
    l2 = np.where(mult > 0, np.log2(np.maximum(mult, 1e-9)), -4.0)
    return lam * float(np.sum(l2 ** 2))


def fit_coord(F, cells, lam, sweeps=2):
    w0 = wvec(W0)
    mult = np.ones(len(MEMBERS))
    best = metric_cells(F, w0 * mult, cells) - penalty(mult, lam)
    for _ in range(sweeps):
        changed = False
        for j in range(len(MEMBERS)):
            cur = mult[j]
            for c in MULTS:
                if c == cur:
                    continue
                mult[j] = c
                if (w0 * mult).sum() <= 0:
                    continue
                v = metric_cells(F, w0 * mult, cells) - penalty(mult, lam)
                if v > best + 1e-6:
                    best, cur, changed = v, c, True
            mult[j] = cur
        if not changed:
            break
    return w0 * mult


def fit_logistic(F, cells, lam):
    """Ridge logistic on top of the v3-16b score: score = s0 + Z @ d, bounded so that w0/W0 + d >= 0."""
    w0 = wvec(W0)
    base = w0 / w0.sum()
    Xn, Xp = [], []
    for k in sorted({c[0] for c in cells}):
        Xn.append(F[k]['Z']['out'])
        for s in [c[1] for c in cells if c[0] == k]:
            Xp.append(F[k]['Z']['sh_' + s])
    Xn, Xp = np.clip(np.concatenate(Xn), -10, 30), np.clip(np.concatenate(Xp), -10, 30)
    X = np.concatenate([Xn, Xp])
    y = np.r_[np.zeros(len(Xn)), np.ones(len(Xp))]
    sw = np.where(y == 1, 0.5 / len(Xp), 0.5 / len(Xn))
    s0 = X @ base

    def f(p):
        a, d = p[0], p[1:]
        t = a + s0 + X @ d
        ll = np.logaddexp(0, -t) * y + np.logaddexp(0, t) * (1 - y)
        g = (1 / (1 + np.exp(-t)) - y) * sw
        return float(np.sum(sw * ll) + lam * np.sum(d ** 2)), np.r_[g.sum(), X.T @ g + 2 * lam * d]
    bounds = [(None, None)] + [(-b, None) for b in base]
    r = minimize(f, np.zeros(1 + len(MEMBERS)), jac=True, method='L-BFGS-B', bounds=bounds)
    w = base + r.x[1:]
    return np.maximum(w, 0) * w0.sum()


def boot_diff(F, k, wa, wb, B=200, seed=0):
    """Paired bootstrap over held-out cosmologies of fold k: metric(wa) - metric(wb), all 19 shifts."""
    rng = np.random.default_rng(seed + k)
    Z, co, cs = F[k]['Z'], F[k]['cos_out'], F[k]['cos_sh']
    cos = np.unique(co)
    out = []
    for _ in range(B):
        pick = rng.choice(cos, len(cos), replace=True)
        io = np.concatenate([np.nonzero(co == c)[0] for c in pick])
        ish = np.concatenate([np.nonzero(cs == c)[0] for c in pick])
        d = []
        for s in lib.SHIFTS:
            P = Z['sh_' + s][ish]
            N = Z['out'][io]
            d.append(G.phase2_score(N @ wa, P @ wa) - G.phase2_score(N @ wb, P @ wb))
        out.append(np.mean(d))
    return float(np.mean(out)), float(np.std(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lam', type=float, nargs='+', default=[0.0005, 0.002])
    ap.add_argument('--lam-log', type=float, default=1e-3)
    ap.add_argument('--boot', type=int, default=150)
    ap.add_argument('--no-logistic', action='store_true')
    ap.add_argument('--out', default='lever1_combiner.json')
    args = ap.parse_args()
    t0 = time.time()
    F = load()
    w0 = wvec(W0)
    all_cells = [(k, s) for k in FOLDS for s in lib.SHIFTS]
    res = dict(members=MEMBERS, w0=W0, families=FAMILIES)
    learners = {f'coord(lam={l})': (lambda cells, l=l: fit_coord(F, cells, l)) for l in args.lam}
    if not args.no_logistic:
        learners[f'logistic(lam={args.lam_log})'] = lambda cells: fit_logistic(F, cells, args.lam_log)
    for name, fit in learners.items():
        print(f'== {name}', flush=True)
        # nested: leave one fold x leave one family out
        nested = {}
        for k in FOLDS:
            for fam, shifts in FAMILIES.items():
                train = [(kk, s) for kk in FOLDS if kk != k for s in lib.SHIFTS if s not in shifts]
                w = fit(train)
                test = [(k, s) for s in shifts]
                nested[f'{k}/{fam}'] = dict(learned=metric_cells(F, w, test), v316b=metric_cells(F, w0, test),
                                            w=(w / w.sum() * w0.sum()).round(3).tolist())
        per_fold = {k: dict(learned=float(np.mean([nested[f'{k}/{f}']['learned'] for f in FAMILIES])),
                            v316b=float(np.mean([nested[f'{k}/{f}']['v316b'] for f in FAMILIES]))) for k in FOLDS}
        per_fam = {f: dict(learned=float(np.mean([nested[f'{k}/{f}']['learned'] for k in FOLDS])),
                           v316b=float(np.mean([nested[f'{k}/{f}']['v316b'] for k in FOLDS]))) for f in FAMILIES}
        # fold-only CV (all families), with a cosmology bootstrap of the difference
        fold_cv = {}
        for k in FOLDS:
            w = fit([(kk, s) for kk in FOLDS if kk != k for s in lib.SHIFTS])
            m, sd = boot_diff(F, k, w, w0, B=args.boot)
            fold_cv[k] = dict(learned=metric_cells(F, w, [(k, s) for s in lib.SHIFTS]),
                              v316b=metric_cells(F, w0, [(k, s) for s in lib.SHIFTS]), boot_diff=m, boot_sd=sd,
                              w=(w / w.sum() * w0.sum()).round(3).tolist())
        wfull = fit(all_cells)
        res[name] = dict(nested=nested, per_fold_nested=per_fold, per_family_nested=per_fam, fold_cv=fold_cv,
                         w_full=dict(zip(MEMBERS, (wfull / wfull.sum() * w0.sum()).round(3).tolist())),
                         in_sample_all=metric_cells(F, wfull, all_cells), v316b_all=metric_cells(F, w0, all_cells))
        nf = sum(v['learned'] > v['v316b'] for v in per_fold.values())
        nfam = sum(v['learned'] > v['v316b'] for v in per_fam.values())
        print(f'  nested per fold  ' + '  '.join(f"{k}: {v['learned']:.4f} vs {v['v316b']:.4f}" for k, v in per_fold.items()), flush=True)
        print(f'  nested per family ' + '  '.join(f"{f}: {v['learned'] - v['v316b']:+.4f}" for f, v in per_fam.items()), flush=True)
        print(f'  folds won {nf}/3, families won {nfam}/10', flush=True)
        print(f'  fold-CV ' + '  '.join(f"{k}: {v['learned'] - v['v316b']:+.4f} (boot {v['boot_diff']:+.4f}+-{v['boot_sd']:.4f})" for k, v in fold_cv.items()), flush=True)
        print(f"  full-data weights {res[name]['w_full']}  in-sample {res[name]['in_sample_all']:.4f} vs {res[name]['v316b_all']:.4f}  [{time.time()-t0:.0f}s]", flush=True)
        (DATA / args.out).write_text(json.dumps(res, indent=1))
    print('wrote', DATA / args.out)


if __name__ == '__main__':
    main()
