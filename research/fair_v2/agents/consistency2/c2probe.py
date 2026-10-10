"""consistency2 stage 1: ridge theta-probes on latent features (cached, resumable).

Probe p: standardized features -> 5 normalized parameters (fit-label mean/std), ridge with alpha chosen by
5-fold cosmology-grouped CV on fit3 (kept cosmologies only inside an LCO set).  Output per set name
(calib/in/out/sh_* for LCO set k; lib.SETS + test for the full fit) in normalized theta units.
Fitting uses fit3 only; nothing touches test except applying the frozen full-data probe.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'consistency'))
import lib  # noqa: E402
from holdout3 import masks, held_set  # noqa: E402

OUT = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/consistency2/probes')
PROBES = {'den_b': ('den_b', slice(None)), 'cnn0e': ('cnn0', slice(0, 256)), 'cnn1e': ('cnn1', slice(0, 256)),
          'dif': ('dif', slice(None)), 'difa': ('difa', slice(None)), 'difk': ('difk', slice(None)),
          'oe': ('oe', slice(None)), 'oee': ('oe', slice(23, 279)), 'oe_head': ('oe_head', slice(None)),
          'difcat': [('dif', slice(None)), ('dife', slice(None))], 'oeho': ('oe_ho', slice(None)),
          'oehoe': ('oe_ho', slice(17, 273)),
          'lat': [('den_b', slice(None)), ('dif', slice(None)), ('difa', slice(None)), ('difk', slice(None))],
          'denbdif': [('den_b', slice(None)), ('dif', slice(None)), ('dife', slice(None))]}
ALPHAS = np.logspace(-6, 1, 8)


def feats(p, name):
    spec = PROBES[p]
    if isinstance(spec, list):
        return np.concatenate([lib.load(s, name)[..., sl] for s, sl in spec], -1)
    src, sl = spec
    return lib.load(src, name)[..., sl]


def ridge_cv(X, y, groups, nfold=5, seed=0):
    """X (N,d) float64, y (N,5). Returns predictor, alpha, cv mse."""
    mu, sd = X.mean(0), X.std(0) + 1e-8
    A = (X - mu) / sd
    ym = y.mean(0)
    Y = y - ym
    ug = np.unique(groups)
    perm = np.random.default_rng(seed).permutation(ug)
    fold_of = {g: i % nfold for i, g in enumerate(perm)}
    fid = np.array([fold_of[g] for g in groups])
    Gs = [A[fid == j].T @ A[fid == j] for j in range(nfold)]
    bs = [A[fid == j].T @ Y[fid == j] for j in range(nfold)]
    Gt, bt = sum(Gs), sum(bs)
    N = len(A)
    I = np.eye(A.shape[1])
    mses = []
    for a in ALPHAS:
        err = 0.0
        for j in range(nfold):
            n_tr = N - (fid == j).sum()
            W = np.linalg.solve(Gt - Gs[j] + a * n_tr * I, bt - bs[j])
            err += ((A[fid == j] @ W - Y[fid == j]) ** 2).sum()
        mses.append(err / Y.size)
    i = int(np.argmin(mses))
    W = np.linalg.solve(Gt + ALPHAS[i] * N * I, bt)
    return (lambda Z: ((Z - mu) / sd) @ W + ym), float(ALPHAS[i]), float(mses[i])


def check_index(p):
    for n in ('fit', 'calib', 'audit'):
        spec = PROBES[p]
        srcs = [s for s, _ in spec] if isinstance(spec, list) else [spec[0]]
        for s in srcs:
            assert np.array_equal(lib.load(s, n, 'index'), lib.load('stats', n, 'index')), (s, n)


def fold_probe(p, k):
    m = masks(k)
    L = {n: lib.labels(n) for n in ('fit', 'calib', 'audit')}
    Lf = L['fit'][~m['fit']]
    tm, ts = Lf.mean(0), Lf.std(0)
    F = {n: feats(p, n) for n in ('fit', 'calib', 'audit')}
    Xf = F['fit'][~m['fit']]
    V = Xf.shape[1]
    yf = np.repeat((Lf - tm) / ts, V, 0)
    gf = np.repeat(lib.load('stats', 'fit', 'index')[~m['fit'], 0], V)
    f, a, mse = ridge_cv(Xf.reshape(-1, Xf.shape[-1]).astype(np.float64), yf, gf)
    d = lambda x: f(x.reshape(-1, x.shape[-1]).astype(np.float64)).astype(np.float32)
    out = {'calib': d(F['calib'][~m['calib']]), 'in': d(F['audit'][~m['audit']]),
           'out': np.concatenate([d(F['calib'][m['calib']]), d(F['audit'][m['audit']])])}
    for s in lib.SHIFTS:
        out['sh_' + s] = d(feats(p, 'shift_' + s)[m['audit']])
    # held-out calib MSE (LCO generalization of the probe)
    yo = np.concatenate([np.repeat((L['calib'][m['calib']] - tm) / ts, 2, 0), (L['audit'][m['audit']] - tm) / ts])
    out['_meta'] = np.array([a, mse, float(np.mean((out['out'] - yo) ** 2)),
                             float(np.mean((out['calib'] - np.repeat((L['calib'][~m['calib']] - tm) / ts, 2, 0)) ** 2))])
    return out


def full_probe(p):
    L = lib.labels('fit')
    tm, ts = L.mean(0), L.std(0)
    Xf = feats(p, 'fit')
    V = Xf.shape[1]
    yf = np.repeat((L - tm) / ts, V, 0)
    gf = np.repeat(lib.load('stats', 'fit', 'index')[:, 0], V)
    f, a, mse = ridge_cv(Xf.reshape(-1, Xf.shape[-1]).astype(np.float64), yf, gf)
    out = {}
    for s in lib.SETS + ['test']:
        x = feats(p, s)
        out[s] = f(x.reshape(-1, x.shape[-1]).astype(np.float64)).astype(np.float32)
    Lc = lib.labels('calib')
    out['_meta'] = np.array([a, mse, float(np.mean((out['calib'] - np.repeat((Lc - tm) / ts, 2, 0)) ** 2))])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probes', default=','.join(PROBES))
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--budget', type=float, default=480)
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    for p in args.probes.split(','):
        check_index(p)
        jobs = [('full', None)] if args.full else [(f'set{k}', k) for k in (0, 1, 2)]
        for tag, k in jobs:
            fn = OUT / f'{p}_{tag}.npz'
            if fn.exists():
                continue
            if time.time() - t0 > args.budget:
                print('budget reached', flush=True)
                return
            t1 = time.time()
            res = full_probe(p) if k is None else fold_probe(p, k)
            np.savez(fn, **res)
            print(f'[{p} {tag}] alpha={res["_meta"][0]:.0e} cvmse={res["_meta"][1]:.3f} '
                  f'{"outmse" if k is not None else "calibmse"}={res["_meta"][2]:.3f} {time.time() - t1:.0f}s', flush=True)


if __name__ == '__main__':
    main()
