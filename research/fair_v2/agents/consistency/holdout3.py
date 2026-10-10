"""Multi-set leave-cosmology-out (LCO) refits for member likelihoods (della-gh), resumable.

Set k (0, 1, 2): 20 cosmologies = permutation(rng 2026)[20k:20k+20], disjoint across sets.
For each family: refit on fit3 rows of the kept cosmologies (variant settings), then score
  calib : kept calib3 rows (2 views)                 -> standardization / covariances
  in    : kept audit3 rows (1 view)                  -> in-sample normal thresholds
  out   : held-out calib3 rows (2 views) + held-out audit3 rows -> LCO InD for the metric
  sh_<s>: the 19 shifts of the held-out audit3 rows -> LCO OoD
Variants: std (fit_v3 settings), cj (+Omega_m/S8 input jitter 0.15), cj2 (0.30), sm (width 128, 1500 steps).
A final cross stage stores cop/E of small and large at each other's best fit (for xSL).
"""
import argparse
import pickle
import time

import numpy as np
import torch

import lib
from jlik import JLikelihood

VARIANTS = {
    'std': dict(jm=(0., 0., 1., 1., .5), width=256, steps=3000),
    'cj': dict(jm=(1.5, 1.5, 1., 1., .5), width=256, steps=3000),
    'cj2': dict(jm=(3., 3., 1., 1., .5), width=256, steps=3000),
    'sm': dict(jm=(0., 0., 1., 1., .5), width=128, steps=1500),
}


def held_set(k):
    return np.random.default_rng(2026).permutation(101)[20 * k:20 * k + 20]


def masks(k):
    held = held_set(k)
    return {n: np.isin(lib.load('stats', n, 'index')[:, 0], held) for n in ('fit', 'calib', 'audit')}


def sets_for(fam, k, shifts=True):
    m = masks(k)
    f = {n: lib.cols(fam, n) for n in ('fit', 'calib', 'audit')}
    out = dict(fit=f['fit'][~m['fit']], calib=f['calib'][~m['calib']], **{'in': f['audit'][~m['audit']]},
               out=np.concatenate([f['calib'][m['calib']].reshape(-1, 1, f['calib'].shape[-1]),
                                   f['audit'][m['audit']]], 0))
    if shifts:
        for s in lib.SHIFTS:
            out['sh_' + s] = lib.cols(fam, 'shift_' + s)[m['audit']]
    return out


def labels_for(k):
    m = masks(k)
    L = {n: lib.labels(n) for n in ('fit', 'calib', 'audit')}
    return L['fit'][~m['fit']]


def score(lk, x):
    n, v, d = x.shape
    E, th = lk.profile(x.reshape(n * v, d))
    return dict(E=E, cop=lk.last_copula, th=((th - lk.t_mean) / lk.t_std).astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', type=int, required=True)
    ap.add_argument('--variant', default='std')
    ap.add_argument('--fams', required=True)
    ap.add_argument('--budget', type=float, default=420)
    args = ap.parse_args()
    t0 = time.time()
    out = lib.OUT / 'lco3' / f'set{args.set}_{args.variant}'
    out.mkdir(parents=True, exist_ok=True)
    V = VARIANTS[args.variant]
    for fam in args.fams.split(','):
        if (out / f'{fam}.npz').exists():
            continue
        if time.time() - t0 > args.budget:
            print('budget reached', flush=True); return
        t1 = time.time()
        S = sets_for(fam, args.set)
        lk = JLikelihood('cuda', 0.02, V['jm']).fit(S['fit'], labels_for(args.set), steps=V['steps'],
                                                    width=V['width'], log=lambda m: None, jitter=0.1)
        res = {}
        for name, x in S.items():
            if name == 'fit':
                continue
            for key, val in score(lk, x).items():
                res[f'{name}/{key}'] = val
        with open(out / f'model_{fam}.pkl', 'wb') as fh:
            pickle.dump(lk, fh)
        np.savez(out / f'{fam}.npz', **res)
        print(f'[set{args.set} {args.variant} {fam}] {time.time() - t1:.0f}s', flush=True)
    # cross stage for xSL
    if (out / 'cross.npz').exists() or not ((out / 'small.npz').exists() and (out / 'large.npz').exists()):
        return
    lks = {}
    for f in ('small', 'large'):
        with open(out / f'model_{f}.pkl', 'rb') as fh:
            lks[f] = pickle.load(fh)
    prev = {f: np.load(out / f'{f}.npz') for f in lks}
    S = {f: sets_for(f, args.set) for f in lks}
    res = {}
    for name in S['small']:
        if name == 'fit':
            continue
        for f, g in (('small', 'large'), ('large', 'small')):
            x = S[f][name]
            n, v, d = x.shape
            xs = lks[f].standardize(x.reshape(n * v, d))
            t = torch.as_tensor(prev[g][f'{name}/th'], device='cuda')
            with torch.no_grad():
                res[f'{name}/cop_{f}@{g}'] = lks[f].copula_energy(xs, t).cpu().numpy()
    np.savez(out / 'cross.npz', **res)
    print('cross done', flush=True)


if __name__ == '__main__':
    main()
