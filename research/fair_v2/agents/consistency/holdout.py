"""Leave-cosmology-out (LCO) / leave-baryon-model-out (LBO) normal-tail checks (della-gh).

LCO: 10 cosmologies (rng 7) removed entirely from fitting AND calibration.
LBO: 32 realization indices = baryon models (rng 8) removed entirely.
For each family: refit model.Likelihood on the reduced fit3 (same hyper-parameters),
then score
  calib_in : reduced calib3 (2 views)            -> standardization
  in       : audit3 maps of kept cosmologies/baryons (1 view)
  out      : every map of the held-out units: fit3 view 0 + calib3 view 0 + audit3
Finally a cross stage stores every family energy at every other family's theta and
at the joint (product) theta, so consistency scores can be rebuilt. Resumable.
"""
import argparse
import pickle
import time

import numpy as np
import torch

import lib
from model import Likelihood
from fit_families import run_model


def split(mode):
    idx = {n: lib.load('stats', n, 'index') for n in ('fit', 'calib', 'audit')}
    if mode == 'lco':
        held = np.random.default_rng(7).choice(101, 10, replace=False)
        col = 0
    else:
        held = np.random.default_rng(8).choice(256, 32, replace=False)
        col = 1
    isout = {n: np.isin(v[:, col], held) for n, v in idx.items()}
    return held, isout


def sets_for(fam, isout):
    """Feature arrays for the holdout protocol: dict name -> (N, V, d)."""
    f = {n: lib.cols(fam, n) for n in ('fit', 'calib', 'audit')}
    return dict(fit=f['fit'][~isout['fit']], calib=f['calib'][~isout['calib']],
                **{'in': f['audit'][~isout['audit']]},
                out=np.concatenate([f['fit'][isout['fit'], :1], f['calib'][isout['calib'], :1],
                                    f['audit'][isout['audit']]], 0))


def labels_for(isout):
    L = {n: lib.labels(n) for n in ('fit', 'calib', 'audit')}
    return dict(fit=L['fit'][~isout['fit']], calib=L['calib'][~isout['calib']], **{'in': L['audit'][~isout['audit']]},
                out=np.concatenate([L['fit'][isout['fit']], L['calib'][isout['calib']], L['audit'][isout['audit']]]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['lco', 'lbo'], required=True)
    ap.add_argument('--fams', default='dog,cnn,psL,psS,onept,extrema,mf,small,mid,large')
    ap.add_argument('--budget', type=float, default=420)
    args = ap.parse_args()
    t_start = time.time()
    out = lib.OUT / 'holdout' / args.mode
    out.mkdir(parents=True, exist_ok=True)
    held, isout = split(args.mode)
    np.save(out / 'held.npy', held)
    fams = args.fams.split(',')
    for fam in fams:
        if (out / f'{fam}.npz').exists():
            continue
        if time.time() - t_start > args.budget:
            print('budget reached', flush=True); return
        t0 = time.time()
        S = sets_for(fam, isout)
        L = labels_for(isout)
        lk = Likelihood('cuda', 0.02).fit(S['fit'], L['fit'], steps=3000, width=256,
                                          log=lambda m: print(f'[{args.mode} {fam}] {m}', flush=True), jitter=0.1)
        res = {}
        for name in ('calib', 'in', 'out'):
            x = S[name]
            n, v, d = x.shape
            for k, val in run_model(lk, x.reshape(n * v, d)).items():
                res[f'{name}/{k}'] = val
            res[f'{name}/labels'] = np.repeat(L[name], v, 0)
        with open(out / f'model_{fam}.pkl', 'wb') as fh:
            pickle.dump(lk, fh)
        np.savez(out / f'{fam}.npz', **res)
        print(f'[{args.mode} {fam}] done {time.time()-t0:.0f}s', flush=True)
    # ---------------------------------------------------------------- cross stage
    if (out / 'cross.npz').exists() or not all((out / f'{f}.npz').exists() for f in fams):
        return
    from scalecons import joint_fit
    lks = {}
    for f in fams:
        with open(out / f'model_{f}.pkl', 'rb') as fh:
            lks[f] = pickle.load(fh)
    lk0 = lks[fams[0]]
    lo = torch.as_tensor(lk0.t_lo, dtype=torch.float32, device='cuda')
    hi = torch.as_tensor(lk0.t_hi, dtype=torch.float32, device='cuda')
    res = {}
    prev = {f: np.load(out / f'{f}.npz') for f in fams}
    for name in ('calib', 'in', 'out'):
        xs = {}
        for f in fams:
            x = sets_for(f, isout)[name]
            n, v, d = x.shape
            xs[f] = lks[f].standardize(x.reshape(n * v, d))
        th = {f: torch.as_tensor(prev[f][f'{name}/th'], device='cuda') for f in fams}
        scale = [f for f in ('small', 'mid', 'large', 'cnn') if f in fams]
        inits = [torch.as_tensor(np.mean([prev[f][f'{name}/reg'] for f in scale], 0), device='cuda'), th['small']]
        th['jointSC'] = joint_fit(lks, xs, scale, inits + [th['cnn']], lo, hi, lk0.prior)
        with torch.no_grad():
            for src, t in th.items():
                res[f'{name}/th_{src}'] = t.cpu().numpy()
                for f in fams:
                    res[f'{name}/E_{f}@{src}'] = lks[f].energy(xs[f], t).cpu().numpy().astype(np.float64)
                    res[f'{name}/cop_{f}@{src}'] = lks[f].copula_energy(xs[f], t).cpu().numpy().astype(np.float64)
        print('cross', name, flush=True)
    np.savez(out / 'cross.npz', **res)


if __name__ == '__main__':
    main()
