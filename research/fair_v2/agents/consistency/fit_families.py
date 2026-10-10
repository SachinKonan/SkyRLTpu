"""Per-family conditional likelihoods + relaxed-box refits (run on della-gh GPU, resumable).

For every statistic family (lib.FAMILIES) fit model.Likelihood on fit3 exactly like
fit_v3 (steps 3000, width 256, jitter 0.1, eig floor 0.02). For every set
(calib, audit, shifts, test) store per map:
  E    profile minimum of gaussian energy + prior (box / hull / Delta_z prior)
  cop  copula energy at the box best fit (fit_v3's raw score)
  th   best-fit theta (normalized units)
  reg  regressor estimate of theta (normalized)
  Er, copr, thr : the same after relaxing the box (expand 0.3 range each side,
                  hull radius x3, Delta_z prior sigma x3), warm-started from th
Also re-profiles the r1 models (Sd, C, S, SC) with the relaxed box (mode r1).
Every quantity is a per-map function of one map.
"""
import argparse
import pickle
import sys
import time

import numpy as np
import torch

import lib
from model import Likelihood, DZ_SIGMA

ALLSETS = lib.SETS + ['test']


def prior_relaxed(lk, t, hull_mult, dz_mult):
    dz = t[:, 4] * float(lk.t_std[4]) + float(lk.t_mean[4])
    pen = (dz / (DZ_SIGMA * dz_mult)) ** 2
    dmin = torch.cdist(t[:, :2], lk.cosmo).min(1).values
    return pen + 50.0 * torch.relu(dmin - lk.hull_r * hull_mult) ** 2


def refine(lk, x, t_init, lo, hi, prior, steps=150, lr=0.05):
    t = t_init.clone().clamp(lo, hi).requires_grad_(True)
    opt = torch.optim.Adam([t], lr=lr)
    for _ in range(steps):
        l = lk.energy(x, t) + prior(t)
        opt.zero_grad(set_to_none=True); l.sum().backward(); opt.step()
        with torch.no_grad():
            t.clamp_(lo, hi)
    with torch.no_grad():
        return (lk.energy(x, t) + prior(t)), t.detach()


def run_model(lk, feats, expand=0.3, hull_mult=3.0, dz_mult=3.0):
    E, th = lk.profile(feats)
    cop = lk.last_copula
    x = lk.standardize(feats)
    with torch.no_grad():
        reg = lk.reg(x).cpu().numpy()
    tn = torch.as_tensor((th - lk.t_mean) / lk.t_std, dtype=torch.float32, device=lk.dev)
    lo = torch.as_tensor(lk.t_lo, dtype=torch.float32, device=lk.dev)
    hi = torch.as_tensor(lk.t_hi, dtype=torch.float32, device=lk.dev)
    w = hi - lo
    lo2, hi2 = lo - expand * w, hi + expand * w
    pr = lambda t: prior_relaxed(lk, t, hull_mult, dz_mult)
    best_l, best_t = refine(lk, x, tn, lo2, hi2, pr)
    l2, t2 = refine(lk, x, torch.as_tensor(reg, device=lk.dev), lo2, hi2, pr)
    better = l2 < best_l
    best_l = torch.where(better, l2, best_l); best_t = torch.where(better[:, None], t2, best_t)
    # never report a relaxed optimum worse than the box one (box is a subset)
    with torch.no_grad():
        box_relaxed_val = lk.energy(x, tn) + pr(tn)
        worse = box_relaxed_val < best_l
        best_l = torch.where(worse, box_relaxed_val, best_l); best_t = torch.where(worse[:, None], tn, best_t)
        copr = lk.copula_energy(x, best_t).cpu().numpy()
    return dict(E=E, cop=cop, th=((th - lk.t_mean) / lk.t_std).astype(np.float32), reg=reg.astype(np.float32),
                Er=best_l.cpu().numpy().astype(np.float64), copr=copr.astype(np.float64),
                thr=best_t.cpu().numpy().astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='families', choices=['families', 'r1'])
    ap.add_argument('--only', default='')
    ap.add_argument('--budget', type=float, default=480, help='seconds; stop starting new work after this')
    args = ap.parse_args()
    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False
    out = lib.OUT / ('fam' if args.mode == 'families' else 'r1box')
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    if args.mode == 'families':
        todo = [f for f in lib.FAMILIES if not args.only or f in args.only.split(',')]
        for fam in todo:
            if time.time() - t_start > args.budget:
                print('budget reached', flush=True); break
            t0 = time.time()
            log = lambda m: print(f'[{fam}] {m}', flush=True)
            if (out / f'{fam}.npz').exists():
                # incremental: score only sets (e.g. newly added shifts) that are missing
                with np.load(out / f'{fam}.npz') as d:
                    res = {k: d[k] for k in d.files}
                missing = [s for s in ALLSETS if f'{s}/E' not in res]
                if not missing:
                    continue
                with open(out / f'model_{fam}.pkl', 'rb') as fh:
                    lk = pickle.load(fh)
                for s in missing:
                    f = lib.cols(fam, s)
                    n, v, d = f.shape
                    for k, val in run_model(lk, f.reshape(n * v, d)).items():
                        res[f'{s}/{k}'] = val
                np.savez(out / f'{fam}.npz', **res)
                log(f'added {missing} in {time.time()-t0:.0f}s')
                continue
            lk = Likelihood(dev, 0.02).fit(lib.cols(fam, 'fit'), lib.labels('fit'), steps=3000, width=256,
                                           log=log, jitter=0.1)
            res = {}
            for s in ALLSETS:
                f = lib.cols(fam, s)
                n, v, d = f.shape
                r = run_model(lk, f.reshape(n * v, d))
                for k, val in r.items():
                    res[f'{s}/{k}'] = val
            with open(out / f'model_{fam}.pkl', 'wb') as fh:
                pickle.dump(lk, fh)
            np.savez(out / f'{fam}.npz', **res)
            log(f'done in {time.time()-t0:.0f}s')
    else:
        spec = {'Sd': ('stats', lib.G['dog']), 'C': ('cnn0', np.arange(261)), 'S': ('stats', lib.G['all']),
                'SC': None}
        for tag in ['Sd', 'C', 'S', 'SC']:
            if args.only and tag not in args.only.split(','):
                continue
            if (out / f'{tag}.npz').exists():
                continue
            if time.time() - t_start > args.budget:
                print('budget reached', flush=True); break
            t0 = time.time()
            with open(lib.ROOT / 'runs' / 'r1' / f'model_{tag}.pkl', 'rb') as fh:
                m = pickle.load(fh)
            lk = m['lk']
            print(tag, m['spec'], flush=True)
            res = {}
            for s in ALLSETS:
                parts = []
                for item in m['spec'].split('|'):
                    src, sub = item.split(':')
                    f = lib.load(src, s)
                    c = np.arange(f.shape[-1]) if sub == 'raw' else np.concatenate([lib.G[g] for g in sub.split('+')])
                    parts.append(f[..., c])
                f = np.concatenate(parts, -1)
                n, v, d = f.shape
                r = run_model(lk, f.reshape(n * v, d))
                for k, val in r.items():
                    res[f'{s}/{k}'] = val
            np.savez(out / f'{tag}.npz', **res)
            print(f'[{tag}] done in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
