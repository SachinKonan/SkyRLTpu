"""Scale-consistency (Dai & Seljak style) cross-evaluations on della-gh.

dlogp_b(src) = -2 log p(x_b | theta_src) + 2 log p(x_b | theta_b)   (b in small, large, cnn ...)
where theta_b is family b's own profile MLE and theta_src comes from another
block or from all data (joint product of blocks, or the r1 full-covariance
models S / SC). Energies are the conditional-Gaussian -2 log p (quad + logdet);
copula energies are stored too. Writes OUT/scale/<tag>.npz. Per-map only.
"""
import argparse
import pickle

import numpy as np
import torch

import lib

FAMS = ['small', 'mid', 'large', 'cnn']
EXTRA = ['dog', 'cnn1']   # evaluated at every theta source, never part of the joint fits


def joint_fit(lks, xs, used, inits, lo, hi, prior):
    best_l, best_t = None, None
    for t0 in inits:
        t = t0.clone().float().clamp(lo, hi).requires_grad_(True)
        opt = torch.optim.Adam([t], lr=0.05)
        for _ in range(200):
            l = sum(lks[f].energy(xs[f], t) for f in used) + prior(t)
            opt.zero_grad(set_to_none=True); l.sum().backward(); opt.step()
            with torch.no_grad():
                t.clamp_(lo, hi)
        with torch.no_grad():
            l = sum(lks[f].energy(xs[f], t) for f in used) + prior(t)
        if best_l is None:
            best_l, best_t = l, t.detach()
        else:
            b = l < best_l
            best_l = torch.where(b, l, best_l); best_t = torch.where(b[:, None], t.detach(), best_t)
    return best_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fams', default=','.join(FAMS))
    ap.add_argument('--tag', default='scale')
    ap.add_argument('--extra', action='store_true', help='also dog/cnn1 families and CNN-prediction theta sources')
    args = ap.parse_args()
    base_fams = args.fams.split(',')
    fams = base_fams + (EXTRA if args.extra else [])
    fam_dir = lib.OUT / 'fam'
    out = lib.OUT / 'scale'
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'{args.tag}.npz'
    res = {}
    if path.exists():
        with np.load(path) as d:
            res = {k: d[k] for k in d.files}
    lks = {}
    for f in fams:
        with open(fam_dir / f'model_{f}.pkl', 'rb') as fh:
            lks[f] = pickle.load(fh)
    lk0 = lks[fams[0]]
    dev = lk0.dev
    lo = torch.as_tensor(lk0.t_lo, dtype=torch.float32, device=dev)
    hi = torch.as_tensor(lk0.t_hi, dtype=torch.float32, device=dev)
    r1th = {}
    for tag in ['S', 'SC', 'Sd', 'C']:
        p = lib.OUT / 'r1box' / f'{tag}.npz'
        if p.exists():
            r1th[tag] = np.load(p)
    stats_f = [f for f in base_fams if f != 'cnn']
    for s in lib.SETS + ['test']:
        if f'{s}/th_small' in res:
            continue
        prev = {f: np.load(fam_dir / f'{f}.npz') for f in fams}
        xs = {}
        for f in fams:
            c = lib.cols(f, s)
            n, v, d = c.shape
            xs[f] = lks[f].standardize(c.reshape(n * v, d))
        th = {f: torch.as_tensor(prev[f][f'{s}/th'], device=dev) for f in fams}
        inits = [torch.as_tensor(np.mean([prev[f][f'{s}/reg'] for f in base_fams], 0), device=dev), th['small']]
        th['jointS'] = joint_fit(lks, xs, stats_f, inits, lo, hi, lk0.prior)
        th['jointSC'] = joint_fit(lks, xs, base_fams, inits + [th['cnn']], lo, hi, lk0.prior)
        if args.extra:
            p0 = lib.load('cnn0', s)[..., -5:].reshape(-1, 5)
            p1 = lib.load('cnn1', s)[..., -5:].reshape(-1, 5)
            for nm, pv in (('pred0', p0), ('pred1', p1), ('predm', 0.5 * (p0 + p1))):
                th[nm] = torch.as_tensor(pv, dtype=torch.float32, device=dev).clamp(lo, hi)
        for tag, d in r1th.items():
            if f'{s}/th' in d.files:
                th['r1' + tag] = torch.as_tensor(d[f'{s}/th'], device=dev)
        with torch.no_grad():
            for src, t in th.items():
                res[f'{s}/th_{src}'] = t.cpu().numpy()
                for f in fams:
                    res[f'{s}/E_{f}@{src}'] = lks[f].energy(xs[f], t).cpu().numpy().astype(np.float64)
                    res[f'{s}/cop_{f}@{src}'] = lks[f].copula_energy(xs[f], t).cpu().numpy().astype(np.float64)
        print(s, flush=True)
        np.savez(path, **res)


if __name__ == '__main__':
    main()
