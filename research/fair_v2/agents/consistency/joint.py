"""Joint fit of the product of per-family likelihoods (della-gh GPU).

theta* = argmin_theta sum_b E_b(theta) + prior(theta) over a partition of the
features into families (cross-family correlations ignored; calibrated on calib3).
Stores per map E_b(theta*) and cop_b(theta*) so that the split tension
  T = sum_b [E_b(theta*) - min_theta E_b(theta)]
(parameter-consistency likelihood-ratio) can be formed. Per-map only.
"""
import argparse
import pickle

import numpy as np
import torch

import lib

PARTS = {
    'P8': ['psL', 'psS', 'dir', 'onept', 'extrema', 'mf', 'dog', 'cnn'],
    'P4': ['psL', 'psS', 'fine', 'coarse'],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', default='P8')
    ap.add_argument('--loo', default='', help='leave this family out of theta* (predictive check)')
    args = ap.parse_args()
    fams = PARTS[args.part]
    tag = args.part + (f'_loo_{args.loo}' if args.loo else '')
    fam_dir = lib.OUT / 'fam'
    out = lib.OUT / 'joint'
    out.mkdir(parents=True, exist_ok=True)
    if (out / f'{tag}.npz').exists():
        return
    lks = {}
    for f in fams:
        with open(fam_dir / f'model_{f}.pkl', 'rb') as fh:
            lks[f] = pickle.load(fh)
    lk0 = lks[fams[0]]
    dev = lk0.dev
    lo = torch.as_tensor(lk0.t_lo, dtype=torch.float32, device=dev)
    hi = torch.as_tensor(lk0.t_hi, dtype=torch.float32, device=dev)
    res = {}
    for s in lib.SETS + ['test']:
        xs = {}
        for f in fams:
            c = lib.cols(f, s)
            n, v, d = c.shape
            xs[f] = lks[f].standardize(c.reshape(n * v, d))
        prev = {f: np.load(fam_dir / f'{f}.npz') for f in fams}

        used = [f for f in fams if f != args.loo]

        def total(t):
            return sum(lks[f].energy(xs[f], t) for f in used) + lk0.prior(t)
        best_l, best_t = None, None
        inits = [torch.as_tensor(np.mean([prev[f][f'{s}/reg'] for f in used], 0), device=dev)]
        inits += [torch.as_tensor(prev[f][f'{s}/th'], device=dev) for f in ('cnn', 'psS', 'onept') if f in used]
        if args.loo:
            inits = inits[:2]
        if 'cnn' not in fams:
            inits += [torch.as_tensor(prev[f][f'{s}/th'], device=dev) for f in fams[:2]]
        for t0 in inits:
            t = t0.clone().float().clamp(lo, hi).requires_grad_(True)
            opt = torch.optim.Adam([t], lr=0.05)
            for _ in range(200):
                l = total(t)
                opt.zero_grad(set_to_none=True); l.sum().backward(); opt.step()
                with torch.no_grad():
                    t.clamp_(lo, hi)
            with torch.no_grad():
                l = total(t)
            if best_l is None:
                best_l, best_t = l, t.detach()
            else:
                b = l < best_l
                best_l = torch.where(b, l, best_l); best_t = torch.where(b[:, None], t.detach(), best_t)
        with torch.no_grad():
            res[f'{s}/thstar'] = best_t.cpu().numpy()
            res[f'{s}/Ltot'] = best_l.cpu().numpy().astype(np.float64)
            res[f'{s}/prior'] = lk0.prior(best_t).cpu().numpy().astype(np.float64)
            for f in fams:
                res[f'{s}/E_{f}'] = lks[f].energy(xs[f], best_t).cpu().numpy().astype(np.float64)
                res[f'{s}/cop_{f}'] = lks[f].copula_energy(xs[f], best_t).cpu().numpy().astype(np.float64)
        print(s, flush=True)
    np.savez(out / f'{tag}.npz', **res)


if __name__ == '__main__':
    main()
