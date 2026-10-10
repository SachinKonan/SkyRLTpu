"""Cross-evaluation (della-gh): energy / copula energy of every family at every family's best fit
and at the CNN predictions (pred0, pred1, predm). Resumable per set.

--holdout lco|lbo uses the holdout refits (holdout/<mode>/model_<fam>.pkl) and the holdout sets
(calib, in, out) instead of the main ones.
"""
import argparse
import pickle

import numpy as np
import torch

import lib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fams', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--holdout', default='')
    args = ap.parse_args()
    fams = args.fams.split(',')
    if args.holdout:
        from holdout import split, sets_for
        mdir = lib.OUT / 'holdout' / args.holdout
        held, isout = split(args.holdout)
        sets = ['calib', 'in', 'out']
        feats = lambda f, s: sets_for(f, isout)[s]
        prevf = lambda f: np.load(mdir / f'{f}.npz')
        preds = lambda s: (sets_for('cnn', isout)[s][..., -5:].reshape(-1, 5),
                           sets_for('cnn1', isout)[s][..., -5:].reshape(-1, 5))
    else:
        mdir = lib.OUT / 'fam'
        sets = lib.SETS + ['test']
        feats = lambda f, s: lib.cols(f, s)
        prevf = lambda f: np.load(mdir / f'{f}.npz')
        preds = lambda s: (lib.load('cnn0', s)[..., -5:].reshape(-1, 5), lib.load('cnn1', s)[..., -5:].reshape(-1, 5))
    out = lib.OUT / 'cross'
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'{args.tag}.npz'
    res = {}
    if path.exists():
        with np.load(path) as d:
            res = {k: d[k] for k in d.files}
    lks = {}
    for f in fams:
        with open(mdir / f'model_{f}.pkl', 'rb') as fh:
            lks[f] = pickle.load(fh)
    lk0 = lks[fams[0]]
    dev = lk0.dev
    lo = torch.as_tensor(lk0.t_lo, dtype=torch.float32, device=dev)
    hi = torch.as_tensor(lk0.t_hi, dtype=torch.float32, device=dev)
    prev = {f: prevf(f) for f in fams}
    for s in sets:
        if f'{s}/done' in res:
            continue
        xs = {}
        for f in fams:
            c = feats(f, s)
            n, v, d = c.shape
            xs[f] = lks[f].standardize(c.reshape(n * v, d))
        th = {f: torch.as_tensor(prev[f][f'{s}/th'], device=dev) for f in fams}
        p0, p1 = preds(s)
        for nm, pv in (('pred0', p0), ('pred1', p1), ('predm', 0.5 * (p0 + p1))):
            th[nm] = torch.as_tensor(pv, dtype=torch.float32, device=dev).clamp(lo, hi)
        with torch.no_grad():
            for src, t in th.items():
                res[f'{s}/th_{src}'] = t.cpu().numpy()
                for f in fams:
                    res[f'{s}/E_{f}@{src}'] = lks[f].energy(xs[f], t).cpu().numpy().astype(np.float32)
                    res[f'{s}/cop_{f}@{src}'] = lks[f].copula_energy(xs[f], t).cpu().numpy().astype(np.float32)
        res[f'{s}/done'] = np.ones(1)
        np.savez(path, **res)
        print(s, flush=True)


if __name__ == '__main__':
    main()
