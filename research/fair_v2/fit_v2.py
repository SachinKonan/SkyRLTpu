"""Fit the conditional-Gaussian summary detector, evaluate, and score the test.

Normal maps only for every fitted quantity:
  fit split   -> mu(theta), D(theta), R, initial regressor
  calib split -> standardization of the profile score vs fitted theta
  audit split -> normal false-positive tail and synthetic-shift diagnostics
Public test maps are scored one map at a time with frozen artifacts.
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from groups import groups
from model import Likelihood, Standardizer, phase2_score


def load(feat_dir, name):
    with np.load(Path(feat_dir) / f'{name}.npz') as f:
        return {k: f[k] for k in f.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', default='/scratch/gpfs/ZHUANGL/sk7524/fair-data/feats-v2')
    ap.add_argument('--out', default='/scratch/gpfs/ZHUANGL/sk7524/fair-data/runs-v2/summary')
    ap.add_argument('--subsets', default='all')
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--stat', default='copula', choices=['gauss', 'copula'])
    ap.add_argument('--eig-floor', type=float, default=0.02)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    fit, calib, audit = (load(args.feats, n) for n in ('fit', 'calib', 'audit'))
    shifts = sorted(p.stem for p in Path(args.feats).glob('shift_*.npz'))
    shift_data = {s: load(args.feats, s) for s in shifts}
    test = load(args.feats, 'test') if args.test else None
    G, d = groups()
    report = {}
    for subset in args.subsets.split(','):
        dim = fit['feats'].shape[-1]
        cols = np.arange(dim) if subset == 'raw' else np.concatenate([G[g] for g in subset.split('+')])
        t0 = time.time()
        log = lambda m: print(f'[{subset}] {m}', flush=True)
        lk = Likelihood(args.device, args.eig_floor).fit(fit['feats'][..., cols], fit['labels'], steps=args.steps, log=log)
        # calibration on fresh realizations (all views)
        cf = calib['feats'][..., cols]
        n, v, _ = cf.shape
        raw_c, th_c = lk.profile(cf.reshape(n * v, -1))
        if args.stat == 'copula':
            raw_c = lk.last_copula
        std = Standardizer().fit(raw_c, th_c, lk.t_mean, lk.t_std)
        res = {}
        raw_a, th_a = lk.profile(audit['feats'][:, 0, cols])
        if args.stat == 'copula':
            raw_a = lk.last_copula
        z_a = std(raw_a, th_a)
        true_a = lk.energy_at(audit['feats'][:, 0, cols], audit['labels'])
        res['normal_quantiles'] = np.quantile(z_a, [.5, .9, .95, .99, .999]).round(3).tolist()
        res['normal_max'] = float(z_a.max())
        res['theta_rmse'] = (np.sqrt(((th_a - audit['labels']) ** 2).mean(0)) / lk.t_std).round(3).tolist()
        # per-cosmology false alarm spread at the pooled 1% threshold
        thr = np.quantile(z_a, .99)
        per = np.array([np.mean(z_a[audit['index'][:, 0] == c] > thr) for c in range(101)])
        res['fa1pct_by_cosmology_max'] = float(per.max())
        scores = {}
        for s in shifts:
            f = shift_data[s]['feats'][:, 0, cols]
            raw, th = lk.profile(f)
            if args.stat == 'copula':
                raw = lk.last_copula
            z = std(raw, th)
            scores[s] = phase2_score(z_a, z)
            scores[s + ':raw'] = phase2_score(raw_a, raw)
        res['shift_scores'] = {k: round(v, 4) for k, v in scores.items()}
        res['shift_mean'] = float(np.mean([scores[s] for s in shifts]))
        res['seconds'] = round(time.time() - t0, 1)
        report[subset] = res
        log(json.dumps({k: res[k] for k in ('normal_quantiles', 'theta_rmse', 'fa1pct_by_cosmology_max',
                                            'shift_mean', 'seconds')}))
        log(' '.join(f'{s.removeprefix("shift_")}={scores[s]:.3f}' for s in shifts))
        if test is not None:
            tf = test['feats'][:, 0, cols]
            raw_t, th_t = lk.profile(tf)
            if args.stat == 'copula':
                raw_t = lk.last_copula
            z_t = std(raw_t, th_t)
            np.savez(out / f'test_{subset}.npz', z=z_t, raw=raw_t, theta=th_t)
            np.savez(out / f'normal_{subset}.npz', z_audit=z_a, raw_audit=raw_a, theta_audit=th_a,
                     z_calib=std(raw_c, th_c), raw_calib=raw_c, theta_calib=th_c)
            with open(out / f'model_{subset}.pkl', 'wb') as fh:
                pickle.dump(dict(lk=lk, std=std, cols=cols), fh)
    (out / 'report.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
