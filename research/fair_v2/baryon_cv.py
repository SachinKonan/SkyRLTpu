"""Generalization to UNSEEN baryon models (v2 realization split = held-out models).

Fits the conditional likelihood on v2 'fit' (192 baryon models) with several
regularization settings and reports, on v2 'audit' (32 unseen models):
  * median / 99% / 99.9% of the energy at the TRUE parameters
  * profile-score tail after standardization on v2 'calib' (32 other unseen models)
"""
import argparse
import json

import numpy as np
import torch

from groups import groups
from model import Likelihood, Standardizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', default='/scratch/gpfs/ZHUANGL/sk7524/fair-data/feats-v2')
    ap.add_argument('--subset', default='all')
    ap.add_argument('--grid', default='256:0:6000:0.05,256:0:6000:0.1,512:0:6000:0.1,256:0:6000:0.2,256:0:3000:0.1')
    args = ap.parse_args()
    G, _ = groups()
    cols = np.concatenate([G[g] for g in args.subset.split('+')])
    load = lambda n: dict(np.load(f'{args.feats}/{n}.npz'))
    fit, calib, audit = load('fit'), load('calib'), load('audit')
    for setting in args.grid.split(','):
        width, wd, steps, jit = setting.split(':')
        lk = Likelihood('cuda')
        lk.fit_wd = float(wd)
        lk.fit(fit['feats'][..., cols], fit['labels'], steps=int(steps), width=int(width), log=lambda m: None,
               weight_decay=float(wd), jitter=float(jit))
        e_true = lk.energy_at(audit['feats'][:, 0, cols], audit['labels'])
        e_fit = lk.energy_at(fit['feats'][::5, 0, cols], fit['labels'][::5])
        cf = calib['feats'][..., cols]; n, v, d = cf.shape
        raw_c, th_c = lk.profile(cf.reshape(n * v, d)); raw_c = lk.last_copula
        std = Standardizer().fit(raw_c, th_c, lk.t_mean, lk.t_std)
        raw_a, th_a = lk.profile(audit['feats'][:, 0, cols]); raw_a = lk.last_copula
        z = std(raw_a, th_a)
        print(json.dumps(dict(setting=setting,
                              e_true_unseen=np.quantile(e_true, [.5, .99, .999]).round(1).tolist(),
                              e_true_seen=np.quantile(e_fit, [.5, .99, .999]).round(1).tolist(),
                              z_unseen=np.quantile(z, [.5, .99, .995, .999]).round(2).tolist())), flush=True)


if __name__ == '__main__':
    main()
