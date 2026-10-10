"""Fit/evaluate conditional-likelihood detectors on v3 (map-level split) features.

--models "S=stats:all,C=cnn0:raw,SC=stats:all|cnn0:raw"
Each model concatenates the named sources' columns (same noisy maps), fits the
conditional likelihood on fit3 normal maps, standardizes the profile score on
calib3 normal maps, reports the audit3 normal tail and synthetic shifts, and
(with --test) writes per-map test scores. Combinations of fitted models are
evaluated with --combos "S+C" (mean of standardized z) or "max:S+C".
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


def load(root, src, name):
    with np.load(Path(root) / src / f'{name}.npz') as f:
        return {k: f[k] for k in f.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3')
    ap.add_argument('--out', default='/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3/runs/default')
    ap.add_argument('--models', default='S=stats:all')
    ap.add_argument('--combos', default='')
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--stat', default='copula', choices=['gauss', 'copula'])
    ap.add_argument('--eig-floor', type=float, default=0.02)
    ap.add_argument('--width', type=int, default=256)
    ap.add_argument('--jitter', type=float, default=0.1)
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    G, _ = groups()
    names = ['fit', 'calib', 'audit'] + sorted(p.stem for p in (Path(args.root) / 'stats').glob('shift_*.npz'))
    if args.test:
        names.append('test')
    cache = {}

    def source(src, name):
        key = (src, name)
        if key not in cache:
            cache[key] = load(args.root, src, name)
        return cache[key]

    def columns(spec, name):
        parts = []
        for item in spec.split('|'):
            src, subset = item.split(':')
            f = source(src, name)['feats']
            cols = np.arange(f.shape[-1]) if subset == 'raw' else np.concatenate([G[g] for g in subset.split('+')])
            parts.append(f[..., cols])
        return np.concatenate(parts, -1)

    ref = source('stats', 'audit')
    labels = {n: source('stats', n)['labels'] for n in ('fit', 'calib', 'audit')}
    shifts = [n for n in names if n.startswith('shift_')]
    report, zs = {}, {}
    for model in args.models.split(','):
        tag, spec = model.split('=')
        log = lambda m: print(f'[{tag}] {m}', flush=True)
        t0 = time.time()
        lk = Likelihood(args.device, args.eig_floor).fit(columns(spec, 'fit'), labels['fit'], steps=args.steps,
                                                         width=args.width, log=log, jitter=args.jitter)

        def score(name):
            f = columns(spec, name)
            n, v, d = f.shape
            raw, th = lk.profile(f.reshape(n * v, d))
            if args.stat == 'copula':
                raw = lk.last_copula
            return raw, th
        raw_c, th_c = score('calib')
        std = Standardizer().fit(raw_c, th_c, lk.t_mean, lk.t_std)
        z = {}
        for name in names[2:]:
            raw, th = score(name)
            z[name] = std(raw, th)
        z['calib'] = std(raw_c, th_c)
        zs[tag] = z
        res = dict(normal_quantiles=np.quantile(z['audit'], [.5, .9, .95, .99, .995, .999]).round(3).tolist(),
                   normal_max=round(float(z['audit'].max()), 2),
                   calib_quantiles=np.quantile(z['calib'], [.99, .995, .999]).round(3).tolist(),
                   shifts={s.removeprefix('shift_'): round(phase2_score(z['audit'], z[s]), 4) for s in shifts})
        res['shift_mean'] = round(float(np.mean(list(res['shifts'].values()))), 4)
        res['seconds'] = round(time.time() - t0, 1)
        report[tag] = res
        log(json.dumps({k: res[k] for k in ('normal_quantiles', 'normal_max', 'calib_quantiles', 'shift_mean')}))
        log(' '.join(f'{k}={v:.3f}' for k, v in res['shifts'].items()))
        with open(out / f'model_{tag}.pkl', 'wb') as fh:
            pickle.dump(dict(lk=lk, std=std, spec=spec, stat=args.stat), fh)
        np.savez(out / f'z_{tag}.npz', **z)
    for combo in filter(None, args.combos.split(',')):
        mode, _, body = combo.rpartition(':')
        parts = body.split('+')
        def comb(name):
            v = np.stack([zs[p][name] for p in parts])
            return v.max(0) if mode == 'max' else v.mean(0)
        za = comb('audit')
        res = dict(normal_quantiles=np.quantile(za, [.5, .9, .95, .99, .995, .999]).round(3).tolist(),
                   shifts={s.removeprefix('shift_'): round(phase2_score(za, comb(s)), 4) for s in shifts})
        res['shift_mean'] = round(float(np.mean(list(res['shifts'].values()))), 4)
        report['combo:' + combo] = res
        print(f'[combo {combo}] {json.dumps(res)}', flush=True)
        if args.test:
            np.save(out / f'test_combo_{combo.replace(":", "_")}.npy', comb('test'))
    (out / 'report.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
