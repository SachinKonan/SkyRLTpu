"""Combine standardized per-map detector scores from different runs.

Each input is a z_<tag>.npz written by fit_v3.py (keys: calib, audit, shift_*, test),
all computed on the same normal maps / shifts. Combination rules are fixed
per-map functions (mean, max, Fisher-style sum of positive parts); normal
calibration uses calib3 only. Reports audit3 normal tail and shift scores and
optionally writes the combined test scores.
"""
import argparse
import json

import numpy as np

from model import phase2_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', nargs='+', help='tag=path/to/z_<tag>.npz')
    ap.add_argument('--rule', default='mean', choices=['mean', 'max', 'posmean', 'rms'])
    ap.add_argument('--weights', default='')
    ap.add_argument('--save-test', default='')
    args = ap.parse_args()
    tags, data = [], []
    for item in args.inputs:
        tag, path = item.split('=')
        tags.append(tag); data.append(dict(np.load(path)))
    w = np.array([float(x) for x in args.weights.split(',')]) if args.weights else np.ones(len(tags))
    w = w / w.sum()
    keys = sorted(set.intersection(*(set(d) for d in data)))

    def comb(key):
        v = np.stack([d[key] for d in data])
        # re-standardize each detector on calib3 normal maps so scales match
        v = (v - mus[:, None]) / sds[:, None]
        if args.rule == 'mean':
            return (w[:, None] * v).sum(0)
        if args.rule == 'max':
            return v.max(0)
        if args.rule == 'posmean':
            return (w[:, None] * np.maximum(v, 0)).sum(0)
        return np.sqrt((w[:, None] * v ** 2).sum(0))

    cal = np.stack([d['calib'] for d in data])
    mus, sds = np.median(cal, 1), np.std(cal, 1)
    za = comb('audit')
    shifts = [k for k in keys if k.startswith('shift_')]
    res = dict(tags=tags, rule=args.rule, weights=w.round(3).tolist(),
               normal_quantiles=np.quantile(za, [.5, .9, .95, .99, .995, .999]).round(3).tolist(),
               shifts={k.removeprefix('shift_'): round(phase2_score(za, comb(k)), 4) for k in shifts})
    res['shift_mean'] = round(float(np.mean(list(res['shifts'].values()))), 4)
    print(json.dumps(res))
    if args.save_test:
        np.save(args.save_test, comb('test'))


if __name__ == '__main__':
    main()
