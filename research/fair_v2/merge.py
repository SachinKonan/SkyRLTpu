"""Weighted mean of standardized detector z-scores from different runs/agents.

Inputs: weight*name=path[::prefix]  (prefix selects keys '<prefix>/<set>' in component files).
All z arrays are per-map scores already standardized on calib3; the mean is a fixed per-map function.
"""
import argparse
import json

import numpy as np

from model import phase2_score

SIGNAL = ['mult+0.05', 'mult-0.05', 'smooth0.5', 'aniso0.7', 'supp0.15', 'boost0.15', 'peakcut0.3',
          'bary9.0', 'bary6.5', 'baryDMO', 'ia0.2']


def load(spec):
    w, rest = spec.split('*', 1) if '*' in spec else ('1', spec)
    name, path = rest.split('=', 1)
    path, _, prefix = path.partition('::')
    d = np.load(path)
    if prefix:
        d = {k.split('/', 1)[1]: d[k] for k in d.files if k.startswith(prefix + '/')}
    else:
        d = {k: d[k] for k in d.files}
    return float(w), name, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', nargs='+')
    ap.add_argument('--save-test', default='')
    args = ap.parse_args()
    parts = [load(s) for s in args.inputs]
    W = sum(w for w, _, _ in parts)
    keys = sorted(set.intersection(*(set(d) for _, _, d in parts)) - {'calib'})
    comb = lambda k: sum(w * d[k] for w, _, d in parts) / W
    za = comb('audit')
    shifts = {k.removeprefix('shift_'): round(phase2_score(za, comb(k)), 4) for k in keys if k.startswith('shift_')}
    sig = [shifts[s] for s in SIGNAL if s in shifts]
    noi = [v for s, v in shifts.items() if s not in SIGNAL]
    q = np.quantile(za, [.9, .99, .995, .999])
    print(json.dumps(dict(members=[f'{w}*{n}' for w, n, _ in parts], n_shifts=len(shifts),
                          mean=round(float(np.mean(list(shifts.values()))), 4),
                          signal=round(float(np.mean(sig)), 4), noise_sys=round(float(np.mean(noi)), 4),
                          q99_9_over_q90=round(float(q[3] / q[0]), 2), tail=q.round(3).tolist())))
    if args.save_test:
        t = comb('test')
        assert t.shape == (10000,) and np.isfinite(t).all()
        np.save(args.save_test, t)


if __name__ == '__main__':
    main()
