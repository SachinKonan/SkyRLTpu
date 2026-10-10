"""fit_v3.py with two conveniences (same Likelihood / Standardizer / phase2_score):
  * evaluates only the shift sets present in EVERY source used (new shifts may lag behind);
  * --reuse: models whose z_<tag>.npz already exists in --out with the same spec are not refit,
    so combos can mix models fitted in earlier invocations.
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from groups import groups  # noqa: E402
from model import Likelihood, Standardizer, phase2_score  # noqa: E402

KEY = ['bary9.0', 'bary6.5', 'baryDMO', 'peakcut0.3', 'srcclust0.15', 'poisson120', 'ia0.2', 'psfpat']


def load(root, src, name):
    with np.load(Path(root) / src / f'{name}.npz') as f:
        return {k: f[k] for k in f.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3')
    ap.add_argument('--out', required=True)
    ap.add_argument('--models', default='')
    ap.add_argument('--combos', default='')
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--eig-floor', type=float, default=0.02)
    ap.add_argument('--width', type=int, default=256)
    ap.add_argument('--jitter', type=float, default=0.1)
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--reuse', action='store_true')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    G, _ = groups()
    specs = dict(m.split('=') for m in filter(None, args.models.split(',')))
    srcs = {item.split(':')[0] for spec in specs.values() for item in spec.split('|')} | {'stats'}
    shift_sets = None
    for s in srcs:
        have = {p.stem for p in (Path(args.root) / s).glob('shift_*.npz')}
        shift_sets = have if shift_sets is None else shift_sets & have
    shifts = sorted(shift_sets)
    names = ['fit', 'calib', 'audit'] + shifts + (['test'] if args.test else [])
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
            if subset == 'raw':
                cols = np.arange(f.shape[-1])
            elif subset[0] in 'xk' and subset[1:2].isdigit():
                # x<i.j-k> drops columns, k<i.j-k> keeps columns ('-' = inclusive range)
                sel = []
                for tok in subset[1:].split('.'):
                    a, _, b = tok.partition('-')
                    sel += list(range(int(a), int(b or a) + 1))
                cols = np.setdiff1d(np.arange(f.shape[-1]), sel) if subset[0] == 'x' else np.array(sel)
            else:
                cols = np.concatenate([G[g] for g in subset.split('+')])
            parts.append(f[..., cols])
        return np.concatenate(parts, -1)

    labels = {n: source('stats', n)['labels'] for n in ('fit', 'calib', 'audit')}
    for n in ('fit', 'calib', 'audit'):
        for s in srcs:
            assert np.array_equal(source(s, n)['index'], source('stats', n)['index']), (s, n)
    rp = out / 'report.json'
    report = json.loads(rp.read_text()) if rp.exists() else {}
    zs = {}
    for tag, spec in specs.items():
        zf, mf = out / f'z_{tag}.npz', out / f'spec_{tag}.txt'
        if args.reuse and zf.exists() and mf.exists() and mf.read_text() == spec:
            z = dict(np.load(zf))
            if all(n in z for n in names[2:]):
                zs[tag] = z
                print(f'[{tag}] reused', flush=True)
                continue
        log = lambda m, tag=tag: print(f'[{tag}] {m}', flush=True)
        t0 = time.time()
        lk = Likelihood(args.device, args.eig_floor).fit(columns(spec, 'fit'), labels['fit'], steps=args.steps,
                                                         width=args.width, log=log, jitter=args.jitter)

        def score(name):
            f = columns(spec, name)
            n, v, d = f.shape
            _, th = lk.profile(f.reshape(n * v, d))
            return lk.last_copula, th
        raw_c, th_c = score('calib')
        std = Standardizer().fit(raw_c, th_c, lk.t_mean, lk.t_std)
        z = {}
        for name in names[2:]:
            raw, th = score(name)
            z[name] = std(raw, th)
        z['calib'] = std(raw_c, th_c)
        zs[tag] = z
        res = dict(spec=spec, dims=int(columns(spec, 'audit').shape[-1]),
                   normal_quantiles=np.quantile(z['audit'], [.5, .9, .95, .99, .995, .999]).round(3).tolist(),
                   normal_max=round(float(z['audit'].max()), 2),
                   calib_quantiles=np.quantile(z['calib'], [.99, .995, .999]).round(3).tolist(),
                   shifts={s.removeprefix('shift_'): round(phase2_score(z['audit'], z[s]), 4) for s in shifts})
        res['shift_mean'] = round(float(np.mean(list(res['shifts'].values()))), 4)
        res['seconds'] = round(time.time() - t0, 1)
        report[tag] = res
        log(json.dumps({k: res[k] for k in ('dims', 'normal_quantiles', 'normal_max', 'calib_quantiles', 'shift_mean')}))
        log(' '.join(f'{k}={v:.3f}' for k, v in res['shifts'].items()))
        with open(out / f'model_{tag}.pkl', 'wb') as fh:
            pickle.dump(dict(lk=lk, std=std, spec=spec, stat='copula'), fh)
        np.savez(zf, **z)
        mf.write_text(spec)
    for combo in filter(None, args.combos.split(',')):
        mode, _, body = combo.rpartition(':')
        parts = body.split('+')
        for p in parts:
            if p not in zs:
                zs[p] = dict(np.load(out / f'z_{p}.npz'))
        def comb(name):
            v = np.stack([zs[p][name] for p in parts])
            return v.max(0) if mode == 'max' else v.mean(0)
        za = comb('audit')
        res = dict(normal_quantiles=np.quantile(za, [.5, .9, .95, .99, .995, .999]).round(3).tolist(),
                   shifts={s.removeprefix('shift_'): round(phase2_score(za, comb(s)), 4) for s in shifts})
        res['shift_mean'] = round(float(np.mean(list(res['shifts'].values()))), 4)
        report['combo:' + combo] = res
        print(f'[combo {combo}] q={res["normal_quantiles"]} mean={res["shift_mean"]} ' +
              ' '.join(f'{k}={res["shifts"].get(k, float("nan")):.3f}' for k in KEY), flush=True)
        if args.test:
            np.save(out / f'test_combo_{combo.replace(":", "_")}.npy', comb('test'))
    rp.write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
