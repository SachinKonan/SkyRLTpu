"""Parity / independence / runtime checks for a package built by package_full.py (public test as input only).

  python verify_full.py --pkg <dir> --mode parity --lo 0 --hi 2000     # features, family fits, component z vs frozen
  python verify_full.py --pkg <dir> --mode indep --ids 0,17,4242       # each map alone vs inside a batch
  python verify_full.py --pkg <dir> --mode full                        # Model() + predict on all 10k, timed
Results go to <pkg>/../_verify/<pkg name>/. Nothing here fits or selects anything.
"""
import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

DATA = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data')
TEST = DATA / 'WIDE12H_bin2_2arcmin_kappa_test_phase2_new_v2.npy'
V3 = DATA / 'v3'
FAM = DATA / 'agents' / 'consistency' / 'fam'
EXPORT3 = DATA / 'agents' / 'consistency' / 'export3'


def load_model(pkg):
    os.environ['FWL_KEEP'] = '1'
    sys.path.insert(0, str(pkg))
    t = time.time()
    mod = importlib.import_module('model')
    m = mod.Model()
    return m, time.time() - t


REF_NAMES = {'consRplus': 'consRplus12', 'consRplus_x': 'consRplus'}   # export_family names -> package names


def ref_components(ref_file=''):
    ref = {}
    if ref_file:
        with np.load(ref_file) as d:
            ref = {REF_NAMES.get(k, k): d[k] for k in d.files if '/' not in k}
    for p in sorted(EXPORT3.glob('components_*.npz')):
        with np.load(p) as d:
            for k in d.files:
                c, s = k.split('/', 1)
                if s == 'test':
                    ref.setdefault(c, d[k])
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkg', required=True)
    ap.add_argument('--mode', choices=['parity', 'indep', 'full'], required=True)
    ap.add_argument('--lo', type=int, default=0)
    ap.add_argument('--hi', type=int, default=10000)
    ap.add_argument('--ids', default='0,1,63,64,100,4242,9999')
    ap.add_argument('--tag', default='')
    ap.add_argument('--ref-components', default='', help='npz of frozen test z per component (export_family names)')
    ap.add_argument('--ref-scores', default='', help='frozen test scores of the packaged recipe (.npy)')
    args = ap.parse_args()
    pkg = Path(args.pkg).resolve()
    out = pkg.parent / '_verify' / pkg.name
    out.mkdir(parents=True, exist_ok=True)
    test = np.load(TEST, mmap_mode='r')
    model, t_init = load_model(pkg)
    print(f'Model() in {t_init:.1f}s', flush=True)
    if args.mode in ('parity', 'full'):
        lo, hi = (0, len(test)) if args.mode == 'full' else (args.lo, args.hi)
        t = time.time()
        s = model.predict(test[lo:hi])
        dt = time.time() - t
        import torch
        gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'
        tag = args.tag or f'{args.mode}_{lo}_{hi}'
        print(f'predict {hi - lo} maps in {dt:.1f}s on {gpu}; finite {np.isfinite(s).all()}', flush=True)
        np.savez(out / f'{tag}.npz', scores=s, lo=lo, hi=hi, seconds=dt, init_seconds=t_init,
                 **{f'z/{c}': z for c, z in model.last_components.items()},
                 **{f'th/{f}': v for f, v in model.last_th.items()},
                 **{f'cop/{f}': v for f, v in model.last_cop.items()})
        # feature-level parity against the frozen feature sources
        for src, f in model.last_feats.items():
            p = V3 / src / 'test.npz'
            if p.exists():
                r = np.load(p)['feats'][lo:hi, 0]
                used = np.abs(f).max(0) > 0          # difa: only the levels a family reads are computed
                f, r = f[:, used], r[:, used]
                rel = np.abs(f - r) / (np.abs(r).mean(0) + 1e-6)
                print(f'  feat {src:9s} median rel err {np.median(rel):.1e}  99% {np.quantile(rel, .99):.1e} '
                      f' max {rel.max():.1e}', flush=True)
        for fam, th in model.last_th.items():
            if not (FAM / f'{fam}.npz').exists():
                continue
            with np.load(FAM / f'{fam}.npz') as d:
                rt, rc = d['test/th'][lo:hi], d['test/cop'][lo:hi]
            dth = np.abs(th - rt).max(1)
            print(f'  fam {fam:8s} |dth| median {np.median(dth):.1e} 99% {np.quantile(dth, .99):.1e} '
                  f'frac>0.05 {(dth > 0.05).mean():.4f}  cop rel med {np.median(np.abs(model.last_cop[fam] - rc) / np.abs(rc)):.1e}',
                  flush=True)
        for v, pv in getattr(model, 'last_preds', {}).items():
            if v == 'pden_b':
                r = np.load(DATA / 'agents' / 'consistency2' / 'probes' / 'den_b_full.npz')['test'][lo:hi]
                print(f'  view {v:9s} max |d| {np.abs(pv - r).max():.2e} median {np.median(np.abs(pv - r)):.1e}', flush=True)
        ref = ref_components(args.ref_components)
        res = {}
        for c, z in model.last_components.items():
            if c in ref:
                r = ref[c][lo:hi]
                d = np.abs(z - r)
                norm = np.abs(r) < 5
                res[c] = dict(spearman=float(spearmanr(z, r)[0]), max_abs=float(d.max()),
                              med_abs=float(np.median(d)), max_abs_normal_range=float(d[norm].max()),
                              q99_abs=float(np.quantile(d, .99)))
                print(f'  comp {c:7s} spearman {res[c]["spearman"]:.5f} |dz| med {res[c]["med_abs"]:.1e} '
                      f'99% {res[c]["q99_abs"]:.1e} max(|z|<5) {res[c]["max_abs_normal_range"]:.2e} '
                      f'max {res[c]["max_abs"]:.2e}', flush=True)
        if args.ref_scores:
            r = np.load(args.ref_scores)[lo:hi]
            d = np.abs(s - r)
            norm = np.abs(r) < 5
            res['_scores'] = dict(spearman=float(spearmanr(s, r)[0]), max_abs=float(d.max()), med_abs=float(np.median(d)),
                                  max_abs_normal_range=float(d[norm].max()), q99_abs=float(np.quantile(d, .99)),
                                  seconds=dt, init_seconds=t_init, gpu=gpu)
            print(f'  SCORES vs {args.ref_scores}: spearman {res["_scores"]["spearman"]:.5f} |dz| med {np.median(d):.1e} '
                  f'99% {np.quantile(d, .99):.1e} max(|z|<5) {d[norm].max():.2e} max {d.max():.2e}', flush=True)
            top = np.argsort(-r)[:100]
            print(f'  top-100 overlap (by reference score) {len(set(top) & set(np.argsort(-s)[:100]))}/100', flush=True)
        if getattr(model, 'timing', None):
            res['_timing'] = model.timing
            print(f'  stage timing {model.timing}', flush=True)
        cfg = json.load(open(pkg / 'config.json'))
        for name, ref_path in (('v3-04', DATA / 'merged' / 'v3-04.npy'), ('BIG_RE2', EXPORT3 / 'test_z_BIG_RE2.npy'),
                               ('HEDGE_RE3', EXPORT3 / 'test_z_HEDGE_RE3.npy')):
            w = json.loads((EXPORT3 / f'recipe_{"v3-04_repro" if name == "v3-04" else name}.txt').read_text())['weights']
            if all(c in model.last_components for c in w):
                z = sum(v * model.last_components[c] for c, v in w.items()) / sum(w.values())
                r = np.load(ref_path)[lo:hi]
                d = np.abs(z - r)
                norm = np.abs(r) < 5
                print(f'  RECIPE {name:9s} spearman {spearmanr(z, r)[0]:.5f} |dz| med {np.median(d):.1e} '
                      f'99% {np.quantile(d, .99):.1e} max(|z|<5) {d[norm].max():.2e} max {d.max():.2e} '
                      f'n={len(r)}', flush=True)
        if cfg['weights']:
            print('  package recipe', cfg['weights'], flush=True)
        json.dump(res, open(out / f'{tag}_components.json', 'w'), indent=1)
    else:
        ids = [int(i) for i in args.ids.split(',')]
        lo = max(0, min(ids) - 5)
        hi = min(len(test), max(ids) + 60)
        blk = model.predict(test[lo:hi])
        comps_blk = dict(model.last_components)
        worst = 0.0
        for i in ids:
            s1 = model.predict(test[i:i + 1])
            d = abs(float(s1[0]) - float(blk[i - lo]))
            dc = {c: abs(float(model.last_components[c][0]) - float(comps_blk[c][i - lo])) for c in comps_blk}
            worst = max(worst, d)
            print(f'  map {i}: alone {s1[0]:.6f} in-batch {blk[i - lo]:.6f} |diff| {d:.2e}  worst comp '
                  f'{max(dc, key=dc.get)} {max(dc.values()):.2e}', flush=True)
        rep = model.predict(test[lo:hi])
        print(f'  same block repeated: max |diff| {np.abs(rep - blk).max():.2e} (run-to-run nondeterminism)', flush=True)
        # the same map placed in a batch of different neighbours (shuffled block)
        rng = np.random.default_rng(0)
        perm = rng.permutation(hi - lo)
        blk2 = model.predict(np.ascontiguousarray(test[lo:hi])[perm])
        inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
        d2 = np.abs(blk2[inv] - blk).max()
        print(f'  shuffled block of {hi - lo}: max |diff| {d2:.2e}; worst alone-vs-batch {worst:.2e}', flush=True)


if __name__ == '__main__':
    main()
