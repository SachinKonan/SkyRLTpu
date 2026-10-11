"""LCO (leave-cosmology-out) numbers for the config-only package variants v3-15b..f (CPU).

  python factory/pkg_variant_lco.py [--out fair-data/factory/pkg_variant_lco.json]

Fold caches and member z come from stack_levers.load_fold (cache_v2 + Oh from the lco3 components); the package
member X (dirpower misfit) is 'Xdp' here, and consRplus is the 13-view consistency (with the den_b and dirpower
votes), exactly as in v3-13a. SS is 0 in every variant. Weights are each variant's config.json weights.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grade as G  # noqa: E402
from stack_levers import DATA, V2, load_fold, summarize, wmean  # noqa: E402

PKG = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/packages')
NAMES = ('v3-15b', 'v3-15c', 'v3-15d', 'v3-15e', 'v3-15f', 'v3-15g', 'v3-15h')
RENAME = {'X': 'Xdp'}


def recipe(name):
    w = json.loads((PKG / name / 'config.json').read_text())['weights']
    r = {RENAME.get(k, k): float(v) for k, v in w.items() if float(v) != 0.0}
    assert 'consRplus12' not in r and 'SS' not in r, r
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=DATA / 'pkg_variant_lco.json')
    args = ap.parse_args()
    views = json.loads((V2 / 'stack.json').read_text())['views']
    folds = {}
    for k in G.FOLDS:
        comp, TH = load_fold(k)
        comp['consRplus'] = G.cons(TH, views, G.NAMES)
        folds[k] = comp
    res = {}
    for name in NAMES:
        r = recipe(name)
        res[name] = dict(summarize([G.metrics(wmean(folds[k], r)) for k in G.FOLDS]), weights=r)
    b = res['v3-15b']
    print(f'{"variant":8s} {"all19":>7s} {"vis":>7s} {"hid":>7s}  {"fold all19":22s}  {"d_fold vs 15b":24s} exc(1/.5/.1%)')
    for name, s in res.items():
        d = [a - c for a, c in zip(s['per_fold_all19'], b['per_fold_all19'])]
        s['d_fold_all19_vs_15b'] = d
        s['worse_than_15b_on_some_fold'] = bool(name != 'v3-15b' and min(d) < 0)
        print(f'{name:8s} {s["all19"]:.4f} {s["visible"]:.4f} {s["hidden"]:.4f}  '
              f'{" ".join(f"{v:.4f}" for v in s["per_fold_all19"]):22s}  {" ".join(f"{v:+.4f}" for v in d):24s} '
              f'{s["exc_pct"]}{"  WORSE ON A FOLD" if s["worse_than_15b_on_some_fold"] else ""}')
    args.out.write_text(json.dumps(res, indent=1))
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
