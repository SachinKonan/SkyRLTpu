"""LCO metric of recipes with OE members added (CPU). Reads the consistency agent's LCO components
(read-only) and this agent's OE refits (agents/outlier2/lco3). L = OE logit standardized on the kept
calib3 maps of each set (constant location/scale)."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'consistency'))
sys.path.insert(0, str(HERE.parents[1]))

import lib  # noqa: E402
from analyze import standardize  # noqa: E402
from holdout3 import masks  # noqa: E402
from lco3_analyze import components, evaluate, combo  # noqa: E402

MY = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/outlier2/lco3')
NAMES = ['calib', 'in', 'out'] + ['sh_' + s for s in lib.SHIFTS]


def L_sets(src, k):
    m = masks(k)
    f = {n: lib.load(src, n)[..., 0] for n in ('calib', 'audit')}
    raw = {'calib': f['calib'][~m['calib']].reshape(-1), 'in': f['audit'][~m['audit']].reshape(-1),
           'out': np.concatenate([f['calib'][m['calib']].reshape(-1), f['audit'][m['audit']].reshape(-1)])}
    for s in lib.SHIFTS:
        raw['sh_' + s] = lib.load(src, 'shift_' + s)[..., 0][m['audit']].reshape(-1)
    mu, sd = raw['calib'].mean(), raw['calib'].std()
    return {n: (v.astype(np.float64) - mu) / sd for n, v in raw.items()}


def comps(k):
    c = components(k, 'std')
    for tag, src in (('O', 'oe'), ('Oh', 'oe_head'), ('Oho', 'oe_ho')):
        p = MY / f'set{k}_std' / f'{src}.npz'
        if p.exists():
            H = dict(np.load(p))
            c[tag] = standardize({n: H[f'{n}/cop'] for n in NAMES}, {n: H[f'{n}/th'] for n in NAMES})
    for tag, src in (('L', 'oe'), ('Lho', 'oe_ho'), ('Lph', 'oe_phys'), ('Ls1', 'oe_s1')):
        if (lib.ROOT / src / 'test.npz').exists():
            c[tag] = L_sets(src, k)
    if all(t in c for t in ('L', 'Lph', 'Ls1')):
        c['Lens'] = {n: (c['L'][n] + c['Lph'][n] + c['Ls1'][n]) / 3 for n in NAMES}
    return c


V4 = {'Sd': 1, 'C': 1, 'Dr': 1, 'xSL': 1, 'consP8': 1}
BIG = {'Dr': 1, 'N2': 1, 'PsBh': .5, 'xSL': 1, 'consR': 2}
R = {
    'BASE Sd+C': {'Sd': 1, 'C': 1},
    'v3-04': V4,
    'v3-04 +O': {**V4, 'O': 1}, 'v3-04 +0.5O': {**V4, 'O': .5}, 'v3-04 +Oh': {**V4, 'Oh': 1},
    'v3-04 +Oho': {**V4, 'Oho': 1},
    'v3-04 +L': {**V4, 'L': 1}, 'v3-04 +0.5L': {**V4, 'L': .5}, 'v3-04 +Lho': {**V4, 'Lho': 1},
    'v3-04 +Lens': {**V4, 'Lens': 1}, 'v3-04 +0.5Lens': {**V4, 'Lens': .5},
    'v3-04 C->O': {'Sd': 1, 'O': 1, 'Dr': 1, 'xSL': 1, 'consP8': 1},
    'v3-04 C->L': {'Sd': 1, 'L': 1, 'Dr': 1, 'xSL': 1, 'consP8': 1},
    'BIG_RE2': BIG, 'BIG_RE2 +0.5L': {**BIG, 'L': .5}, 'BIG_RE2 +1L': {**BIG, 'L': 1},
    'BIG_RE2 +0.5O': {**BIG, 'O': .5}, 'BIG_RE2 +0.5Lens': {**BIG, 'Lens': .5},
    'O+Dr+Sd+C': {'O': 1, 'Dr': 1, 'Sd': 1, 'C': 1}, 'Dr+Sd+C': {'Dr': 1, 'Sd': 1, 'C': 1},
}


def main():
    cs = {k: comps(k) for k in (0, 1, 2)}
    avail = set.intersection(*[set(c) for c in cs.values()])
    print('available:', sorted(avail))
    keys = ['bary6.5', 'baryDMO', 'bary9.0', 'supp0.15', 'ia0.2', 'srcclust0.15', 'psfpat', 'gradient']

    def rep(name, w):
        if not all(c in avail for c in w):
            return
        rs = [evaluate(combo(cs[k], w)) for k in (0, 1, 2)]
        exc = np.mean([r['exc'] for r in rs], 0)
        m19 = np.mean([r['m19'] for r in rs])
        print(f'{name:18s} LCO m19={m19:.4f} (sets ' + '/'.join(f'{r["m19"]:.3f}' for r in rs) +
              f') exceed99/99.5/99.9={exc[0]:.2f}/{exc[1]:.2f}/{exc[2]:.2f}% | ' +
              ' '.join(f'{s[:7]}={np.mean([r["sh"][s] for r in rs]):.3f}' for s in keys), flush=True)
    print('=== single members')
    for c in ('Sd', 'C', 'Dr', 'xSL', 'consP8', 'O', 'Oh', 'Oho', 'L', 'Lho', 'Lph', 'Ls1', 'Lens'):
        rep(c, {c: 1})
    print('=== recipes')
    for n, w in R.items():
        rep(n, w)


if __name__ == '__main__':
    main()
