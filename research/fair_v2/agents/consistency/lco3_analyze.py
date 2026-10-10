"""LCO-based evaluation of member combinations (CPU).

Per held-out cosmology set k: InD = held-out normal maps ('out'), OoD = shifts of held-out audit3
maps ('sh_*'); thresholds for exceedance from in-sample kept audit3 maps ('in').
"""
import argparse
import itertools
import json

import numpy as np

import lib
from analyze import standardize, mahal, mean_of
from holdout3 import masks
from model import phase2_score

NAMES_BASE = ['calib', 'in', 'out']


def pred_sets(src, k):
    m = masks(k)
    f = {n: lib.load(src, n)[..., -5:] for n in ('calib', 'audit')}
    out = {'calib': f['calib'][~m['calib']].reshape(-1, 5), 'in': f['audit'][~m['audit']].reshape(-1, 5),
           'out': np.concatenate([f['calib'][m['calib']].reshape(-1, 5), f['audit'][m['audit']].reshape(-1, 5)])}
    for s in lib.SHIFTS:
        out['sh_' + s] = lib.load(src, 'shift_' + s)[..., -5:][m['audit']].reshape(-1, 5)
    return out


def cons(TH, fs, names):
    vec, cen = {}, {}
    for n in names:
        A = np.stack([TH[f][n] for f in fs], 1)
        vec[n] = (A - A.mean(1, keepdims=True))[:, 1:].reshape(len(A), -1)
        cen[n] = A.mean(1)
    m2 = mahal(vec)
    return standardize({n: np.log(m2[n] + 1.0) for n in names}, cen)


MEMBERS = {'Sd': 'dog', 'C': 'cnn', 'C1': 'cnn1', 'Dr': 'den_r', 'N2': 'N2', 'K': 'stx_pkcl', 'PsBh': 'PsBh',
           'De': 'den_e', 'Dw': 'den_w', 'SS': 'dn_SS', 'Oh': 'oe_head'}


def components(k, variant='std', std_dir=None):
    d = lib.OUT / 'lco3' / f'set{k}_{variant}'
    sd = lib.OUT / 'lco3' / f'set{k}_std'
    names = NAMES_BASE + ['sh_' + s for s in lib.SHIFTS]
    comp = {}
    H = {}
    for f in set(MEMBERS.values()) | {'psL', 'psS', 'onept', 'extrema', 'mf', 'b0', 'b1', 'b2', 'b3', 'den_sd',
                                      'small', 'large'}:
        p = (d / f'{f}.npz') if (d / f'{f}.npz').exists() else (sd / f'{f}.npz')
        if p.exists():
            H[f] = dict(np.load(p))
    for m, f in MEMBERS.items():
        if f in H:
            comp[m] = standardize({n: H[f][f'{n}/cop'] for n in names}, {n: H[f][f'{n}/th'] for n in names})
    TH = {f: {n: H[f][f'{n}/th'] for n in names} for f in H}
    TH['pred0'] = pred_sets('cnn0', k)
    TH['pred1'] = pred_sets('cnn1', k)
    p8 = ['psL', 'psS', 'onept', 'extrema', 'mf', 'dog', 'cnn', 'pred0']
    if all(f in TH for f in p8):
        comp['consP8'] = cons(TH, p8, names)
    rb = ['b0', 'b1', 'b2', 'b3', 'cnn', 'cnn1', 'pred0', 'pred1', 'dog', 'den_r', 'den_sd']
    if all(f in TH for f in rb):
        comp['consR'] = cons(TH, rb, names)
    cp = sd / 'cross.npz'
    if cp.exists():
        X = dict(np.load(cp))
        raw = {n: (X[f'{n}/cop_small@large'] - H['small'][f'{n}/cop']) +
               (X[f'{n}/cop_large@small'] - H['large'][f'{n}/cop']) for n in names}
        comp['xSL'] = standardize(raw, {n: 0.5 * (TH['small'][n] + TH['large'][n]) for n in names})
    return comp


def evaluate(z):
    t = np.quantile(z['in'], [.99, .995, .999])
    exc = [float(np.mean(z['out'] > x)) * 100 for x in t]
    sh = {s: phase2_score(z['out'], z['sh_' + s]) for s in lib.SHIFTS}
    return dict(exc=exc, m15=float(np.mean([sh[s] for s in lib.SHIFTS15])),
                m19=float(np.mean(list(sh.values()))), sh=sh)


def combo(comp, w):
    return mean_of(*[comp[c] for c in w], w=list(w.values()))


RECIPES = {
    'BASE Sd+C': {'Sd': 1, 'C': 1},
    'REC1 (LB .5669)': {'Sd': 1, 'C': 1, 'xSL': .5, 'consP8': .5},
    'v3-04 (LB .6461)': {'Sd': 1, 'C': 1, 'Dr': 1, 'xSL': 1, 'consP8': 1},
    'b (export2)': {'Sd': 1, 'C': 1, 'xSL': 1, 'consR': 1},
    'v3-04 + consR': {'Sd': 1, 'C': 1, 'Dr': 1, 'xSL': 1, 'consP8': 1, 'consR': 1},
    'v3-04 half C': {'Sd': 1, 'C': .5, 'Dr': 1, 'xSL': 1, 'consP8': 1},
    'v3-04 no C': {'Sd': 1, 'Dr': 1, 'xSL': 1, 'consP8': 1},
    'v3-04 +N2': {'Sd': 1, 'C': 1, 'Dr': 1, 'N2': 1, 'xSL': 1, 'consP8': 1},
    'v3-04 +N2+K': {'Sd': 1, 'C': 1, 'Dr': 1, 'N2': 1, 'K': 1, 'xSL': 1, 'consP8': 1},
    'v3-04 +PsBh': {'Sd': 1, 'C': 1, 'Dr': 1, 'PsBh': 1, 'xSL': 1, 'consP8': 1},
    'cons-heavy': {'Sd': 1, 'C': .5, 'Dr': 1, 'N2': 1, 'xSL': 1.5, 'consP8': 1, 'consR': 1},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', default='0,1,2')
    ap.add_argument('--variant', default='std')
    ap.add_argument('--search', action='store_true')
    args = ap.parse_args()
    ks = [int(k) for k in args.sets.split(',')]
    comps = {k: components(k, args.variant) for k in ks}
    avail = set.intersection(*[set(c) for c in comps.values()])
    print('components available in all sets:', sorted(avail))

    def report(name, w):
        if not all(c in avail for c in w):
            return None
        rs = [evaluate(combo(comps[k], w)) for k in ks]
        exc = np.mean([r['exc'] for r in rs], 0)
        m15, m19 = np.mean([r['m15'] for r in rs]), np.mean([r['m19'] for r in rs])
        per = ' '.join(f'{r["m19"]:.3f}' for r in rs)
        keys = ['bary6.5', 'baryDMO', 'bary9.0', 'supp0.15', 'ia0.2', 'srcclust0.15', 'psfpat']
        shs = ' '.join(f'{s[:6]}={np.mean([r["sh"][s] for r in rs]):.3f}' for s in keys)
        print(f'{name:28s} LCO m19={m19:.4f} (sets {per}) m15={m15:.4f} exceed={exc[0]:.2f}/{exc[1]:.2f}/{exc[2]:.2f}% | {shs}',
              flush=True)
        return m19, exc
    print('=== single components (LCO)')
    for c in sorted(avail):
        report(c, {c: 1})
    print('=== recipes (LCO)')
    for name, w in RECIPES.items():
        report(name, w)
    if args.search:
        print('=== coarse weight search (maximize LCO m19 with 1%-exceedance <= 1.6%)')
        mem = [c for c in ('Sd', 'C', 'Dr', 'N2', 'K', 'PsBh', 'xSL', 'consP8', 'consR') if c in avail]
        grid = {c: (0, .5, 1, 2) for c in mem}
        best = []
        rng = np.random.default_rng(0)
        cands = [dict(zip(mem, ws)) for ws in itertools.product(*[grid[c] for c in mem])]
        rng.shuffle(cands)
        for w in cands[:900]:
            w = {c: v for c, v in w.items() if v > 0}
            if len(w) < 2:
                continue
            rs = [evaluate(combo(comps[k], w)) for k in ks]
            m19 = float(np.mean([r['m19'] for r in rs])); exc = np.mean([r['exc'] for r in rs], 0)
            best.append((m19, exc.tolist(), w))
        best.sort(key=lambda x: -x[0])
        for m19, exc, w in best[:15]:
            print(f'  m19={m19:.4f} exceed={exc[0]:.2f}/{exc[1]:.2f}/{exc[2]:.2f}% {w}')
        (lib.OUT / 'lco3' / f'search_{args.variant}.json').write_text(json.dumps(best[:50]))


if __name__ == '__main__':
    main()
