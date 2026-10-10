"""Leave-cosmology-out tail check for the round-2 recipes (CPU)."""
import numpy as np

import lib
from analyze import standardize, mahal, mean_of
from holdout import split, sets_for

NAMES = ('calib', 'in', 'out')
Q = [.99, .995, .999]


def cons(TH, fs):
    vec, cen = {}, {}
    for n in NAMES:
        A = np.stack([TH[f][n] for f in fs], 1)
        vec[n] = (A - A.mean(1, keepdims=True))[:, 1:].reshape(len(A), -1)
        cen[n] = A.mean(1)
    m2 = mahal(vec)
    return standardize({n: np.log(m2[n] + 1.0) for n in NAMES}, cen)


def main(mode='lco'):
    d = lib.OUT / 'holdout' / mode
    held, isout = split(mode)
    fams = [f for f in ('dog', 'cnn', 'cnn1', 'psL', 'psS', 'onept', 'extrema', 'mf', 'b0', 'b1', 'b2', 'b3', 'den_r',
                        'den_sd') if (d / f'{f}.npz').exists()]
    H = {f: dict(np.load(d / f'{f}.npz')) for f in fams}
    TH = {f: {n: H[f][f'{n}/th'] for n in NAMES} for f in fams}
    TH['pred0'] = {n: sets_for('cnn', isout)[n][..., -5:].reshape(-1, 5) for n in NAMES}
    TH['pred1'] = {n: sets_for('cnn1', isout)[n][..., -5:].reshape(-1, 5) for n in NAMES}
    comp = {}
    for f, nm in (('dog', 'Sd'), ('cnn', 'C'), ('cnn1', 'C1')):
        comp[nm] = standardize({n: H[f][f'{n}/cop'] for n in NAMES}, TH[f])
    X = dict(np.load(d / 'cross.npz'))
    raw = {n: (X[f'{n}/cop_small@large'] - X[f'{n}/cop_small@small']) +
           (X[f'{n}/cop_large@small'] - X[f'{n}/cop_large@large']) for n in NAMES}
    comp['xSL'] = standardize(raw, {n: 0.5 * (X[f'{n}/th_small'] + X[f'{n}/th_large']) for n in NAMES})
    comp['consP8'] = cons(TH, ['psL', 'psS', 'onept', 'extrema', 'mf', 'dog', 'cnn', 'pred0'])
    comp['consP10'] = cons(TH, ['psL', 'psS', 'onept', 'extrema', 'mf', 'dog', 'cnn', 'pred0', 'cnn1', 'pred1'])
    if all(f in TH for f in ('b0', 'b1', 'b2', 'b3', 'den_r', 'den_sd')):
        comp['consR_bands_glob'] = cons(TH, ['b0', 'b1', 'b2', 'b3', 'cnn', 'cnn1', 'pred0', 'pred1', 'dog', 'den_r',
                                             'den_sd'])
    cp = lib.OUT / 'cross' / f'r2_{mode}.npz'
    if cp.exists():
        Y = dict(np.load(cp))
        if all(f'{n}/done' in Y for n in NAMES):
            B = ['b0', 'b1', 'b2', 'b3']
            raw = {n: sum(Y[f'{n}/cop_{bi}@{bj}'] - Y[f'{n}/cop_{bi}@{bi}'] for bi in B for bj in B if bi != bj)
                   for n in NAMES}
            comp['xB4_cop'] = standardize(raw, {n: np.mean([TH[b][n] for b in B], 0) for n in NAMES})
            pm = {n: 0.5 * (TH['pred0'][n] + TH['pred1'][n]) for n in NAMES}
            for kind in ('E', 'cop'):
                zs = [standardize({n: Y[f'{n}/{kind}_{f}@predm'] - Y[f'{n}/{kind}_{f}@{f}'] for n in NAMES}, pm)
                      for f in ('cnn', 'cnn1')]
                comp[f'PIcnn_{kind}'] = mean_of(*zs)
    recipes = {
        'BASE': {'Sd': 1, 'C': 1},
        'REC1': {'Sd': 1, 'C': 1, 'xSL': .5, 'consP8': .5},
        'a': {'Sd': 1, 'C': 1, 'xSL': 1, 'consP8': 1},
        'b': {'Sd': 1, 'C': 1, 'xSL': 1, 'consR_bands_glob': 1},
        'c': {'Sd': .5, 'C': .5, 'xSL': 1, 'consR_bands_glob': 1},
        'b+xB4': {'Sd': 1, 'C': 1, 'xSL': 1, 'consR_bands_glob': 1, 'xB4_cop': 1},
        'b+PIcnnE': {'Sd': 1, 'C': 1, 'xSL': 1, 'consR_bands_glob': 1, 'PIcnn_E': 1},
        'c+xB4': {'Sd': .5, 'C': .5, 'xSL': 1, 'consR_bands_glob': 1, 'xB4_cop': 1},
        'c+PIcnnE': {'Sd': .5, 'C': .5, 'xSL': 1, 'consR_bands_glob': 1, 'PIcnn_E': 1},
    }
    print(f'===== {mode}: exceedance of held-out maps over in-sample thresholds (nominal 1.0/0.5/0.1%)')
    for k in comp:
        z = comp[k]; t = np.quantile(z['in'], Q)
        print(f'  component {k:18s} in q99/99.5/99.9 ' + '/'.join(f'{v:.2f}' for v in t) + '  exceed ' +
              ' '.join(f'{100 * np.mean(z["out"] > x):.2f}%' for x in t))
    for name, w in recipes.items():
        if not all(c in comp for c in w):
            continue
        z = mean_of(*[comp[c] for c in w], w=list(w.values()))
        t = np.quantile(z['in'], Q)
        print(f'  recipe {name:20s} in q99/99.5/99.9 ' + '/'.join(f'{v:.2f}' for v in t) + '  exceed ' +
              ' '.join(f'{100 * np.mean(z["out"] > x):.2f}%' for x in t))


if __name__ == '__main__':
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else 'lco')
