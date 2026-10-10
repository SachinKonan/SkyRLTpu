"""Full-data components and export of LCO-chosen recipes to OUT/export3/ (CPU).

--recipe 'tag={"Sd":1,"C":0.5,...}' (repeatable, ';'-separated); --var 'Sd=cj,C=cj' chooses a
fit variant (fam_<variant>/) for likelihood members. Fitting/standardization on fit3/calib3 only.
"""
import argparse
import json

import numpy as np

import lib
from analyze import standardize, ev, mean_of, fam, mahal

MEMBERS = {'Sd': 'dog', 'C': 'cnn', 'C1': 'cnn1', 'Dr': 'den_r', 'N2': 'N2', 'K': 'stx_pkcl', 'PsBh': 'PsBh',
           'De': 'den_e', 'Dw': 'den_w'}


def cons(TH, fs, sets):
    vec, cen = {}, {}
    for s in sets:
        A = np.stack([TH[f][s] for f in fs], 1)
        vec[s] = (A - A.mean(1, keepdims=True))[:, 1:].reshape(len(A), -1)
        cen[s] = A.mean(1)
    m2 = mahal(vec)
    return standardize({s: np.log(m2[s] + 1.0) for s in sets}, cen)


def components(var):
    sets = lib.SETS + ['test']
    comp = {}
    for m, f in MEMBERS.items():
        v = var.get(m)
        if v:
            with np.load(lib.OUT / f'fam_{v}' / f'{f}.npz') as d:
                F = {k: d[k] for k in d.files}
        else:
            F = fam(f)
        comp[m] = standardize({s: F[f'{s}/cop'] for s in sets}, {s: F[f'{s}/th'] for s in sets})
    need = ['psL', 'psS', 'onept', 'extrema', 'mf', 'dog', 'cnn', 'cnn1', 'b0', 'b1', 'b2', 'b3', 'den_r', 'den_sd']
    TH = {}
    for f in need:
        F = fam(f)
        TH[f] = {s: F[f'{s}/th'] for s in sets}
    TH['pred0'] = {s: lib.load('cnn0', s)[..., -5:].reshape(-1, 5) for s in sets}
    TH['pred1'] = {s: lib.load('cnn1', s)[..., -5:].reshape(-1, 5) for s in sets}
    comp['consP8'] = cons(TH, ['psL', 'psS', 'onept', 'extrema', 'mf', 'dog', 'cnn', 'pred0'], sets)
    comp['consR'] = cons(TH, ['b0', 'b1', 'b2', 'b3', 'cnn', 'cnn1', 'pred0', 'pred1', 'dog', 'den_r', 'den_sd'], sets)
    with np.load(lib.OUT / 'scale' / 'scale2.npz') as d:
        X = {k: d[k] for k in d.files if k.endswith('@small') or k.endswith('@large') or '/th_small' in k
             or '/th_large' in k}
    raw = {s: (X[f'{s}/cop_small@large'] - X[f'{s}/cop_small@small']) +
           (X[f'{s}/cop_large@small'] - X[f'{s}/cop_large@large']) for s in sets}
    comp['xSL'] = standardize(raw, {s: 0.5 * (X[f'{s}/th_small'] + X[f'{s}/th_large']) for s in sets})
    return comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--recipe', required=True)
    ap.add_argument('--var', default='')
    ap.add_argument('--export', action='store_true')
    args = ap.parse_args()
    var = dict(kv.split('=') for kv in filter(None, args.var.split(',')))
    comp = components(var)
    d = lib.OUT / 'export3'
    d.mkdir(exist_ok=True)
    for spec in filter(None, args.recipe.split(';')):
        tag, _, js = spec.partition('=')
        w = json.loads(js)
        z = mean_of(*[comp[c] for c in w], w=list(w.values()))
        r = ev(z)
        print(f'{tag:30s} audit m15={r["shift_mean"]:.4f} new4={r.get("new_mean", float("nan")):.3f} '
              f'q99/99.5/99.9={r["nq"][2]:.2f}/{r["nq"][3]:.2f}/{r["nq"][4]:.2f} w={w} var={var}', flush=True)
        if args.export:
            assert z['test'].shape == (10000,) and np.isfinite(z['test']).all()
            np.save(d / f'test_z_{tag}.npy', z['test'].astype(np.float64))
            np.savez(d / f'z_{tag}.npz', **z)
            np.savez(d / f'components_{tag}.npz', **{f'{c}/{k}': v for c in w for k, v in comp[c].items()})
            (d / f'recipe_{tag}.txt').write_text(json.dumps(dict(weights=w, variants=var)) + '\n')
            print('  exported', d / f'test_z_{tag}.npy')


if __name__ == '__main__':
    main()
