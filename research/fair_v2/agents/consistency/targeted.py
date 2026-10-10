"""Coordinator request: low-dof targeted tests on ia0.2, bary9.0, bary6.5, baryDMO, supp0.15.

A  5-dim (or sub-) parameter-difference Mahalanobis between two estimators, covariance from calib3,
   global or theta-dependent (mean and per-dim log-variance quadratic in the average theta, then the
   correlation of the whitened residual).
B  plug-in dlogp: E_f(theta_src) - E_f(theta_f) with theta_src from the CNN predictions / other blocks.
C  prior-box/hull relaxation energy drop E_box - E_relaxed (per family).
Every score is standardized on calib3 (Standardizer vs theta) and evaluated on audit3 vs shifts.
"""
import argparse

import numpy as np

import lib
from analyze import standardize, ev, mean_of, fam, mahal, raw_theta, TM, TS, boot_diff
from model import design

KEY = ['ia0.2', 'bary9.0', 'bary6.5', 'baryDMO', 'supp0.15']


def mahal_cond(d, center, ridge=1.0):
    """theta-dependent Mahalanobis of differences d (dict set -> (N,p)) given centers (normalized theta)."""
    X = {k: design(raw_theta(v), TM, TS) for k, v in center.items()}
    A = X['calib'].T @ X['calib'] + ridge * np.eye(X['calib'].shape[1])
    B = np.linalg.solve(A, X['calib'].T @ d['calib'])
    r = {k: d[k] - X[k] @ B for k in d}
    S = np.linalg.solve(A, X['calib'].T @ np.log(r['calib'] ** 2 + 1e-12))
    u = {k: r[k] / np.exp(0.5 * (X[k] @ S)) for k in d}
    return mahal(u)


def row(name, z):
    r = ev(z)
    sh = r['shifts']
    keys = ' '.join(f'{k}={sh[k]:.3f}' for k in KEY if k in sh)
    nm = r.get('new_mean', float('nan'))
    print(f'{name:46s} mean15={r["shift_mean"]:.4f} new4={nm:.3f} q99/99.5/99.9={r["nq"][2]:.2f}/{r["nq"][3]:.2f}/'
          f'{r["nq"][4]:.2f} | {keys}', flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scale', default='scale2')
    args = ap.parse_args()
    sets = lib.SETS + ['test']
    F = {f: fam(f) for f in ('psL', 'psS', 'onept', 'extrema', 'mf', 'dog', 'cnn', 'cnn1', 'small', 'mid', 'large')}
    with np.load(lib.OUT / 'scale' / f'{args.scale}.npz') as d:
        X = {k: d[k] for k in d.files}
    sets = [s for s in sets if f'{s}/th_small' in X]
    TH = {f: {s: F[f][f'{s}/th'] for s in sets} for f in F}
    for src in ('jointS', 'jointSC', 'pred0', 'pred1', 'predm'):
        TH[src] = {s: X[f'{s}/th_{src}'] for s in sets}
    Z = {}
    for f in ('dog', 'cnn', 'cnn1'):
        Z[f] = standardize({s: F[f][f'{s}/cop'] for s in sets}, TH[f])
    base = mean_of(Z['dog'], Z['cnn'])
    base3 = mean_of(Z['dog'], Z['cnn'], Z['cnn1'])
    print('=== reference')
    row('BASE mean:Sd+C', base)
    row('C1 alone (cnn1 likelihood)', Z['cnn1'])
    row('BASE3 mean:Sd+C+C1', base3)
    out = {}
    print('=== A: parameter-difference Mahalanobis (global | theta-dependent covariance)')
    pairs = [('pred0', 'dog'), ('pred0', 'jointS'), ('predm', 'jointS'), ('cnn', 'dog'), ('cnn', 'jointS'),
             ('cnn1', 'jointS'), ('pred0', 'pred1'), ('cnn', 'cnn1'), ('small', 'large'), ('pred0', 'large'),
             ('pred0', 'small')]
    dimsets = {'all5': [0, 1, 2, 3, 4], 'cos': [0, 1], 'cosT': [0, 1, 2], 'bary': [2, 3]}
    for a, b in pairs:
        for dn, dims in dimsets.items():
            d = {s: (TH[a][s] - TH[b][s])[:, dims] for s in sets}
            c = {s: 0.5 * (TH[a][s] + TH[b][s]) for s in sets}
            for kind in ('glob', 'cond'):
                m2 = mahal(d) if kind == 'glob' else mahal_cond(d, c)
                z = standardize({s: np.log(m2[s] + 1.0) for s in sets}, c)
                out[f'A {a}-{b} {dn} {kind}'] = z
                row(f'A {a}-{b} {dn} {kind}', z)
    print('=== A2: multi-estimator consistency (consP8 and with cnn1)')
    for nm, fs in [('consP8', ['psL', 'psS', 'onept', 'extrema', 'mf', 'dog', 'cnn', 'pred0']),
                   ('consP10', ['psL', 'psS', 'onept', 'extrema', 'mf', 'dog', 'cnn', 'pred0', 'cnn1', 'pred1']),
                   ('consP9m', ['psL', 'psS', 'onept', 'extrema', 'mf', 'dog', 'cnn', 'cnn1', 'predm'])]:
        for kind in ('glob', 'cond'):
            vec, cen = {}, {}
            for s in sets:
                A = np.stack([TH[f][s] for f in fs], 1)
                vec[s] = (A - A.mean(1, keepdims=True))[:, 1:].reshape(len(A), -1)
                cen[s] = A.mean(1)
            m2 = mahal(vec) if kind == 'glob' else mahal_cond(vec, cen)
            z = standardize({s: np.log(m2[s] + 1.0) for s in sets}, cen)
            out[f'{nm} {kind}'] = z
            row(f'{nm} {kind}', z)
    print('=== B: plug-in dlogp  E_f(theta_src) - E_f(theta_f)')
    for f in ('dog', 'small', 'large', 'cnn', 'cnn1'):
        for src in ('pred0', 'predm', 'jointS', 'large', 'small', 'cnn'):
            if src == f or f'audit/E_{f}@{src}' not in X:
                continue
            for kind in ('E', 'cop'):
                raw = {s: X[f'{s}/{kind}_{f}@{src}'] - X[f'{s}/{kind}_{f}@{f}'] for s in sets}
                z = standardize(raw, TH[src])
                out[f'B {kind} {f}|{src}'] = z
                row(f'B {kind} {f}|{src}', z)
    raw = {s: (X[f'{s}/cop_small@large'] - X[f'{s}/cop_small@small']) +
           (X[f'{s}/cop_large@small'] - X[f'{s}/cop_large@large']) for s in sets}
    out['xSL'] = standardize(raw, {s: 0.5 * (TH['small'][s] + TH['large'][s]) for s in sets})
    row('xSL (small<->large, both directions)', out['xSL'])
    print('=== C: prior box/hull relaxation energy drop (E_box - E_relaxed)')
    for f in ('dog', 'cnn', 'cnn1', 'small', 'large'):
        for nm, tr in (('theta-std', TH[f]), ('global-std', None)):
            raw = {s: np.log(np.maximum(F[f][f'{s}/E'] - F[f][f'{s}/Er'], 0) + 1.0) for s in sets}
            if tr is None:
                m, sd = raw['calib'].mean(), raw['calib'].std()
                z = {s: (raw[s] - m) / sd for s in sets}
            else:
                z = standardize(raw, tr)
            out[f'C drop {f} {nm}'] = z
            row(f'C drop {f} {nm}', z)
    print('=== combos (weights 0.5 for each added component)')
    combos = {
        'BASE+0.5xSL+0.5consP8 (frozen rec.)': (base, ['xSL', 'consP8 glob'], 2),
        'BASE+0.5xSL+0.5consP10': (base, ['xSL', 'consP10 glob'], 2),
        'BASE3+0.5xSL+0.5consP10': (base3, ['xSL', 'consP10 glob'], 3),
        'BASE3+0.5xSL+0.5consP8': (base3, ['xSL', 'consP8 glob'], 3),
    }
    best_single = sorted([k for k in out if k.startswith('A ') or k.startswith('B ')],
                         key=lambda k: -ev(out[k])['shift_mean'])[:6]
    for k in best_single:
        combos[f'BASE+0.5xSL+0.5consP8+0.5[{k}]'] = (base, ['xSL', 'consP8 glob', k], 2)
    for name, (b, comps, nb) in combos.items():
        z = mean_of(b, *[out[c] for c in comps], w=[nb] + [0.5] * len(comps))
        out['COMBO ' + name] = z
        row(name, z)
    np.save(lib.OUT / 'targeted_keys.npy', np.array(list(out)))
    import pickle
    with open(lib.OUT / 'targeted_z.pkl', 'wb') as fh:
        pickle.dump(dict(out=out, base=base, base3=base3), fh)


if __name__ == '__main__':
    main()
