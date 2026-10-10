"""Round 2 (after LB +0.040 for BASE+0.5xSL+0.5consP8): richer multi-view theta consistency,
4-band scale tensions, plug-in energies at the CNN predictions, weight variants. Export to export2/.

All fitting (Standardizer, covariances) on calib3 only; evaluation on audit3 vs 19 shifts.
"""
import argparse
from pathlib import Path

import numpy as np

import lib
from analyze import standardize, ev, mean_of, fam, mahal, boot_diff
from targeted import mahal_cond

KEY = ['ia0.2', 'bary9.0', 'bary6.5', 'baryDMO', 'supp0.15', 'boost0.15', 'peakcut0.3']
STX = ['stx_bisp', 'stx_ipow', 'stx_nois', 'stx_pkcl', 'stx_prof', 'stx_scat', 'stx_tail']
BANDS = ['b0', 'b1', 'b2', 'b3']


def row(name, z, extra=''):
    r = ev(z)
    sh = r['shifts']
    print(f'{name:44s} m15={r["shift_mean"]:.4f} new4={r.get("new_mean", float("nan")):.3f} '
          f'q99/99.5/99.9={r["nq"][2]:.2f}/{r["nq"][3]:.2f}/{r["nq"][4]:.2f} | ' +
          ' '.join(f'{k[:6]}={sh[k]:.3f}' for k in KEY if k in sh) + extra, flush=True)
    return r


def cons(TH, fs, sets, kind='glob'):
    vec, cen = {}, {}
    for s in sets:
        A = np.stack([TH[f][s] for f in fs], 1)
        vec[s] = (A - A.mean(1, keepdims=True))[:, 1:].reshape(len(A), -1)
        cen[s] = A.mean(1)
    m2 = mahal(vec) if kind == 'glob' else mahal_cond(vec, cen)
    return standardize({s: np.log(m2[s] + 1.0) for s in sets}, cen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--export', default='')
    ap.add_argument('--boot', default='')
    args = ap.parse_args()
    sets = lib.SETS + ['test']
    have = [f for f in lib.FAMILIES if (lib.OUT / 'fam' / f'{f}.npz').exists()]
    F = {f: fam(f) for f in have}
    have = [f for f in have if all(f'{s}/th' in F[f] for s in sets)]
    print('families:', have)
    TH = {f: {s: F[f][f'{s}/th'] for s in sets} for f in have}
    TH['pred0'] = {s: lib.load('cnn0', s)[..., -5:].reshape(-1, 5) for s in sets}
    TH['pred1'] = {s: lib.load('cnn1', s)[..., -5:].reshape(-1, 5) for s in sets}
    comp = {}
    for f, nm in (('dog', 'Sd'), ('cnn', 'C'), ('cnn1', 'C1')):
        comp[nm] = standardize({s: F[f][f'{s}/cop'] for s in sets}, TH[f])
    with np.load(lib.OUT / 'scale' / 'scale2.npz') as d:
        X = {k: d[k] for k in d.files if '@small' in k or '@large' in k or 'th_small' in k or 'th_large' in k}
    raw = {s: (X[f'{s}/cop_small@large'] - X[f'{s}/cop_small@small']) +
           (X[f'{s}/cop_large@small'] - X[f'{s}/cop_large@large']) for s in sets}
    comp['xSL'] = standardize(raw, {s: 0.5 * (X[f'{s}/th_small'] + X[f'{s}/th_large']) for s in sets})
    P8 = ['psL', 'psS', 'onept', 'extrema', 'mf', 'dog', 'cnn', 'pred0']
    comp['consP8'] = cons(TH, P8, sets)
    comp['consP10'] = cons(TH, P8 + ['cnn1', 'pred1'], sets)
    den = [f for f in ('den_r', 'den_sd') if f in TH]
    stx = [f for f in STX if f in TH]
    bands = [f for f in BANDS if f in TH]
    rich = {
        'consR_den': P8 + ['cnn1', 'pred1'] + den,
        'consR_stx': P8 + ['cnn1', 'pred1'] + stx,
        'consR_all': P8 + ['cnn1', 'pred1'] + den + stx,
        'consR_allb': P8 + ['cnn1', 'pred1'] + den + stx + bands,
        'consR_bands': bands + ['cnn', 'cnn1', 'pred0', 'pred1', 'dog'] + den,
    }
    print('=== reference / components')
    for k in ('Sd', 'C', 'C1', 'xSL', 'consP8', 'consP10'):
        row(k, comp[k])
    for nm, fs in rich.items():
        if len(set(fs)) < len(fs) or any(f not in TH for f in fs):
            continue
        for kind in ('glob', 'cond'):
            comp[f'{nm}_{kind}'] = cons(TH, fs, sets, kind)
            row(f'{nm}_{kind} (K={len(fs)})', comp[f'{nm}_{kind}'])
    # single new-view pairs vs the CNN mean prediction (which views disagree most on OoD?)
    print('=== each view vs CNN mean prediction (5-dim, cond cov)')
    TH['predm'] = {s: 0.5 * (TH['pred0'][s] + TH['pred1'][s]) for s in sets}
    for f in den + stx + bands:
        d = {s: TH[f][s] - TH['predm'][s] for s in sets}
        c = {s: 0.5 * (TH[f][s] + TH['predm'][s]) for s in sets}
        m2 = mahal_cond(d, c)
        z = standardize({s: np.log(m2[s] + 1.0) for s in sets}, c)
        comp[f'pair_{f}-predm'] = z
        row(f'pair {f}-predm', z)
    for f in den + stx:
        comp[f'fam_{f}'] = standardize({s: F[f][f'{s}/cop'] for s in sets}, TH[f])
        row(f'fam {f} (own likelihood)', comp[f'fam_{f}'])
    cp = lib.OUT / 'cross' / 'r2.npz'
    if cp.exists():
        with np.load(cp) as d:
            Y = {k: d[k] for k in d.files}
        cs = [s for s in sets if f'{s}/done' in Y]
        if len(cs) == len(sets):
            print('=== 4-band scale tensions and plug-in energies')
            fams_c = sorted({k.split('_', 1)[1].split('@')[0] for k in Y if k.startswith('audit/cop_')})
            for kind in ('cop', 'E'):
                if all(b in fams_c for b in BANDS):
                    raw = {s: sum(Y[f'{s}/{kind}_{bi}@{bj}'] - Y[f'{s}/{kind}_{bi}@{bi}']
                                  for bi in BANDS for bj in BANDS if bi != bj) for s in sets}
                    cen = {s: np.mean([TH[b][s] for b in BANDS], 0) for s in sets}
                    comp[f'xB4_{kind}'] = standardize(raw, cen)
                    row(f'xB4 {kind} (all band pairs)', comp[f'xB4_{kind}'])
                    raw = {s: sum(Y[f'{s}/{kind}_{bi}@{bj}'] - Y[f'{s}/{kind}_{bi}@{bi}']
                                  for bi, bj in (('b0', 'b1'), ('b1', 'b0'), ('b1', 'b2'), ('b2', 'b1'),
                                                 ('b2', 'b3'), ('b3', 'b2'))) for s in sets}
                    comp[f'xB4adj_{kind}'] = standardize(raw, cen)
                    row(f'xB4 {kind} (adjacent pairs)', comp[f'xB4adj_{kind}'])
                # plug-in at the CNN mean prediction, averaged over families as z's
                for setname, pf in (('PIcnn', ['cnn', 'cnn1']), ('PIstats', ['dog', 'small', 'large']),
                                    ('PIall', ['cnn', 'cnn1', 'dog', 'small', 'large'] + den + stx)):
                    pf = [f for f in pf if f in fams_c]
                    zs = [standardize({s: Y[f'{s}/{kind}_{f}@predm'] - Y[f'{s}/{kind}_{f}@{f}'] for s in sets},
                                      TH['predm']) for f in pf]
                    comp[f'{setname}_{kind}'] = mean_of(*zs)
                    row(f'{setname} {kind} plug-in @predm (K={len(pf)})', comp[f'{setname}_{kind}'])
    # ------------------------------------------------------------------ recipes
    best_rich = max([k for k in comp if k.startswith('consR_')], key=lambda k: ev(comp[k])['shift_mean'],
                    default='consP10')
    print('best rich consistency by m15:', best_rich)
    recipes = {
        'BASE': {'Sd': 1, 'C': 1},
        'REC1 (LB 0.5669)': {'Sd': 1, 'C': 1, 'xSL': .5, 'consP8': .5},
        'a: Sd+C+1xSL+1consP8': {'Sd': 1, 'C': 1, 'xSL': 1, 'consP8': 1},
        'b: Sd+C+1xSL+1consRich': {'Sd': 1, 'C': 1, 'xSL': 1, best_rich: 1},
        'c: .5Sd+.5C+1xSL+1consRich': {'Sd': .5, 'C': .5, 'xSL': 1, best_rich: 1},
        'Sd+C+1xSL+1consP10': {'Sd': 1, 'C': 1, 'xSL': 1, 'consP10': 1},
    }
    for extra in ('xB4_cop', 'xB4adj_cop', 'PIcnn_E', 'PIall_E', 'PIall_cop', 'PIstats_cop'):
        if extra in comp:
            recipes[f'b+1{extra}'] = {'Sd': 1, 'C': 1, 'xSL': 1, best_rich: 1, extra: 1}
            recipes[f'c+1{extra}'] = {'Sd': .5, 'C': .5, 'xSL': 1, best_rich: 1, extra: 1}
    print('=== recipes')
    Z = {}
    for name, w in recipes.items():
        Z[name] = mean_of(*[comp[k] for k in w], w=list(w.values()))
        row(name, Z[name])
    for name in filter(None, args.boot.split(';')):
        m, sd, pp = boot_diff(Z[name], None, Z['REC1 (LB 0.5669)'], None, n=100)
        print(f'  boot vs REC1 {name:40s} {m:+.4f} +- {sd:.4f} P>0={pp:.2f}')
    d = lib.OUT / 'export2'
    d.mkdir(exist_ok=True)
    for spec in filter(None, args.export.split(';')):
        name, _, fname = spec.partition('=')
        z = Z[name]
        assert z['test'].shape == (10000,) and np.isfinite(z['test']).all()
        np.save(d / f'test_z_{fname}.npy', z['test'].astype(np.float64))
        np.savez(d / f'z_{fname}.npz', **z)
        np.savez(d / f'components_{fname}.npz',
                 **{f'{c}/{k}': v for c in recipes[name] for k, v in comp[c].items()})
        (d / f'recipe_{fname}.txt').write_text(f'{name}: {recipes[name]}\n')
        print('exported', name, '->', d / f'test_z_{fname}.npy')


if __name__ == '__main__':
    main()
