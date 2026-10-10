"""Frozen candidate recipes: evaluate on audit3 vs shifts, paired bootstrap vs BASE, export test z.

Components (all fitted on fit3, standardized on calib3, per map):
  Sd     copula profile score of the DoG-wavelet model (== r1 Sd, bit-identical refit)
  C      copula profile score of the CNN-embedding model (== r1 C)
  consP8 internal consistency: Mahalanobis distance (calib3 covariance) between the
         profile MLEs of 6 statistic families (psL, psS, onept, extrema, mf, dog), the CNN
         model and the CNN's own predicted parameters; log(1+d2), standardized vs mean theta
  xSL    scale consistency (Dai & Seljak style, both directions):
         [cop_small(theta_large) - cop_small(theta_small)] + [cop_large(theta_small) - cop_large(theta_large)]
         small = PS k>median + all stats at smoothing 0,1 px + fine DoG bands; large = the rest
         except the 2 px scale; standardized vs (theta_small + theta_large)/2
Score = weighted mean of the standardized z's.
"""
import argparse

import numpy as np

import lib
from analyze import standardize, ev, mean_of, get, fam, mahal, boot_diff

RECIPES = {
    'BASE': {'Sd': 1, 'C': 1},
    'BASE+0.5xSL': {'Sd': 1, 'C': 1, 'xSL': 0.5},
    'BASE+1consP8': {'Sd': 1, 'C': 1, 'consP8': 1},
    'BASE+0.5xSL+0.5consP8': {'Sd': 1, 'C': 1, 'xSL': 0.5, 'consP8': 0.5},
    'BASE+0.5xSL+1consP8': {'Sd': 1, 'C': 1, 'xSL': 0.5, 'consP8': 1},
    'BASE+0.5xSL+0.5consP10': {'Sd': 1, 'C': 1, 'xSL': 0.5, 'consP10': 0.5},
    'BASE3': {'Sd': 1, 'C': 1, 'C1': 1},
    'BASE3+0.5xSL+0.5consP10': {'Sd': 1, 'C': 1, 'C1': 1, 'xSL': 0.5, 'consP10': 0.5},
    'BASE3+0.5xSL+0.5consP8': {'Sd': 1, 'C': 1, 'C1': 1, 'xSL': 0.5, 'consP8': 0.5},
    # optional, prior-dependent: typicality of the CNN-model best fit under the InD theta_hat density
    'BASE+0.5xSL+0.5consP8+0.5typC': {'Sd': 1, 'C': 1, 'xSL': 0.5, 'consP8': 0.5, 'typC': 0.5},
}
SCALE = 'scale2'


def components(sets):
    F = {f: fam(f) for f in ('psL', 'psS', 'onept', 'extrema', 'mf', 'dog', 'cnn', 'cnn1')}
    comp = {}
    comp['Sd'] = standardize({s: F['dog'][f'{s}/cop'] for s in sets}, {s: F['dog'][f'{s}/th'] for s in sets})
    comp['C'] = standardize({s: F['cnn'][f'{s}/cop'] for s in sets}, {s: F['cnn'][f'{s}/th'] for s in sets})
    comp['C1'] = standardize({s: F['cnn1'][f'{s}/cop'] for s in sets}, {s: F['cnn1'][f'{s}/th'] for s in sets})
    E = {f: {s: F[f][f'{s}/th'] for s in sets} for f in F}
    E['cnnpred'] = {s: lib.load('cnn0', s)[..., -5:].reshape(-1, 5) for s in sets}
    E['cnn1pred'] = {s: lib.load('cnn1', s)[..., -5:].reshape(-1, 5) for s in sets}
    for name, fs in (('consP8', ['psL', 'psS', 'onept', 'extrema', 'mf', 'dog', 'cnn', 'cnnpred']),
                     ('consP10', ['psL', 'psS', 'onept', 'extrema', 'mf', 'dog', 'cnn', 'cnnpred', 'cnn1',
                                  'cnn1pred'])):
        vec = {}
        for s in sets:
            A = np.stack([E[f][s] for f in fs], 1)
            vec[s] = (A - A.mean(1, keepdims=True))[:, 1:].reshape(len(A), -1)
        m2 = mahal(vec)
        center = {s: np.stack([E[f][s] for f in fs], 1).mean(1) for s in sets}
        comp[name] = standardize({s: np.log(m2[s] + 1.0) for s in sets}, center)
    # theta_hat typicality: -log density of the C-model profile MLE under a 12-component GMM fitted on calib3
    from sklearn.mixture import GaussianMixture
    gm = GaussianMixture(12, covariance_type='full', random_state=0, reg_covar=1e-3).fit(E['cnn']['calib'])
    raw = {s: -gm.score_samples(E['cnn'][s]) for s in sets}
    m, sd = raw['calib'].mean(), raw['calib'].std()
    comp['typC'] = {s: (raw[s] - m) / sd for s in sets}
    with np.load(lib.OUT / 'scale' / f'{SCALE}.npz') as d:
        X = {k: d[k] for k in d.files}
    raw = {s: (X[f'{s}/cop_small@large'] - X[f'{s}/cop_small@small']) +
           (X[f'{s}/cop_large@small'] - X[f'{s}/cop_large@large']) for s in sets}
    comp['xSL'] = standardize(raw, {s: 0.5 * (X[f'{s}/th_small'] + X[f'{s}/th_large']) for s in sets})
    return comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boot', type=int, default=200)
    ap.add_argument('--export', default='')
    args = ap.parse_args()
    sets = lib.SETS + ['test']
    comp = components(sets)
    Z = {}
    for name, w in RECIPES.items():
        Z[name] = mean_of(*[comp[k] for k in w], w=list(w.values()))
    for k in ('Sd', 'C', 'C1', 'consP8', 'consP10', 'xSL', 'typC'):
        print(lib.fmt(f'component {k}', ev(comp[k])))
    print('corr on audit normal:', {f'{a}-{b}': round(float(np.corrcoef(comp[a]['audit'], comp[b]['audit'])[0, 1]), 3)
                                    for a in comp for b in comp if a < b})
    for name, z in Z.items():
        r = ev(z)
        print(lib.fmt(name, r), flush=True)
        print('     ', ' '.join(f'{k}={v:.3f}' for k, v in r['shifts'].items()))
        if args.boot and name not in ('BASE', 'BASE3'):
            ref = 'BASE3' if name.startswith('BASE3') else 'BASE'
            m, sd, pp = boot_diff(z, None, Z[ref], None, n=args.boot)
            print(f'      (vs {ref})', end='')
            print(f'      paired bootstrap shift_mean diff vs BASE: {m:+.4f} +- {sd:.4f}  P(>0)={pp:.2f}', flush=True)
    for name in filter(None, args.export.split(',')):
        z = Z[name]
        d = lib.OUT / 'export'; d.mkdir(exist_ok=True)
        assert z['test'].shape == (10000,) and np.isfinite(z['test']).all()
        np.save(d / f'test_z_{name}.npy', z['test'].astype(np.float64))
        np.savez(d / f'z_{name}.npz', **{k: v for k, v in z.items() if k != 'test'})
        np.savez(d / f'components_{name}.npz', **{f'{c}/{k}': v for c in RECIPES[name] for k, v in comp[c].items()})
        print('exported', name, '->', d / f'test_z_{name}.npy')


if __name__ == '__main__':
    main()
