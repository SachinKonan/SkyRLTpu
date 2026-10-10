"""Normal-tail quantiles under leave-cosmology-out / leave-baryon-model-out refits (CPU)."""
import numpy as np

import lib
from analyze import standardize, mahal, mean_of
from holdout import split, sets_for

Q = [.5, .9, .99, .995, .999]
NAMES = ('calib', 'in', 'out')


def q(z):
    return ' '.join(f'{v:6.2f}' for v in np.quantile(z, Q))


def main():
    for mode in ('lco', 'lbo'):
        d = lib.OUT / 'holdout' / mode
        if not (d / 'dog.npz').exists():
            continue
        held, isout = split(mode)
        fams = [f for f in ('dog', 'cnn', 'cnn1', 'psL', 'psS', 'onept', 'extrema', 'mf', 'small', 'mid', 'large')
                if (d / f'{f}.npz').exists()]
        H = {f: dict(np.load(d / f'{f}.npz')) for f in fams}
        print(f'===== {mode}: held-out units {sorted(held.tolist())}')
        print(f'{"score":40s} {"set":5s} n     ' + ' '.join(f'{x:>6}' for x in ['q50', 'q90', 'q99', 'q99.5', 'q99.9']))
        Z = {}
        for f in fams:
            raw = {n: H[f][f'{n}/cop'] for n in NAMES}
            th = {n: H[f][f'{n}/th'] for n in NAMES}
            Z[f] = standardize(raw, th)
        rows = {}
        for f in ('dog', 'cnn'):
            if f in Z:
                rows[f'fam {f} ({"Sd" if f == "dog" else "C"})'] = Z[f]
        if 'dog' in Z and 'cnn' in Z:
            rows['BASE mean:Sd+C'] = mean_of(Z['dog'], Z['cnn'])
        if 'cnn1' in Z:
            rows['C1'] = Z['cnn1']
            rows['BASE3 mean:Sd+C+C1'] = mean_of(Z['dog'], Z['cnn'], Z['cnn1'])
        p8 = [f for f in ('psL', 'psS', 'onept', 'extrema', 'mf', 'dog', 'cnn') if f in H]
        if len(p8) == 7:
            cp = {}
            for n in NAMES:
                x = sets_for('cnn', isout)[n][..., -5:]
                cp[n] = x.reshape(-1, 5)
            E = {f: {n: H[f][f'{n}/th'] for n in NAMES} for f in p8}
            E['cnnpred'] = cp
            fs = list(E)
            vec = {}
            for n in NAMES:
                A = np.stack([E[f][n] for f in fs], 1)
                D = A - A.mean(1, keepdims=True)
                vec[n] = D[:, 1:].reshape(len(D), -1)
            m2 = mahal(vec)
            center = {n: np.stack([E[f][n] for f in fs], 1).mean(1) for n in NAMES}
            zc = standardize({n: np.log(m2[n] + 1.0) for n in NAMES}, center)
            rows['cons th P8 all5'] = zc
            rows['BASE+1*cons th P8 all5'] = mean_of(Z['dog'], Z['cnn'], zc)
        cp = d / 'cross.npz'
        if cp.exists():
            X = dict(np.load(cp))
            raw = {n: (X[f'{n}/cop_small@large'] - X[f'{n}/cop_small@small']) +
                   (X[f'{n}/cop_large@small'] - X[f'{n}/cop_large@large']) for n in NAMES}
            xsl = standardize(raw, {n: 0.5 * (X[f'{n}/th_small'] + X[f'{n}/th_large']) for n in NAMES})
            rows['xSL'] = xsl
            rows['BASE+0.5xSL'] = mean_of(Z['dog'], Z['cnn'], xsl, w=[1, 1, .5])
            if 'cons th P8 all5' in rows:
                rows['BASE+0.5xSL+0.5consP8'] = mean_of(Z['dog'], Z['cnn'], xsl, rows['cons th P8 all5'],
                                                        w=[1, 1, .5, .5])
                from sklearn.mixture import GaussianMixture
                gm = GaussianMixture(12, covariance_type='full', random_state=0, reg_covar=1e-3).fit(H['cnn']['calib/th'])
                raw = {n: -gm.score_samples(H['cnn'][f'{n}/th']) for n in NAMES}
                typ = {n: (raw[n] - raw['calib'].mean()) / raw['calib'].std() for n in NAMES}
                rows['typC'] = typ
                rows['BASE+0.5xSL+0.5consP8+0.5typC'] = mean_of(Z['dog'], Z['cnn'], xsl, rows['cons th P8 all5'], typ,
                                                                w=[1, 1, .5, .5, .5])
                if 'cnn1' in Z:
                    rows['BASE3+0.5xSL+0.5consP8'] = mean_of(Z['dog'], Z['cnn'], Z['cnn1'], xsl,
                                                             rows['cons th P8 all5'], w=[1, 1, 1, .5, .5])
            for f, src in [('small', 'jointSC'), ('small', 'large'), ('large', 'small')]:
                for kind in ('E', 'cop'):
                    key = f'{kind}_{f}@{src}'
                    if f'in/{key}' not in X:
                        continue
                    raw = {n: X[f'{n}/{key}'] - X[f'{n}/{kind}_{f}@{f}'] for n in NAMES}
                    z = standardize(raw, {n: X[f'{n}/th_{src}'] for n in NAMES})
                    rows[f'dlogp[{kind}] {f}|{src}'] = z
                    if 'dog' in Z:
                        rows[f'BASE+1*dlogp[{kind}] {f}|{src}'] = mean_of(Z['dog'], Z['cnn'], z)
        for name, z in rows.items():
            for n in ('in', 'out'):
                print(f'{name:40s} {n:5s} {len(z[n]):5d} {q(z[n])}')
            # rank-level check: fraction of held-out maps above the in-sample 99/99.5/99.9% thresholds
            t = np.quantile(z['in'], [.99, .995, .999])
            print(f'{"":40s} out-exceedance of in-thresholds (expect 1.0/0.5/0.1%): ' +
                  ' '.join(f'{100 * np.mean(z["out"] > x):.2f}%' for x in t))


if __name__ == '__main__':
    main()
