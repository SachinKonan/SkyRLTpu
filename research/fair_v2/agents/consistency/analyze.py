"""Build and evaluate consistency / split / boundary / calibration candidates (CPU, numpy).

Everything is fitted on calib3 (standardization, covariance of estimate
differences, Gaussianizing tail maps) and evaluated on audit3 normal vs every
synthetic shift with the official functional. Test arrays are carried along
untouched and only written for the frozen final candidates (--export).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats as st

import lib
from model import Standardizer, DZ_SIGMA

ALL = lib.SETS + ['test']  # sets missing for a source are skipped by callers
FAMDIR = lib.OUT / 'fam'
LABF = lib.labels('fit')
TM, TS = LABF.mean(0), LABF.std(0)
COS = (np.unique(LABF[:, :2], axis=0) - TM[:2]) / TS[:2]
_dd = np.sqrt(((COS[:, None] - COS[None]) ** 2).sum(-1)) + 1e9 * np.eye(len(COS))
HULL_R = float(np.median(_dd.min(1)))
T_LO = (LABF.min(0) - TM) / TS
T_HI = (LABF.max(0) - TM) / TS


def prior_np(t):
    dz = t[:, 4] * TS[4] + TM[4]
    dmin = np.sqrt(((t[:, None, :2] - COS[None]) ** 2).sum(-1)).min(1)
    return (dz / DZ_SIGMA) ** 2 + 50.0 * np.maximum(dmin - HULL_R, 0) ** 2


def raw_theta(tn):
    return tn * TS + TM


def standardize(raw, theta):
    """raw, theta: dict set -> array (theta normalized). Fit Standardizer on calib."""
    s = Standardizer().fit(raw['calib'], raw_theta(theta['calib']), TM, TS)
    return {k: s(raw[k], raw_theta(theta[k])) for k in raw}


def gaussianize(z, tail_frac=0.05):
    """Monotone map fitted on calib: Phi^-1(ECDF) in the bulk, exponential tails beyond the
    calib (1-tail_frac) quantiles (upper tail extrapolated continuously)."""
    c = np.sort(z['calib'])
    n = len(c)
    qs = (np.arange(n) + 0.5) / n
    u0 = np.quantile(c, 1 - tail_frac)
    exc = c[c > u0] - u0
    beta = exc.mean()            # exponential tail scale
    def f(x):
        p = np.interp(x, c, qs)
        sf = np.where(x > u0, tail_frac * np.exp(-(x - u0) / beta), 1 - p)
        sf = np.clip(sf, 1e-300, 1 - 1e-12)
        return -st.norm.ppf(sf)  # = Phi^-1(1 - sf)
    return {k: f(v) for k, v in z.items()}


def tailcal(z, theta, q=0.95, lin=True):
    """Conditional tail calibration: quantile-regress the median and the q-quantile of the
    standardized score on theta (calib) and map z -> (z - m(theta)) / (q(theta) - m(theta))."""
    from sklearn.linear_model import QuantileRegressor
    from model import design
    X = {k: design(raw_theta(v), TM, TS) for k, v in theta.items()}
    if lin:
        X = {k: v[:, :6] for k, v in X.items()}
    fm = QuantileRegressor(quantile=0.5, alpha=1e-4, solver='highs', fit_intercept=False).fit(X['calib'], z['calib'])
    fq = QuantileRegressor(quantile=q, alpha=1e-4, solver='highs', fit_intercept=False).fit(X['calib'], z['calib'])
    out = {}
    for k in z:
        m, h = fm.predict(X[k]), fq.predict(X[k])
        out[k] = (z[k] - m) / np.maximum(h - m, 0.2)
    return out


def ev(z):
    return lib.evaluate(z['audit'], {s: z['shift_' + s] for s in lib.SHIFTS if 'shift_' + s in z})


def boot_diff(za, zb, base_a, base_b, n=200, seed=0):
    """Paired bootstrap over audit maps (normal and shifted versions share the map index):
    distribution of shift_mean(candidate) - shift_mean(base)."""
    from model import phase2_score
    rng = np.random.default_rng(seed)
    N = len(za['audit'])
    out = []
    for _ in range(n):
        i = rng.integers(0, N, N)
        a = np.mean([phase2_score(za['audit'][i], za['shift_' + s][i]) for s in lib.SHIFTS15])
        b = np.mean([phase2_score(base_a['audit'][i], base_a['shift_' + s][i]) for s in lib.SHIFTS15])
        out.append(a - b)
    out = np.array(out)
    return float(out.mean()), float(out.std()), float((out > 0).mean())


def mean_of(*zs, w=None):
    w = np.ones(len(zs)) if w is None else np.asarray(w)
    return {k: sum(wi * zi[k] for wi, zi in zip(w, zs)) / w.sum() for k in zs[0]}


def mahal(vecs, ridge=1e-3):
    """vecs: dict set -> (N, p). Covariance from calib; returns squared distance dict."""
    c = vecs['calib']
    mu = c.mean(0)
    C = np.cov(c - mu, rowvar=False)
    C = C + ridge * np.trace(C) / len(C) * np.eye(len(C))
    Ci = np.linalg.inv(C)
    return {k: np.einsum('ij,jk,ik->i', v - mu, Ci, v - mu) for k, v in vecs.items()}


def fam(f):
    with np.load(FAMDIR / f'{f}.npz') as d:
        return {k: d[k] for k in d.files}


def get(d, key):
    return {s: d[f'{s}/{key}'] for s in ALL}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--export', default='')
    ap.add_argument('--boot', default='')
    args = ap.parse_args()
    results = {}
    rows = []

    REG = {}

    def report(name, z, keep=True):
        r = ev(z)
        results[name] = r
        REG[name] = z
        print(lib.fmt(name, r), flush=True)
        return z

    zSd, zC = lib.r1_z('Sd'), lib.r1_z('C')
    base = report('BASE mean:Sd+C', mean_of(zSd, zC))
    report('r1 Sd', zSd); report('r1 C', zC)
    fams = [f for f in lib.FAMILIES if (FAMDIR / f'{f}.npz').exists()]
    F = {f: fam(f) for f in fams}
    Z = {}
    # ---------------------------------------------------------------- (b) per family
    print('--- per-family profile scores (copula energy at box best fit)')
    for f in fams:
        Z[f] = standardize(get(F[f], 'cop'), get(F[f], 'th'))
        report(f'fam {f}', Z[f])
    P8 = [f for f in ['psL', 'psS', 'dir', 'onept', 'extrema', 'mf', 'dog', 'cnn'] if f in fams]
    G = {f: gaussianize(Z[f]) for f in fams}
    print('--- (b) split-likelihood combinations over', P8)
    report('split mean z P8', mean_of(*[Z[f] for f in P8]))
    report('split stouffer(gauss) P8', mean_of(*[G[f] for f in P8]))
    report('split max(gauss) P8', {k: np.max([G[f][k] for f in P8], 0) for k in ALL})
    fisher = {k: sum(-2 * st.norm.logsf(G[f][k]) for f in P8) for k in ALL}
    report('split fisher P8', fisher)
    stats_only = [f for f in P8 if f != 'cnn']
    report('split stouffer stats-fams + C', mean_of(mean_of(*[G[f] for f in stats_only]), gaussianize(zC)))
    # -------------------------------------------------------------- (a) consistency
    print('--- (a) internal consistency of parameter estimates')
    cnnpred = {s: lib.load('cnn0', s)[..., -5:].reshape(-1, 5) for s in ALL}
    est_sets = {
        'reg': {f: get(F[f], 'reg') for f in fams},
        'th': {f: get(F[f], 'th') for f in fams},
    }
    est_sets['reg']['cnnpred'] = cnnpred
    est_sets['th']['cnnpred'] = cnnpred
    cons = {}
    for kind, E in est_sets.items():
        for famset_name, famset in [('P8', [f for f in P8 if f != 'dir'] + ['cnnpred']),
                                    ('scale', ['psL', 'psS', 'fine', 'coarse']),
                                    ('stats-vs-cnn', ['onept', 'extrema', 'mf', 'dog', 'psS', 'cnnpred']),
                                    ('ps-vs-cnn', ['psL', 'psS', 'cnnpred'])]:
            famset = [f for f in famset if f in E]
            if len(famset) < 2:
                continue
            for dims_name, dims in [('all5', [0, 1, 2, 3, 4]), ('cos', [0, 1]), ('cosT', [0, 1, 2])]:
                vec = {}
                for s in ALL:
                    A = np.stack([E[f][s][:, dims] for f in famset], 1)  # (N, K, p)
                    D = A - A.mean(1, keepdims=True)
                    vec[s] = D[:, 1:].reshape(len(D), -1)   # drop one (sum to zero)
                m2 = mahal(vec)
                center = {s: np.stack([E[f][s] for f in famset], 1).mean(1) for s in ALL}
                raw = {s: np.log(m2[s] + 1.0) for s in ALL}
                z = standardize(raw, center)
                name = f'cons {kind} {famset_name} {dims_name}'
                cons[name] = z
                report(name, z)
    # deep-ensemble disagreement between two CNN seeds (when a second seed exists)
    for c2 in [lib.ROOT / 'cnn1', Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3tmp_cnna/cnn0')]:
        if all((c2 / f'{s}.npz').exists() for s in ALL):
            p2 = {}
            for s in ALL:
                with np.load(c2 / f'{s}.npz') as d:
                    p2[s] = d['feats'][..., -5:].reshape(-1, 5)
            # same noisy maps? (stats recomputed by the same pipeline must match bit-for-bit)
            for dims_name, dims in [('all5', [0, 1, 2, 3, 4]), ('cos', [0, 1])]:
                vec = {s: (cnnpred[s] - p2[s])[:, dims] for s in ALL}
                m2 = mahal(vec)
                center = {s: 0.5 * (cnnpred[s] + p2[s]) for s in ALL}
                z = standardize({s: np.log(m2[s] + 1.0) for s in ALL}, center)
                cons[f'ens-disagree {c2.parent.name}/{c2.name} {dims_name}'] = z
                report(f'ens-disagree {dims_name}', z)
            # stats-family regressors vs mean of both CNN seeds
            break
    # ------------------------------------------------------------ tension (joint)
    jp = lib.OUT / 'joint' / 'P8.npz'
    if jp.exists():
        print('--- (a/b) split tension: sum_b E_b(theta*) - sum_b min E_b')
        with np.load(jp) as d:
            J = {k: d[k] for k in d.files}
        jf = [k.split('/E_')[1] for k in J if k.startswith('audit/E_')]
        T = {}
        for s in ALL:
            own = sum(F[f][f'{s}/E'] - prior_np(F[f][f'{s}/th']) for f in jf)
            T[s] = sum(J[f'{s}/E_{f}'] for f in jf) - own
        th = {s: J[f'{s}/thstar'] for s in ALL}
        zT = standardize({s: np.log(np.maximum(T[s], 0) + 5.0) for s in ALL}, th)
        cons['tension P8'] = zT
        report('tension P8', zT)
        zJ = standardize({s: sum(J[f'{s}/cop_{f}'] for f in jf) for s in ALL}, th)
        report('joint-product cop at theta*', zJ)
        cons['jointprod'] = zJ
        for f in jf:
            d = {s: J[f'{s}/E_{f}'] - (F[f][f'{s}/E'] - prior_np(F[f][f'{s}/th'])) for s in ALL}
            report(f'tension part {f}', standardize(d, th))
    # ---------------------------------------------------------------- (c) boundary
    print('--- (c) boundary: energy drop when the prior box is relaxed')
    bnd = {}
    R1TH = {}
    for tag in ['Sd', 'C', 'S', 'SC']:
        p = lib.OUT / 'r1box' / f'{tag}.npz'
        if not p.exists():
            continue
        with np.load(p) as d:
            B = {k: d[k] for k in d.files}
        th = get(B, 'th')
        R1TH[tag] = th
        drop = {s: B[f'{s}/E'] - B[f'{s}/Er'] for s in ALL}
        zb = standardize({s: np.log(np.maximum(drop[s], 0) + 1.0) for s in ALL}, th)
        bnd[tag] = zb
        report(f'box-drop {tag}', zb)
        pinned = {s: ((th[s] <= T_LO + 1e-4) | (th[s] >= T_HI - 1e-4)).sum(1).astype(float) for s in ALL}
        print(f'    pinned coords: audit mean {pinned["audit"].mean():.3f}; ' +
              ' '.join(f'{k[:6]}={pinned["shift_"+k].mean():.2f}' for k in lib.SHIFTS))
        zr = standardize(get(B, 'copr'), get(B, 'thr'))
        report(f'relaxed-cop {tag}', zr)
        bnd[tag + 'relcop'] = zr
    # -------------------------------------------------------------- combos w/ base
    print('--- combinations with the current best (mean of z: Sd, C, new)')
    cands = dict(cons)
    cands.update({f'fam {f}': Z[f] for f in fams if f not in ('cnn', 'dog')})
    cands['split mean stats-fams'] = mean_of(*[Z[f] for f in stats_only])
    cands.update({f'box {k}': v for k, v in bnd.items()})
    for name, z in cands.items():
        for w in (0.5, 1.0):
            report(f'BASE+{w}*[{name}]', mean_of(zSd, zC, z, w=[1, 1, w]))
    print('--- (d) calibration: gaussianized members before averaging')
    gS, gC = gaussianize(zSd), gaussianize(zC)
    report('gauss mean:Sd+C', mean_of(gS, gC))
    report('gauss max:Sd+C', {k: np.maximum(gS[k], gC[k]) for k in ALL})
    report('fisher Sd+C', {k: -2 * (st.norm.logsf(gS[k]) + st.norm.logsf(gC[k])) for k in ALL})
    if 'Sd' in R1TH and 'C' in R1TH:
        for q in (0.9, 0.97):
            for lin in (True, False):
                tS, tC = tailcal(zSd, R1TH['Sd'], q, lin), tailcal(zC, R1TH['C'], q, lin)
                report(f'tailcal q{q} lin={lin} Sd', tS)
                report(f'tailcal q{q} lin={lin} C', tC)
                report(f'tailcal q{q} lin={lin} mean:Sd+C', mean_of(tS, tC))
                report(f'tailcal q{q} lin={lin} gauss mean:Sd+C', mean_of(gaussianize(tS), gaussianize(tC)))
    (lib.OUT / 'results.json').write_text(json.dumps(results, indent=1))
    if args.boot:
        print('--- paired bootstrap (200 reps over audit maps) of shift_mean difference vs BASE')
        for name in args.boot.split(';'):
            m, sd, pp = boot_diff(REG[name], None, base, None)
            print(f'  {name:45s} diff={m:+.4f} +- {sd:.4f}  P(>0)={pp:.2f}', flush=True)
    for name in filter(None, args.export.split(';')):
        z = REG[name]
        slug = ''.join(c if c.isalnum() or c in '+-.' else '_' for c in name)
        d = lib.OUT / 'export'; d.mkdir(exist_ok=True)
        assert z['test'].shape == (10000,) and np.isfinite(z['test']).all()
        np.save(d / f'test_z_{slug}.npy', z['test'].astype(np.float64))
        np.savez(d / f'z_{slug}.npz', **z)
        print('exported', name, '->', d / f'test_z_{slug}.npy')


if __name__ == '__main__':
    main()
