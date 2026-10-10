"""Round 4: linear (ridge) theta-probes on reconstruction / latent features as extra consistency views.

Probe: standardized features -> 5 normalized parameters, ridge alpha chosen on calib3 MSE.
LCO: refit inside each fold on fit3 minus the held-out cosmologies (alpha on the fold's calib).
consRplus = calib3-covariance Mahalanobis over the consR views (4 scale bands, Sd, C, C1 profiles,
cnn0/cnn1 predictions, den_r, den_s:dog) plus selected probe views.
"""
import argparse
import json

import numpy as np

import lib
from analyze import standardize, mahal, mean_of, ev, fam
from holdout3 import masks
from lco3_analyze import components as lco_components, evaluate, pred_sets
from model import phase2_score

PROBES = {'den_b': ('den_b', slice(None)), 'cnn0e': ('cnn0', slice(0, 256)), 'cnn1e': ('cnn1', slice(0, 256)),
          'dif': ('dif', slice(None)), 'difa': ('difa', slice(None)), 'difk': ('difk', slice(None)),
          'oe': ('oe', slice(None)), 'oe_head': ('oe_head', slice(None))}
ALPHAS = [1e-1, 1, 10, 100, 1e3, 1e4]
RB = ['b0', 'b1', 'b2', 'b3', 'cnn', 'cnn1', 'pred0', 'pred1', 'dog', 'den_r', 'den_sd']


def feats(p, name):
    src, sl = PROBES[p]
    return lib.load(src, name)[..., sl]


def ridge_fit(Xf, yf, Xc, yc):
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-8
    A = (Xf - mu) / sd
    ym = yf.mean(0)
    G = A.T @ A
    b = A.T @ (yf - ym)
    best = None
    for a in ALPHAS:
        W = np.linalg.solve(G + a * len(A) / 1000 * np.eye(len(G)), b)
        mse = np.mean((((Xc - mu) / sd) @ W + ym - yc) ** 2)
        if best is None or mse < best[0]:
            best = (mse, a, W)
    _, a, W = best
    return lambda X: ((X - mu) / sd) @ W + ym, a, best[0]


def fold_probe(p, k):
    m = masks(k)
    L = {n: lib.labels(n) for n in ('fit', 'calib', 'audit')}
    Lf = L['fit'][~m['fit']]
    tm, ts = Lf.mean(0), Lf.std(0)
    F = {n: feats(p, n) for n in ('fit', 'calib', 'audit')}
    Xf = F['fit'][~m['fit']]
    yf = np.repeat((Lf - tm) / ts, Xf.shape[1], 0)
    Xf = Xf.reshape(-1, Xf.shape[-1]).astype(np.float64)
    Xc = F['calib'][~m['calib']]
    yc = np.repeat((L['calib'][~m['calib']] - tm) / ts, Xc.shape[1], 0)
    Xc = Xc.reshape(-1, Xc.shape[-1]).astype(np.float64)
    f, a, mse = ridge_fit(Xf, yf, Xc, yc)
    d = lambda x: f(x.reshape(-1, x.shape[-1]).astype(np.float64))
    out = {'calib': d(F['calib'][~m['calib']]), 'in': d(F['audit'][~m['audit']]),
           'out': np.concatenate([d(F['calib'][m['calib']]), d(F['audit'][m['audit']])])}
    for s in lib.SHIFTS:
        out['sh_' + s] = d(feats(p, 'shift_' + s)[m['audit']])
    return out, a, mse


def full_probe(p):
    L = {n: lib.labels(n) for n in ('fit', 'calib')}
    tm, ts = L['fit'].mean(0), L['fit'].std(0)
    Xf = feats(p, 'fit')
    yf = np.repeat((L['fit'] - tm) / ts, Xf.shape[1], 0)
    Xc = feats(p, 'calib')
    yc = np.repeat((L['calib'] - tm) / ts, Xc.shape[1], 0)
    f, a, mse = ridge_fit(Xf.reshape(-1, Xf.shape[-1]).astype(np.float64), yf,
                          Xc.reshape(-1, Xc.shape[-1]).astype(np.float64), yc)
    return {s: f(feats(p, s).reshape(-1, feats(p, s).shape[-1]).astype(np.float64)) for s in lib.SETS + ['test']}, a, mse


def cons(TH, fs, names):
    vec, cen = {}, {}
    for n in names:
        A = np.stack([TH[f][n] for f in fs], 1)
        vec[n] = (A - A.mean(1, keepdims=True))[:, 1:].reshape(len(A), -1)
        cen[n] = A.mean(1)
    m2 = mahal(vec)
    return standardize({n: np.log(m2[n] + 1.0) for n in names}, cen)


def fold_TH(k):
    d = lib.OUT / 'lco3' / f'set{k}_std'
    names = ['calib', 'in', 'out'] + ['sh_' + s for s in lib.SHIFTS]
    TH = {}
    for f in RB:
        if f.startswith('pred'):
            continue
        H = np.load(d / f'{f}.npz')
        TH[f] = {n: H[f'{n}/th'] for n in names}
    TH['pred0'] = pred_sets('cnn0', k)
    TH['pred1'] = pred_sets('cnn1', k)
    return TH, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--export', default='')
    args = ap.parse_args()
    ks = (0, 1, 2)
    comps, THs = {}, {}
    print('=== probe fit quality (calib MSE in normalized units; alpha)')
    PTH = {k: {} for k in ks}
    for p in PROBES:
        for k in ks:
            PTH[k][p], a, mse = fold_probe(p, k)
        print(f'  {p:8s} fold0 alpha={a} mse={mse:.3f}', flush=True)
    for k in ks:
        comps[k] = lco_components(k, 'std')
        TH, names = fold_TH(k)
        TH.update(PTH[k])
        THs[k] = (TH, names)

    def lco(w, extra_views=None, tag=None):
        rs = []
        for k in ks:
            c = dict(comps[k])
            if extra_views is not None:
                TH, names = THs[k]
                c['consRplus'] = cons(TH, RB + extra_views, names)
            rs.append(evaluate(mean_of(*[c[x] for x in w], w=list(w.values()))))
        exc = np.mean([r['exc'] for r in rs], 0)
        return np.mean([r['m19'] for r in rs]), [r['m19'] for r in rs], exc

    print('=== consistency views (LCO, standalone)')
    res_views = {}
    print('  consR (bands)            m19=%.4f sets=%s exc=%s' % (lambda r: (r[0], np.round(r[1], 3), np.round(r[2], 2)))(lco({'consR': 1})))
    for p in PROBES:
        r = lco({'consRplus': 1}, [p])
        res_views[p] = r[0]
        print(f'  consR+{p:8s}           m19={r[0]:.4f} sets={np.round(r[1], 3)} exc={np.round(r[2], 2)}', flush=True)
    good = [p for p in PROBES if res_views[p] > lco({'consR': 1})[0] + 0.002]
    print('  helpful probes:', good)
    plus_sets = {'plus_good': good} if good else {}
    if len(good) > 1:
        best1 = max(good, key=lambda p: res_views[p])
        plus_sets['plus_best1'] = [best1]
    for nm, views in plus_sets.items():
        r = lco({'consRplus': 1}, views)
        print(f'  consRplus[{nm}]={views} m19={r[0]:.4f} sets={np.round(r[1], 3)} exc={np.round(r[2], 2)}')
    BIG = {'Dr': 1, 'N2': 1, 'PsBh': .5, 'xSL': 1, 'consR': 2}
    print('=== recipes (LCO mean, per set, exceedance 1/0.5/0.1%)')
    rows = []
    r = lco(BIG)
    print(f'  BIG_RE2                         m19={r[0]:.4f} sets={np.round(r[1], 3)} exc={np.round(r[2], 2)}')
    rows.append(('BIG_RE2', BIG, None, r))
    for nm, views in plus_sets.items():
        for wc in (1.5, 2, 3):
            w = {'Dr': 1, 'N2': 1, 'PsBh': .5, 'xSL': 1, 'consRplus': wc}
            r = lco(w, views)
            print(f'  BIG[{nm} w={wc}]                m19={r[0]:.4f} sets={np.round(r[1], 3)} exc={np.round(r[2], 2)}', flush=True)
            rows.append((f'BIG_{nm}_w{wc}', w, views, r))
        for w in ({'Dr': 1, 'N2': 1, 'PsBh': .5, 'xSL': .5, 'consRplus': 2},
                  {'Dr': 1.5, 'N2': 1, 'PsBh': .5, 'xSL': 1, 'consRplus': 2},
                  {'Dr': 1, 'N2': 1.5, 'PsBh': .5, 'xSL': 1, 'consRplus': 2},
                  {'Dr': 1, 'N2': 1, 'PsBh': .5, 'xSL': 1, 'consRplus': 2, 'consR': 1}):
            r = lco(w, views)
            print(f'  {nm} {w} m19={r[0]:.4f} sets={np.round(r[1], 3)} exc={np.round(r[2], 2)}', flush=True)
            rows.append((f'{nm}_{len(rows)}', w, views, r))
    (lib.OUT / 'probes_rows.json').write_text(json.dumps([(a, b, c, [float(r[0]), [float(x) for x in r[1]],
                                                            [float(x) for x in r[2]]]) for a, b, c, r in rows]))
    # ------------------------------------------------------------ export (full data)
    if args.export:
        sel = json.loads(args.export)  # {tag: [weights, views]}
        from export3 import components as full_components
        comp = full_components({})
        sets = lib.SETS + ['test']
        TH = {}
        need = ['b0', 'b1', 'b2', 'b3', 'cnn', 'cnn1', 'dog', 'den_r', 'den_sd']
        for f in need:
            F = fam(f)
            TH[f] = {s: F[f'{s}/th'] for s in sets}
        TH['pred0'] = {s: lib.load('cnn0', s)[..., -5:].reshape(-1, 5) for s in sets}
        TH['pred1'] = {s: lib.load('cnn1', s)[..., -5:].reshape(-1, 5) for s in sets}
        d = lib.OUT / 'export4'
        d.mkdir(exist_ok=True)
        for tag, (w, views) in sel.items():
            c = dict(comp)
            if views:
                for p in views:
                    TH[p], a, mse = full_probe(p)
                c['consRplus'] = cons(TH, RB + views, sets)
            z = mean_of(*[c[x] for x in w], w=list(w.values()))
            r = ev(z)
            print(f'export {tag}: audit m15={r["shift_mean"]:.4f} new4={r.get("new_mean", 0):.3f} '
                  f'q99/99.5/99.9={r["nq"][2]:.2f}/{r["nq"][3]:.2f}/{r["nq"][4]:.2f}', flush=True)
            assert z['test'].shape == (10000,) and np.isfinite(z['test']).all()
            np.save(d / f'test_z_{tag}.npy', z['test'].astype(np.float64))
            np.savez(d / f'z_{tag}.npz', **z)
            np.savez(d / f'components_{tag}.npz', **{f'{x}/{k}': v for x in w for k, v in c[x].items()})
            (d / f'recipe_{tag}.txt').write_text(json.dumps(dict(weights=w, probe_views=views)) + '\n')


if __name__ == '__main__':
    main()
