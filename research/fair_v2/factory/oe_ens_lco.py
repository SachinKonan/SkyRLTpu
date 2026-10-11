"""LCO numbers for an OE-network ensemble (Ol3) and OE-embedding Mahalanobis members (Oe) on top of v3-15d (CPU).

  python factory/oe_ens_lco.py

Sources (fair-data/v3): oe (run_all14), oe_s1 (seed 1), oe_phys -- all 14 generator families; oe_ho -- only 7
families (tell, nz, shear, psf, noise, mask, nonlin), the honest check against generator overlap.
  Ol_<net>: robust z of the OoD logit (column 0), constants from the fold's calib normals (as factory/ol_lco.py).
  Ol3: robust z (fold calib) of the mean of Ol_oe, Ol_oe_s1, Ol_oe_phys.
  Oe_<net>: log Mahalanobis distance of the 256-d pooled embedding; mean and OAS-shrunk covariance from the fold's
            fit3 normal rows (held-out cosmologies excluded), then robust z on the fold's calib normals.
  Oe3: robust z of the mean of the three Oe.
The OE nets were trained on all fit3 cosmologies; only the Oe statistics are refit per fold.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'agents' / 'consistency'))
import grade as G  # noqa: E402
import lib  # noqa: E402
from holdout3 import masks  # noqa: E402
from ol_lco import recipe  # noqa: E402
from stack_levers import DATA, load_fold, summarize, wmean  # noqa: E402

NETS = {'oe': 23, 'oe_s1': 23, 'oe_phys': 23, 'oe_ho': 17}       # embedding start column
NEVER_SEEN_BY_HO = ['bary9.0', 'bary6.5', 'baryDMO', 'peakcut0.3', 'ia0.2', 'srcclust0.15', 'poisson120', 'psfpat']


def robust_z(raw):
    c = raw['calib']
    med = np.median(c)
    mad = 1.4826 * np.median(np.abs(c - med))
    return {n: (v - med) / mad for n, v in raw.items()}


def fold_sets(k, src, fn):
    """Member raw values per LCO set name (same row conventions as ol_lco.fold_member)."""
    m = masks(k)
    f = {n: fn(lib.load(src, n)) for n in ('calib', 'audit')}
    raw = {'calib': f['calib'][~m['calib']].reshape(-1), 'in': f['audit'][~m['audit']].reshape(-1),
           'out': np.concatenate([f['calib'][m['calib']].reshape(-1), f['audit'][m['audit']].reshape(-1)])}
    for s in lib.SHIFTS:
        raw['sh_' + s] = fn(lib.load(src, 'shift_' + s))[m['audit']].reshape(-1)
    return raw


def oas(X):
    n, p = X.shape
    mu_x = X.mean(0)
    Xc = X - mu_x
    S = Xc.T @ Xc / n
    mu = np.trace(S) / p
    alpha = np.mean(S ** 2)
    den = (n + 1) * (alpha - mu ** 2 / p)
    sh = 1.0 if den == 0 else min((alpha + mu ** 2) / den, 1.0)
    return mu_x, (1 - sh) * S + sh * mu * np.eye(p), sh


def emb_fn(k, src, start):
    m = masks(k)
    fit = lib.load(src, 'fit')[~m['fit']]                       # (n, V, d)
    X = fit[..., start:start + 256].reshape(-1, 256).astype(np.float64)
    mu, C, sh = oas(X)
    P = np.linalg.inv(C)

    def fn(x):
        e = x[..., start:start + 256].astype(np.float64) - mu
        return np.log(np.einsum('...i,ij,...j->...', e, P, e))
    return fn, sh


def main():
    folds, info = {}, {}
    for k in G.FOLDS:
        comp, TH = load_fold(k)
        views = json.loads((DATA / 'cache_v2' / 'stack.json').read_text())['views']
        comp['consRplus'] = G.cons(TH, views, G.NAMES)
        ol, oe = {}, {}
        for src, start in NETS.items():
            ol[src] = robust_z(fold_sets(k, src, lambda x: x[..., 0]))
            fn, sh = emb_fn(k, src, start)
            info[f'oas_shrinkage_{src}_fold{k}'] = float(sh)
            oe[src] = robust_z(fold_sets(k, src, fn))
            comp[f'Ol_{src}'], comp[f'Oe_{src}'] = ol[src], oe[src]
        three = ('oe', 'oe_s1', 'oe_phys')
        comp['Ol'] = ol['oe']
        comp['Ol3'] = robust_z({n: np.mean([ol[s][n] for s in three], 0) for n in ol['oe']})
        comp['Oe3'] = robust_z({n: np.mean([oe[s][n] for s in three], 0) for n in oe['oe']})
        for mname in ('Ol3', 'Oe3', 'Ol_oe_ho', 'Oe_oe_ho'):
            assert all(len(comp[mname][n]) == len(comp['Oh'][n]) for n in G.NAMES), mname
        folds[k] = comp
    base = recipe('v3-15d')
    rec = {f'{m} alone': {m: 1.0} for m in ('Ol_oe', 'Ol_oe_s1', 'Ol_oe_phys', 'Ol3', 'Ol_oe_ho',
                                             'Oe_oe', 'Oe_oe_s1', 'Oe_oe_phys', 'Oe3', 'Oe_oe_ho')}
    rec.update({
        'v3-15d': base,
        'v3-16b (15d + Ol 3)': dict(base, Ol=3.0),
        'v3-16c (15d + Ol 6)': dict(base, Ol=6.0),
        '15d + Ol3 3': dict(base, Ol3=3.0),
        '15d + Ol3 6': dict(base, Ol3=6.0),
        '15d + Ol3 6 + Oe3 1.5': dict(base, Ol3=6.0, Oe3=1.5),
        '15d + Ol3 6 + Oe3 3': dict(base, Ol3=6.0, Oe3=3.0),
        '15d + Ol 6 + Oe3 1.5': dict(base, Ol=6.0, Oe3=1.5),
        '15d + Ol_ho 3': dict(base, Ol_oe_ho=3.0),
    })
    res = {name: dict(summarize([G.metrics(wmean(folds[k], r)) for k in G.FOLDS]), weights=r)
           for name, r in rec.items()}
    ref = res['v3-16b (15d + Ol 3)']
    b15 = res['v3-15d']
    print(f'{"recipe":26s} {"all19":>7s} {"vis":>7s} {"hid":>7s}  {"per fold all19":22s} {"d_fold vs 16b":22s} exc(1/.5/.1%)')
    for name, s in res.items():
        d = [a - c for a, c in zip(s['per_fold_all19'], ref['per_fold_all19'])]
        s['d_fold_vs_16b'] = d
        s['worse_than_16b'] = bool(not name.endswith('alone') and name != 'v3-15d'
                                   and (min(d) < 0 or s['hidden'] < ref['hidden']))
        flag = '  WORSE THAN 16b' if s['worse_than_16b'] else ''
        print(f'{name:26s} {s["all19"]:.4f} {s["visible"]:.4f} {s["hidden"]:.4f}  '
              f'{" ".join(f"{v:.4f}" for v in s["per_fold_all19"]):22s} {" ".join(f"{v:+.4f}" for v in d):22s} '
              f'{s["exc_pct"]}{flag}')
    print('\nper-shift gain vs v3-15d on shifts oe_ho never trained on (Ol_ho 3 | Ol 3 (=16b)):')
    overlap = {}
    for sname in NEVER_SEEN_BY_HO:
        a = res['15d + Ol_ho 3']['per_shift'][sname] - b15['per_shift'][sname]
        c = ref['per_shift'][sname] - b15['per_shift'][sname]
        overlap[sname] = dict(Ol_ho3=round(a, 4), Ol3_16b=round(c, 4))
        print(f'  {sname:14s} {a:+.4f} | {c:+.4f}')
    out = DATA / 'oe_ens_lco.json'
    out.write_text(json.dumps(dict(results=res, ho_unseen_shift_gains=overlap, info=info), indent=1))
    print('wrote', out)


if __name__ == '__main__':
    main()
