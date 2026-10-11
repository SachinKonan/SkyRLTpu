"""Stack-level levers on the LCO harness (CPU): prior-centre vote view / vote eccentricity, and tail-type
combination of specialist members.

  python factory/stack_levers.py [--out fair-data/factory/stack_levers.json]

Recipes (fold caches from build_stack_v2.py, Oh from lco3 components):
  v2      = v3-12 without Oh + dirpower (misfit 'Xdp' w2 + 'dirpower' vote view in consRplus)
  v3-13a  = v2 + Oh 1.5
Lever 1 (null-vote effect): 'prior' = a constant vote at the training prior centre (0 in normalized theta)
  added to the consRplus views; 'ECC' = log(1 + Mahalanobis^2 of the mean consRplus vote from the prior centre),
  covariance = calib second moment about the centre, standardized on calib (unconditional: conditioning on
  theta would regress the eccentricity away).
Lever 2 (specialists): Ramp = search_A/ramp4n misfit member (+ its ridge vote), Xdp = dirpower misfit.
  'mean'      : Ramp at weight 2 in the weighted mean + ramp4n vote view (how the grader scored it);
  'sp(l,c)'   : recipe mean + l * softplus(max(z_Ramp, z_Xdp) - c);
  'lse(t)'    : t * log(sum_m w_m exp(z_m / t) / sum_m w_m) over the recipe members + Ramp (w 2).
All configs are reported; hidden shifts are only used as the final gate.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grade as G  # noqa: E402
from analyze import mahal  # noqa: E402
from lco3_analyze import components as lco_components  # noqa: E402

DATA = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/factory')
V2 = DATA / 'cache_v2'
RAMP = DATA / 'runs' / 'A_ramp4n_full'


def softplus(x):
    return np.logaddexp(0.0, x)


def load_fold(k):
    with open(V2 / f'fold{k}.pkl', 'rb') as fh:
        C = pickle.load(fh)
    comp, TH = dict(C['comp']), dict(C['TH'])
    comp['Oh'] = lco_components(k, 'std')['Oh']
    with open(RAMP / f'fold{k}_full.pkl', 'rb') as fh:
        R = pickle.load(fh)
    assert R['take'].all(), 'ramp4n full run must cover every held shift row'
    comp['Ramp'] = R['mis']
    TH['ramp4n'] = R['ridge']
    TH['prior'] = {n: np.zeros_like(TH['b0'][n]) for n in G.NAMES}
    for m, z in comp.items():
        assert all(len(z[n]) == len(TH['b0'][n]) for n in G.NAMES), m
    return comp, TH


def ecc_member(TH, views):
    cen = {n: np.mean([TH[f][n] for f in views], 0).astype(np.float64) for n in G.NAMES}
    c = cen['calib']
    S = c.T @ c / len(c)
    S += 1e-3 * np.trace(S) / len(S) * np.eye(len(S))
    Si = np.linalg.inv(S)
    raw = {n: np.log1p(np.einsum('ij,jk,ik->i', v, Si, v)) for n, v in cen.items()}
    mu, sd = raw['calib'].mean(), raw['calib'].std()
    return {n: (v - mu) / sd for n, v in raw.items()}


def wmean(comp, w):
    W = sum(w.values())
    return {n: sum(wi * comp[m][n] for m, wi in w.items()) / W for n in G.NAMES}


def lse(comp, w, t):
    """Weighted soft maximum, computed stably (shifted z reach the hundreds)."""
    W = sum(w.values())
    out = {}
    for n in G.NAMES:
        a = np.stack([comp[m][n] / t + np.log(wi) for m, wi in w.items()])
        mx = a.max(0)
        out[n] = t * (mx + np.log(np.exp(a - mx).sum(0)) - np.log(W))
    return out


def configs(recipe, views):
    """name -> fn(comp, TH, cache) returning the score dict for one fold."""
    def base(c, T, x):
        return wmean(c, recipe)

    def with_view(extra):
        def f(c, T, x):
            cc = dict(c)
            cc['consRplus'] = x['cons_' + extra]
            return wmean(cc, recipe)
        return f

    def with_ecc(we, prior=False):
        def f(c, T, x):
            cc = dict(c)
            if prior:
                cc['consRplus'] = x['cons_prior']
            cc['ECC'] = x['ecc']
            return wmean(cc, {**recipe, 'ECC': we})
        return f

    def ramp_mean(c, T, x):
        cc = dict(c)
        cc['consRplus'] = x['cons_ramp4n']
        return wmean(cc, {**recipe, 'Ramp': 2})

    def sp(lam, cut):
        def f(c, T, x):
            z = wmean(c, recipe)
            return {n: z[n] + lam * softplus(np.maximum(c['Ramp'][n], c['Xdp'][n]) - cut) for n in G.NAMES}
        return f

    def lse_cfg(t):
        def f(c, T, x):
            return lse(c, {**recipe, 'Ramp': 2}, t)
        return f

    return {
        'base': base,
        'L1 prior view': with_view('prior'),
        'L1 ECC w0.5': with_ecc(0.5),
        'L1 ECC w1': with_ecc(1.0),
        'L1 prior view + ECC w0.5': with_ecc(0.5, prior=True),
        'L2 Ramp in mean (w2 + vote)': ramp_mean,
        'L2 sp(l=0.5,c=2)': sp(0.5, 2.0),
        'L2 sp(l=1,c=2)': sp(1.0, 2.0),
        'L2 sp(l=1,c=3)': sp(1.0, 3.0),
        'L2 sp(l=2,c=3)': sp(2.0, 3.0),
        'L2 lse(t=2)': lse_cfg(2.0),
    }


def summarize(rows):
    agg = lambda which: [float(np.mean([r[0][s] for s in which])) for r in rows]
    a19, vis, hid = agg(G.lib.SHIFTS), agg(G.VISIBLE), agg(G.HIDDEN)
    return dict(all19=float(np.mean(a19)), visible=float(np.mean(vis)), hidden=float(np.mean(hid)),
                per_fold_all19=a19, per_fold_visible=vis, per_fold_hidden=hid,
                exc_pct=np.mean([r[1] for r in rows], 0).round(3).tolist(),
                per_shift={s: float(np.mean([r[0][s] for r in rows])) for s in G.lib.SHIFTS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=DATA / 'stack_levers.json')
    args = ap.parse_args()
    st = json.loads((V2 / 'stack.json').read_text())
    views = st['views']
    recipes = {'v2': dict(st['stack']), 'v3-13a': {**st['stack'], 'Oh': 1.5}}
    folds = {}
    for k in G.FOLDS:
        comp, TH = load_fold(k)
        comp['consRplus'] = G.cons(TH, views, G.NAMES)
        x = dict(cons_prior=G.cons(TH, views + ['prior'], G.NAMES),
                 cons_ramp4n=G.cons(TH, views + ['ramp4n'], G.NAMES),
                 ecc=ecc_member(TH, views))
        folds[k] = (comp, TH, x)
        print(f'fold {k} loaded', flush=True)
    res = {}
    for rname, recipe in recipes.items():
        for cname, fn in configs(recipe, views).items():
            rows = [G.metrics(fn(*folds[k])) for k in G.FOLDS]
            res[rname, cname] = summarize(rows)
    out = {}
    for rname in recipes:
        b = res[rname, 'base']
        print(f'\n== recipe {rname}: base all19 {b["all19"]:.4f} visible {b["visible"]:.4f} hidden {b["hidden"]:.4f} '
              f'exc {b["exc_pct"]}')
        print(f'{"config":30s} {"d_all19":>8s} {"d_vis":>8s} {"d_hid":>8s}  {"d_fold_all19":24s} exc(1/.5/.1%)  gate')
        for (r, cname), s in res.items():
            if r != rname:
                continue
            d = {key: s[key] - b[key] for key in ('all19', 'visible', 'hidden')}
            dfold = [a - c for a, c in zip(s['per_fold_all19'], b['per_fold_all19'])]
            gate = d['all19'] >= 0.004 and min(dfold) > 0 and d['hidden'] >= 0
            print(f'{cname:30s} {d["all19"]:+8.4f} {d["visible"]:+8.4f} {d["hidden"]:+8.4f}  '
                  f'{" ".join(f"{v:+.4f}" for v in dfold):24s} {s["exc_pct"]}  {"PASS" if gate else ""}')
            out[f'{rname} | {cname}'] = dict(s, d_all19=d['all19'], d_visible=d['visible'], d_hidden=d['hidden'],
                                             d_fold_all19=dfold, gate=bool(gate))
    args.out.write_text(json.dumps(out, indent=1))
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
