"""Lever 3: which held-out-universe NORMAL maps score highest under v3-16b, and why (CPU).

  python factory/lever3_tail.py [--out fair-data/factory/lever3_tail.json]

Per LCO fold k, 'out' = normal maps of the 20 held-out cosmologies (calib rows x 2 noise views, then audit rows),
exactly the normal set the metric uses (G.metrics: phase2_score(z['out'], z['sh_*'])). Member z come from the fold
caches (stack_levers.load_fold: cache_v2 + Oh) plus the Ol fold member (ol_lco.fold_member); recipe = v3-16b.
For the top 0.1-5% of 'out' by the combined score it reports labels (cosmology, baryon pair, dz), distance of the
true (Om, S8) to the training cosmologies, the member whose weighted z dominates, and whether the misfit members'
profile theta-hat sits on the search box / hull-penalty boundary. Then a small, pre-registered set of config-only
reweightings (each member x0.5 and x0, one at a time) is scored on all folds (all-19 / visible / hidden / tail).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'agents' / 'consistency'))
import grade as G  # noqa: E402
import lib  # noqa: E402
from analyze import TM, TS, COS, HULL_R  # noqa: E402
from holdout3 import masks, held_set  # noqa: E402
from ol_lco import fold_member, ol_raw  # noqa: E402
from stack_levers import DATA, V2, load_fold, summarize, wmean  # noqa: E402

W16B = {'Sd': .5, 'Dr': 2.5, 'xSL': 1., 'consP8': .5, 'consRplus': 3., 'N2': .5, 'PsBh': .5, 'Oh': 1.5,
        'Xdp': 2., 'Ol': 3.}
MISFIT_FAMS = {'Sd': 'dog', 'Dr': 'den_r', 'N2': 'N2', 'PsBh': 'PsBh', 'Oh': 'oe_head'}
PN = ['Om', 'S8', 'T_AGN', 'f0', 'dz']


def out_labels(k):
    """Labels and (cosmology, realization) for the 'out' rows of fold k, in the cached order."""
    m = masks(k)
    lab_c, lab_a = lib.labels('calib'), lib.labels('audit')
    idx_c, idx_a = lib.load('stats', 'calib', 'index'), lib.load('stats', 'audit', 'index')
    nv = lib.load('stats', 'calib').shape[1]                       # calib noise views per row
    lab = np.concatenate([np.repeat(lab_c[m['calib']], nv, 0), lab_a[m['audit']]])
    idx = np.concatenate([np.repeat(idx_c[m['calib']], nv, 0), idx_a[m['audit']]])
    return lab, idx


def fam_theta(k, fam):
    d = np.load(lib.OUT / 'lco3' / f'set{k}_std' / f'{fam}.npz')
    return d['out/th']


def boundary_flags(th):
    """th: normalized theta-hat (N, 5). Box edges for T_AGN, f0 (fit3 label range); hull penalty for (Om, S8)."""
    LAB = lib.labels('fit')
    lo, hi = (LAB.min(0) - TM) / TS, (LAB.max(0) - TM) / TS
    eps = 0.02 * (hi - lo)
    box = ((th[:, 2:4] - lo[2:4]) < eps[2:4]) | ((hi[2:4] - th[:, 2:4]) < eps[2:4])
    dmin = np.sqrt(((th[:, None, :2] - COS[None]) ** 2).sum(-1)).min(1)
    return dict(at_box=box.any(1), beyond_hull=dmin > HULL_R, dmin=dmin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=DATA / 'lever3_tail.json')
    args = ap.parse_args()
    views = json.loads((V2 / 'stack.json').read_text())['views']
    folds, rows = {}, []
    for k in G.FOLDS:
        comp, TH = load_fold(k)
        comp['consRplus'] = G.cons(TH, views, G.NAMES)
        comp['Ol'] = fold_member(k, ol_raw)
        folds[k] = comp
        s = wmean(comp, W16B)['out']
        lab, idx = out_labels(k)
        assert len(lab) == len(s), (len(lab), len(s))
        W = sum(W16B.values())
        contrib = {m: W16B[m] * comp[m]['out'] / W for m in W16B}
        th = {m: fam_theta(k, f) for m, f in MISFIT_FAMS.items()}
        for m, t in th.items():
            assert len(t) == len(s), m
        bf = {m: boundary_flags(t) for m, t in th.items()}
        true_n = (lab - TM) / TS
        true_dmin = np.sqrt(((true_n[:, None, :2] - COS[None]) ** 2).sum(-1))
        true_dmin[true_dmin < 1e-9] = np.inf          # held-out cosmology: its own point is not in COS anyway
        for i in range(len(s)):
            rows.append(dict(fold=k, score=float(s[i]), cosmo=int(idx[i, 0]), real=int(idx[i, 1]),
                             lab=lab[i].tolist(), true_dmin=float(true_dmin[i].min()),
                             contrib={m: float(c[i]) for m, c in contrib.items()},
                             z={m: float(comp[m]['out'][i]) for m in W16B},
                             at_box={m: bool(b['at_box'][i]) for m, b in bf.items()},
                             beyond_hull={m: bool(b['beyond_hull'][i]) for m, b in bf.items()}))
        print(f'fold {k}: {len(s)} held-out normals', flush=True)

    # ranks within each fold (the metric's FPR is per fold)
    res = dict(per_band={}, members={}, concentration={})
    for k in G.FOLDS:
        r = [x for x in rows if x['fold'] == k]
        order = np.argsort([-x['score'] for x in r])
        for rank, j in enumerate(order):
            r[j]['u'] = (rank + 0.5) / len(r)
    allr = rows
    bands = {'top0.1-1%': (0.001, 0.01), 'top1-5%': (0.01, 0.05), 'top0-5%': (0.0, 0.05)}
    for bname, (a, b) in bands.items():
        sel = [x for x in allr if a <= x['u'] < b] if a > 0 else [x for x in allr if x['u'] < b]
        rest = [x for x in allr if x['u'] >= 0.05]
        def mean(f, xs):
            return float(np.mean([f(x) for x in xs])) if xs else None
        dom = {}
        for x in sel:
            m = max(x['contrib'], key=x['contrib'].get)
            dom[m] = dom.get(m, 0) + 1
        res['per_band'][bname] = dict(
            n=len(sel),
            dominant_member={m: round(c / len(sel), 3) for m, c in sorted(dom.items(), key=lambda t: -t[1])},
            mean_contrib_top={m: round(mean(lambda x: x['contrib'][m], sel), 3) for m in W16B},
            mean_contrib_rest={m: round(mean(lambda x: x['contrib'][m], rest), 3) for m in W16B},
            labels_top_mean=np.round(np.mean([x['lab'] for x in sel], 0), 4).tolist(),
            labels_rest_mean=np.round(np.mean([x['lab'] for x in rest], 0), 4).tolist(),
            labels_top_sd=np.round(np.std([x['lab'] for x in sel], 0), 4).tolist(),
            labels_all_sd=np.round(np.std([x['lab'] for x in allr], 0), 4).tolist(),
            true_dmin_top=round(mean(lambda x: x['true_dmin'], sel), 3),
            true_dmin_rest=round(mean(lambda x: x['true_dmin'], rest), 3),
            at_box_top={m: round(mean(lambda x: x['at_box'][m], sel), 3) for m in MISFIT_FAMS},
            at_box_rest={m: round(mean(lambda x: x['at_box'][m], rest), 3) for m in MISFIT_FAMS},
            beyond_hull_top={m: round(mean(lambda x: x['beyond_hull'][m], sel), 3) for m in MISFIT_FAMS},
            beyond_hull_rest={m: round(mean(lambda x: x['beyond_hull'][m], rest), 3) for m in MISFIT_FAMS})
    # concentration over held-out cosmologies and baryon realizations (top 5% vs expectation)
    top = [x for x in allr if x['u'] < 0.05]
    by_c, n_c = {}, {}
    for x in allr:
        n_c[(x['fold'], x['cosmo'])] = n_c.get((x['fold'], x['cosmo']), 0) + 1
    for x in top:
        by_c[(x['fold'], x['cosmo'])] = by_c.get((x['fold'], x['cosmo']), 0) + 1
    share = sorted(((v / n_c[c], c) for c, v in by_c.items()), reverse=True)
    res['concentration']['top5_by_cosmology'] = [dict(fold=c[0], cosmo=c[1], frac_of_its_maps=round(f, 3),
                                                      n_top=by_c[c], n=n_c[c],
                                                      lab=next(x['lab'][:2] for x in allr if (x['fold'], x['cosmo']) == c))
                                                 for f, c in share[:12]]
    res['concentration']['n_cosmologies_with_top5'] = len(by_c)
    res['concentration']['top5_share_in_worst3_cosmologies'] = round(sum(sorted(by_c.values())[-3:]) / len(top), 3)
    reals = {}
    for x in top:
        reals[x['real']] = reals.get(x['real'], 0) + 1
    res['concentration']['top5_distinct_baryon_realizations'] = len(reals)
    res['concentration']['top5_max_per_realization'] = max(reals.values())

    # pre-registered config-only reweightings: each member x0.5 and x0, one at a time
    cfgs = {'v3-16b': dict(W16B)}
    for m in W16B:
        cfgs[f'{m} x0.5'] = dict(W16B, **{m: W16B[m] * 0.5})
        cfgs[f'{m} x0'] = {mm: w for mm, w in W16B.items() if mm != m}
    res['reweight'] = {}
    base = None
    for name, w in cfgs.items():
        s = summarize([G.metrics(wmean(folds[k], w)) for k in G.FOLDS])
        if name == 'v3-16b':
            base = s
        res['reweight'][name] = dict(all19=round(s['all19'], 4), visible=round(s['visible'], 4),
                                     hidden=round(s['hidden'], 4),
                                     d_fold=[round(a - b, 4) for a, b in zip(s['per_fold_all19'], base['per_fold_all19'])],
                                     d_hidden_fold=[round(a - b, 4) for a, b in zip(s['per_fold_hidden'], base['per_fold_hidden'])],
                                     exc_pct=s['exc_pct'], weights=w)
    args.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != 'reweight'}, indent=1))
    print(f'{"config":14s} {"all19":>7s} {"vis":>7s} {"hid":>7s}  d_fold(all19)            d_fold(hidden)           exc(1/.5/.1%)')
    for name, r in res['reweight'].items():
        print(f'{name:14s} {r["all19"]:.4f} {r["visible"]:.4f} {r["hidden"]:.4f}  '
              f'{" ".join(f"{v:+.4f}" for v in r["d_fold"]):24s} {" ".join(f"{v:+.4f}" for v in r["d_hidden_fold"]):24s} {r["exc_pct"]}')
    print('wrote', args.out)


if __name__ == '__main__':
    main()
