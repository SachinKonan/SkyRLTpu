"""LCO numbers for direct outlier-exposure members on top of v3-15d (CPU).

  python factory/ol_lco.py

Ol = the OE classifier's own OoD logit (oe_head column 0); Of = logsumexp(distortion-family logits) - normal-class
logit (oe_head columns 2..17 vs 1). Each is a per-map robust z with constants frozen on the fold's calib normals
(median, 1.4826 MAD), the same rule the package uses with full calib3. Rows follow lco3_analyze.pred_sets.
The OE head itself was trained on all fit3 cosmologies and on generators that overlap the proxy's shifts, so its
LCO is optimistic (as for Oh).
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
from stack_levers import DATA, load_fold, summarize, wmean  # noqa: E402

PKG = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/packages')


def ol_raw(x):
    return x[..., 0]


def of_raw(x):
    fam = x[..., 2:18]
    m = fam.max(-1, keepdims=True)
    return (m[..., 0] + np.log(np.exp(fam - m).sum(-1))) - x[..., 1]


def robust_z(raw, calib):
    med = np.median(calib)
    mad = 1.4826 * np.median(np.abs(calib - med))
    return {n: (v - med) / mad for n, v in raw.items()}, (float(med), float(mad))


def fold_member(k, fn):
    m = masks(k)
    f = {n: fn(lib.load('oe_head', n)) for n in ('calib', 'audit')}
    raw = {'calib': f['calib'][~m['calib']].reshape(-1), 'in': f['audit'][~m['audit']].reshape(-1),
           'out': np.concatenate([f['calib'][m['calib']].reshape(-1), f['audit'][m['audit']].reshape(-1)])}
    for s in lib.SHIFTS:
        raw['sh_' + s] = fn(lib.load('oe_head', 'shift_' + s))[m['audit']].reshape(-1)
    z, _ = robust_z(raw, raw['calib'])
    return z


def recipe(name, extra=None):
    w = json.loads((PKG / name / 'config.json').read_text())['weights']
    r = {('Xdp' if k == 'X' else k): float(v) for k, v in w.items() if float(v) != 0.0}
    r.update(extra or {})
    return r


def main():
    folds = {}
    for k in G.FOLDS:
        comp, TH = load_fold(k)
        views = json.loads((DATA / 'cache_v2' / 'stack.json').read_text())['views']
        comp['consRplus'] = G.cons(TH, views, G.NAMES)
        comp['Ol'] = fold_member(k, ol_raw)
        comp['Of'] = fold_member(k, of_raw)
        for mname in ('Ol', 'Of'):
            assert all(len(comp[mname][n]) == len(comp['Oh'][n]) for n in G.NAMES), mname
        folds[k] = comp
    rec = {
        'Ol alone': {'Ol': 1.0}, 'Of alone': {'Of': 1.0}, 'Oh alone': {'Oh': 1.0},
        'v3-15d': recipe('v3-15d'), 'v3-15h (Oh 3)': recipe('v3-15h'),
        'v3-16a (15d + Ol 1.5)': recipe('v3-15d', {'Ol': 1.5}), 'v3-16b (15d + Ol 3)': recipe('v3-15d', {'Ol': 3.0}),
        '15d + Of 1.5': recipe('v3-15d', {'Of': 1.5}), '15d + Of 3': recipe('v3-15d', {'Of': 3.0}),
        # after v3-16b (Ol 3) scored 0.7037 on the LB: how far to push Ol
        '15d + Ol 4.5': recipe('v3-15d', {'Ol': 4.5}), '15d + Ol 6': recipe('v3-15d', {'Ol': 6.0}),
        '15d + Ol 9': recipe('v3-15d', {'Ol': 9.0}), '15d + Ol 12': recipe('v3-15d', {'Ol': 12.0}),
    }
    res = {}
    base = None
    print(f'{"recipe":24s} {"all19":>7s} {"vis":>7s} {"hid":>7s}  folds                     d_folds vs 15d          exc(1/.5/.1%)')
    for name, r in rec.items():
        s = summarize([G.metrics(wmean(folds[k], r)) for k in G.FOLDS])
        if name == 'v3-15d':
            base = s
        res[name] = dict(s, weights=r)
    for name, s in res.items():
        d = [a - c for a, c in zip(s['per_fold_all19'], base['per_fold_all19'])]
        s['d_fold_vs_15d'] = d
        print(f'{name:24s} {s["all19"]:.4f} {s["visible"]:.4f} {s["hidden"]:.4f}  '
              f'{" ".join(f"{v:.4f}" for v in s["per_fold_all19"])}  {" ".join(f"{v:+.4f}" for v in d)}  {s["exc_pct"]}')
    out = DATA / 'ol_lco_push.json'
    out.write_text(json.dumps(res, indent=1))
    print('wrote', out)


if __name__ == '__main__':
    main()
