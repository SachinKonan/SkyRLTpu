"""LCO comparison of fine-tuned OE OoD heads (Ol') against the original OE logit (Ol), CPU.

  python factory/olft_lco.py TAG [TAG ...]

Ol' = head(silu(e)) on the run_all14 pooled embedding e (columns 23..278 of the `oe` source), with the fold's
calib-normal robust z exactly as Ol in ol_lco.py. Reports Ol/Ol' alone and inside the v3-16b recipe (v3-15d + O* 3)
and v3-16c (O* 6): all-19 / visible / hidden / per fold / tail, plus per-shift deltas split by whether the shift
kind overlaps the OE generator families (the closest thing to a leave-shift-family-out view; the OE backbone
itself trained on all 14 families and all fit3 cosmologies, so only RELATIVE changes are meaningful).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ol_lco as O  # noqa: E402
from ol_lco import G, lib, masks, robust_z, recipe, load_fold, summarize, wmean, DATA  # noqa: E402

POOL = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/outlier/ft_pool')
# shift kinds whose physics the OE generators imitate closely (training-family overlap)
OVERLAP = {'srcclust0.15', 'poisson120', 'psfpat', 'ia0.2', 'noise+0.05', 'noise-0.05', 'noisegrad0.15', 'noise_t5',
           'mult+0.05', 'mult-0.05', 'smooth0.5', 'aniso0.7', 'baryDMO', 'bary6.5', 'bary9.0', 'supp0.15',
           'boost0.15'}


def head_fn(w, b):
    def f(x):
        e = x[..., 23:279].astype(np.float32)
        return (e / (1 + np.exp(-e))) @ w + b
    return f


def fold_member_oe(k, fn):
    m = masks(k)
    f = {n: fn(lib.load('oe', n)) for n in ('calib', 'audit')}
    raw = {'calib': f['calib'][~m['calib']].reshape(-1), 'in': f['audit'][~m['audit']].reshape(-1),
           'out': np.concatenate([f['calib'][m['calib']].reshape(-1), f['audit'][m['audit']].reshape(-1)])}
    for s in lib.SHIFTS:
        raw['sh_' + s] = fn(lib.load('oe', 'shift_' + s))[m['audit']].reshape(-1)
    z, _ = robust_z(raw, raw['calib'])
    return z


def main():
    tags = sys.argv[1:]
    heads = {}
    for t in tags:
        h = torch.load(POOL / f'head_{t}.pt', weights_only=False)
        heads[t] = head_fn(h['w'].numpy().astype(np.float32), float(h['b'].numpy().ravel()[0]))
    folds = {}
    for k in G.FOLDS:
        comp, TH = load_fold(k)
        views = json.loads((DATA / 'cache_v2' / 'stack.json').read_text())['views']
        comp['consRplus'] = G.cons(TH, views, G.NAMES)
        comp['Ol'] = O.fold_member(k, O.ol_raw)
        for t, fn in heads.items():
            comp['Ol_' + t] = fold_member_oe(k, fn)
        folds[k] = comp
    rec = {'Ol alone': {'Ol': 1.0}, 'v3-16b (Ol 3)': recipe('v3-15d', {'Ol': 3.0}),
           'v3-16c (Ol 6)': recipe('v3-15d', {'Ol': 6.0})}
    for t in tags:
        rec[f"Ol'[{t}] alone"] = {'Ol_' + t: 1.0}
        rec[f"16b with Ol'[{t}] 3"] = recipe('v3-15d', {'Ol_' + t: 3.0})
        rec[f"16c with Ol'[{t}] 6"] = recipe('v3-15d', {'Ol_' + t: 6.0})
    res = {name: dict(summarize([G.metrics(wmean(folds[k], r)) for k in G.FOLDS]), weights=r) for name, r in rec.items()}
    print(f'{"recipe":28s} {"all19":>7s} {"vis":>7s} {"hid":>7s}  {"per fold":22s} {"overlap":>7s} {"nonovl":>7s} exc(1/.5/.1%)')
    for name, s in res.items():
        ov = np.mean([s['per_shift'][x] for x in lib.SHIFTS if x in OVERLAP])
        no = np.mean([s['per_shift'][x] for x in lib.SHIFTS if x not in OVERLAP])
        s['overlap_mean'], s['nonoverlap_mean'] = float(ov), float(no)
        print(f'{name:28s} {s["all19"]:.4f} {s["visible"]:.4f} {s["hidden"]:.4f}  '
              f'{" ".join(f"{v:.4f}" for v in s["per_fold_all19"]):22s} {ov:.4f} {no:.4f} {s["exc_pct"]}')
    for t in tags:
        for a, b in ((f"Ol'[{t}] alone", 'Ol alone'), (f"16b with Ol'[{t}] 3", 'v3-16b (Ol 3)'),
                     (f"16c with Ol'[{t}] 6", 'v3-16c (Ol 6)')):
            d = {x: res[a]['per_shift'][x] - res[b]['per_shift'][x] for x in lib.SHIFTS}
            worse = sorted(d.items(), key=lambda kv: kv[1])[:4]
            better = sorted(d.items(), key=lambda kv: -kv[1])[:4]
            print(f'  {a} vs {b}: folds {[round(x - y, 4) for x, y in zip(res[a]["per_fold_all19"], res[b]["per_fold_all19"])]} '
                  f'best {[(k_, round(v, 3)) for k_, v in better]} worst {[(k_, round(v, 3)) for k_, v in worse]}')
    (DATA / 'olft_lco.json').write_text(json.dumps(res, indent=1))
    print('wrote', DATA / 'olft_lco.json')


if __name__ == '__main__':
    main()
