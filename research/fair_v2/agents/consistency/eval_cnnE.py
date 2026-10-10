"""LCO test of the 12-CNN TPU ensemble as consistency views (coordinator, tonight's last check)."""
import numpy as np

import lib
from lco3_analyze import components, evaluate, combo, cons, masks, pred_sets, NAMES_BASE

K = 12
names = NAMES_BASE + ['sh_' + s for s in lib.SHIFTS]
RB = ['b0', 'b1', 'b2', 'b3', 'cnn', 'cnn1', 'pred0', 'pred1', 'dog', 'den_r', 'den_sd']


def ens_sets(k):
    """Per-net and mean predictions of the cnnE source for LCO set k."""
    m = masks(k)
    f = {n: lib.load('cnnE', n) for n in ('calib', 'audit')}
    def shape(a):
        return a.reshape(-1, K, 5)
    out = {'calib': shape(f['calib'][~m['calib']]), 'in': shape(f['audit'][~m['audit']]),
           'out': np.concatenate([shape(f['calib'][m['calib']]), shape(f['audit'][m['audit']])])}
    for s in lib.SHIFTS:
        out['sh_' + s] = shape(lib.load('cnnE', 'shift_' + s)[m['audit']])
    return out


rows = {}
for k in (0, 1, 2):
    comp = components(k, 'std')
    d = lib.OUT / 'lco3' / f'set{k}_std'
    TH = {f: {n: dict(np.load(d / f'{f}.npz'))[f'{n}/th'] for n in names} for f in RB if f not in ('pred0', 'pred1')}
    TH['pred0'] = pred_sets('cnn0', k)
    TH['pred1'] = pred_sets('cnn1', k)
    E = ens_sets(k)
    TH['predE'] = {n: E[n].mean(1) for n in names}
    for j in range(K):
        TH[f'p{j}'] = {n: E[n][:, j] for n in names}
    comp['consR_E'] = cons(TH, RB + ['predE'], names)
    comp['consR_Erep'] = cons(TH, [v for v in RB if v not in ('pred0', 'pred1')] + ['predE'], names)
    comp['consCNN12'] = cons(TH, [f'p{j}' for j in range(K)], names)
    rows[k] = comp

SDp = {'Sd': .5, 'C': 1, 'Dr': 2.5, 'xSL': 1, 'consP8': .5, 'N2': .5, 'PsBh': .5, 'SS': 1, 'Oh': 1.5}
R = {
    'consR alone': {'consR': 1}, 'consR_E alone': {'consR_E': 1}, 'consR_Erep alone': {'consR_Erep': 1},
    'consCNN12 alone': {'consCNN12': 1},
    'v3-12-like (consR 3)': {**SDp, 'consR': 3},
    'v3-12-like consR->consR_E': {**SDp, 'consR_E': 3},
    'v3-12-like consR->consR_Erep': {**SDp, 'consR_Erep': 3},
    'v3-12-like + consCNN12 1': {**SDp, 'consR': 3, 'consCNN12': 1},
    'clean (no Oh) consR 3': {**{k: v for k, v in SDp.items() if k != 'Oh'}, 'consR': 3},
    'clean (no Oh) consR_E 3': {**{k: v for k, v in SDp.items() if k != 'Oh'}, 'consR_E': 3},
}
for name, w in R.items():
    rs = [evaluate(combo(rows[k], w)) for k in (0, 1, 2)]
    s = np.array([r['m19'] for r in rs]); exc = np.mean([r['exc'] for r in rs], 0)
    print(f'{name:32s} mean {s.mean():.4f} sets {s.round(3).tolist()} exc {exc.round(2).tolist()}', flush=True)
