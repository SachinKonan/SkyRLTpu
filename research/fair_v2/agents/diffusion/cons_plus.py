"""Do the field-level per-band theta estimates add to the consistency agent's consP8?

Re-uses (read-only) the consistency agent's helpers and family theta files; my families
(difb_lo/mid/hi, difa profiles from bandcons.py) are normalized with the same fit3 TM/TS.
Recipe = (z_Sd + z_C + 0.5 z_xSL + 0.5 z_cons)/3 with the exported Sd/C/xSL components.
"""
import sys
from pathlib import Path
import numpy as np
CONS = Path(__file__).resolve().parents[1] / 'consistency'
sys.path.insert(0, str(CONS)); sys.path.insert(0, str(CONS.parents[1]))
import analyze as A  # noqa: E402
import lib  # noqa: E402

sets = lib.SETS + ['test']
TH = {}
for f in ['psL', 'psS', 'onept', 'extrema', 'mf', 'dog', 'cnn']:
    d = A.fam(f)
    TH[f] = {s: d[f'{s}/th'] for s in sets}
TH['pred0'] = {s: lib.load('cnn0', s)[..., -5:].reshape(-1, 5) for s in sets}
B = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/diffusion/bandcons/fam')
for f in ['lo', 'mid', 'hi', 'FA']:
    d = np.load(B / f'{f}.npz')
    TH['dif_' + f] = {s: (d[f'{s}/th'] - A.TM) / A.TS for s in sets}


def cons(fs):
    vec, cen = {}, {}
    for s in sets:
        X = np.stack([TH[f][s] for f in fs], 1)
        vec[s] = (X - X.mean(1, keepdims=True))[:, 1:].reshape(len(X), -1)
        cen[s] = X.mean(1)
    m2 = A.mahal(vec)
    return A.standardize({s: np.log(m2[s] + 1.0) for s in sets}, cen)


C = np.load(lib.OUT / 'export' / 'components_BASE+0.5xSL+0.5consP8.npz')
comp = {k: {s: C[f'{k}/{s}'] for s in sets if f'{k}/{s}' in C} for k in ('Sd', 'C', 'xSL', 'consP8')}
FAz = dict(np.load('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/diffusion/fit/z_FA.npz'))
P8 = ['psL', 'psS', 'onept', 'extrema', 'mf', 'dog', 'cnn', 'pred0']
variants = {'consP8(re)': P8, 'P8+bands': P8 + ['dif_lo', 'dif_mid', 'dif_hi'],
            'P8+bands+FA': P8 + ['dif_lo', 'dif_mid', 'dif_hi', 'dif_FA'], 'P8+FA': P8 + ['dif_FA'],
            'bands+cnn+pred0': ['dif_lo', 'dif_mid', 'dif_hi', 'cnn', 'pred0']}
keys = [k for k in comp['Sd'] if k != 'test']
print('exported consP8   ', lib.fmt('', A.ev(comp['consP8'])))
for name, fs in variants.items():
    z = cons(fs)
    print(f'{name:18s}', lib.fmt('', A.ev(z)), flush=True)
    rec = {s: (comp['Sd'][s] + comp['C'][s] + .5 * comp['xSL'][s] + .5 * z[s]) / 3 for s in keys}
    print(f'  REC({name})', lib.fmt('', A.ev(rec)), flush=True)
    rec2 = {s: (comp['Sd'][s] + comp['C'][s] + .5 * comp['xSL'][s] + .5 * z[s] + .5 * FAz[s]) / 3.5 for s in keys}
    print(f'  REC({name})+.5FA', lib.fmt('', A.ev(rec2)), flush=True)
rec = {s: (comp['Sd'][s] + comp['C'][s] + .5 * comp['xSL'][s] + .5 * comp['consP8'][s]) / 3 for s in keys}
print('REC1 (exported comps)', lib.fmt('', A.ev(rec)))
