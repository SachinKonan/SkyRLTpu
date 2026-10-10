"""LCO proxy for the diffusion (field-level denoising) members, reusing (read-only) the consistency
agent's leave-cosmology-out refits: lco3/set{k}_std/{dif,difa,difk}.npz already exist.

Members: F = dif (ELBO-error band powers per noise level), FA = difa, FK = difk (posterior-mean features).
Reports single-member LCO m19 / exceedance, v3-04 + w*F*, BIG_RE2 + w*F*, and consistency scores
that add the diffusion theta families to consR / consP8.
"""
import sys
from pathlib import Path

import numpy as np

CONS = Path(__file__).resolve().parents[1] / 'consistency'
sys.path.insert(0, str(CONS)); sys.path.insert(0, str(CONS.parents[1]))
import lib  # noqa: E402
import lco3_analyze as L  # noqa: E402
from analyze import standardize  # noqa: E402

KS = (0, 1, 2)
DIF = {'F': 'dif', 'FA': 'difa', 'FK': 'difk'}
names = L.NAMES_BASE + ['sh_' + s for s in lib.SHIFTS]


def comps(k):
    c = L.components(k, 'std')
    d = lib.OUT / 'lco3' / f'set{k}_std'
    TH = {}
    for m, f in DIF.items():
        H = dict(np.load(d / f'{f}.npz'))
        c[m] = standardize({n: H[f'{n}/cop'] for n in names}, {n: H[f'{n}/th'] for n in names})
        TH[f] = {n: H[f'{n}/th'] for n in names}
    for f in ['psL', 'psS', 'onept', 'extrema', 'mf', 'dog', 'cnn', 'b0', 'b1', 'b2', 'b3', 'cnn1', 'den_r',
              'den_sd']:
        H = dict(np.load(d / f'{f}.npz'))
        TH[f] = {n: H[f'{n}/th'] for n in names}
    TH['pred0'] = L.pred_sets('cnn0', k)
    TH['pred1'] = L.pred_sets('cnn1', k)
    p8 = ['psL', 'psS', 'onept', 'extrema', 'mf', 'dog', 'cnn', 'pred0']
    rb = ['b0', 'b1', 'b2', 'b3', 'cnn', 'cnn1', 'pred0', 'pred1', 'dog', 'den_r', 'den_sd']
    c['consP8+FA'] = L.cons(TH, p8 + ['difa'], names)
    c['consR+FA'] = L.cons(TH, rb + ['difa'], names)
    c['consR+F+FK'] = L.cons(TH, rb + ['dif', 'difk'], names)
    c['consF'] = L.cons(TH, ['dif', 'difk', 'cnn', 'pred0'], names)
    return c, TH


C, THS = {}, {}
for k in KS:
    C[k], THS[k] = comps(k)


def report(name, w):
    rs = [L.evaluate(L.combo(C[k], w)) for k in KS]
    exc = np.mean([r['exc'] for r in rs], 0)
    m19 = np.mean([r['m19'] for r in rs])
    per = ' '.join(f'{r["m19"]:.3f}' for r in rs)
    keys = ['bary6.5', 'baryDMO', 'bary9.0', 'supp0.15', 'boost0.15', 'ia0.2', 'srcclust0.15', 'psfpat', 'aniso0.7',
            'smooth0.5']
    shs = ' '.join(f'{s[:6]}={np.mean([r["sh"][s] for r in rs]):.3f}' for s in keys)
    print(f'{name:30s} LCO m19={m19:.4f} ({per}) exc={exc[0]:.2f}/{exc[1]:.2f}/{exc[2]:.2f}% | {shs}', flush=True)
    return m19


print('=== single')
for m in ['Sd', 'C', 'Dr', 'N2', 'F', 'FA', 'FK', 'consP8', 'consR', 'consP8+FA', 'consR+FA', 'consR+F+FK', 'consF']:
    report(m, {m: 1})
V304 = {'Sd': 1, 'C': 1, 'Dr': 1, 'xSL': 1, 'consP8': 1}
BIG = {'Dr': 1, 'N2': 1, 'PsBh': .5, 'xSL': 1, 'consR': 2}
print('=== v3-04 family')
report('v3-04', V304)
for m in ['F', 'FA', 'FK']:
    for w in (.5, 1):
        report(f'v3-04 + {w}{m}', {**V304, m: w})
report('v3-04 consP8->consP8+FA', {**{a: b for a, b in V304.items() if a != 'consP8'}, 'consP8+FA': 1})
print('=== Dr+Sd+C vs F+Dr+Sd+C (task combo)')
report('Dr+Sd+C', {'Dr': 1, 'Sd': 1, 'C': 1})
report('F+Dr+Sd+C', {'F': 1, 'Dr': 1, 'Sd': 1, 'C': 1})
report('FA+Dr+Sd+C', {'FA': 1, 'Dr': 1, 'Sd': 1, 'C': 1})
print('=== BIG_RE2 family')
report('BIG_RE2', BIG)
for m in ['F', 'FA', 'FK']:
    for w in (.5, 1):
        report(f'BIG_RE2 + {w}{m}', {**BIG, m: w})
report('BIG_RE2 consR->consR+FA', {**{a: b for a, b in BIG.items() if a != 'consR'}, 'consR+FA': 2})
report('BIG_RE2 consR->consR+F+FK', {**{a: b for a, b in BIG.items() if a != 'consR'}, 'consR+F+FK': 2})
report('BIG_RE2 + 1consF', {**BIG, 'consF': 1})
