"""LCO proxy for the noise-level diffusion members (lco_nl.py refits) on top of v3-04 / BIG_RE2."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'consistency')); sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
import lib  # noqa: E402
import lco3_analyze as L  # noqa: E402
from analyze import standardize  # noqa: E402

KS = (0, 1, 2)
names = L.NAMES_BASE + ['sh_' + s for s in lib.SHIFTS]
MY = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/diffusion2/lco3')


def comps(k):
    c = L.components(k, 'std')
    d = lib.OUT / 'lco3' / f'set{k}_std'
    TH = {}
    for m, f in {'F': 'dif', 'FA': 'difa'}.items():
        Hh = dict(np.load(d / f'{f}.npz'))
        c[m] = standardize({n: Hh[f'{n}/cop'] for n in names}, {n: Hh[f'{n}/th'] for n in names})
    for t in ('nlo', 'SS', 'LS', 'nhi'):
        p = MY / f'set{k}' / f'dn_{t}.npz'
        if p.exists():
            Hh = dict(np.load(p))
            c[t] = standardize({n: Hh[f'{n}/cop'] for n in names}, {n: Hh[f'{n}/th'] for n in names})
            TH[t] = {n: Hh[f'{n}/th'] for n in names}
    for f in ('small', 'large'):
        Hh = dict(np.load(d / f'{f}.npz'))
        TH[f] = {n: Hh[f'{n}/th'] for n in names}
    if 'SS' in TH and 'LS' in TH:
        c['consSSLS'] = L.cons(TH, ['SS', 'LS'], names)
        c['consSL+SSLS'] = L.cons(TH, ['SS', 'LS', 'small', 'large'], names)
    if 'nlo' in TH and 'nhi' in TH:
        c['consNL'] = L.cons(TH, ['nlo', 'nhi'], names)
    return c


C = {k: comps(k) for k in KS}
avail = set.intersection(*[set(c) for c in C.values()])


def report(name, w):
    if not all(m in avail for m in w):
        return
    rs = [L.evaluate(L.combo(C[k], w)) for k in KS]
    exc = np.mean([r['exc'] for r in rs], 0)
    m19 = np.mean([r['m19'] for r in rs])
    per = ' '.join(f'{r["m19"]:.3f}' for r in rs)
    keys = ['bary6.5', 'baryDMO', 'bary9.0', 'supp0.15', 'boost0.15', 'ia0.2', 'srcclust0.15', 'psfpat', 'aniso0.7']
    shs = ' '.join(f'{s[:6]}={np.mean([r["sh"][s] for r in rs]):.3f}' for s in keys)
    print(f'{name:28s} LCO m19={m19:.4f} ({per}) exc={exc[0]:.2f}/{exc[1]:.2f}/{exc[2]:.2f}% | {shs}', flush=True)


print('=== single')
for m in ('FA', 'nlo', 'SS', 'LS', 'nhi', 'consSSLS', 'consNL', 'consSL+SSLS', 'xSL'):
    report(m, {m: 1})
V304 = {'Sd': 1, 'C': 1, 'Dr': 1, 'xSL': 1, 'consP8': 1}
BIG = {'Dr': 1, 'N2': 1, 'PsBh': .5, 'xSL': 1, 'consR': 2}
for base, W in (('v3-04', V304), ('BIG_RE2', BIG)):
    print(f'=== {base}')
    report(base, W)
    for m in ('FA', 'nlo', 'SS', 'consSSLS', 'consNL', 'consSL+SSLS'):
        for w in (.5, 1):
            report(f'{base} + {w}{m}', {**W, m: w})
    report(f'{base} + .5nlo + .5consSSLS', {**W, 'nlo': .5, 'consSSLS': .5})
