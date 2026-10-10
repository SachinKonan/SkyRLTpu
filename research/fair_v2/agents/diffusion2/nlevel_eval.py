"""Standardize the noise-level families (nlevel.py) into z-scores and compare recipes on the synthetic shifts.

z members (each z_<tag>.npz: calib, audit, shift_*, test; standardized on calib3 like fit_v3):
  nlo, nhi, LS, SS        copula profile score of each family
  xNL, xSSLS              cross-fit energy (A at B's theta - A at own theta, both directions), like xSL
  consNL, consSSLS        log1p Mahalanobis^2 of the theta difference (calib covariance)
Recipes use full-fit components of v3-04 (export3/components_v3-04_repro) and BIG_RE2, plus diffusion
fit_v3 members F (dif) and FA (difa) from agents/diffusion/fit.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import Standardizer, phase2_score  # noqa: E402

V3 = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3')
O = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/diffusion2')
NL = O / 'nlevel' / 'fam'
lab = np.load(V3 / 'stats' / 'fit.npz')['labels']
TM, TS = lab.mean(0), lab.std(0)
shifts = sorted(p.stem for p in (V3 / 'stats').glob('shift_*.npz'))
SETS = ['calib', 'audit'] + shifts + ['test']
KEY = ['bary6.5', 'baryDMO', 'bary9.0', 'supp0.15', 'boost0.15', 'ia0.2', 'srcclust0.15', 'psfpat', 'aniso0.7',
       'smooth0.5']


def std(raw, th):
    S = Standardizer().fit(raw['calib'], th['calib'], TM, TS)
    return {s: S(raw[s], th[s]) for s in SETS}


def ev(name, z):
    sh = {s.removeprefix('shift_'): phase2_score(z['audit'], z[s]) for s in shifts}
    q = np.quantile(z['audit'], [.99, .995, .999])
    print(f'{name:34s} m19={np.mean(list(sh.values())):.4f} q={q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f} | ' +
          ' '.join(f'{k[:6]}={sh[k]:.3f}' for k in KEY), flush=True)
    return sh


Z, TH, COP = {}, {}, {}
for t in ('nlo', 'nhi', 'LS', 'SS'):
    if not (NL / f'{t}.npz').exists():
        continue
    d = np.load(NL / f'{t}.npz')
    TH[t] = {s: d[f'{s}/th'].astype(np.float64) for s in SETS}
    COP[t] = {s: d[f'{s}/cop'] for s in SETS}
    Z[t] = std(COP[t], TH[t])
for a, b, tag in (('nlo', 'nhi', 'NL'), ('SS', 'LS', 'SSLS')):
    fp = NL / f'cross_{a}_{b}.npz'
    if not fp.exists():
        continue
    X = np.load(fp)
    raw = {s: (X[f'{s}/cop_{a}@{b}'] - COP[a][s]) + (X[f'{s}/cop_{b}@{a}'] - COP[b][s]) for s in SETS}
    cen = {s: 0.5 * (TH[a][s] + TH[b][s]) for s in SETS}
    Z['x' + tag] = std(raw, cen)
    dv = {s: (TH[a][s] - TH[b][s]) / TS for s in SETS}
    mu, C = dv['calib'].mean(0), np.cov(dv['calib'].T)
    Ci = np.linalg.inv(C + 1e-3 * np.trace(C) / 5 * np.eye(5))
    Z['cons' + tag] = std({s: np.log1p(np.einsum('ni,ij,nj->n', dv[s] - mu, Ci, dv[s] - mu)) for s in SETS}, cen)
(O / 'z').mkdir(exist_ok=True)
for k, z in Z.items():
    np.savez(O / 'z' / f'z_{k}.npz', **z)

print('=== diffusion noise-level members (full fit, synthetic shifts)')
for k in Z:
    ev(k, Z[k])


def comps(path):
    c = np.load(path)
    out = {}
    for k in c.files:
        m, s = k.split('/', 1)
        out.setdefault(m, {})[s] = c[k]
    return out


M = comps('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/consistency/export3/components_v3-04_repro.npz')
M.update(comps('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/consistency/export3/components_BIG_RE2.npz'))
for t in ('F', 'FA', 'FK'):
    M[t] = dict(np.load(f'/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/diffusion/fit/z_{t}.npz'))
M.update(Z)


def mix(w):
    W = sum(w.values())
    return {s: sum(v * M[m][s] for m, v in w.items()) / W for s in ['audit'] + shifts}


V304 = {'Sd': 1, 'C': 1, 'Dr': 1, 'xSL': 1, 'consP8': 1}
BIG = {'Dr': 1, 'N2': 1, 'PsBh': .5, 'xSL': 1, 'consR': 2}
print('=== single reference members')
for m in ('Sd', 'C', 'Dr', 'xSL', 'consP8', 'consR', 'N2', 'F', 'FA', 'FK'):
    ev(m, M[m])
print('=== recipes')
ev('Dr+Sd+C', mix({'Dr': 1, 'Sd': 1, 'C': 1}))
ev('F+Dr+Sd+C', mix({'F': 1, 'Dr': 1, 'Sd': 1, 'C': 1}))
ev('FA+Dr+Sd+C', mix({'FA': 1, 'Dr': 1, 'Sd': 1, 'C': 1}))
ev('v3-04', mix(V304))
for extra in ('F', 'FA', 'xNL', 'xSSLS', 'consNL', 'consSSLS'):
    if extra in M:
        for w in (.5, 1):
            ev(f'v3-04 + {w}{extra}', mix({**V304, extra: w}))
ev('BIG_RE2', mix(BIG))
for extra in ('F', 'FA', 'xNL', 'xSSLS'):
    if extra in M:
        for w in (.5, 1):
            ev(f'BIG_RE2 + {w}{extra}', mix({**BIG, extra: w}))
