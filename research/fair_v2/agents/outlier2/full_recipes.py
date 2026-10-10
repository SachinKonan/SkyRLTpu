"""Full-data (fit3/calib3-fitted) recipes with OE members: audit tail, region tail, shifts; writes frozen z files."""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, '../..')
import data as D
from model import phase2_score
from analyze_L import region, evaluate, SETS
E3 = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/consistency/export3')
OUT = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/outlier2/runs/recipes'); OUT.mkdir(parents=True, exist_ok=True)
def comps(p):
    c = {}
    with np.load(p) as f:
        for k in f.files:
            m, s = k.split('/'); c.setdefault(m, {})[s] = f[k]
    return c
c = comps(E3 / 'components_v3-04_repro.npz'); c.update({k: v for k, v in comps(E3 / 'components_BIG_RE2.npz').items() if k not in c})
R1 = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/outlier2/runs/r1')
for t, p in (('O', R1 / 'z_O.npz'), ('Oh', R1 / 'z_Oh.npz'), ('Oho', '/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/outlier/fit_ho/z_O.npz')):
    with np.load(p) as f: c[t] = {k: f[k] for k in f.files}
def mix(w):
    tot = sum(w.values()); return {n: sum(v * c[k][n] for k, v in w.items()) / tot for n in SETS}
V4 = {'Sd': 1, 'C': 1, 'Dr': 1, 'xSL': 1, 'consP8': 1}; BIG = {'Dr': 1, 'N2': 1, 'PsBh': .5, 'xSL': 1, 'consR': 2}
# sanity: reproduce exported test z
for name, w in (('v3-04_repro', V4), ('BIG_RE2', BIG)):
    ref = np.load(E3 / f'test_z_{name}.npy'); z = mix(w)['test']
    print(name, 'reproduce export3 test z: corr %.6f max|diff| %.2e' % (np.corrcoef(ref, z)[0, 1], np.abs(ref - z).max()))
R = {'v3-04': V4, 'v3-04+O': {**V4, 'O': 1}, 'v3-04+Oh': {**V4, 'Oh': 1}, 'v3-04+Oho': {**V4, 'Oho': 1},
     'BIG_RE2': BIG, 'BIG_RE2+Oh': {**BIG, 'Oh': 1}, 'BIG_RE2+0.5Oh': {**BIG, 'Oh': .5}, 'BIG_RE2+Oho': {**BIG, 'Oho': 1},
     'BIG_RE2+O': {**BIG, 'O': 1}, 'Oho': {'Oho': 1}}
lab_cal = np.repeat(np.load(D.ROOT / 'v3' / 'stats' / 'calib.npz')['labels'], 2, axis=0) if hasattr(D, 'ROOT') else None
lab_cal = np.repeat(np.load('/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3/stats/calib.npz')['labels'], 2, axis=0)
lab_aud = np.load('/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3/stats/audit.npz')['labels']
for n, w in R.items():
    z = mix(w); r = evaluate(z); rg = region(z, lab_cal, lab_aud)
    print(f'{n:14s} audit q50/90/99/99.5/99.9={"/".join(f"{v:.2f}" for v in r["q"])} m19={r["m19"]:.4f} phys7={r["phys"]:.4f}')
    for nm in ('T_AGN', 'f0', 'Delta_z'):
        print(f'      {nm:8s} >99 by bin: ' + ' '.join(f'{x["gt99"]:.3f}' for x in rg[nm]) + '  >99.9: ' + ' '.join(f'{x["gt999"]:.4f}' for x in rg[nm]))
    np.savez(OUT / f'z_{n}.npz', **z)
    (OUT / f'recipe_{n}.json').write_text(json.dumps(w))
