"""Do diffusion-feature thetas from different scales / noise levels disagree on baryon-type shifts?

For a pair of theta families (A, B), per map dth = th_A - th_B (normalized theta units).
  effect d_p(shift) = (mean dth_p[shift] - mean dth_p[audit]) / std dth_p[calib]     (per parameter)
  pair score = log1p(Mahalanobis^2 of dth, covariance + mean from calib) -> phase2 vs audit (raw ranking)
Compares diffusion radial bands (lo/mid/hi), diffusion noise levels (nlo/nhi, SS/LS) and the stats
small/large split that xSL uses. Full fit3 fits (not LCO). Read-only on other agents' files.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import phase2_score  # noqa: E402

V3 = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3')
lab = np.load(V3 / 'stats' / 'fit.npz')['labels']
TM, TS = lab.mean(0), lab.std(0)
SH = ['bary6.5', 'baryDMO', 'bary9.0', 'supp0.15', 'boost0.15', 'ia0.2', 'smooth0.5', 'aniso0.7', 'psfpat',
      'srcclust0.15', 'mult+0.05']
SETS = ['calib', 'audit'] + ['shift_' + s for s in SH]
P = ['Om', 'S8', 'TAGN', 'f0', 'Dz']


def raw_fam(path):
    d = np.load(path)
    return {s: (d[f'{s}/th'] - TM) / TS for s in SETS}


def norm_fam(path):
    d = np.load(path)
    return {s: d[f'{s}/th'] for s in SETS}


B = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/diffusion/bandcons/fam')
CF = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/consistency/fam')
NL = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/diffusion2/nlevel/fam')
TH = {f'dif_{t}': raw_fam(B / f'{t}.npz') for t in ('lo', 'mid', 'hi', 'FA')}
for t in ('small', 'large', 'dog', 'cnn'):
    TH[f'st_{t}'] = norm_fam(CF / f'{t}.npz')
for t in ('nlo', 'nhi', 'SS', 'LS'):
    if (NL / f'{t}.npz').exists():
        TH[f'dif_{t}'] = raw_fam(NL / f'{t}.npz')

PAIRS = [('st_small', 'st_large'), ('dif_lo', 'dif_hi'), ('dif_lo', 'dif_mid'), ('dif_mid', 'dif_hi'),
         ('dif_nlo', 'dif_nhi'), ('dif_SS', 'dif_LS'), ('dif_FA', 'st_dog'), ('dif_FA', 'st_cnn'),
         ('dif_hi', 'st_small'), ('dif_lo', 'st_large'), ('dif_LS', 'st_large'), ('dif_SS', 'st_small')]
for a, b in PAIRS:
    if a not in TH or b not in TH:
        continue
    d = {s: TH[a][s] - TH[b][s] for s in SETS}
    mu, C = d['calib'].mean(0), np.cov(d['calib'].T)
    Ci = np.linalg.inv(C + 1e-3 * np.trace(C) / 5 * np.eye(5))
    sc = {s: np.log1p(np.einsum('ni,ij,nj->n', d[s] - mu, Ci, d[s] - mu)) for s in SETS}
    sd = d['calib'].std(0)
    print(f'\n{a} - {b}: calib std of dth = ' + ' '.join(f'{p}={x:.2f}' for p, x in zip(P, sd)))
    for s in SH:
        eff = (d['shift_' + s].mean(0) - d['audit'].mean(0)) / sd
        print(f'  {s:13s} pair-score={phase2_score(sc["audit"], sc["shift_" + s]):.3f}  effect ' +
              ' '.join(f'{p}={e:+.2f}' for p, e in zip(P, eff)))
