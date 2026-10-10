"""Paired per-feature effect size of synthetic shifts: shift_X and audit share clean maps (different noise).
d = mean(f_shift - f_audit) / sd(f_audit - f_fitmodel) approximated by the sd of audit features within cosmology."""
import sys
import numpy as np
V3 = '/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3'
src = sys.argv[1]
shifts = sys.argv[2].split(',') if len(sys.argv) > 2 else ['bary9.0', 'bary6.5', 'baryDMO', 'peakcut0.3', 'supp0.15', 'boost0.15']
a = np.load(f'{V3}/{src}/audit.npz'); fa = a['feats'][:, 0].astype(np.float64); idx = a['index']
# within-cosmology sd (removes cosmology-driven variance)
res = fa.copy()
for c in np.unique(idx[:, 0]):
    s = idx[:, 0] == c
    res[s] -= fa[s].mean(0)
sd = res.std(0) + 1e-12
print(src, 'dims', fa.shape[1])
for sh in shifts:
    try:
        fs = np.load(f'{V3}/{src}/shift_{sh}.npz')['feats'][:, 0]
    except FileNotFoundError:
        continue
    d = (fs - fa).mean(0) / sd
    k = np.argsort(-np.abs(d))[:8]
    print(f'  {sh:12s} max|d|={np.abs(d).max():6.2f}  rms d={np.sqrt((d**2).mean()):5.2f}  top', ' '.join(f'{j}:{d[j]:+.2f}' for j in k))
