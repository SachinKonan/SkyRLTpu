"""Paired effect sizes: shift vs audit on the same clean maps (first n maps), per feature family."""
import sys, numpy as np
from pathlib import Path
from make_sources import cols
O = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/diffusion')
fd = Path(sys.argv[1]) if len(sys.argv) > 1 else O / 'feats_smoke_bf16'
V3 = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3')
n = 512
au = np.concatenate([np.load(fd / f'parts/audit_v0_c{c}.npz')['feats'] for c in range(1)])[:n]
st_a = np.load(V3 / 'stats/audit.npz')['feats'][:n, 0]
cn_a = np.load(V3 / 'cnn0/audit.npz')['feats'][:n, 0]
def dprime(a, b):
    d = b - a
    s = d.std(0) / np.sqrt(2) + 1e-12
    return d.mean(0) / s
def multi(a, b):
    # Gaussian Mahalanobis of the mean paired shift in units of single-map noise (shrunk covariance)
    d = b - a; mu = d.mean(0); C = np.cov(d.T) / 2
    C = 0.9 * C + 0.1 * np.diag(np.diag(C)) + 1e-12 * np.eye(len(C))
    return float(np.sqrt(mu @ np.linalg.solve(C, mu)))
print(f'{"shift":18s} | dif max|d|  multi | difk max|d| multi | stats max|d| multi | cnn0 max|d| multi')
for sh in ['peakcut0.3', 'bary6.5', 'supp0.15', 'boost0.15', 'gradient', 'srcclust0.15', 'poisson120', 'ia0.2', 'psfpat']:
    p = fd / f'parts/shift_{sh}_v0_c0.npz'
    if not p.exists():
        continue
    f = np.load(p)['feats'][:n]
    row = []
    for kind in ('dif', 'difk'):
        c = cols(kind)
        dd = dprime(au[:, c], f[:, c]); row += [np.abs(dd).max(), multi(au[:, c], f[:, c])]
    sp = V3 / f'stats/shift_{sh}.npz'
    if sp.exists():
        s = np.load(sp)['feats'][:n, 0]; keep = st_a.std(0) > 0
        dd = dprime(st_a[:, keep], s[:, keep]); row += [np.abs(dd).max(), multi(st_a[:, keep], s[:, keep])]
        c2 = np.load(V3 / f'cnn0/shift_{sh}.npz')['feats'][:n, 0]
        dd = dprime(cn_a, c2); row += [np.abs(dd).max(), multi(cn_a, c2)]
    print(f'{sh:18s} | ' + ' | '.join(f'{row[i]:6.2f} {row[i+1]:6.2f}' for i in range(0, len(row), 2)))
