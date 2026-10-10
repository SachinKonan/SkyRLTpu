"""Held-out theta information per family: ridge regression on quadratic-free standardized features
(fit3, both views) -> theta; R^2 on calib3 per parameter (Om, S8, T_AGN, f0, dz)."""
import numpy as np
V3 = '/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3'
DROP = {'stx_nois': [7, 8, 25, 26, 53]}
print('source        dims   R2: Om     S8     T_AGN  f0     dz')
for src in ['stats', 'cnn0', 'stx_scat', 'stx_prof', 'stx_pkcl', 'stx_ipow', 'stx_bisp', 'stx_nois', 'stx_tail']:
    a = np.load(f'{V3}/{src}/fit.npz'); b = np.load(f'{V3}/{src}/calib.npz')
    X = a['feats'].astype(np.float64); Y = a['labels']; Xc = b['feats'].astype(np.float64); Yc = b['labels']
    if src in DROP:
        X = np.delete(X, DROP[src], -1); Xc = np.delete(Xc, DROP[src], -1)
    n, v, d = X.shape
    X = X.reshape(n * v, d); Y = np.repeat(Y, v, 0); Xc = Xc.reshape(-1, d); Yc = np.repeat(Yc, 2, 0)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd; Zc = (Xc - mu) / sd
    Z = np.c_[Z, np.ones(len(Z))]; Zc = np.c_[Zc, np.ones(len(Zc))]
    W = np.linalg.solve(Z.T @ Z + 1.0 * np.eye(Z.shape[1]), Z.T @ Y)
    P = Zc @ W
    r2 = 1 - ((P - Yc) ** 2).mean(0) / Yc.var(0)
    print(f'{src:12s} {d:5d}  ' + ' '.join(f'{x:6.3f}' for x in r2), flush=True)
