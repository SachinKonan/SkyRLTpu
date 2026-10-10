"""Save frozen full-data ridge-probe parameters (for packaging): theta_norm = ((x - mu) / sd) @ W + ym,
x = feature columns of the source (see c2probe.PROBES); theta normalized by fit3 label mean/std."""
import sys
import numpy as np
import c2probe as P
import lib

for p in sys.argv[1].split(','):
    L = lib.labels('fit'); tm, ts = L.mean(0), L.std(0)
    Xf = P.feats(p, 'fit'); V = Xf.shape[1]
    X = Xf.reshape(-1, Xf.shape[-1]).astype(np.float64)
    y = np.repeat((L - tm) / ts, V, 0)
    g = np.repeat(lib.load('stats', 'fit', 'index')[:, 0], V)
    f, a, mse = P.ridge_cv(X, y, g)
    mu, sd = X.mean(0), X.std(0) + 1e-8
    A = (X - mu) / sd; ym = y.mean(0)
    W = np.linalg.solve(A.T @ A + a * len(A) * np.eye(A.shape[1]), A.T @ (y - ym))
    with np.load(P.OUT / f'{p}_full.npz') as H:
        chk = np.abs(((P.feats(p, 'calib').reshape(-1, X.shape[1]) - mu) / sd) @ W + ym - H['calib']).max()
    np.savez(P.OUT / f'{p}_full_params.npz', W=W, mu=mu, sd=sd, ym=ym, alpha=a, label_mean=tm, label_std=ts)
    print(p, 'alpha', a, 'reproduces cached calib probe to', chk)
