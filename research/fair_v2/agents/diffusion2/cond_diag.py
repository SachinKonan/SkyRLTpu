"""Quick signal check of the incomplete parameter-conditional features (feats_cond: calib/audit/shifts only,
no fit/test). Linear-in-theta_hat (cnn0 predictions) Gaussian residual Mahalanobis, fitted on calib3 only."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import phase2_score  # noqa: E402
V3 = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3')
FC = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/diffusion/feats_cond')
shifts = sorted(p.stem for p in FC.glob('shift_*.npz'))
PER = 34
cols = {'all': np.arange(136), 'condeff': np.concatenate([np.arange(k * PER + 16, k * PER + 32) for k in range(4)])}
def load(n):
    f = np.load(FC / f'{n}.npz')['feats']; t = np.load(V3 / 'cnn0' / f'{n}.npz')['feats'][..., -5:]
    return f.reshape(-1, f.shape[-1]), t.reshape(-1, 5)
data = {n: load(n) for n in ['calib', 'audit'] + shifts}
for name, c in cols.items():
    Xc, Tc = data['calib']; Xc = Xc[:, c]
    A = np.c_[np.ones(len(Tc)), Tc, Tc ** 2]
    B, *_ = np.linalg.lstsq(A, Xc, rcond=None)
    R = Xc - A @ B; mu = R.mean(0); C = np.cov(R.T); C += 1e-3 * np.trace(C) / len(C) * np.eye(len(C)); Ci = np.linalg.inv(C)
    def sc(n):
        X, T = data[n]; r = X[:, c] - np.c_[np.ones(len(T)), T, T ** 2] @ B - mu
        return np.einsum('ni,ij,nj->n', r, Ci, r)
    za = sc('audit'); sh = {s.removeprefix('shift_'): phase2_score(za, sc(s)) for s in shifts}
    print(name, 'm19=%.4f' % np.mean(list(sh.values())), ' '.join(f'{k[:6]}={v:.3f}' for k, v in sh.items()))
