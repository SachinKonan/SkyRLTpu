"""Supervised ceiling diagnostic (synthetic shifts only, never test): shrinkage-LDA trained to separate
shift_X from audit on a feature set, 5-fold CV split by map (paired maps stay together); reports the
phase-2 metric of held-out scores. Tells whether a feature set CONTAINS per-map information about a shift
(an unsupervised detector cannot exceed this much)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from model import phase2_score

V3 = '/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3'
SHIFTS = sys.argv[2].split(',') if len(sys.argv) > 2 else ['bary9.0', 'bary6.5', 'baryDMO', 'peakcut0.3', 'ia0.2', 'poisson120', 'psfpat', 'supp0.15']
SETS = {k: v.split('|') for k, v in (s.split('=') for s in sys.argv[1].split(','))}
cache = {}


def feats(src, name):
    if (src, name) not in cache:
        import os
        ds = f'/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/statistics/dshift/{src}/{name}.npz'
        f = np.load(ds if os.path.exists(ds) else f'{V3}/{src}/{name}.npz')['feats'][:, 0].astype(np.float64)
        cache[(src, name)] = f
    return cache[(src, name)]


def lda_cv(X0, X1, lams=(0.03, 0.3, 3.0), k=5):
    n = len(X0)
    fold = np.arange(n) % k
    best = None
    for lam in lams:
        s0 = np.zeros(n); s1 = np.zeros(n)
        for f in range(k):
            tr = fold != f; te = ~tr
            A = np.concatenate([X0[tr], X1[tr]])
            mu, sd = A.mean(0), A.std(0) + 1e-9
            a0 = (X0[tr] - mu) / sd; a1 = (X1[tr] - mu) / sd
            S = np.cov(np.concatenate([a0 - a0.mean(0), a1 - a1.mean(0)]).T)
            S += lam * np.eye(len(S))
            w = np.linalg.solve(S, a1.mean(0) - a0.mean(0))
            s0[te] = ((X0[te] - mu) / sd) @ w; s1[te] = ((X1[te] - mu) / sd) @ w
        sc = max(phase2_score(s0, s1), phase2_score(-s0, -s1))   # either tail
        best = sc if best is None else max(best, sc)
    return best


print('set'.ljust(14) + ' '.join(s[:10].rjust(10) for s in SHIFTS), flush=True)
for tag, srcs in SETS.items():
    X0 = np.concatenate([feats(s, 'audit') for s in srcs], 1)
    keep = X0.std(0) > 1e-9
    row = []
    for sh in SHIFTS:
        X1 = np.concatenate([feats(s, f'shift_{sh}') for s in srcs], 1)
        row.append(lda_cv(X0[:, keep], X1[:, keep]))
    print(tag.ljust(14) + ' '.join(f'{v:10.3f}' for v in row), flush=True)
