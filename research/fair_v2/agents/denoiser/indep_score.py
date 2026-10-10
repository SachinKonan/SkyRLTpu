"""Rescore one-at-a-time den features (indep_feats.npy) with frozen fit_v3 models; compare with batch z."""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from groups import groups  # noqa: E402

A = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/denoiser')
ids = [0, 1, 17, 101, 1007, 4095, 7001, 9999]
f = np.load(A / 'indep_feats.npy')
cols = {'den': slice(0, 1090), 'den_r': slice(0, 112), 'den_s': slice(112, 578), 'den_b': slice(578, 1090)}
G, _ = groups()
for run, tag in (('f1', 'D'), ('f2', 'Dr'), ('f2', 'Db'), ('f3', 'Ds')):
    mdl = pickle.load(open(A / 'runs' / run / f'model_{tag}.pkl', 'rb'))
    src, sub = mdl['spec'].split(':')
    x = f[:, cols[src]]
    if sub != 'raw':
        x = x[:, np.concatenate([G[g] for g in sub.split('+')])]
    single = []
    for r in range(len(ids)):
        raw, th = mdl['lk'].profile(x[r:r + 1])
        if mdl['stat'] == 'copula':
            raw = mdl['lk'].last_copula
        single.append(mdl['std'](raw, th)[0])
    batch = np.load(A / 'runs' / run / f'z_{tag}.npz')['test'][ids]
    print(tag, 'max |z_single - z_batch| =', float(np.abs(np.array(single) - batch).max()))
