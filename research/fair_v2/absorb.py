"""Which parameters absorb each shift? Paired change of profiled theta (shift vs normal, same clean maps)."""
import pickle, sys, json
import numpy as np
from groups import groups
R = '/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3/'
tag = sys.argv[1]
m = pickle.load(open(R + f'runs/r2/model_{tag}.pkl', 'rb')); lk = m['lk']
G, _ = groups()
def cols(name):
    parts = []
    for item in m['spec'].split('|'):
        src, subset = item.split(':')
        f = np.load(R + f'{src}/{name}.npz')['feats'][:, 0]
        parts.append(f if subset == 'raw' else f[:, np.concatenate([G[g] for g in subset.split('+')])])
    return np.concatenate(parts, 1)
lab = np.load(R + 'stats/audit.npz')['labels']
_, th_a = lk.profile(cols('audit'))
names = ['Om', 'S8', 'TAGN', 'f0', 'dz']
err = (th_a - lab) / lk.t_std
print(tag, 'normal: theta_hat - true (std units) mean', err.mean(0).round(2), 'sd', err.std(0).round(2))
for s in sys.argv[2].split(','):
    _, th = lk.profile(cols('shift_' + s))
    d = (th - th_a) / lk.t_std
    print(f'{s:12s} paired shift of theta_hat (std units) mean', d.mean(0).round(2), ' sd', d.std(0).round(2),
          ' frac at box edge (TAGN,f0):', np.mean(np.isclose(th[:, 2], lab[:, 2].min(), atol=0.02) | np.isclose(th[:, 2], lab[:, 2].max(), atol=0.02)).round(2),
          np.mean((th[:, 3] < 0.0005) | (th[:, 3] > 0.026)).round(2))
