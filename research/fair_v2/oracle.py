"""Oracle diagnostic: detection at TRUE parameters vs profiled parameters."""
import pickle, sys, json
import numpy as np, torch
from model import phase2_score
R = '/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3/'
run = R + 'runs/r2/'
out = {}
for tag in sys.argv[1].split(','):
    m = pickle.load(open(run + f'model_{tag}.pkl', 'rb')); lk = m['lk']
    from groups import groups
    G, _ = groups()
    def cols(name):
        parts = []
        for item in m['spec'].split('|'):
            src, subset = item.split(':')
            d = np.load(R + f'{src}/{name}.npz')
            f = d['feats'][:, 0]
            parts.append(f if subset == 'raw' else f[:, np.concatenate([G[g] for g in subset.split('+')])])
        return np.concatenate(parts, 1), d['labels']
    def oracle(name):
        f, lab = cols(name)
        x = lk.standardize(f)
        t = torch.as_tensor((lab - lk.t_mean) / lk.t_std, dtype=torch.float32, device=lk.dev)
        with torch.no_grad():
            return lk.copula_energy(x, t).cpu().numpy()
    a = oracle('audit')
    z = np.load(run + f'z_{tag}.npz')
    res = {}
    for s in ['ia0.2', 'poisson120', 'bary9.0', 'bary6.5', 'baryDMO', 'peakcut0.3', 'mult+0.05', 'supp0.15', 'psfpat']:
        res[s] = dict(oracle=round(phase2_score(a, oracle('shift_' + s)), 3),
                      profile=round(phase2_score(z['audit'], z['shift_' + s]), 3))
    out[tag] = res
    print(tag, json.dumps(res), flush=True)
