"""LCO refits (consistency agent's holdout3 protocol, std variant, imported read-only) for the diffusion
noise-level families of nlevel.py. Output: fair-data/agents/diffusion2/lco3/set{k}/{fam}.npz (resumable)."""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'consistency')); sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
import lib  # noqa: E402
import holdout3 as H  # noqa: E402
from jlik import JLikelihood  # noqa: E402
from nlevel import fam_cols  # noqa: E402

for t, c in fam_cols().items():
    lib.FAMILIES['dn_' + t] = ('difa', c)
OUT = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/diffusion2/lco3')
t0 = time.time()
V = H.VARIANTS['std']
for k in (0, 1, 2):
    out = OUT / f'set{k}'; out.mkdir(parents=True, exist_ok=True)
    for fam in ('dn_nlo', 'dn_SS', 'dn_LS', 'dn_nhi'):
        if (out / f'{fam}.npz').exists():
            continue
        if time.time() - t0 > 420:
            print('budget reached', flush=True); sys.exit(0)
        t1 = time.time()
        S = H.sets_for(fam, k)
        lk = JLikelihood('cuda', 0.02, V['jm']).fit(S['fit'], H.labels_for(k), steps=V['steps'], width=V['width'],
                                                    log=lambda m: None, jitter=0.1)
        res = {}
        for name, x in S.items():
            if name == 'fit':
                continue
            for key, val in H.score(lk, x).items():
                res[f'{name}/{key}'] = val
        np.savez(out / f'{fam}.npz', **res)
        print(f'[set{k} {fam}] {time.time() - t1:.0f}s', flush=True)
print('ALL DONE', flush=True)
