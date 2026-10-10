import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import data as D
V3 = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3')
names = ['fit', 'calib', 'audit'] + ['shift_' + s for s in D.SHIFTS] + ['test']
want_v = dict(fit=2, calib=2, test=1)
for src in ['stx_scat', 'stx_prof', 'stx_pkcl', 'stx_ipow', 'stx_bisp', 'stx_nois', 'stx_tail']:
    bad = []; dims = set()
    for n in names:
        p = V3 / src / f'{n}.npz'
        if not p.exists():
            bad.append(f'missing {n}'); continue
        z = np.load(p); f = z['feats']; dims.add(f.shape[-1])
        if f.shape[1] != want_v.get(n, 1): bad.append(f'views {n} {f.shape}')
        if not np.isfinite(f).all(): bad.append(f'nonfinite {n} {(~np.isfinite(f)).sum()}')
        if n != 'test':
            ref = np.load(V3 / 'stats' / f'{n}.npz')['index']
            if not np.array_equal(ref, z['index']): bad.append(f'index {n}')
        elif f.shape[0] != 10000: bad.append('test rows')
    print(src, 'dims', dims, 'sets', len(names), 'OK' if not bad else bad)
