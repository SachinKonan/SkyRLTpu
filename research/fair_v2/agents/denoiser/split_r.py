"""Derived sources from den_r: den_rr (residual spectra/moments/cross, cols 0-79) and
den_rt (SURE + tile + tail block, cols 80-111). Pure column slicing of the same per-map features."""
from pathlib import Path

import numpy as np

R = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3')
for name, sl in (('den_rr', slice(0, 80)), ('den_rt', slice(80, 112))):
    (R / name).mkdir(exist_ok=True)
    for p in sorted((R / 'den_r').glob('*.npz')):
        with np.load(p) as f:
            d = {k: f[k] for k in f.files}
        assert d['feats'].shape[-1] == 112, d['feats'].shape
        d['feats'] = d['feats'][..., sl]
        np.savez(R / name / p.name, **d)
    print(name, len(list((R / name).glob('*.npz'))))
