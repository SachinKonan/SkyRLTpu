"""difc = all conditional features (4 levels x 34); difcl = only the conditioning-effect columns
(log band power of kc - ku and the per-band relative gain), 4 x 16."""
import sys
from pathlib import Path
import numpy as np
src, root = Path(sys.argv[1]), Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3')
PER = 34
cl = np.concatenate([np.arange(k * PER + 16, k * PER + 32) for k in range(4)])
for fn in sorted(src.glob('*.npz')):
    d = dict(np.load(fn))
    for name, c in (('difc', None), ('difcl', cl)):
        (root / name).mkdir(exist_ok=True)
        f = d['feats'] if c is None else d['feats'][..., c]
        np.savez(root / name / fn.name, **{**d, 'feats': f.astype(np.float32)})
    print(fn.name, flush=True)
