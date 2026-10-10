"""Write full-data standardized z of single likelihood members (from fam/<family>.npz) in merge.py format."""
import sys
import numpy as np
import lib
from analyze import standardize, fam

out = sys.argv[1]
for m, f in [x.split('=') for x in sys.argv[2:]]:
    sets = lib.SETS + ['test']
    F = fam(f)
    z = standardize({s: F[f'{s}/cop'] for s in sets}, {s: F[f'{s}/th'] for s in sets})
    np.savez(f'{out}/z_{m}.npz', **z)
    print(m, f, {k: v.shape for k, v in list(z.items())[:2]}, 'finite', all(np.isfinite(v).all() for v in z.values()))
