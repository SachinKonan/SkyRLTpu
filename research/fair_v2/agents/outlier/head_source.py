"""Derived feature source: only the OE head outputs (ood logit, family logits, param preds)
of an existing OE source, same maps/rows. Usage: head_source.py SRC DST"""
import sys
from pathlib import Path
import numpy as np
root = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3')
src, dst = root / sys.argv[1], root / sys.argv[2]
dst.mkdir(exist_ok=True)
for p in sorted(src.glob('*.npz')):
    if '.tmp' in p.name:
        continue
    with np.load(p) as f:
        d = {k: f[k] for k in f.files}
    nfh = d['feats'].shape[-1] - 1 - 5 - 256
    d['feats'] = np.ascontiguousarray(d['feats'][..., :1 + nfh + 5])
    np.savez(dst / p.name, **d)
(dst / 'SOURCE.txt').write_text(f'columns 0..{nfh + 5} of {src} (ood logit, family-head logits, param preds)\n')
print('wrote', dst, d['feats'].shape)
