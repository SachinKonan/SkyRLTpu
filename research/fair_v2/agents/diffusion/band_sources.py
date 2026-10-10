"""Per-scale-band subsets of the field-level features (for per-band theta profiling).

From the extracted per-level features (see make_sources.py layout) keep, for every
diffusion level, the log band powers of the ELBO error e_k (levels with sigma_t>0),
of r_k = x - kh_k and of kh_k, restricted to a group of radial bands:
  difb_lo  : bands 0-2  (f < 0.04 cyc/pix, wavelength > 50 arcmin)
  difb_mid : bands 3-4  (0.04-0.12 cyc/pix, 17-50 arcmin)
  difb_hi  : bands 5-7  (f > 0.12 cyc/pix, < 17 arcmin)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract import LEVELS, EDGES  # noqa: E402

NB = len(EDGES) - 1
PER = 3 * NB + 5
GROUPS = dict(lo=[0, 1, 2], mid=[3, 4], hi=[5, 6, 7])


def cols(bands):
    out = []
    for k, st in enumerate(LEVELS):
        b = k * PER
        if st > 0:
            out += [b + j for j in bands]
        out += [b + NB + j for j in bands] + [b + 2 * NB + j for j in bands]
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=Path, required=True)
    ap.add_argument('--root', type=Path, default=Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3'))
    args = ap.parse_args()
    for fn in sorted(args.src.glob('*.npz')):
        with np.load(fn) as z:
            d = {k: z[k] for k in z.files}
        for g, bands in GROUPS.items():
            out = args.root / f'difb_{g}'
            out.mkdir(parents=True, exist_ok=True)
            np.savez(out / fn.name, **{**d, 'feats': d['feats'][..., cols(bands)].astype(np.float32)})
        print(fn.name, flush=True)


if __name__ == '__main__':
    main()
