"""Slice extracted field-level features into v3 feature sources.

Per diffusion level (29 columns): [0:8] log band power of the ELBO error e_k,
[8:16] of r_k = x - kh_k, [16:24] of kh_k, [24:26] skew/kurt of kh_k,
[26:28] skew/kurt of r_k, [28] mean e_k^2.
Sources written under ROOT:
  dif   ELBO decomposition: levels>=1 x 8 bands of log |e_k|^2
  difk  posterior-mean features: r_k, kh_k band powers + moments, all levels
  difa  dif + difk
  dife  scalar denoising ELBO  sum_k dlog(sigma) * mean(e_k^2) / sigma_k^2
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract import LEVELS, EDGES  # noqa: E402

NB = len(EDGES) - 1
PER = 3 * NB + 5


def cols(kind):
    out = []
    for k, st in enumerate(LEVELS):
        b = k * PER
        if kind in ('dif', 'difa') and st > 0:
            out += list(range(b, b + NB))
        if kind in ('difk', 'difa'):
            out += list(range(b + NB, b + 3 * NB + 4))
    return np.array(out)


def elbo(f):
    lv = np.array(LEVELS)
    ks = np.nonzero(lv > 0)[0]
    lg = np.log(lv[ks])
    w = np.gradient(lg)  # trapezoid-like weights in log sigma
    tot = f[..., ks * PER + PER - 1]
    return (tot / lv[ks] ** 2 * w).sum(-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=Path, required=True)
    ap.add_argument('--root', type=Path, default=Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3'))
    ap.add_argument('--suffix', default='')
    args = ap.parse_args()
    for fn in sorted(args.src.glob('*.npz')):
        with np.load(fn) as z:
            d = {k: z[k] for k in z.files}
        f = d['feats']
        for kind in ('dif', 'difk', 'difa', 'dife'):
            out = args.root / (kind + args.suffix)
            out.mkdir(parents=True, exist_ok=True)
            g = elbo(f) if kind == 'dife' else f[..., cols(kind)]
            np.savez(out / fn.name, **{**d, 'feats': g.astype(np.float32)})
        print(fn.name, f.shape, flush=True)


if __name__ == '__main__':
    main()
