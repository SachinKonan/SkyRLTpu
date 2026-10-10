"""Tiny synthetic dataset for tests/demos only; never use to make scientific claims."""
import argparse
from pathlib import Path
import numpy as np


def make_toy(out, n_cosmo=5, n_real=32, h=32, w=16, seed=123):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    mask = np.ones((h, w), dtype=bool)
    mask[:2] = False
    mask[:, :1] = False
    mask[-2:] = False
    arr = np.empty((n_cosmo, n_real, int(mask.sum())), dtype=np.float16)
    labels = np.empty((n_cosmo, n_real, 5), dtype=np.float64)
    xx = np.linspace(-1, 1, h)[:, None]
    yy = np.linspace(-1, 1, w)[None, :]
    for c in range(n_cosmo):
        theta = np.array([.2 + .1*c/n_cosmo, .72 + .1*c/n_cosmo])
        for r in range(n_real):
            field = .03 * (1 + .2*c) * np.sin(4*xx + 3*yy + r*.12)
            field += .01 * rng.normal(size=(h, w))
            arr[c,r] = field[mask].astype(np.float16)
            labels[c,r] = [*theta, 7.5, .01, 0.0]
    np.save(out / "WIDE12H_bin2_2arcmin_mask.npy", mask)
    np.save(out / "sampled_WIDE12H_bin2_2arcmin_kappa.npy", arr)
    np.save(out / "sampled_label.npy", labels)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--n-cosmo", type=int, default=5)
    p.add_argument("--n-real", type=int, default=32)
    args = p.parse_args()
    print(make_toy(args.out, args.n_cosmo, args.n_real))


if __name__ == "__main__":
    main()
