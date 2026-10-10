"""Full-data refit of member families with a holdout3 variant (e.g. cosmology jitter) -> OUT/fam_<variant>/."""
import argparse
import pickle
import time

import numpy as np

import lib
from holdout3 import VARIANTS
from jlik import JLikelihood


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True)
    ap.add_argument('--fams', required=True)
    args = ap.parse_args()
    V = VARIANTS[args.variant]
    out = lib.OUT / f'fam_{args.variant}'
    out.mkdir(parents=True, exist_ok=True)
    for fam in args.fams.split(','):
        if (out / f'{fam}.npz').exists():
            continue
        t0 = time.time()
        lk = JLikelihood('cuda', 0.02, V['jm']).fit(lib.cols(fam, 'fit'), lib.labels('fit'), steps=V['steps'],
                                                    width=V['width'], log=lambda m: None, jitter=0.1)
        res = {}
        for s in lib.SETS + ['test']:
            x = lib.cols(fam, s)
            n, v, d = x.shape
            E, th = lk.profile(x.reshape(n * v, d))
            res[f'{s}/cop'] = lk.last_copula
            res[f'{s}/E'] = E
            res[f'{s}/th'] = ((th - lk.t_mean) / lk.t_std).astype(np.float32)
        with open(out / f'model_{fam}.pkl', 'wb') as fh:
            pickle.dump(lk, fh)
        np.savez(out / f'{fam}.npz', **res)
        print(f'[{args.variant} {fam}] {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
