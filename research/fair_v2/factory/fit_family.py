"""Full-data family likelihood exactly as fit_families.py (families mode), but resumable per set.

  python factory/fit_family.py dirpower [--budget 450]

Fits model.Likelihood on fit3 (steps 3000, width 256, jitter 0.1, eig floor 0.02), pickles it, then scores every
set of lib.SETS + test with fit_families.run_model, saving after each set. Output is the same
consistency/fam/<family>.npz + model_<family>.pkl that analyze.fam() reads. Test is only scored.
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'agents' / 'consistency'))
import lib  # noqa: E402
from fit_families import ALLSETS, run_model  # noqa: E402
from model import Likelihood  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('family')
    ap.add_argument('--budget', type=float, default=450)
    args = ap.parse_args()
    fam = args.family
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False          # as fit_families.main
    out = lib.OUT / 'fam'
    final, part, mp = out / f'{fam}.npz', out / f'{fam}.partial.npz', out / f'model_{fam}.pkl'
    if final.exists():
        print('already done', flush=True)
        return
    if mp.exists():
        with open(mp, 'rb') as fh:
            lk = pickle.load(fh)
    else:
        lk = Likelihood('cuda', 0.02).fit(lib.cols(fam, 'fit'), lib.labels('fit'), steps=3000, width=256,
                                          log=lambda m: print(f'[{fam}] {m}', flush=True), jitter=0.1)
        with open(mp.with_suffix('.tmp'), 'wb') as fh:
            pickle.dump(lk, fh)
        mp.with_suffix('.tmp').replace(mp)
        print(f'fitted in {time.time()-t0:.0f}s', flush=True)
    res = {}
    if part.exists():
        with np.load(part) as d:
            res = {k: d[k] for k in d.files}
    for s in ALLSETS:
        if f'{s}/E' in res:
            continue
        if time.time() - t0 > args.budget:
            print('budget reached', flush=True)
            return
        t = time.time()
        f = lib.cols(fam, s)
        n, v, d = f.shape
        for k, val in run_model(lk, f.reshape(n * v, d)).items():
            res[f'{s}/{k}'] = val
        np.savez(out / f'{fam}.partial.tmp.npz', **res)
        (out / f'{fam}.partial.tmp.npz').replace(part)
        print(f'{s} {n}x{v} {time.time()-t:.0f}s', flush=True)
    np.savez(out / f'{fam}.tmp.npz', **res)
    (out / f'{fam}.tmp.npz').replace(final)
    part.unlink()
    print(f'done {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
