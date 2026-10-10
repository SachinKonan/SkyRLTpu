"""Leave-cosmology-out refits of the OE likelihood members (della-gh), resumable.

Same protocol as agents/consistency/holdout3.py (variant std, sets 0/1/2, 20 held-out
cosmologies each) but for OE feature sources, written to agents/outlier2/lco3/set{k}_std/.
The consistency agent's files are only read.
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'consistency'))
sys.path.insert(0, str(HERE.parents[1]))

import lib  # noqa: E402
from holdout3 import VARIANTS, masks, labels_for, score  # noqa: E402
from jlik import JLikelihood  # noqa: E402

OUT = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/outlier2/lco3')
SRC = {'oe': ('oe', None), 'oe_head': ('oe_head', None), 'oe_ho': ('oe_ho', None), 'oe_ho_head': ('oe_ho_head', None)}


def feats(src, name):
    return lib.load(src, name)


def sets_for(src, k):
    m = masks(k)
    f = {n: feats(src, n) for n in ('fit', 'calib', 'audit')}
    out = dict(fit=f['fit'][~m['fit']], calib=f['calib'][~m['calib']], **{'in': f['audit'][~m['audit']]},
               out=np.concatenate([f['calib'][m['calib']].reshape(-1, 1, f['calib'].shape[-1]),
                                   f['audit'][m['audit']]], 0))
    for s in lib.SHIFTS:
        out['sh_' + s] = feats(src, 'shift_' + s)[m['audit']]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', default='0,1,2')
    ap.add_argument('--srcs', default='oe,oe_head')
    ap.add_argument('--budget', type=float, default=420)
    args = ap.parse_args()
    t0 = time.time()
    V = VARIANTS['std']
    for k in [int(x) for x in args.sets.split(',')]:
        out = OUT / f'set{k}_std'
        out.mkdir(parents=True, exist_ok=True)
        for src in args.srcs.split(','):
            if (out / f'{src}.npz').exists():
                continue
            if time.time() - t0 > args.budget:
                print('budget reached', flush=True)
                return
            t1 = time.time()
            S = sets_for(src, k)
            lk = JLikelihood('cuda', 0.02, V['jm']).fit(S['fit'], labels_for(k), steps=V['steps'],
                                                        width=V['width'], log=lambda m: None, jitter=0.1)
            res = {}
            for name, x in S.items():
                if name == 'fit':
                    continue
                for key, val in score(lk, x).items():
                    res[f'{name}/{key}'] = val
            np.savez(out / f'{src}.npz', **res)
            print(f'[set{k} {src}] {time.time() - t1:.0f}s', flush=True)
    print('all done', flush=True)


if __name__ == '__main__':
    main()
