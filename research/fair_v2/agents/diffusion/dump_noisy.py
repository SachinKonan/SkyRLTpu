"""Dump the exact noisy maps of pipeline_v3 (fit3 x2, calib3 x2, audit3, shifts) as float16.

Copies pipeline_v3.embed.run's noise loop (batches of 64, gen.manual_seed(seed*1000+v))
and must run on della-gh (GH200) so the CUDA Philox draws are bit-identical to the
stats/cnn0 sources. add_noise/noisy_shift already quantize to float16, so storing
float16 is lossless. Output: OUT/<name>_v<v>.npy (N, 132019) float16, index order.
--verify recomputes stats for the first audit batch and compares with v3/stats.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import data as D  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/diffusion/noisy'))
    ap.add_argument('--budget', type=float, default=7.0)
    ap.add_argument('--verify', action='store_true')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    dev = torch.device('cuda')
    mask = D.mask()
    m = torch.as_tensor(mask, device=dev, dtype=torch.float32)
    lab = D.labels()
    train = np.load(D.TRAIN, mmap_mode='r')
    gen = torch.Generator(device=dev)
    jobs = [('audit', 'audit3', 1, 33, None), ('calib', 'calib3', 2, 32, None)]
    jobs += [('shift_' + s, 'audit3', 1, 300 + i, s) for i, s in enumerate(D.SHIFTS)]
    jobs += [('fit', 'fit3', 2, 31, None)]
    for name, part, views, seed, shift in jobs:
        idx = D.index(part)
        for v in range(views):
            fn = args.out / f'{name}_v{v}.npy'
            if fn.exists():
                continue
            if (time.time() - t0) / 60 > args.budget:
                print('budget reached', flush=True)
                return
            t = time.time()
            tmp = args.out / f'{name}_v{v}.tmp.npy'
            mm = open_memmap(tmp, mode='w+', dtype=np.float16, shape=(len(idx), int(mask.sum())))
            gen.manual_seed(seed * 1000 + v)
            for lo in range(0, len(idx), 64):
                sl = idx[lo:lo + 64]
                full = torch.zeros((len(sl), 1424, 176), device=dev)
                full[:, m > 0] = torch.as_tensor(np.asarray(train[sl[:, 0], sl[:, 1]]), device=dev).float()
                x = (D.noisy_shift(shift, full, m, gen, lab[sl[:, 0], sl[:, 1]]) if shift
                     else D.add_noise(full, m, gen))
                mm[lo:lo + len(sl)] = x[:, m > 0].half().cpu().numpy()
            mm.flush(); del mm
            tmp.replace(fn)
            print(f'{name} v{v}: {len(idx)} maps in {time.time()-t:.0f}s', flush=True)
    if args.verify:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from stats import Stats
        st = Stats(mask, dev, edges=dict(np.load('/scratch/gpfs/ZHUANGL/sk7524/fair-data/feats-v2/edges.npz')))
        x = np.load(args.out / 'audit_v0.npy', mmap_mode='r')[:64]
        full = torch.zeros((64, 1424, 176), device=dev)
        full[:, m > 0] = torch.as_tensor(np.asarray(x), device=dev).float()
        mine = st(full).cpu().numpy()
        ref = np.load('/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3/stats/audit.npz')['feats'][:64, 0]
        print('verify audit stats max abs diff', float(np.abs(mine - ref).max()), 'max rel',
              float((np.abs(mine - ref) / (np.abs(ref) + 1e-6)).max()), flush=True)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
