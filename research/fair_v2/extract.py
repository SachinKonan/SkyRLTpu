"""Compute frozen summary statistics for every split, shift and the public test.

Writes OUT/{edges.npz, <name>.npz}. Test maps are used only as per-map inference
input: features are computed one map at a time in fixed batches, no aggregate.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

import data as D
from stats import Stats


def batches(n, size):
    for lo in range(0, n, size):
        yield lo, min(n, lo + size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(D.ROOT / 'feats-v2'))
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--fit-views', type=int, default=2)
    ap.add_argument('--calib-views', type=int, default=2)
    ap.add_argument('--skip-test', action='store_true')
    ap.add_argument('--only', default='')
    ap.add_argument('--mmap', action='store_true')
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dev.type == 'cpu':
        torch.set_num_threads(int(__import__('os').environ.get('SLURM_CPUS_PER_TASK', '8')))
    t0 = time.time()
    train = D.load_train(ram=not args.mmap)
    lab = D.labels()
    mask = D.mask()
    mask_t = torch.as_tensor(mask, device=dev, dtype=torch.float32)
    print(f'loaded train {train.shape} in {time.time()-t0:.1f}s', flush=True)
    stats = Stats(mask, dev)
    gen = torch.Generator(device=dev)

    def clean(idx):
        x = torch.as_tensor(train[idx[:, 0], idx[:, 1]], device=dev).float()
        full = torch.zeros((len(idx), D.H if hasattr(D, 'H') else 1424, 176), device=dev)
        full[:, mask_t > 0] = x
        return full

    edge_file = out / 'edges.npz'
    if edge_file.exists():
        stats.edges = {k: torch.as_tensor(v, device=dev) for k, v in np.load(edge_file).items()}
    else:
        idx = D.index('fit')
        sel = idx[np.random.default_rng(0).choice(len(idx), 512, replace=False)]
        gen.manual_seed(12345)
        maps = torch.cat([D.add_noise(clean(sel[lo:hi]), mask_t, gen) for lo, hi in batches(len(sel), 64)])
        edges = stats.fit_edges(maps)
        np.savez(edge_file, **edges)
        del maps
        print('edges frozen', flush=True)

    def run(name, idx, views, shift=None, seed=0):
        path = out / f'{name}.npz'
        if path.exists() or (args.only and name not in args.only.split(',')):
            return
        t = time.time(); feats = []
        for v in range(views):
            gen.manual_seed(seed * 1000 + v)
            part = []
            for lo, hi in batches(len(idx), args.batch):
                c = clean(idx[lo:hi])
                x = D.noisy_shift(shift, c, mask_t, gen) if shift else D.add_noise(c, mask_t, gen)
                part.append(stats(x).cpu().numpy())
            feats.append(np.concatenate(part))
        f = np.stack(feats, 1).astype(np.float32)   # (N, views, d)
        np.savez(path, feats=f, index=idx, labels=lab[idx[:, 0], idx[:, 1]])
        print(f'{name}: {f.shape} {time.time()-t:.1f}s finite={np.isfinite(f).all()}', flush=True)

    run('fit', D.index('fit'), args.fit_views, seed=1)
    run('calib', D.index('calib'), args.calib_views, seed=2)
    run('audit', D.index('audit'), 1, seed=3)
    for i, name in enumerate(D.SHIFTS):
        run('shift_' + name, D.index('audit'), 1, shift=name, seed=100 + i)
    if not args.skip_test and not (out / 'test.npz').exists() and (not args.only or 'test' in args.only.split(',')):
        t = time.time()
        test = np.load(D.TEST, mmap_mode='r')
        part = []
        for lo, hi in batches(len(test), args.batch):
            x = torch.as_tensor(np.asarray(test[lo:hi]), device=dev).float()
            full = torch.zeros((hi - lo, 1424, 176), device=dev)
            full[:, mask_t > 0] = x
            part.append(stats(full).cpu().numpy())
        f = np.concatenate(part).astype(np.float32)
        np.savez(out / 'test.npz', feats=f[:, None])
        print(f'test: {f.shape} {time.time()-t:.1f}s finite={np.isfinite(f).all()}', flush=True)


if __name__ == '__main__':
    main()
