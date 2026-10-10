"""Resumable extraction of the new statistic families on the SAME noisy maps as v3 `stats`/`cnn0`.

Noise scheme copied from pipeline_v3.embed.run: per (set, view) gen.manual_seed(seed*1000+v), maps in
index order, batches of 64, D.add_noise / D.noisy_shift per batch. Work is checkpointed per chunk of
CHUNK batches (generator + global CUDA RNG state saved, the latter for the Chi2 draws of noise_t5).
The first batch of every chunk is verified against the saved v3 `stats` features (same noisy map =>
same stats). Test: batches of 64 zero-padded to fixed shape.

  python extract_fam.py --budget 480            # run until ~budget seconds, then exit; rerun to resume
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import data as D  # noqa: E402
from stats import Stats  # noqa: E402
from famfeats import Fam, NoiseFam, FAMILIES  # noqa: E402

V3 = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3')
WORK = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/statistics/work')
EDGES = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/statistics/edges_fam.npz')
STATS_EDGES = '/scratch/gpfs/ZHUANGL/sk7524/fair-data/feats-v2/edges.npz'
BATCH = 64
CHUNK = 40
DEFAULT_SEED = 67280421310721      # torch default CUDA generator seed (fresh process)


def plan():
    sets = [('fit', 'fit3', 2, 31, None), ('calib', 'calib3', 2, 32, None), ('audit', 'audit3', 1, 33, None)]
    sets += [('shift_' + s, 'audit3', 1, 300 + i, s) for i, s in enumerate(D.SHIFTS)]
    return sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=float, default=480)
    ap.add_argument('--prefix', default='stx_')
    ap.add_argument('--only', default='')
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--kind', default='main', choices=['main', 'nois'])
    args = ap.parse_args()
    global WORK, EDGES
    if args.kind == 'nois':
        fams = ('nois', 'tail')
        WORK = WORK / 'nois'
        EDGES = EDGES.with_name('edges_nois.npz')
    else:
        fams = FAMILIES
    t_start = time.time()
    dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    init_state = torch.cuda.get_rng_state()
    mask = D.mask(); m = torch.as_tensor(mask, device=dev, dtype=torch.float32)
    lab = D.labels()
    train = np.load(D.TRAIN, mmap_mode='r')
    WORK.mkdir(parents=True, exist_ok=True)
    Cls = NoiseFam if args.kind == 'nois' else Fam
    if not EDGES.exists():
        fam = Cls(mask, dev)
        idx = D.index('fit3')
        rng = np.random.default_rng(999)
        sel = np.sort(rng.choice(len(idx), 768, replace=False))
        g = torch.Generator(device=dev); g.manual_seed(999)
        maps = []
        for lo in range(0, len(sel), BATCH):
            sl = idx[sel[lo:lo + BATCH]]
            full = torch.zeros((len(sl), 1424, 176), device=dev)
            full[:, m > 0] = torch.as_tensor(np.asarray(train[sl[:, 0], sl[:, 1]]), device=dev).float()
            maps.append(D.add_noise(full, m, g))
        e = fam.fit_nedges(torch.cat(maps)) if args.kind == 'nois' else fam.fit_edges(torch.cat(maps))
        np.savez(EDGES, **e)
        print('edges', {k: np.round(v, 4).tolist() for k, v in e.items()}, flush=True)
        del maps
    fam = Cls(mask, dev, dict(np.load(EDGES)))
    stats = Stats(mask, dev, edges=dict(np.load(STATS_EDGES)))
    gen = torch.Generator(device=dev)

    def check(x, name, v, rows):
        ref = np.load(V3 / 'stats' / f'{name}.npz')['feats'][rows, v]
        got = stats(x).cpu().numpy()[:len(rows)]
        err = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-9)
        if name == 'shift_noise_t5':
            # its Chi2 draws use the global CUDA RNG whose state in the original v3 run is unknown:
            # Gaussian part identical, t-scaling not reproducible -> not bit-identical (documented)
            return err
        if not np.isfinite(err) or err > 1e-3:
            raise RuntimeError(f'noise mismatch {name} v{v} rows {rows[0]}: rel err {err:.2e}')
        return err

    def out_done(name):
        return all((V3 / (args.prefix + f) / f'{name}.npz').exists() for f in fams)

    def save(name, feats, idx=None):
        extra = {} if idx is None else dict(index=idx, labels=lab[idx[:, 0], idx[:, 1]])
        for f in fams:
            d = V3 / (args.prefix + f); d.mkdir(parents=True, exist_ok=True)
            np.savez(d / f'{name}.npz', feats=feats[f].astype(np.float32), **extra)

    sets = plan()
    if args.only:
        sets = [s for s in sets if s[0] in args.only.split(',')]
    for name, part, views, seed, shift in sets:
        if out_done(name):
            continue
        if not (V3 / 'stats' / f'{name}.npz').exists():
            print(f'skip {name}: no stats reference yet', flush=True)
            continue
        idx = D.index(part)
        nb = (len(idx) + BATCH - 1) // BATCH
        nchunk = (nb + CHUNK - 1) // CHUNK
        for v in range(views):
            for c in range(nchunk):
                ck = WORK / f'{name}_v{v}_c{c:03d}.npz'
                if ck.exists():
                    continue
                if time.time() - t_start > args.budget:
                    print(f'budget reached at {name} v{v} c{c}', flush=True)
                    return
                t = time.time()
                if c == 0:
                    gen.manual_seed(seed * 1000 + v)
                    torch.cuda.set_rng_state(init_state)
                else:
                    prev = np.load(WORK / f'{name}_v{v}_c{c - 1:03d}.npz')
                    gen.set_state(torch.as_tensor(prev['gen_state']))
                    torch.cuda.set_rng_state(torch.as_tensor(prev['cuda_state']))
                acc = {f: [] for f in fams}
                err = None
                for bi in range(c * CHUNK, min(nb, (c + 1) * CHUNK)):
                    lo = bi * BATCH
                    sl = idx[lo:lo + BATCH]
                    full = torch.zeros((len(sl), 1424, 176), device=dev)
                    full[:, m > 0] = torch.as_tensor(np.asarray(train[sl[:, 0], sl[:, 1]]), device=dev).float()
                    x = (D.noisy_shift(shift, full, m, gen, lab[sl[:, 0], sl[:, 1]]) if shift
                         else D.add_noise(full, m, gen))
                    if err is None:
                        err = check(x, name, v, np.arange(lo, lo + len(sl)))
                    o = fam(x, fams)
                    for f in fams:
                        acc[f].append(o[f].cpu().numpy())
                np.savez(ck, gen_state=gen.get_state().cpu().numpy(), cuda_state=torch.cuda.get_rng_state().cpu().numpy(),
                         verify_err=err, **{f: np.concatenate(acc[f]) for f in fams})
                nf = {f: int((~np.isfinite(np.concatenate(acc[f]))).sum()) for f in fams}
                print(f'{name} v{v} chunk {c + 1}/{nchunk} {time.time() - t:.1f}s verify {err:.1e} nonfinite {nf}', flush=True)
        feats = {}
        for f in fams:
            per_view = []
            for v in range(views):
                per_view.append(np.concatenate([np.load(WORK / f'{name}_v{v}_c{c:03d}.npz')[f] for c in range(nchunk)]))
            feats[f] = np.stack(per_view, 1)
            bad = ~np.isfinite(feats[f])
            if bad.any():
                print(f'WARNING {name} {f}: {bad.sum()} non-finite values', flush=True)
        save(name, feats, idx)
        print(f'saved {name} {[feats[f].shape for f in fams]}', flush=True)
    if args.test and not out_done('test'):
        test = np.load(D.TEST, mmap_mode='r')
        nb = (len(test) + BATCH - 1) // BATCH
        nchunk = (nb + CHUNK - 1) // CHUNK
        for c in range(nchunk):
            ck = WORK / f'test_c{c:03d}.npz'
            if ck.exists():
                continue
            if time.time() - t_start > args.budget:
                print(f'budget reached at test c{c}', flush=True)
                return
            t = time.time()
            acc = {f: [] for f in fams}
            err = None
            for bi in range(c * CHUNK, min(nb, (c + 1) * CHUNK)):
                lo = bi * BATCH
                n = min(BATCH, len(test) - lo)
                full = torch.zeros((BATCH, 1424, 176), device=dev)
                full[:n, m > 0] = torch.as_tensor(np.asarray(test[lo:lo + n]), device=dev).float()
                if err is None:
                    err = check(full, 'test', 0, np.arange(lo, lo + n))
                o = fam(full, fams)
                for f in fams:
                    acc[f].append(o[f][:n].cpu().numpy())
            np.savez(ck, verify_err=err, **{f: np.concatenate(acc[f]) for f in fams})
            print(f'test chunk {c + 1}/{nchunk} {time.time() - t:.1f}s verify {err:.1e}', flush=True)
        feats = {f: np.concatenate([np.load(WORK / f'test_c{c:03d}.npz')[f] for c in range(nchunk)])[:, None]
                 for f in fams}
        save('test', feats)
        print('saved test', flush=True)
    print(f'done in {time.time() - t_start:.0f}s', flush=True)


if __name__ == '__main__':
    main()
