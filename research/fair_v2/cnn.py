"""Supervised CNN on full-resolution noisy maps -> learned summaries.

Trained ONLY on normal fit-split maps (fresh shape noise every step) to predict
all five physical parameters. Its pooled embedding is then treated as another
summary vector by the conditional-Gaussian profile detector.
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import data as D


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.skip = None if (cin == cout and stride == 1) else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        h = F.silu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return F.silu(h + (x if self.skip is None else self.skip(x)))


class Net(nn.Module):
    def __init__(self, widths=(32, 64, 128, 192, 256, 256), emb=256):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(2, widths[0], 3, 1, 1, bias=False), nn.BatchNorm2d(widths[0]), nn.SiLU())
        blocks, cin = [], widths[0]
        for i, w in enumerate(widths):
            blocks += [Block(cin, w, 1 if i == 0 else 2), Block(w, w, 1)]
            cin = w
        self.blocks = nn.Sequential(*blocks)
        self.emb = nn.Sequential(nn.Linear(2 * cin, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.head = nn.Sequential(nn.SiLU(), nn.Linear(emb, 5))

    def embed(self, x, m):
        h = self.blocks(self.stem(torch.stack([x / 0.03, m.expand_as(x)], 1)))
        mm = F.adaptive_avg_pool2d(m[None, None], h.shape[-2:])[0, 0]
        w = mm / mm.sum()
        mean = (h * w).sum((2, 3))
        sd = (((h - mean[:, :, None, None]) ** 2) * w).sum((2, 3)).clamp(min=1e-8).sqrt()
        return self.emb(torch.cat([mean, sd], 1))

    def forward(self, x, m):
        e = self.embed(x, m)
        return self.head(e), e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/scratch/gpfs/ZHUANGL/sk7524/fair-data/runs-v2/cnn')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--embed-only', action='store_true')
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda')
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    mask = D.mask(); m = torch.as_tensor(mask, device=dev, dtype=torch.float32)
    lab = D.labels()
    train = np.load(D.TRAIN, mmap_mode='r')
    idx_fit = D.index('fit')
    t0 = time.time()
    # Fit split resident on GPU as float16 valid-pixel vectors.
    xfit = torch.empty((len(idx_fit), int(mask.sum())), dtype=torch.float16, device=dev)
    for lo in range(0, len(idx_fit), 2048):
        sl = idx_fit[lo:lo + 2048]
        xfit[lo:lo + len(sl)] = torch.as_tensor(np.asarray(train[sl[:, 0], sl[:, 1]]), device=dev)
    theta = lab[idx_fit[:, 0], idx_fit[:, 1]]
    t_mean, t_std = theta.mean(0), theta.std(0)
    tfit = torch.as_tensor((theta - t_mean) / t_std, dtype=torch.float32, device=dev)
    print(f'loaded fit split {tuple(xfit.shape)} in {time.time()-t0:.0f}s', flush=True)
    net = Net().to(dev).to(memory_format=torch.channels_last)
    gen = torch.Generator(device=dev); gen.manual_seed(args.seed)

    def batch_maps(rows, noise=True):
        full = torch.zeros((len(rows), 1424, 176), device=dev)
        full[:, m > 0] = xfit[rows].float()
        return D.add_noise(full, m, gen) if noise else full

    ckpt = out / 'net.pt'
    if not args.embed_only:
        steps = args.epochs * (len(idx_fit) // args.batch)
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
        step = 0
        for ep in range(args.epochs):
            perm = torch.randperm(len(idx_fit), generator=gen, device=dev)
            tl, tn = 0.0, 0
            net.train()
            for lo in range(0, len(perm) - args.batch + 1, args.batch):
                rows = perm[lo:lo + args.batch]
                x = batch_maps(rows)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    pred, _ = net(x, m)
                loss = F.mse_loss(pred.float(), tfit[rows])
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step(); sched.step(); step += 1
                tl += loss.item() * len(rows); tn += len(rows)
            print(f'epoch {ep} train mse {tl/tn:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s', flush=True)
            torch.save(dict(net=net.state_dict(), t_mean=t_mean, t_std=t_std), ckpt)
    state = torch.load(ckpt, map_location=dev, weights_only=False)
    net.load_state_dict(state['net']); net.eval()
    del xfit; torch.cuda.empty_cache()
    embed_all(net, m, mask, out, dev)


@torch.no_grad()
def embed_all(net, m, mask, out, dev, views=2):
    """Embeddings for fit/calib/audit (fresh noise), shifts, and the public test."""
    train = np.load(D.TRAIN, mmap_mode='r')
    lab = D.labels()
    gen = torch.Generator(device=dev)

    def run(idx, nviews, seed, shift=None):
        res = []
        for v in range(nviews):
            gen.manual_seed(seed * 1000 + v)
            part = []
            for lo in range(0, len(idx), 128):
                sl = idx[lo:lo + 128]
                full = torch.zeros((len(sl), 1424, 176), device=dev)
                full[:, m > 0] = torch.as_tensor(np.asarray(train[sl[:, 0], sl[:, 1]]), device=dev).float()
                x = D.noisy_shift(shift, full, m, gen) if shift else D.add_noise(full, m, gen)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    pred, e = net(x, m)
                part.append(torch.cat([e.float(), pred.float()], 1).cpu().numpy())
            res.append(np.concatenate(part))
        return np.stack(res, 1).astype(np.float32)

    for name, part, nv, seed in (('fit', 'fit', views, 11), ('calib', 'calib', views, 12), ('audit', 'audit', 1, 13)):
        idx = D.index(part)
        np.savez(out / f'{name}.npz', feats=run(idx, nv, seed), index=idx, labels=lab[idx[:, 0], idx[:, 1]])
        print('embedded', name, flush=True)
    idx = D.index('audit')
    for i, s in enumerate(D.SHIFTS):
        np.savez(out / f'shift_{s}.npz', feats=run(idx, 1, 200 + i, s), index=idx, labels=lab[idx[:, 0], idx[:, 1]])
    test = np.load(D.TEST, mmap_mode='r')
    part = []
    for lo in range(0, len(test), 128):
        full = torch.zeros((min(128, len(test) - lo), 1424, 176), device=dev)
        full[:, m > 0] = torch.as_tensor(np.asarray(test[lo:lo + 128]), device=dev).float()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred, e = net(full, m)
        part.append(torch.cat([e.float(), pred.float()], 1).cpu().numpy())
    np.savez(out / 'test.npz', feats=np.concatenate(part)[:, None].astype(np.float32))
    print('embedded test', flush=True)


if __name__ == '__main__':
    main()
