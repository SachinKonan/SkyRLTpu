"""Train the U-Net denoiser on fit3 maps (fresh shape noise every step); resumable.

Loss: MSE on observed pixels in units of the noise variance (1.0 = identity map).
Validation each epoch on 256 calib3 maps with fixed noise.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import data as D  # noqa: E402
from unet import UNet  # noqa: E402

SIG2 = D.NOISE_SIGMA ** 2


def load_gpu(train, idx, dev):
    x = torch.empty((len(idx), train.shape[-1]), dtype=torch.float16, device=dev)
    for lo in range(0, len(idx), 2048):
        sl = idx[lo:lo + 2048]
        x[lo:lo + len(sl)] = torch.as_tensor(np.asarray(train[sl[:, 0], sl[:, 1]]), device=dev)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/denoiser/unet'))
    ap.add_argument('--epochs', type=int, default=24)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--widths', type=int, nargs='+', default=[32, 64, 128, 256, 256])
    ap.add_argument('--time-limit', type=float, default=3300.0)
    ap.add_argument('--max-steps', type=int, default=0, help='smoke test: stop after this many steps')
    ap.add_argument('--nfit', type=int, default=0, help='smoke test: use only the first n fit maps')
    args = ap.parse_args()
    t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    mask = D.mask(); m = torch.as_tensor(mask, device=dev, dtype=torch.float32)
    sel = m > 0
    train = np.load(D.TRAIN, mmap_mode='r')
    idx = D.index('fit3')
    if args.nfit:
        idx = idx[:args.nfit]
    xfit = load_gpu(train, idx, dev)
    vidx = D.index('calib3')[::10][:256]
    xval = load_gpu(train, vidx, dev)
    print(f'loaded fit {len(idx)} val {len(vidx)} in {time.time()-t0:.0f}s', flush=True)

    torch.manual_seed(args.seed)
    net = UNet(tuple(args.widths)).to(dev).to(memory_format=torch.channels_last)
    print('params', sum(p.numel() for p in net.parameters()), flush=True)
    steps_per_epoch = len(idx) // args.batch
    total = args.epochs * steps_per_epoch
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.05)
    gen = torch.Generator(device=dev); gen.manual_seed(args.seed)
    ckpt = args.out / 'ckpt.pt'
    start = 0
    if ckpt.exists():
        st = torch.load(ckpt, map_location=dev, weights_only=False)
        net.load_state_dict(st['net']); opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched'])
        gen.set_state(st['gen'].cpu() if hasattr(st['gen'], 'cpu') else st['gen'])
        start = st['epoch'] + 1
        print(f'resumed at epoch {start}', flush=True)

    vgen = torch.Generator(device=dev)

    @torch.no_grad()
    def validate():
        net.eval()
        vgen.manual_seed(999)
        se, cv, n = 0.0, 0.0, 0
        for lo in range(0, len(xval), 32):
            full = torch.zeros((min(32, len(xval) - lo), 1424, 176), device=dev)
            full[:, sel] = xval[lo:lo + 32].float()
            x = D.add_noise(full, m, vgen)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                y, _ = net(x, m)
            se += float(((y - full) ** 2)[:, sel].sum())
            c = full[:, sel]
            cv += float(((c - c.mean(1, keepdim=True)) ** 2).sum())
            n += c.numel()
        net.train()
        return se / n / SIG2, 1 - se / cv

    ep_time = None
    for ep in range(start, args.epochs):
        te = time.time()
        if ep_time is not None and (time.time() - t0) + ep_time > args.time_limit:
            print(f'time budget reached before epoch {ep}; exiting for resume', flush=True)
            return
        perm = torch.randperm(len(idx), generator=gen, device=dev)
        net.train()
        tl, tn = 0.0, 0
        for k, lo in enumerate(range(0, steps_per_epoch * args.batch, args.batch)):
            rows = perm[lo:lo + args.batch]
            full = torch.zeros((len(rows), 1424, 176), device=dev)
            full[:, sel] = xfit[rows].float()
            x = D.add_noise(full, m, gen)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                y, _ = net(x, m)
            loss = ((y - full) ** 2)[:, sel].mean() / SIG2
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
            tl += loss.item(); tn += 1
            if k % 200 == 0:
                print(f'  ep {ep} step {k}/{steps_per_epoch} loss {tl/tn:.4f} {time.time()-t0:.0f}s', flush=True)
            if args.max_steps and k + 1 >= args.max_steps:
                break
        vm, r2 = validate()
        ep_time = time.time() - te
        print(f'epoch {ep} train {tl/tn:.4f} val mse/sig2 {vm:.4f} val R2 {r2:.4f} lr {sched.get_last_lr()[0]:.2e} '
              f'epoch {ep_time:.0f}s total {time.time()-t0:.0f}s', flush=True)
        st = dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), gen=gen.get_state(),
                  epoch=ep, widths=args.widths, val=vm, r2=r2)
        torch.save(st, args.out / 'ckpt.tmp'); (args.out / 'ckpt.tmp').replace(ckpt)
        torch.save(dict(net=net.state_dict(), widths=args.widths, epoch=ep, val=vm, r2=r2), args.out / f'net_ep{ep:02d}.pt')
        if args.max_steps:
            return
    print('done', flush=True)


if __name__ == '__main__':
    main()
