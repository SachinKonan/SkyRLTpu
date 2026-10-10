"""Train the noise-conditional denoiser on fit3 normal maps (resumable).

y = kappa + s * eps on observed pixels, s = sqrt(sigma_n^2 + sigma_t^2) with
sigma_t = 0 (prob p0) or log-uniform in [tmin, tmax]. Clean-kappa targets give
the same optimum as noisy-x targets (Gaussian convolution identity) with far
lower gradient variance. EDM loss weighting, EMA weights for evaluation.
"""
import argparse
import copy
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import data as D  # noqa: E402
from unet import UNet, denoise, SIGMA_DATA  # noqa: E402


def load_fit_gpu(train, idx, dev, limit=None):
    if limit:
        idx = idx[:limit]
    x = torch.empty((len(idx), train.shape[-1]), dtype=torch.float16, device=dev)
    for lo in range(0, len(idx), 2048):
        sl = idx[lo:lo + 2048]
        x[lo:lo + len(sl)] = torch.as_tensor(np.asarray(train[sl[:, 0], sl[:, 1]]), device=dev)
    return x


def save(path, **kw):
    tmp = path.with_suffix('.tmp')
    torch.save(kw, tmp)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/diffusion/run1'))
    ap.add_argument('--budget', type=float, default=50.0, help='minutes of training in this process')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=4e-4)
    ap.add_argument('--ema', type=float, default=0.999)
    ap.add_argument('--tmin', type=float, default=0.004)
    ap.add_argument('--tmax', type=float, default=1.5)
    ap.add_argument('--p0', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0, help='debug: only this many fit maps')
    ap.add_argument('--compile', action='store_true')
    ap.add_argument('--ch', type=int, nargs='+', default=[96, 160, 256])
    ap.add_argument('--nres', type=int, nargs='+', default=[2, 2, 2])
    ap.add_argument('--ps', type=int, default=4)
    ap.add_argument('--cond', action='store_true', help='condition on (jittered) standardized theta + flag')
    ap.add_argument('--p-drop', type=float, default=0.25)
    ap.add_argument('--init', default='', help='initialize (EMA weights, strict=False) from this checkpoint')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    dev = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    m = torch.as_tensor(D.mask(), device=dev, dtype=torch.float32)
    train = np.load(D.TRAIN, mmap_mode='r')
    idx = D.index('fit3')
    xfit = load_fit_gpu(train, idx, dev, args.limit or None)
    N = len(xfit)
    lab = D.labels()[idx[:, 0], idx[:, 1]]
    if args.limit:
        lab = lab[:args.limit]
    th = torch.as_tensor((lab - lab.mean(0)) / lab.std(0), dtype=torch.float32, device=dev)
    # per-parameter jitter = CNN (cnn0) prediction RMS error on calib3, so training matches the
    # estimated-theta conditioning used at evaluation
    jit = torch.tensor([0.26, 0.37, 0.75, 0.49, 0.95], device=dev)
    print(f'fit maps {N} loaded in {time.time()-t0:.0f}s', flush=True)
    torch.manual_seed(args.seed)
    cfg = dict(ch=tuple(args.ch), nres=tuple(args.nres), ps=args.ps, cond_dim=6 if args.cond else 0)
    net = UNet(**cfg).to(dev)
    ema = copy.deepcopy(net).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0)
    ck = args.out / 'ckpt.pt'
    step, epoch = 0, 0
    if ck.exists():
        st = torch.load(ck, map_location=dev, weights_only=False)
        net.load_state_dict(st['net']); ema.load_state_dict(st['ema']); opt.load_state_dict(st['opt'])
        step, epoch = st['step'], st['epoch']
        for g in opt.param_groups:
            g['lr'] = args.lr
        print(f'resumed step {step} epoch {epoch}', flush=True)
    elif args.init:
        st = torch.load(args.init, map_location=dev, weights_only=False)
        print('init from', args.init, net.load_state_dict(st['ema'], strict=False), flush=True)
        ema.load_state_dict(net.state_dict())
    print(f'params {sum(p.numel() for p in net.parameters())/1e6:.2f}M', flush=True)
    fwd = torch.compile(net) if args.compile else net
    sn = D.NOISE_SIGMA
    gen = torch.Generator(device=dev)
    lt0, lt1 = math.log(args.tmin), math.log(args.tmax)
    nb = N // args.batch
    tstart = time.time()
    stop = False
    while not stop:
        gen.manual_seed(args.seed * 100000 + epoch)
        perm = torch.randperm(N, generator=gen, device=dev)
        net.train()
        tl, tn = 0.0, 0
        bins = torch.zeros(6, device=dev); bcnt = torch.zeros(6, device=dev)
        te = time.time()
        for b in range(nb):
            rows = perm[b * args.batch:(b + 1) * args.batch]
            B = len(rows)
            kappa = torch.zeros((B, 1424, 176), device=dev)
            kappa[:, m > 0] = xfit[rows].float()
            u = torch.rand(B, generator=gen, device=dev)
            st_ = torch.exp(lt0 + u * (lt1 - lt0))
            st_ = torch.where(torch.rand(B, generator=gen, device=dev) < args.p0, torch.zeros_like(st_), st_)
            s = torch.sqrt(sn * sn + st_ * st_)
            y = (kappa + s[:, None, None] * torch.randn(kappa.shape, generator=gen, device=dev)) * m
            # EDM: lambda(s) = (s^2 + sd^2) / (s sd)^2
            lam = (s * s + SIGMA_DATA ** 2) / (s * SIGMA_DATA) ** 2
            cond = None
            if args.cond:
                tc = th[rows] + jit * torch.randn((B, 5), generator=gen, device=dev)
                keep = (torch.rand(B, generator=gen, device=dev) >= args.p_drop).float()[:, None]
                cond = torch.cat([tc * keep, keep], 1)
            Dk = denoise(fwd, y, s, m, cond=cond)
            per = (((Dk - kappa) ** 2) * m).sum((1, 2)) / m.sum() * lam
            loss = per.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            warm = min(1.0, (step + 1) / 500)
            for g in opt.param_groups:
                g['lr'] = args.lr * warm
            opt.step()
            step += 1
            with torch.no_grad():
                d = min(args.ema, (1 + step) / (10 + step))
                for pe, pn in zip(ema.parameters(), net.parameters()):
                    pe.lerp_(pn, 1 - d)
                for be, bn in zip(ema.buffers(), net.buffers()):
                    be.copy_(bn)
                k = ((torch.log(s) - math.log(sn)) / (math.log(1.6) - math.log(sn)) * 6).long().clamp(0, 5)
                bins.index_add_(0, k, per.detach()); bcnt.index_add_(0, k, torch.ones_like(per))
            tl += loss.item() * B if b % 50 == 0 else 0.0
            tn += B if b % 50 == 0 else 0
            if b == 20 and epoch == 0:
                torch.cuda.synchronize()
                print(f'  20 steps: {(time.time()-te)/21*1000:.0f} ms/step', flush=True)
            if (time.time() - tstart) / 60 > args.budget:
                stop = True
                break
        if not stop:
            epoch += 1
        save(ck, net=net.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), step=step, epoch=epoch, cfg=cfg,
             args=vars(args))
        bl = (bins / bcnt.clamp(min=1)).tolist()
        print(f'epoch {epoch} step {step} loss {tl/max(tn,1):.4f} by-s-bin ' + ' '.join(f'{v:.3f}' for v in bl)
              + f' | {time.time()-te:.0f}s, total {time.time()-t0:.0f}s', flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
