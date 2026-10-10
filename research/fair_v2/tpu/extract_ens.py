"""Ensemble-denoiser residual features (den_r layout, 112 dims) on the EXACT dumped noisy maps.

y = mean_k D_k(x) over several U-Net denoisers (TPU-trained, converted to torch); features =
ResidualFeats(x, y) (80) + SURE block with the first model's divergence (6) + tile block (26),
identical definitions to agents/denoiser/extract_den2.py. Inputs come from the noisy-map dumps
(agents/diffusion/noisy, bit-identical to pipeline_v3) so any GPU can run any set.
Writes <root>/<name>/<set>.npz with feats (N, V, 112), index, labels; test from the test file.
"""
import argparse
import fnmatch
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / 'agents' / 'denoiser'))
import data as D  # noqa: E402
from unet import UNet  # noqa: E402
from extract_den2 import ResidualFeats  # noqa: E402

NOISY = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/diffusion/noisy')
BATCH = 64
VIEWS = {'fit': ('fit3', 2), 'calib': ('calib3', 2), 'audit': ('audit3', 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs='+', required=True)
    ap.add_argument('--root', type=Path, default=Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3'))
    ap.add_argument('--name', required=True)
    ap.add_argument('--only', default='')
    ap.add_argument('--budget', type=float, default=1e9)
    args = ap.parse_args()
    t0 = time.time()
    dev = torch.device('cuda')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    mask = D.mask(); m = torch.as_tensor(mask, device=dev, dtype=torch.float32)
    sel = m > 0
    lab = D.labels()
    nets = []
    for ck in args.ckpts:
        st = torch.load(ck, map_location=dev, weights_only=False)
        net = UNet(tuple(st['widths'])).to(dev).to(memory_format=torch.channels_last)
        net.load_state_dict(st['net']); net.eval()
        nets.append(net)
    out = args.root / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / 'ckpts.txt').write_text('\n'.join(args.ckpts) + '\n')
    rf = ResidualFeats(m, dev)
    sig = D.NOISE_SIGMA
    pg = torch.Generator(); pg.manual_seed(12345)
    probe = ((torch.randint(0, 2, (1424, 176), generator=pg) * 2 - 1).float().to(dev)) * m
    ysd = float(np.load(args.root / 'den' / 'edges_denoised.npz')['sd0'][0])   # same frozen scale as den_r
    ybins = torch.as_tensor([-1.0, 1.0, 2.0], device=dev) * ysd

    @torch.no_grad()
    def den(net, x):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            y, _ = net(x, m)
        return y.float() * m

    @torch.no_grad()
    def sure(x, y, y1):
        y2 = den(nets[0], x + sig * probe)
        g = probe * (y2 - y1) / sig
        gv = g[:, sel]; yv = y[:, sel]; rv = (x - y)[:, sel]
        bidx = torch.bucketize(yv, ybins)
        cnt = torch.zeros((len(x), 4), device=dev).scatter_add_(1, bidx, torch.ones_like(yv)).clamp(min=1)
        dbin = torch.zeros((len(x), 4), device=dev).scatter_add_(1, bidx, gv) / cnt
        r2 = (rv * rv).mean(1) / sig ** 2
        return torch.cat([torch.stack([r2, gv.mean(1)], 1), dbin], 1)

    tiles = []
    for t in (8, 16):
        cov = torch.nn.functional.avg_pool2d(m[None, None], t)[0, 0]
        vt = cov >= 0.5
        th, tw = cov.shape
        fy = torch.fft.fftfreq(th, device=dev)[:, None]
        fx = torch.fft.rfftfreq(tw, device=dev)[None, :]
        rr = torch.sqrt(fy ** 2 + fx ** 2)
        bands = [(rr > lo) & (rr <= hi) for lo, hi in ((0.0, 0.03), (0.03, 0.08), (0.08, 0.2))]
        tiles.append((t, cov, vt, bands))
    tq = torch.as_tensor([0.05, 0.5, 0.95], device=dev)
    tails = torch.as_tensor([2.5, 3.0, 4.0], device=dev)

    @torch.no_grad()
    def tile_feats(x, y):
        r = (x - y) * m
        res = []
        for t, cov, vt, bands in tiles:
            pool = lambda a: torch.nn.functional.avg_pool2d(a[:, None], t)[:, 0] / cov.clamp(min=1e-6)
            v = pool(r * r) / sig ** 2
            ym = pool(y * m)
            ys = (pool(y * y * m) - ym ** 2).clamp(min=0).sqrt()
            vv = v[:, vt]; ymv = ym[:, vt]; ysv = ys[:, vt]
            mu = vv.mean(1, keepdim=True)
            vc = vv - mu
            sd = vc.pow(2).mean(1).sqrt()

            def corr(a):
                ac = a - a.mean(1, keepdim=True)
                return (vc * ac).mean(1) / (sd * ac.pow(2).mean(1).sqrt() + 1e-12)
            q = torch.quantile(vv, tq, dim=1).T
            vm = torch.where(vt, v / mu[:, :, None] - 1, torch.zeros_like(v))
            f = torch.fft.rfft2(vm)
            p = f.real ** 2 + f.imag ** 2
            lowk = torch.stack([p[:, bd].mean(1) for bd in bands], 1) / float(vt.sum())
            res += [torch.log(mu), (sd / mu[:, 0])[:, None], torch.log(q), torch.stack([corr(ymv), corr(ysv)], 1),
                    torch.log(lowk + 1e-12)]
        z = r[:, sel] / sig
        res += [torch.log1p((z[:, :, None] > tails).float().sum(1)), torch.log1p((z[:, :, None] < -tails).float().sum(1))]
        return torch.cat(res, 1)

    @torch.no_grad()
    def features(x):
        ys = [den(n, x) for n in nets]
        y = sum(ys) / len(ys)
        return torch.cat([rf(x, y), sure(x, y, ys[0]), tile_feats(x, y)], 1).cpu().numpy()

    def run_array(arr):
        ps = []
        for lo in range(0, len(arr), BATCH):
            n = min(BATCH, len(arr) - lo)
            full = torch.zeros((BATCH, 1424, 176), device=dev)
            full[:n, sel] = torch.as_tensor(np.asarray(arr[lo:lo + n]), device=dev).float()
            ps.append(features(full)[:n])
        return np.concatenate(ps).astype(np.float32)

    only = list(filter(None, args.only.split(',')))
    want = lambda n: not only or any(fnmatch.fnmatch(n, p) for p in only)
    names = list(VIEWS) + ['shift_' + s for s in D.SHIFTS] + ['test']
    for name in names:
        if not want(name) or (out / f'{name}.npz').exists():
            continue
        if time.time() - t0 > args.budget:
            print('budget reached', flush=True); break
        t = time.time()
        if name == 'test':
            f = run_array(np.load(D.TEST, mmap_mode='r'))[:, None]
            np.savez(out / 'test.npz', feats=f)
        else:
            part, views = VIEWS.get(name, ('audit3', 1))
            idx = D.index(part)
            parts = []
            for v in range(views):
                pf = out / '_partial' / f'{name}_v{v}.npy'
                if pf.exists():
                    parts.append(np.load(pf)); continue
                if time.time() - t0 > args.budget:
                    print(f'budget reached before {name} view {v}', flush=True); return
                a = run_array(np.load(NOISY / f'{name}_v{v}.npy', mmap_mode='r'))
                pf.parent.mkdir(exist_ok=True); np.save(pf, a)
                parts.append(a)
                print(f'{name} view {v} done {time.time()-t:.0f}s', flush=True)
            f = np.stack(parts, 1)
            np.savez(out / f'{name}.npz', feats=f, index=idx, labels=lab[idx[:, 0], idx[:, 1]])
        print(f'{name}: {f.shape} {time.time()-t:.0f}s finite={np.isfinite(f).all()}', flush=True)
    print(f'done {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
