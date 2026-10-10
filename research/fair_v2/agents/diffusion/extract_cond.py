"""Field-level parameter-consistency features from the theta-conditional diffusion model.

One network gives D(y, s | theta) and, with the condition dropped, D(y, s). For each map the
condition is the map's own CNN estimate theta_hat (cnn0 predictions on the SAME noisy map,
standardized units). Per diffusion level sigma_t (fixed seeded eps_k shared by all maps):
  ku = D(y_k, s_k), kc = D(y_k, s_k | theta_hat)
  per radial band b: log P_b(x - ku), log P_b(x - kc), log P_b(kc - ku),
                     relative gain (P_b(x-ku) - P_b(x-kc)) / P_b(x-ku)   (band log-likelihood-ratio proxy)
  + skew/kurtosis of kc.
If a map's scales are mutually consistent with one theta (InD), conditioning on theta_hat improves
every band in the usual way; scale-dependent distortions change the per-band gain pattern.
Writes RAW/<set>.npz (feats (N,V,d), index, labels) from the dumped bit-identical noisy maps.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import data as D  # noqa: E402
from unet import UNet, denoise  # noqa: E402

LEVELS = [0.0, 0.02, 0.08, 0.32]
EDGES = [0.0, 0.01, 0.02, 0.04, 0.07, 0.12, 0.2, 0.32, 0.72]
SUB = 32
V3 = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3')


class CondFeaturizer:
    def __init__(self, ckpt, dev, m, bf16=True):
        st = torch.load(ckpt, map_location=dev, weights_only=False)
        self.net = UNet(**st['cfg']).to(dev)
        self.net.load_state_dict(st['ema']); self.net.eval()
        if bf16:
            self.net = self.net.to(torch.bfloat16)
        self.bf16, self.m, self.dev = bf16, m, dev
        g = torch.Generator().manual_seed(20261011)
        self.eps = [torch.randn((1424, 176), generator=g).to(dev) * m for _ in LEVELS]
        H, W = m.shape
        fy = torch.fft.fftfreq(H, device=dev)[:, None]
        fx = torch.fft.rfftfreq(W, device=dev)[None, :]
        r = torch.sqrt(fy ** 2 + fx ** 2).flatten()
        nb = len(EDGES) - 1
        b = torch.bucketize(r, torch.tensor(EDGES[1:-1], device=dev))
        B = torch.zeros((r.numel(), nb), device=dev)
        B[torch.arange(r.numel(), device=dev), b] = 1.0
        B[0] = 0.0
        self.B = B / B.sum(0, keepdim=True)
        self.npix = float(m.sum())

    def bandp(self, f):
        return (torch.fft.rfft2(f * self.m).abs().pow(2).flatten(1) / self.npix) @ self.B

    def moments(self, f):
        v = f[:, self.m > 0]
        c = v - v.mean(1, keepdim=True)
        var = c.pow(2).mean(1)
        return torch.stack([c.pow(3).mean(1) / var.pow(1.5), c.pow(4).mean(1) / var.pow(2) - 3], 1)

    @torch.no_grad()
    def __call__(self, x, th):
        """x (n,H,W) noisy maps, th (n,5) standardized theta_hat -> (n, d)."""
        n = len(x)
        sn = D.NOISE_SIGMA
        out = []
        for lo in range(0, n, SUB):
            k = min(SUB, n - lo)
            xb = torch.zeros((SUB,) + x.shape[1:], device=self.dev); xb[:k] = x[lo:lo + k]
            tb = torch.zeros((SUB, 5), device=self.dev); tb[:k] = th[lo:lo + k]
            cond = torch.cat([tb, torch.ones((SUB, 1), device=self.dev)], 1)
            unc = torch.zeros((SUB, 6), device=self.dev)
            fs = []
            for lv, st in enumerate(LEVELS):
                s = torch.full((SUB,), (sn * sn + st * st) ** 0.5, device=self.dev)
                y = xb + st * self.eps[lv]
                ku = denoise(self.net, y, s, self.m, bf16=self.bf16, cond=unc)
                kc = denoise(self.net, y, s, self.m, bf16=self.bf16, cond=cond)
                pu, pc, pd = self.bandp(xb - ku), self.bandp(xb - kc), self.bandp(kc - ku)
                fs += [torch.log(pu + 1e-30), torch.log(pc + 1e-30), torch.log(pd + 1e-30),
                       (pu - pc) / (pu + 1e-30), self.moments(kc)]
            out.append(torch.cat(fs, 1)[:k])
        return torch.cat(out).cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--noisy', type=Path, default=Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/diffusion/noisy'))
    ap.add_argument('--budget', type=float, default=45.0)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    dev = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    m = torch.as_tensor(D.mask(), device=dev, dtype=torch.float32)
    lab = D.labels()
    fz = CondFeaturizer(args.ckpt, dev, m)
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [('calib', 'calib3', 2), ('audit', 'audit3', 1)] + [('shift_' + s, 'audit3', 1) for s in D.SHIFTS]
    jobs += [('fit', 'fit3', 2)]
    for name, part, views in jobs:
        if (args.out / f'{name}.npz').exists():
            continue
        t = time.time()
        idx = D.index(part)
        n = args.limit or len(idx)
        th_all = np.load(V3 / 'cnn0' / f'{name}.npz')['feats'][..., -5:]
        F = []
        for v in range(views):
            src = np.load(args.noisy / f'{name}_v{v}.npy', mmap_mode='r')
            fs = []
            for lo in range(0, n, 64):
                k = min(64, n - lo)
                x = torch.zeros((k, 1424, 176), device=dev)
                x[:, m > 0] = torch.as_tensor(np.asarray(src[lo:lo + k]), device=dev).float()
                fs.append(fz(x, torch.as_tensor(th_all[lo:lo + k, v], device=dev)))
            F.append(np.concatenate(fs))
        np.savez(args.out / f'{name}.npz', feats=np.stack(F, 1), index=idx[:n], labels=lab[idx[:n, 0], idx[:n, 1]])
        print(f'{name} {n}x{views} {time.time()-t:.0f}s', flush=True)
        if (time.time() - t0) / 60 > args.budget:
            print('budget reached', flush=True)
            return
    if not (args.out / 'test.npz').exists():
        test = np.load(D.TEST, mmap_mode='r')
        n = args.limit or len(test)
        th_all = np.load(V3 / 'cnn0' / 'test.npz')['feats'][:, 0, -5:]
        fs = []
        for lo in range(0, n, 64):
            k = min(64, n - lo)
            x = torch.zeros((64, 1424, 176), device=dev)   # fixed batch shape, zero padding
            x[:k, m > 0] = torch.as_tensor(np.asarray(test[lo:lo + k]), device=dev).float()
            th = torch.zeros((64, 5), device=dev); th[:k] = torch.as_tensor(th_all[lo:lo + k], device=dev)
            fs.append(fz(x, th)[:k])
        np.savez(args.out / 'test.npz', feats=np.concatenate(fs)[:, None])
        print(f'test {n}', flush=True)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
