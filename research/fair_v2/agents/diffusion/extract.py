"""Field-level denoising-ELBO features for every v3 set (resumable, per map).

For a noisy observed map x and K diffusion levels sigma_t (FIXED seeded noise
tensor eps_k shared by every map, so the result is a deterministic function of
the one map):
    y_k   = x + sigma_t eps_k                     (observed pixels only)
    kh_k  = D(y_k, s_k),  s_k = sqrt(sigma_n^2 + sigma_t^2)   posterior mean of kappa
    xh_k  = (sigma_t^2/s^2) kh_k + (sigma_n^2/s^2) y_k         posterior mean of x (denoiser of p(x))
    e_k   = x - xh_k                                           denoising (ELBO) error
Per level: log radial-band power of e_k, of r_k = x - kh_k, of kh_k; skew and
kurtosis of kh_k and r_k on observed pixels; mean e_k^2.
Noisy inputs come from dump_noisy.py (pipeline_v3's exact noise loop run on
della-gh GH200), so they are bit-identical to the stats/cnn0 sources; this
script itself can then run on any GPU. Test: fixed-shape zero-padded batches.
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

LEVELS = [0.0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
EDGES = [0.0, 0.01, 0.02, 0.04, 0.07, 0.12, 0.2, 0.32, 0.72]
SUB = 32        # fixed sub-batch shape for every forward pass
CHUNK = 8       # batches of 64 per resumable part


class Featurizer:
    def __init__(self, ckpt, dev, m, levels=LEVELS, bf16=False):
        st = torch.load(ckpt, map_location=dev, weights_only=False)
        cfg = st.get('cfg', {})
        self.net = UNet(**cfg).to(dev)
        self.net.load_state_dict(st['ema'])
        self.net.eval()
        self.bf16 = bf16
        if bf16:
            self.net = self.net.to(torch.bfloat16)
        self.m = m
        self.dev = dev
        self.levels = levels
        g = torch.Generator().manual_seed(20261010)
        self.eps = [torch.randn((1424, 176), generator=g).to(dev) * m for _ in levels]
        H, W = m.shape
        fy = torch.fft.fftfreq(H, device=dev)[:, None]
        fx = torch.fft.rfftfreq(W, device=dev)[None, :]
        r = torch.sqrt(fy ** 2 + fx ** 2).flatten()
        nb = len(EDGES) - 1
        b = torch.bucketize(r, torch.tensor(EDGES[1:-1], device=dev))
        B = torch.zeros((r.numel(), nb), device=dev)
        B[torch.arange(r.numel(), device=dev), b] = 1.0
        B[0] = 0.0  # drop the DC mode
        self.B = B / B.sum(0, keepdim=True)
        self.npix = float(m.sum())
        self.dim = len(levels) * (3 * nb + 5)

    def band(self, f):
        P = torch.fft.rfft2(f * self.m).abs().pow(2).flatten(1) / self.npix
        return torch.log(P @ self.B + 1e-30)

    def moments(self, f):
        v = f[:, self.m > 0]
        c = v - v.mean(1, keepdim=True)
        var = c.pow(2).mean(1)
        return torch.stack([c.pow(3).mean(1) / var.pow(1.5), c.pow(4).mean(1) / var.pow(2) - 3], 1)

    @torch.no_grad()
    def __call__(self, x):
        """x (n, H, W) noisy masked maps -> (n, dim) features, fixed SUB-sized passes."""
        n = len(x)
        out = []
        sn = D.NOISE_SIGMA
        for lo in range(0, n, SUB):
            xb = torch.zeros((SUB,) + x.shape[1:], device=self.dev)
            k = min(SUB, n - lo)
            xb[:k] = x[lo:lo + k]
            fs = []
            for lv, st in enumerate(self.levels):
                s = (sn * sn + st * st) ** 0.5
                y = xb + st * self.eps[lv]
                kh = denoise(self.net, y, torch.full((SUB,), s, device=self.dev), self.m, bf16=self.bf16)
                xh = (st * st / (s * s)) * kh + (sn * sn / (s * s)) * y
                e = (xb - xh) * self.m
                r = (xb - kh) * self.m
                fs += [self.band(e) if st > 0 else torch.zeros((SUB, len(EDGES) - 1), device=self.dev),
                       self.band(r), self.band(kh), self.moments(kh), self.moments(r),
                       (e.pow(2).sum((1, 2)) / self.npix)[:, None]]
            out.append(torch.cat(fs, 1)[:k])
        return torch.cat(out).cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--budget', type=float, default=8.0, help='minutes')
    ap.add_argument('--fit-views', type=int, default=2)
    ap.add_argument('--only', default='', help='comma list of set names to process')
    ap.add_argument('--noisy', type=Path, default=Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/diffusion/noisy'))
    ap.add_argument('--bf16', action='store_true')
    ap.add_argument('--max-chunks', type=int, default=0, help='diagnostics only: first chunks of each set')
    args = ap.parse_args()
    t0 = time.time()
    dev = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    m = torch.as_tensor(D.mask(), device=dev, dtype=torch.float32)
    lab = D.labels()
    fz = Featurizer(args.ckpt, dev, m, bf16=args.bf16)
    parts = args.out / 'parts'
    parts.mkdir(parents=True, exist_ok=True)
    BATCH = 64

    def over():
        return (time.time() - t0) / 60 > args.budget

    def run(name, part, views, seed, shift=None):
        """Features of the dumped noisy maps (dump_noisy.py, bit-identical to pipeline_v3). True when complete."""
        idx = D.index(part)
        nch = -(-len(idx) // (BATCH * CHUNK))
        for v in range(views):
            src = np.load(args.noisy / f'{name}_v{v}.npy', mmap_mode='r')
            assert len(src) == len(idx)
            for c in range(min(nch, args.max_chunks or nch)):
                if (parts / f'{name}_v{v}_c{c}.npz').exists():
                    continue
                t = time.time()
                fs = []
                for lo in range(c * BATCH * CHUNK, min(len(idx), (c + 1) * BATCH * CHUNK), BATCH):
                    n = min(BATCH, len(idx) - lo)
                    x = torch.zeros((n, 1424, 176), device=dev)
                    x[:, m > 0] = torch.as_tensor(np.asarray(src[lo:lo + n]), device=dev).float()
                    fs.append(fz(x))
                np.savez(parts / f'{name}_v{v}_c{c}.npz', feats=np.concatenate(fs))
                print(f'{name} v{v} chunk {c+1}/{nch} {time.time()-t:.0f}s', flush=True)
                if over():
                    return False
        if args.max_chunks:
            return True
        F = np.stack([np.concatenate([np.load(parts / f'{name}_v{v}_c{c}.npz')['feats'] for c in range(nch)])
                      for v in range(views)], 1)
        np.savez(args.out / f'{name}.npz', feats=F, index=idx, labels=lab[idx[:, 0], idx[:, 1]])
        print(f'{name} complete {F.shape}', flush=True)
        return True

    def run_test():
        test = np.load(D.TEST, mmap_mode='r')
        nch = -(-len(test) // (BATCH * CHUNK))
        for c in range(nch):
            if (parts / f'test_c{c}.npz').exists():
                continue
            t = time.time()
            fs = []
            for lo in range(c * BATCH * CHUNK, min(len(test), (c + 1) * BATCH * CHUNK), BATCH):
                n = min(BATCH, len(test) - lo)
                full = torch.zeros((BATCH, 1424, 176), device=dev)
                full[:n, m > 0] = torch.as_tensor(np.asarray(test[lo:lo + n]), device=dev).float()
                fs.append(fz(full)[:n])
            np.savez(parts / f'test_c{c}.npz', feats=np.concatenate(fs))
            print(f'test chunk {c+1}/{nch} {time.time()-t:.0f}s', flush=True)
            if over():
                return False
        F = np.concatenate([np.load(parts / f'test_c{c}.npz')['feats'] for c in range(nch)])[:, None]
        np.savez(args.out / 'test.npz', feats=F)
        print(f'test complete {F.shape}', flush=True)
        return True

    jobs = [('calib', 'calib3', 2, 32, None), ('audit', 'audit3', 1, 33, None)]
    jobs += [('shift_' + s, 'audit3', 1, 300 + i, s) for i, s in enumerate(D.SHIFTS)]
    jobs += [('fit', 'fit3', args.fit_views, 31, None)]
    only = set(filter(None, args.only.split(',')))
    for name, part, views, seed, shift in jobs:
        if only and name not in only:
            continue
        if (args.out / f'{name}.npz').exists():
            continue
        if not run(name, part, views, seed, shift):
            print('budget reached', flush=True)
            return
    if (not only or 'test' in only) and not (args.out / 'test.npz').exists():
        if not run_test():
            print('budget reached', flush=True)
            return
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
