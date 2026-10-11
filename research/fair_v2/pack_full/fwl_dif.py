"""Field-level denoising-ELBO features (source `difa`) for inference (copied from agents/diffusion/extract.py
and make_sources.py).

For a noisy map x and diffusion level sigma_t (a FIXED seeded noise tensor eps_k shared by every map, so the
result is a deterministic function of the one map): y = x + sigma_t eps_k; kh = D(y, s) with
s = sqrt(sigma_n^2 + sigma_t^2); xh = (sigma_t^2/s^2) kh + (sigma_n^2/s^2) y; e = x - xh; r = x - kh.
Per level (29): log radial-band power of e (8), r (8), kh (8); skew/kurt of kh (2) and r (2); mean e^2 (1).
`difa` keeps, per level, [e bands if sigma_t > 0] + r, kh bands + kh, r moments (216 columns).
Only the levels in `need` are computed (the SS family reads levels 0-3); the others are left at 0 and are
never read by any packaged family.
"""
import math

import numpy as np
import torch

from fwl_dunet import UNet, denoise

LEVELS = [0.0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
EDGES = [0.0, 0.01, 0.02, 0.04, 0.07, 0.12, 0.2, 0.32, 0.72]
NOISE_SIGMA = 0.4 / math.sqrt(2 * 30 * 4)
SUB = 32
NB = len(EDGES) - 1
PER = 3 * NB + 5


def difa_cols():
    out = []
    for k, st in enumerate(LEVELS):
        b = k * PER
        if st > 0:
            out += list(range(b, b + NB))
        out += list(range(b + NB, b + 3 * NB + 4))
    return np.array(out)


def levels_for(cols):
    """Diffusion levels whose features a set of difa columns reads."""
    raw = difa_cols()[np.asarray(cols)]
    return sorted({int(c) // PER for c in raw})


class DifFeats:
    def __init__(self, st, m, dev, need):
        self.net = UNet(**st['cfg']).to(dev)
        self.net.load_state_dict({k: v.float() for k, v in st['ema'].items()})
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.m, self.dev, self.need = m, dev, sorted(need)
        g = torch.Generator().manual_seed(20261010)
        self.eps = [torch.randn((1424, 176), generator=g).to(dev) * m for _ in LEVELS]   # all drawn, in order
        H, W = m.shape
        fy = torch.fft.fftfreq(H, device=dev)[:, None]
        fx = torch.fft.rfftfreq(W, device=dev)[None, :]
        r = torch.sqrt(fy ** 2 + fx ** 2).flatten()
        b = torch.bucketize(r, torch.tensor(EDGES[1:-1], device=dev))
        B = torch.zeros((r.numel(), NB), device=dev)
        B[torch.arange(r.numel(), device=dev), b] = 1.0
        B[0] = 0.0
        self.B = B / B.sum(0, keepdim=True)
        self.npix = float(m.sum())
        self.cols = torch.as_tensor(difa_cols(), device=dev)

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
        """x (n, H, W) noisy masked maps -> (n, 216) difa features (fixed SUB-sized passes)."""
        n = len(x)
        out = []
        sn = NOISE_SIGMA
        for lo in range(0, n, SUB):
            xb = torch.zeros((SUB,) + x.shape[1:], device=self.dev)
            k = min(SUB, n - lo)
            xb[:k] = x[lo:lo + k]
            fs = []
            for lv, st in enumerate(LEVELS):
                if lv not in self.need:
                    fs.append(torch.zeros((SUB, PER), device=self.dev))
                    continue
                s = (sn * sn + st * st) ** 0.5
                y = xb + st * self.eps[lv]
                kh = denoise(self.net, y, torch.full((SUB,), s, device=self.dev), self.m)
                xh = (st * st / (s * s)) * kh + (sn * sn / (s * s)) * y
                e = (xb - xh) * self.m
                r = (xb - kh) * self.m
                fs += [self.band(e) if st > 0 else torch.zeros((SUB, NB), device=self.dev),
                       self.band(r), self.band(kh), self.moments(kh), self.moments(r),
                       (e.pow(2).sum((1, 2)) / self.npix)[:, None]]
            out.append(torch.cat(fs, 1)[:k])
        return torch.cat(out)[:, self.cols]
