"""Noise-conditional U-Net denoiser (EDM preconditioning) for masked kappa maps.

The denoiser D(y, s) estimates E[kappa | y] for y = kappa + s * eps on the
observed pixels (eps white Gaussian, s >= shape-noise sigma). Because shape
noise is itself white Gaussian, the observed noisy map x = kappa + n diffused
by sigma_t is exactly kappa + s * eps with s = sqrt(sigma_n^2 + sigma_t^2), so
this one network gives the score / denoiser of p(x) at every diffusion level:
    E[x | x + sigma_t eps] = (sigma_t^2/s^2) D(y, s) + (sigma_n^2/s^2) y.
GroupNorm only (no BatchNorm): the output for one map never depends on others.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SIGMA_DATA = 0.011


class ResBlock(nn.Module):
    def __init__(self, cin, cout, edim):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.e = nn.Linear(edim, 2 * cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = self.c1(F.silu(self.n1(x)))
        sc, sh = self.e(emb).chunk(2, 1)
        h = self.n2(h) * (1 + sc[:, :, None, None]) + sh[:, :, None, None]
        h = self.c2(F.silu(h))
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(self, ch=(96, 160, 256), nres=(2, 2, 2), edim=256, nfreq=16, ps=4, cond_dim=0):
        super().__init__()
        self.cond_dim = cond_dim
        if cond_dim:
            # parameter conditioning (zero-init: starts as the unconditional model)
            self.cemb = nn.Linear(cond_dim, edim)
            nn.init.zeros_(self.cemb.weight); nn.init.zeros_(self.cemb.bias)
        self.ps = ps
        self.register_buffer('freqs', math.pi * 2.0 ** torch.linspace(-2, 3, nfreq))
        self.emb = nn.Sequential(nn.Linear(2 * nfreq, edim), nn.SiLU(), nn.Linear(edim, edim), nn.SiLU())
        self.stem = nn.Conv2d(2 * ps * ps, ch[0], 3, padding=1)
        self.down, self.pool = nn.ModuleList(), nn.ModuleList()
        c = ch[0]
        for i, w in enumerate(ch):
            self.down.append(nn.ModuleList([ResBlock(c if j == 0 else w, w, edim) for j in range(nres[i])]))
            c = w
            if i < len(ch) - 1:
                self.pool.append(nn.Conv2d(w, w, 3, stride=2, padding=1))
        self.mid = nn.ModuleList([ResBlock(c, c, edim), ResBlock(c, c, edim)])
        self.up, self.upc = nn.ModuleList(), nn.ModuleList()
        for i in reversed(range(len(ch))):
            w = ch[i]
            self.up.append(nn.ModuleList([ResBlock(c + w if j == 0 else w, w, edim) for j in range(nres[i])]))
            c = w
            if i > 0:
                self.upc.append(nn.Conv2d(w, ch[i - 1], 3, padding=1))
                c = ch[i - 1]
        self.out_n = nn.GroupNorm(8, ch[0])
        self.out = nn.Conv2d(ch[0], ps * ps, 3, padding=1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x, m, cnoise, cond=None):
        """x (B,H,W) scaled input, m (H,W) mask, cnoise (B,) noise conditioning, cond (B,cond_dim)."""
        a = cnoise[:, None] * self.freqs[None]
        emb = self.emb(torch.cat([a.sin(), a.cos()], 1))
        if cond is not None:
            emb = emb + self.cemb(cond.to(emb.dtype))
        h = self.stem(F.pixel_unshuffle(torch.stack([x, m.expand_as(x)], 1), self.ps))
        skips = []
        for i, blocks in enumerate(self.down):
            for b in blocks:
                h = b(h, emb)
            skips.append(h)
            if i < len(self.pool):
                h = self.pool[i](h)
        for b in self.mid:
            h = b(h, emb)
        for k, blocks in enumerate(self.up):
            h = torch.cat([h, skips.pop()], 1)
            for b in blocks:
                h = b(h, emb)
            if k < len(self.upc):
                h = self.upc[k](F.interpolate(h, scale_factor=2, mode='nearest'))
        return F.pixel_shuffle(self.out(F.silu(self.out_n(h))), self.ps)[:, 0]


def denoise(net, y, s, m, sd=SIGMA_DATA, bf16=False, cond=None):
    """EDM-preconditioned denoiser: estimate of clean kappa given y = kappa + s*eps."""
    s = s.float()
    s2 = s * s + sd * sd
    c_skip = (sd * sd / s2)[:, None, None]
    c_out = (s * sd / s2.sqrt())[:, None, None]
    c_in = (1 / s2.sqrt())[:, None, None]
    if bf16:  # whole net in bf16 (inference only)
        Fx = net((c_in * y).bfloat16(), m.bfloat16(), (torch.log(s) / 4).bfloat16(), cond)
    else:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            Fx = net(c_in * y, m, torch.log(s) / 4, cond)
    return (c_skip * y + c_out * Fx.float()) * m
