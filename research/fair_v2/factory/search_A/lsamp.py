"""Large-scale clustering amplitude at several smoothing scales, after removing low-order footprint modes.

log rms of the map low-passed with Gaussians of sigma 8, 16, 32 and 64 px (16-128 arcmin), measured on the
footprint interior after projecting out the mean, linear, quadratic and cross modes. These scales are set
by linear large-scale structure and are insensitive to baryonic feedback and pixel-scale systematics, so a
parameter estimate from them disagrees with small-scale families when small-scale physics is distorted.
"""
import math

import torch

H, W = 1424, 176
SCALES = (8.0, 16.0, 32.0, 64.0)
_cache = {}


def _setup(mask):
    if mask.device in _cache:
        return _cache[mask.device]
    dev = mask.device
    yy = torch.linspace(-1, 1, H, device=dev, dtype=torch.float64)[:, None].expand(H, W)
    xx = torch.linspace(-1, 1, W, device=dev, dtype=torch.float64)[None, :].expand(H, W)
    sel = mask.flatten().nonzero()[:, 0]
    B = torch.stack([torch.ones_like(yy), yy, xx, 1.5 * yy * yy - 0.5, yy * xx, 1.5 * xx * xx - 0.5],
                    -1).reshape(-1, 6)[sel]
    Q, _ = torch.linalg.qr(B)
    r2 = torch.fft.fftfreq(H, device=dev)[:, None] ** 2 + torch.fft.rfftfreq(W, device=dev)[None, :] ** 2
    m = mask.float()
    gs, inners = [], []
    for s in SCALES:
        g = torch.exp(-2 * math.pi ** 2 * s * s * r2).float()
        ms = torch.fft.irfft2(torch.fft.rfft2(m) * g, s=(H, W))
        gs.append(g / ms.clamp(min=1e-3).mean() * 0 + g)
        inners.append(((ms > 0.9) & mask).flatten()[sel])
    _cache[dev] = (sel, Q.float(), gs, inners)
    return _cache[dev]


def features(maps, mask):
    sel, Q, gs, inners = _setup(mask)
    B = len(maps)
    x = maps.reshape(B, -1)[:, sel]
    res = torch.zeros_like(maps).reshape(B, -1)
    res[:, sel] = x - (x @ Q) @ Q.T
    fr = torch.fft.rfft2(res.reshape(maps.shape))
    out = []
    for g, inner in zip(gs, inners):
        lp = torch.fft.irfft2(fr * g, s=(H, W)).reshape(B, -1)[:, sel]
        out.append(0.5 * torch.log((lp[:, inner] ** 2).mean(1) + 1e-14))
    return torch.stack(out, 1)
