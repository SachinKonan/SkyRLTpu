"""Self-normalized low-order additive-pattern probe: linear, quadratic and cross modes (5) / own rms.

The linear-trend coefficients of the map along and across the footprint (least squares over the footprint)
are divided by the map's own large-scale fluctuation level, measured as the rms of a 16-px low-pass of the
map after removing the mean, linear and quadratic modes. Normal maps give a ratio whose spread does not
depend on the clustering amplitude, so the per-map parameter fit cannot hide an additive gradient by moving
to a higher-amplitude cosmology. Also returns the log fluctuation level itself.
"""
import math

import torch

H, W = 1424, 176
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
    g = torch.exp(-2 * math.pi ** 2 * 16.0 ** 2 * r2).float()
    m = mask.float()
    ms = torch.fft.irfft2(torch.fft.rfft2(m) * g, s=(H, W))
    inner = ((ms > 0.9) & mask).flatten()[sel]
    _cache[dev] = (sel, Q.float(), g, inner)
    return _cache[dev]


def features(maps, mask):
    sel, Q, g, inner = _setup(mask)
    x = maps.reshape(len(maps), -1)[:, sel]
    c = x @ Q
    res = torch.zeros_like(maps).reshape(len(maps), -1)
    res[:, sel] = x - c @ Q.T
    lp = torch.fft.irfft2(torch.fft.rfft2(res.reshape(maps.shape)) * g, s=(H, W)).reshape(len(maps), -1)[:, sel]
    ref = torch.sqrt((lp[:, inner] ** 2).mean(1) * len(sel))       # comparable to a mode coefficient
    return torch.cat([c[:, 1:6] / ref[:, None], torch.log(ref)[:, None]], 1)
