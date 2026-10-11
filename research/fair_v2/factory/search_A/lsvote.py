"""Large-scale view: baryon-insensitive large-scale power plus a self-normalized gradient probe.

Log power of the footprint-masked map in 10 log-spaced |k| bands below 0.03 cycles/px (scales > ~1 deg, set
by the cosmic-variance-limited large-scale structure and insensitive to baryonic feedback), so a parameter
estimate from this family disagrees with small-scale families when small-scale physics is distorted. Plus
the linear-trend coefficients along/across the footprint divided by the map's own large-scale rms (after
removing the low-order modes), which flags additive gradients without being absorbed by the amplitude fit.
"""
import math

import numpy as np
import torch

H, W = 1424, 176
KEDGES = np.geomspace(0.0012, 0.03, 11)
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
    r = torch.sqrt(r2).float()
    band = torch.bucketize(r, torch.as_tensor(KEDGES, device=dev, dtype=torch.float32)) - 1
    nb = len(KEDGES) - 1
    idx = torch.where((band >= 0) & (band < nb), band, torch.full_like(band, nb)).flatten()
    cnt = torch.bincount(idx, minlength=nb + 1)[:nb].clamp(min=1).float()
    _cache[dev] = (sel, Q.float(), g, inner, idx, cnt, nb)
    return _cache[dev]


def features(maps, mask):
    sel, Q, g, inner, idx, cnt, nb = _setup(mask)
    B = len(maps)
    f = torch.fft.rfft2(maps)
    p = (f.real ** 2 + f.imag ** 2).reshape(B, -1)
    bp = torch.zeros((B, nb + 1), device=maps.device).index_add_(1, idx, p)[:, :nb] / cnt
    x = maps.reshape(B, -1)[:, sel]
    c = x @ Q
    res = torch.zeros_like(maps).reshape(B, -1)
    res[:, sel] = x - c @ Q.T
    lp = torch.fft.irfft2(torch.fft.rfft2(res.reshape(maps.shape)) * g, s=(H, W)).reshape(B, -1)[:, sel]
    ref = torch.sqrt((lp[:, inner] ** 2).mean(1) * len(sel))
    return torch.cat([torch.log(bp + 1e-12), c[:, 1:3] / ref[:, None], torch.log(ref)[:, None]], 1)
