"""Self-normalized large-scale gradient probe + robust noise-distribution summary (level, kurtosis, L1/L2).

The linear-trend coefficients of the map along and across the footprint (least squares over the footprint)
are divided by the map's own large-scale fluctuation level, measured as the rms of a 16-px low-pass of the
map after removing the mean, linear and quadratic modes. Normal maps give a ratio whose spread does not
depend on the clustering amplitude, so the per-map parameter fit cannot hide an additive gradient by moving
to a higher-amplitude cosmology. Also returns the log fluctuation level itself.
"""
import math

import torch
import torch.nn.functional as F

H, W = 1424, 176
_cache = {}
_ncache = {}
SIG2 = (0.4 / math.sqrt(240)) ** 2 * (1 + 1 / 8)
THRESH = 1.5


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


def _nsetup(mask):
    if mask.device in _ncache:
        return _ncache[mask.device]
    dev = mask.device
    m = mask.float()
    full = F.conv2d(m[None, None], torch.ones(1, 1, 3, 3, device=dev), padding=1)[0, 0] >= 8.999
    nb = torch.ones(1, 1, 3, 3, device=dev) / 8
    nb[0, 0, 1, 1] = 0
    r2 = torch.fft.fftfreq(H, device=dev)[:, None] ** 2 + torch.fft.rfftfreq(W, device=dev)[None, :] ** 2
    _ncache[dev] = (nb, full, torch.exp(-2 * math.pi ** 2 * 4.0 * r2))
    return _ncache[dev]


def _noise(maps, mask):
    """Noise level, excess kurtosis and L1/L2 of the pixel-minus-neighbour-mean residual away from peaks."""
    nb, full, g = _nsetup(mask)
    d = maps - F.conv2d(maps[:, None], nb, padding=1)[:, 0]
    xs = torch.fft.irfft2(torch.fft.rfft2(maps) * g, s=(H, W))
    n0 = full.sum()
    mu = (xs * full).sum((1, 2)) / n0
    sd = torch.sqrt(((xs - mu[:, None, None]) ** 2 * full).sum((1, 2)) / n0)
    w = (full & ((xs - mu[:, None, None]).abs() < THRESH * sd[:, None, None])).double()
    n = w.sum((1, 2))
    dd = d.double()
    m2 = (dd * dd * w).sum((1, 2)) / n
    z = dd / m2.sqrt()[:, None, None]
    k4 = (z ** 4 * w).sum((1, 2)) / n - 3
    l1 = (z.abs() * w).sum((1, 2)) / n / math.sqrt(2 / math.pi)
    return torch.stack([torch.log(m2 / SIG2), k4, l1], 1).float()


def features(maps, mask):
    sel, Q, g, inner = _setup(mask)
    x = maps.reshape(len(maps), -1)[:, sel]
    c = x @ Q
    res = torch.zeros_like(maps).reshape(len(maps), -1)
    res[:, sel] = x - c @ Q.T
    lp = torch.fft.irfft2(torch.fft.rfft2(res.reshape(maps.shape)) * g, s=(H, W)).reshape(len(maps), -1)[:, sel]
    ref = torch.sqrt((lp[:, inner] ** 2).mean(1) * len(sel))       # comparable to a mode coefficient
    return torch.cat([c[:, 1:3] / ref[:, None], torch.log(ref)[:, None], _noise(maps, mask)], 1)
