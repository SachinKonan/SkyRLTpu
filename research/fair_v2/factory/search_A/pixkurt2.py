"""Robust per-pixel noise non-Gaussianity: moments of the pixel-minus-neighbour-mean residual away from peaks.

d = x - mean(8 neighbours) on pixels whose 3x3 neighbourhood is inside the footprint and whose 2-px
smoothed value lies within 1.5 standard deviations of the map's smoothed mean. Excluding the peaks and
voids removes most of the signal's non-Gaussian leakage (within-cosmology spread of the excess kurtosis
drops ~7x), leaving a probe of the noise's own distribution: level, excess kurtosis, L1/L2 ratio and
scale-free tail fractions, plus the residual's lag-1 correlations along each axis (noise whiteness).
"""
import math

import torch
import torch.nn.functional as F

H, W = 1424, 176
SIG2 = (0.4 / math.sqrt(240)) ** 2 * (1 + 1 / 8)
THRESH = 1.5
_cache = {}


def _setup(mask):
    if mask.device in _cache:
        return _cache[mask.device]
    dev = mask.device
    m = mask.float()
    k3 = torch.ones(1, 1, 3, 3, device=dev)
    full = F.conv2d(m[None, None], k3, padding=1)[0, 0] >= 8.999
    k5 = torch.ones(1, 1, 5, 5, device=dev)
    full5 = F.conv2d(m[None, None], k5, padding=2)[0, 0] >= 24.999      # neighbours' neighbourhoods too
    nb = torch.ones(1, 1, 3, 3, device=dev) / 8
    nb[0, 0, 1, 1] = 0
    r2 = torch.fft.fftfreq(H, device=dev)[:, None] ** 2 + torch.fft.rfftfreq(W, device=dev)[None, :] ** 2
    g = torch.exp(-2 * math.pi ** 2 * 4.0 * r2)                              # Gaussian, sigma 2 px
    _cache[dev] = (nb, full, full5, g)
    return _cache[dev]


def features(maps, mask):
    nb, full, full5, g = _setup(mask)
    B = len(maps)
    d = maps - F.conv2d(maps[:, None], nb, padding=1)[:, 0]
    xs = torch.fft.irfft2(torch.fft.rfft2(maps) * g, s=(H, W))
    fv = full.expand(B, H, W)
    n0 = full.sum()
    mu = (xs * full).sum((1, 2)) / n0
    sd = torch.sqrt(((xs - mu[:, None, None]) ** 2 * full).sum((1, 2)) / n0)
    w = (fv & ((xs - mu[:, None, None]).abs() < THRESH * sd[:, None, None])).double()
    n = w.sum((1, 2))
    dd = d.double()
    m2 = (dd * dd * w).sum((1, 2)) / n
    z = dd / m2.sqrt()[:, None, None]
    z2 = z * z
    k4 = (z2 * z2 * w).sum((1, 2)) / n - 3
    l1 = (z.abs() * w).sum((1, 2)) / n / math.sqrt(2 / math.pi)
    t3 = ((z.abs() > 3).double() * w).sum((1, 2)) / n / 2.7e-3
    t4 = ((z.abs() > 4).double() * w).sum((1, 2)) / n / 6.33e-5
    # lag-1 correlations of d inside the 5x5-interior (both pixels' residuals fully defined)
    w5 = full5.double()
    n5 = w5.sum()
    cy = (dd[:, 1:, :] * dd[:, :-1, :] * w5[1:, :] * w5[:-1, :]).sum((1, 2)) / (w5[1:, :] * w5[:-1, :]).sum()
    cx = (dd[:, :, 1:] * dd[:, :, :-1] * w5[:, 1:] * w5[:, :-1]).sum((1, 2)) / (w5[:, 1:] * w5[:, :-1]).sum()
    v5 = (dd * dd * w5).sum((1, 2)) / n5
    return torch.stack([torch.log(m2 / SIG2), k4, l1, torch.log(t3 + 0.05), torch.log(t4 + 0.05),
                        cy / v5, cx / v5], 1).float()
