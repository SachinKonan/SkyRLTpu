"""Per-pixel noise non-Gaussianity: moments of the pixel-minus-neighbour-mean residual.

d = x - mean(8 neighbours) on pixels whose 3x3 neighbourhood lies inside the footprint. For the true
white Gaussian noise d is Gaussian with variance sigma^2 (1 + 1/8); the signal leaks in only weakly. A
per-pixel scale mixture of the noise (galaxy-count fluctuations, heavy-tailed shapes) raises the kurtosis
of d, and the weights here keep ~79% of a per-pixel excess kurtosis (a 2x2 Haar filter keeps only 25%).
Features: log(m2 / expected), excess kurtosis, standardized 6th moment, L1/L2 ratio, scale-free tail
fractions, globally and in 4 slabs along the long axis (non-stationarity).
"""
import math

import torch
import torch.nn.functional as F

H, W = 1424, 176
SIG2 = (0.4 / math.sqrt(240)) ** 2 * (1 + 1 / 8)
NSLAB = 4
_cache = {}


def _setup(mask):
    if mask.device in _cache:
        return _cache[mask.device]
    m = mask.float()[None, None]
    k = torch.ones(1, 1, 3, 3, device=m.device)
    full = F.conv2d(m, k, padding=1)[0, 0] >= 8.999          # whole 3x3 neighbourhood inside
    nb = torch.ones(1, 1, 3, 3, device=m.device) / 8
    nb[0, 0, 1, 1] = 0
    rows = torch.arange(H, device=m.device)[:, None].expand(H, W)
    slab = torch.clamp(rows * NSLAB // H, max=NSLAB - 1)
    sel = [full] + [full & (slab == s) for s in range(NSLAB)]
    _cache[mask.device] = (nb, [s.flatten().nonzero()[:, 0] for s in sel])
    return _cache[mask.device]


def _moments(d):
    d = d.double()
    m2 = (d * d).mean(1)
    z = d / m2.sqrt()[:, None]
    z2 = z * z
    k4 = (z2 * z2).mean(1) - 3
    k6 = (z2 * z2 * z2).mean(1) - 15
    l1 = z.abs().mean(1) / math.sqrt(2 / math.pi)
    t3 = (z.abs() > 3).double().mean(1) / 2.7e-3
    t4 = (z.abs() > 4).double().mean(1) / 6.33e-5
    return torch.stack([torch.log(m2 / SIG2), k4, k6, l1, torch.log(t3 + 0.05), torch.log(t4 + 0.05)], 1)


def features(maps, mask):
    nb, sels = _setup(mask)
    d = (maps - F.conv2d(maps[:, None], nb, padding=1)[:, 0]).reshape(len(maps), -1)
    out = [_moments(d[:, sels[0]])]
    for s in sels[1:]:
        out.append(_moments(d[:, s])[:, :3])
    return torch.cat(out, 1).float()
