"""Spatial non-stationarity of band power across the footprint (self-normalized).

The map is split into difference-of-Gaussian bands (pixel, 1-2, 2-4, 4-8, 8-16 px). For each band the local
power is averaged over the covered pixels of 32x32-px tiles that are at least 75% inside the footprint. Features per band: log coefficient of
variation of the tile powers across tiles, the linear trend of log tile power along the long axis and across
it (in units of its scatter), and the across-tile correlation of each band's log power with the next band's.
Homogeneous signal plus white noise fixes these up to cosmic variance, nearly independent of the
parameters; spatially varying systematics (depth, PSF patterns, patchy noise) move them.
"""
import math

import torch
import torch.nn.functional as F

H, W = 1424, 176
SCALES = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0)
T = 32
_cache = {}


def _setup(mask):
    if mask.device in _cache:
        return _cache[mask.device]
    dev = mask.device
    r2 = torch.fft.fftfreq(H, device=dev)[:, None] ** 2 + torch.fft.rfftfreq(W, device=dev)[None, :] ** 2
    g = [torch.exp(-2 * math.pi ** 2 * s * s * r2) for s in SCALES]
    bands = [g[i] - g[i + 1] for i in range(len(g) - 1)]
    m = mask.float()
    cov = F.avg_pool2d(m[None, None], T, ceil_mode=False)[0, 0]          # (44, 5) tile coverage
    ok = (cov > 0.75).flatten()
    th, tw = cov.shape
    ty = (torch.arange(th, device=dev, dtype=torch.float32)[:, None].expand(th, tw)).flatten()[ok]
    tx = (torch.arange(tw, device=dev, dtype=torch.float32)[None, :].expand(th, tw)).flatten()[ok]
    ty = (ty - ty.mean()) / ty.std()
    tx = (tx - tx.mean()) / tx.std().clamp(min=1e-6)
    _cache[dev] = (torch.stack(bands), ok, ty, tx, cov.flatten()[ok])
    return _cache[dev]


def features(maps, mask):
    bands, ok, ty, tx, cv0 = _setup(mask)
    B = len(maps)
    fr = torch.fft.rfft2(maps)
    out, logs = [], []
    for b in bands:
        y = torch.fft.irfft2(fr * b, s=(H, W)) * mask
        p = F.avg_pool2d((y * y)[:, None], T)[:, 0].reshape(B, -1)[:, ok] / cv0   # (B, tiles), per covered pixel
        lp = torch.log(p + 1e-14)
        mu = p.mean(1)
        cv = p.std(1) / mu
        lc = lp - lp.mean(1, keepdim=True)
        sd = lc.std(1).clamp(min=1e-6)
        out += [torch.log(cv), (lc * ty).mean(1) / sd, (lc * tx).mean(1) / sd]
        logs.append(lc / sd[:, None])
    for i in range(len(logs) - 1):
        out.append((logs[i] * logs[i + 1]).mean(1))
    return torch.stack(out, 1)
