"""Existing family re-expressed: pseudo power spectrum of the masked map in 28 log-spaced |k| bands.

Same definition as stats.py (columns groups()['ps']).
"""
import numpy as np
import torch

H, W = 1424, 176
PS_EDGES = np.geomspace(0.004, 0.72, 29)
_cache = {}


def _setup(mask):
    if mask.device in _cache:
        return _cache[mask.device]
    dev = mask.device
    fy = torch.fft.fftfreq(H, device=dev)[:, None]
    fx = torch.fft.rfftfreq(W, device=dev)[None, :]
    r = torch.sqrt(fy ** 2 + fx ** 2)
    nb = len(PS_EDGES) - 1
    b = torch.bucketize(r, torch.as_tensor(PS_EDGES, device=dev, dtype=torch.float32)) - 1
    b = torch.where((b >= 0) & (b < nb), b, torch.full_like(b, nb))
    cnt = torch.bincount(b.flatten(), minlength=nb + 1)[:nb].clamp(min=1).float()
    _cache[dev] = (mask.float(), b.flatten(), cnt, float(mask.sum()))
    return _cache[dev]


def features(maps, mask):
    m, b, cnt, npix = _setup(mask)
    f = torch.fft.rfft2(maps * m)
    p = (f.real ** 2 + f.imag ** 2) / npix
    nb = len(cnt)
    ps = torch.zeros((len(maps), nb + 1), device=maps.device).index_add_(1, b, p.reshape(len(maps), -1))[:, :nb] / cnt
    return torch.log(ps + 1e-30)
