"""Additive large-scale gradient probe: least-squares linear-trend coefficients of the map over the footprint.

Projections of the map on the footprint-orthonormalized linear modes along the long axis and across it
(after removing the footprint mean), in units of the per-pixel noise. Normal maps scatter around 0 with a
cosmology-dependent cosmic variance; an additive gradient shifts the coefficient. Two dimensions only, so
the family's misfit is not diluted by uninformative columns.
"""
import math

import torch

H, W = 1424, 176
SIG = 0.4 / math.sqrt(240)
_cache = {}


def _setup(mask):
    if mask.device in _cache:
        return _cache[mask.device]
    dev = mask.device
    yy = torch.linspace(-1, 1, H, device=dev, dtype=torch.float64)[:, None].expand(H, W)
    xx = torch.linspace(-1, 1, W, device=dev, dtype=torch.float64)[None, :].expand(H, W)
    sel = mask.flatten().nonzero()[:, 0]
    B = torch.stack([torch.ones_like(yy), yy, xx], -1).reshape(-1, 3)[sel]
    Q, _ = torch.linalg.qr(B)
    _cache[dev] = (sel, Q[:, 1:].float())
    return _cache[dev]


def features(maps, mask):
    sel, Q = _setup(mask)
    return maps.reshape(len(maps), -1)[:, sel] @ Q / SIG
