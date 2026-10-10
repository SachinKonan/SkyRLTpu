"""Existing family re-expressed: difference-of-Gaussian wavelet l1 norms in SNR bins + band moments.

Same definition as stats.py (columns groups()['dog'], 56 dims); band scales frozen from normal fit maps.
"""
import math

import torch
import torch.nn.functional as F

H, W = 1424, 176
SCALES = (0.0, 1.0, 2.0, 4.0, 8.0)
DOG = ((0, 1), (1, 2), (2, 3), (3, 4))
DOG_SNR = (-4., -3., -2., -1., 0., 1., 2., 3., 4., 5.)
DOG_S = (0.02347128838300705, 0.00570716755464673, 0.0034574621822685003, 0.0022787442430853844)
_cache = {}


def _setup(mask):
    if mask.device in _cache:
        return _cache[mask.device]
    m = mask.float()
    fy = torch.fft.fftfreq(H, device=m.device)[:, None]
    fx = torch.fft.rfftfreq(W, device=m.device)[None, :]
    r = torch.sqrt(fy ** 2 + fx ** 2)
    kernels = [torch.exp(-2 * math.pi ** 2 * s * s * r * r) for s in SCALES]
    mk = torch.fft.rfft2(m)
    smask = [torch.fft.irfft2(mk * g, s=(H, W)) for g in kernels]
    interior = []
    for i, s in enumerate(SCALES):
        inside = (m > 0) if s == 0 else ((smask[i] > 0.95) & (m > 0))
        padded = F.pad(inside.float()[None, None], (1, 1, 1, 1), value=0.)
        interior.append(-F.max_pool2d(-padded, 3, 1, 0)[0, 0] > 0.5)
    _cache[mask.device] = (m, kernels, smask, interior, torch.as_tensor(DOG_SNR, device=m.device))
    return _cache[mask.device]


def features(maps, mask):
    m, kernels, smask, interior, edges = _setup(mask)
    f = torch.fft.rfft2(maps * m)
    fields = [maps * m if s == 0 else torch.fft.irfft2(f * kernels[i], s=(H, W)) / smask[i].clamp(min=0.05) * m
              for i, s in enumerate(SCALES)]
    out = []
    for j, (a, b) in enumerate(DOG):
        w = (fields[a] - fields[b])[:, interior[b]]
        s = DOG_S[j]
        idx = torch.bucketize(w / s, edges)
        l1 = torch.zeros((len(maps), len(DOG_SNR) + 1), device=maps.device).scatter_add_(1, idx, w.abs() / s)
        w2 = (w * w).mean(1)
        out += [torch.log1p(l1), torch.stack([torch.log(w2 / s ** 2), (w ** 3).mean(1) / w2 ** 1.5,
                                             (w ** 4).mean(1) / w2 ** 2], 1)]
    return torch.cat(out, 1)
