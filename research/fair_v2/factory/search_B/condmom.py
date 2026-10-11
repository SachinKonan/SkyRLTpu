"""Two-scale conditional moments: how the small-scale field depends on the large-scale field (rank-free copula).

For scale pairs (small s, large L) the mask-normalized smoothed fields are standardized on the interior. In 7
bins of the large-scale field (in rms units), features are the conditional mean of the small-scale field minus
its linear (Gaussian) prediction, and the conditional variance of the linear residual relative to its overall
variance. For a Gaussian field both are flat; gravity bends them (squeezed bispectrum / position-dependent
power), and the bend is fixed by physics. Rescaling the map changes nothing.
"""
import math

import torch

H, W = 1424, 176
PAIRS = ((1.0, 4.0), (2.0, 8.0))
EDGES = (-1.5, -0.75, 0.0, 0.75, 1.5, 2.5)
_cache = {}


def _setup(mask):
    dev = mask.device
    if dev in _cache:
        return _cache[dev]
    m = mask.float()
    fy = torch.fft.fftfreq(H, device=dev)[:, None]
    fx = torch.fft.rfftfreq(W, device=dev)[None, :]
    r2 = fy ** 2 + fx ** 2
    mk = torch.fft.rfft2(m)
    scales = sorted({s for p in PAIRS for s in p})
    ks = {s: torch.exp(-2 * math.pi ** 2 * s * s * r2) for s in scales}
    sm = {s: torch.fft.irfft2(mk * k, s=(H, W)) for s, k in ks.items()}
    inside = {L: ((sm[L] > 0.85) & mask).flatten().nonzero().squeeze(1) for _, L in PAIRS}
    _cache[dev] = (m, ks, sm, inside, torch.as_tensor(EDGES, device=dev))
    return _cache[dev]


def _std(v):
    v = v - v.mean(1, keepdim=True)
    return v / v.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-12)


def features(maps, mask):
    m, ks, sm, inside, edges = _setup(mask)
    B = maps.shape[0]
    f = torch.fft.rfft2(maps * m)
    fields = {s: (torch.fft.irfft2(f * k, s=(H, W)) / sm[s].clamp(min=0.05)).reshape(B, -1) for s, k in ks.items()}
    nb = len(EDGES) + 1
    out = []
    for s, L in PAIRS:
        idx = inside[L]
        ys, yl = _std(fields[s][:, idx]), _std(fields[L][:, idx])
        beta = (ys * yl).mean(1, keepdim=True)
        res = ys - beta * yl
        b = torch.bucketize(yl, edges)
        cnt = torch.zeros((B, nb), device=maps.device).scatter_add_(1, b, torch.ones_like(yl)).clamp(min=1.0)
        cm = torch.zeros((B, nb), device=maps.device).scatter_add_(1, b, res) / cnt
        cv = torch.zeros((B, nb), device=maps.device).scatter_add_(1, b, res * res) / cnt
        rv = res.pow(2).mean(1, keepdim=True).clamp(min=1e-12)
        out += [cm, torch.log(cv.clamp(min=1e-6) / rv)]
    return torch.cat(out, 1)
