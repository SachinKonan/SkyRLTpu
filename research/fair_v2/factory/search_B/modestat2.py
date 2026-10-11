"""Distribution of Fourier-mode powers within |k| bands, with the per-band maximum (modestat + outlier modes).

For a Gaussian field each mode's power divided by its band mean is ~exponential (unit mean), whatever the
spectrum, so the band's normalized second and third moments are fixed numbers (2 and 6, smeared slightly by the
fixed footprint). Non-linear structure couples modes at low k; coherent additive patterns, gradients and
non-stationary noise concentrate power in few modes. Features per band: log mean(p^2), log mean(p^3) and the
fraction of modes with p > 6, and log of the largest p, where p = |F|^2 / band mean.
"""
import math

import torch

H, W = 1424, 176
EDGES = torch.logspace(math.log10(0.003), math.log10(0.5), 13).tolist()
_cache = {}


def _setup(mask):
    dev = mask.device
    if dev in _cache:
        return _cache[dev]
    m = mask.float()
    fy = torch.fft.fftfreq(H, device=dev)[:, None]
    fx = torch.fft.rfftfreq(W, device=dev)[None, :]
    k = torch.sqrt(fy ** 2 + fx ** 2).flatten()
    nb = len(EDGES) - 1
    band = torch.bucketize(k, torch.as_tensor(EDGES, device=dev)) - 1
    keep = ((band >= 0) & (band < nb)).nonzero().squeeze(1)
    band = band[keep]
    cnt = torch.bincount(band, minlength=nb).float()
    _cache[dev] = (m, keep, band, cnt, nb)
    return _cache[dev]


def features(maps, mask):
    m, keep, band, cnt, nb = _setup(mask)
    B = maps.shape[0]
    f = torch.fft.rfft2(maps * m)
    p = (f.real ** 2 + f.imag ** 2).reshape(B, -1)[:, keep]
    idx = band[None, :].expand(B, -1)
    def bmean(v):
        return torch.zeros((B, nb), device=maps.device).scatter_add_(1, idx, v) / cnt
    mu = bmean(p).clamp(min=1e-30)
    q = p / mu.gather(1, idx)
    mx = torch.full((B, nb), 0.0, device=maps.device).scatter_reduce_(1, idx, q, 'amax', include_self=True)
    return torch.cat([torch.log(bmean(q * q)), torch.log(bmean(q ** 3)), bmean((q > 6).float()),
                      torch.log(mx.clamp(min=1e-3))], 1)
