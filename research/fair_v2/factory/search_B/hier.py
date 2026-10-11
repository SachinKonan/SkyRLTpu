"""Noise-debiased skewness k3 / s^3 and kurtosis k4 / s^4 of the smoothed field (amplitude-free).

Gaussian smoothing at 1, 2, 4, 8, 16 px (mask-normalized), on pixels where the smoothed footprint > 0.9. Shape noise is white and Gaussian with known
sigma, so it adds nothing to the connected moments k3, k4 and a known amount to the variance; subtracting it
gives the signal variance s^2. Gravity nearly fixes the reduced moments (weakly cosmology dependent); baryons, smoothing
and non-gravitational distortions change them differently.
"""
import math

import torch

H, W = 1424, 176
SCALES = (1.0, 2.0, 4.0, 8.0, 16.0)
SIGMA_N = 0.4 / math.sqrt(240.0)
_cache = {}


def _setup(mask):
    dev = mask.device
    if dev in _cache:
        return _cache[dev]
    m = mask.float()
    fy = torch.fft.fftfreq(H, device=dev)[:, None]
    fx = torch.fft.rfftfreq(W, device=dev)[None, :]
    r2 = fy ** 2 + fx ** 2
    fxf = torch.fft.fftfreq(W, device=dev)[None, :]
    r2f = fy ** 2 + fxf ** 2
    mk = torch.fft.rfft2(m)
    ks, sms, nv = [], [], []
    for s in SCALES:
        k = torch.exp(-2 * math.pi ** 2 * s * s * r2)
        ks.append(k)
        sms.append(torch.fft.irfft2(mk * k, s=(H, W)))
        nv.append(float(SIGMA_N ** 2 * torch.exp(-4 * math.pi ** 2 * s * s * r2f).mean()))
    inside = [(sm > 0.9) & mask for sm in sms]
    _cache[dev] = (m, ks, sms, nv, inside)
    return _cache[dev]


def features(maps, mask):
    m, ks, sms, nv, inside = _setup(mask)
    f = torch.fft.rfft2(maps * m)
    out = []
    for k, sm, n2, ins in zip(ks, sms, nv, inside):
        d = (torch.fft.irfft2(f * k, s=(H, W)) / sm.clamp(min=0.05))[:, ins]
        d = d - d.mean(1, keepdim=True)
        v = d.pow(2).mean(1)
        k3 = d.pow(3).mean(1)
        k4 = d.pow(4).mean(1) - 3 * v * v
        s2 = torch.maximum(v - n2, 0.1 * v)
        out += [k3 / s2 ** 1.5, torch.asinh(k4 / s2 ** 2)]
    return torch.stack(out, 1)
