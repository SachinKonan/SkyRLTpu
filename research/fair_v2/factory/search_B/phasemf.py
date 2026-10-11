"""Minkowski functionals of the phase-only (whitened) field.

eps_kc = irfft2(F/|F| * [|k| < kc]) of the footprint-masked map (unit rms on the interior) for kc = 0.025,
0.05, 0.1 cycles/px. At thresholds nu = -1.5, -0.5, 0.5, 1.5: area fraction V0, boundary length V1 (smoothed
delta) and Euler characteristic density V2 (curvature form). The whitened field has a fixed spectrum, so for
Gaussian phases all MFs are fixed numbers; gravitational phase coupling moves them. Isotropic Fourier filters
and rescalings leave every feature unchanged.
"""
import math

import torch

H, W = 1424, 176
CUTS = (0.025, 0.05, 0.1)
NUS = (-1.5, -0.5, 0.5, 1.5)
DELTA_H = 0.15
_cache = {}


def _setup(mask):
    dev = mask.device
    if dev in _cache:
        return _cache[dev]
    m = mask.float()
    fy = torch.fft.fftfreq(H, device=dev)[:, None].expand(H, W // 2 + 1)
    fx = torch.fft.rfftfreq(W, device=dev)[None, :].expand(H, W // 2 + 1)
    k = torch.sqrt(fy ** 2 + fx ** 2)
    sel = [(k < kc).float() for kc in CUTS]
    ops = dict(x=2j * math.pi * fx, y=2j * math.pi * fy)
    ops['xx'], ops['yy'], ops['xy'] = ops['x'] * ops['x'], ops['y'] * ops['y'], ops['x'] * ops['y']
    g = torch.exp(-2 * math.pi ** 2 * 8.0 ** 2 * k ** 2)
    inside = ((torch.fft.irfft2(torch.fft.rfft2(m) * g, s=(H, W)) > 0.85) & mask).flatten().nonzero().squeeze(1)
    _cache[dev] = (m, sel, ops, inside, torch.as_tensor(NUS, device=dev))
    return _cache[dev]


def features(maps, mask):
    m, sel, ops, inside, nus = _setup(mask)
    B = maps.shape[0]
    f = torch.fft.rfft2(maps * m)
    u = f / f.abs().clamp(min=1e-12)
    out = []
    for s in sel:
        us = u * s
        e = torch.fft.irfft2(us, s=(H, W)).reshape(B, -1)[:, inside]
        mu = e.mean(1, keepdim=True)
        sd = (e - mu).pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-12)
        e = (e - mu) / sd
        d = {n: torch.fft.irfft2(us * op, s=(H, W)).reshape(B, -1)[:, inside] / sd for n, op in ops.items()}
        g2 = d['x'] ** 2 + d['y'] ** 2
        gs = g2.mean(1, keepdim=True)
        curv = (2 * d['x'] * d['y'] * d['xy'] - d['x'] ** 2 * d['yy'] - d['y'] ** 2 * d['xx']) / (g2 + 1e-3 * gs)
        gn = g2.sqrt() / gs.sqrt()
        cn = curv / gs.sqrt()
        for nu in NUS:
            dl = torch.exp(-0.5 * ((e - nu) / DELTA_H) ** 2) / (DELTA_H * math.sqrt(2 * math.pi))
            out += [(e > nu).float().mean(1), (gn * dl).mean(1), (cn * dl).mean(1)]
    return torch.stack(out, 1)
