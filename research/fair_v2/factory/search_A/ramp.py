"""Large-scale mode residuals: masked least-squares fit of each map to low-order polynomial modes.

Basis (orthonormalized over the footprint): Legendre P1..P4 along the long axis, P1..P2 across, and the
y*x cross term. Signal large-scale modes are random with a cosmology-dependent variance; an additive
large-scale systematic (gradient, plane-like pattern) shifts specific coefficients deterministically.
Features: the signed coefficients in units of the per-pixel noise, plus the log energy in the
long-axis modes, the cross-axis modes, and the low-pass field's residual variance after removing them.
"""
import math

import torch
import torch.nn.functional as F

H, W = 1424, 176
SIG = 0.4 / math.sqrt(240)
_cache = {}


def _legendre(t, n):
    p = [torch.ones_like(t), t]
    for k in range(1, n):
        p.append(((2 * k + 1) * t * p[k] - k * p[k - 1]) / (k + 1))
    return p[1:n + 1]


def _setup(mask):
    if mask.device in _cache:
        return _cache[mask.device]
    dev = mask.device
    yy = torch.linspace(-1, 1, H, device=dev, dtype=torch.float64)[:, None].expand(H, W)
    xx = torch.linspace(-1, 1, W, device=dev, dtype=torch.float64)[None, :].expand(H, W)
    sel = mask.flatten().nonzero()[:, 0]
    py = _legendre(yy, 4)
    px = _legendre(xx, 2)
    B = torch.stack([torch.ones_like(yy)] + py + px + [yy * xx], -1).reshape(-1, 8)[sel]
    Q, _ = torch.linalg.qr(B)                                   # orthonormal over the footprint
    # low-pass (sigma 16 px) for the residual large-scale variance
    r = torch.sqrt(torch.fft.fftfreq(H, device=dev)[:, None] ** 2 + torch.fft.rfftfreq(W, device=dev)[None, :] ** 2)
    g = torch.exp(-2 * math.pi ** 2 * 16.0 ** 2 * r ** 2).float()
    m = mask.float()
    ms = torch.fft.irfft2(torch.fft.rfft2(m) * g, s=(H, W))
    inner = ((ms > 0.9) & mask).flatten()[sel]
    _cache[dev] = (sel, Q.float(), g, inner)
    return _cache[dev]


def features(maps, mask):
    sel, Q, g, inner = _setup(mask)
    x = maps.reshape(len(maps), -1)[:, sel]
    c = x @ Q                                                    # (B, 8) projections, c[:, 0] = mean mode
    lp = torch.fft.irfft2(torch.fft.rfft2(maps) * g, s=(H, W)).reshape(len(maps), -1)[:, sel]
    res = lp - c @ Q.T
    rv = (res[:, inner] ** 2).mean(1)
    ey = (c[:, 1:5] ** 2).sum(1)
    ex = (c[:, 5:8] ** 2).sum(1)
    return torch.cat([c / SIG, torch.log(ey + SIG ** 2)[:, None], torch.log(ex + SIG ** 2)[:, None],
                      torch.log(rv + 1e-12)[:, None]], 1)
