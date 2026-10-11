"""Large-scale additive-mode detector with a power-spectrum-whitened (GLS-like) template fit.

Templates: Legendre P1..P4 along the long axis, P1..P2 across, the cross term, and cos/sin with 1-3
periods along the long axis, all restricted to the footprint. Map and templates are whitened by
1/sqrt(P_fid(k)), the mean power of normal noisy maps (frozen constants from fit maps), so the coefficient
fit downweights the cosmic-variance-dominated lowest modes and uses the footprint-edge steps an additive
pattern creates. Features: whitened coefficients (signed, unit variance under the fiducial model), the
plain least-squares long-axis coefficients, and log energies of both groups.
"""
import math

import numpy as np
import torch

H, W = 1424, 176
SIG = 0.4 / math.sqrt(240)
EDGES = [0.0, 0.000492, 0.000592, 0.000714, 0.00086, 0.001037, 0.001249, 0.001506, 0.001814, 0.002186,
         0.002635, 0.003175, 0.003826, 0.004611, 0.005557, 0.006696, 0.008069, 0.009724, 0.011719, 0.014122,
         0.017018, 0.020508, 0.024714, 0.029783, 0.035891, 0.043251, 0.052122, 0.062811, 0.075692, 0.091216,
         0.109922, 0.132466, 0.159632, 0.19237, 0.231822, 0.279365, 0.336659, 0.405702, 0.488904, 0.589171,
         0.71]
LOGP = [None, None, 8.8965, None, None, None, 8.7846, None, 8.4535, None, 8.1364, 8.0822, 7.9782, 7.8196,
        7.8947, 7.4942, 7.2634, 7.1377, 6.9174, 6.6765, 6.5429, 6.2985, 6.1066, 5.9288, 5.7586, 5.6024,
        5.4595, 5.3215, 5.1933, 5.0749, 4.9657, 4.8673, 4.7821, 4.7087, 4.6488, 4.602, 4.568, 4.5471,
        4.5341, 4.5245]
_cache = {}


def _legendre(t, n):
    p = [torch.ones_like(t), t]
    for k in range(1, n):
        p.append(((2 * k + 1) * t * p[k] - k * p[k - 1]) / (k + 1))
    return p[1:n + 1]


def _whitener(dev):
    centers = np.sqrt(np.maximum(np.array(EDGES[:-1]), 1e-4) * np.array(EDGES[1:]))
    ok = np.array([v is not None for v in LOGP])
    logp = np.interp(np.log(centers), np.log(centers[ok]), np.array([v for v in LOGP if v is not None]))
    r = np.sqrt(np.fft.fftfreq(H)[:, None] ** 2 + np.fft.rfftfreq(W)[None, :] ** 2)
    lp = np.interp(np.log(np.maximum(r, 1e-5)), np.log(centers), logp)
    return torch.as_tensor(np.exp(-0.5 * lp), device=dev, dtype=torch.float32)


def _setup(mask):
    if mask.device in _cache:
        return _cache[mask.device]
    dev = mask.device
    m = mask.float()
    yy = torch.linspace(-1, 1, H, device=dev)[:, None].expand(H, W)
    xx = torch.linspace(-1, 1, W, device=dev)[None, :].expand(H, W)
    t = math.pi * (yy + 1)
    T = _legendre(yy, 4) + _legendre(xx, 2) + [yy * xx]
    T += [f(j * t) for j in (1, 2, 3) for f in (torch.cos, torch.sin)]
    T = torch.stack([v * m for v in T])                                   # (14, H, W)
    F = _whitener(dev)
    Tw = torch.fft.irfft2(torch.fft.rfft2(T) * F, s=(H, W)).reshape(len(T), -1)
    G = Tw @ Tw.T                                                         # whitened Gram matrix
    Gi = torch.linalg.inv(G.double()).float()
    L = torch.linalg.cholesky(G.double()).float()                         # c^T G c = |L^T c|^2
    sel = mask.flatten().nonzero()[:, 0]
    B = torch.stack([torch.ones(H, W, device=dev)] + _legendre(yy, 4), -1).reshape(-1, 5)[sel].double()
    Q, _ = torch.linalg.qr(B)
    _cache[dev] = (F, Tw, Gi, L, sel, Q.float())
    return _cache[dev]


def features(maps, mask):
    F, Tw, Gi, L, sel, Q = _setup(mask)
    xw = torch.fft.irfft2(torch.fft.rfft2(maps) * F, s=(H, W)).reshape(len(maps), -1)
    c = (xw @ Tw.T) @ Gi                                                  # GLS coefficients (B, 14)
    u = c @ L                                                             # unit-variance coordinates
    ls = maps.reshape(len(maps), -1)[:, sel] @ Q[:, 1:]                   # plain LS long-axis modes (B, 4)
    e_poly = (u[:, :7] ** 2).sum(1)
    e_four = (u[:, 7:] ** 2).sum(1)
    return torch.cat([u, ls / SIG, torch.log(e_poly + 1e-3)[:, None], torch.log(e_four + 1e-3)[:, None],
                      torch.log((ls ** 2).sum(1) / SIG ** 2 + 1e-3)[:, None]], 1)
