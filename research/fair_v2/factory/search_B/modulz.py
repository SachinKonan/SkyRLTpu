"""Position-dependent small-scale power (modul2) with Fisher-z / log transforms for Gaussian, linear-in-theta features.

Large-scale fields L_s (masked Gaussian smoothing at s = 8, 16 px, unit rms on the deep interior) and local
energies E_{j,s} = smooth_s(w_j^2) of four small-scale bands w_j (x-G1, G1-G2, G2-G4, G4-G8). Features per
(j, s): corr(L_s, E_{j,s}) and the fractional response cov(L_s, E_{j,s}) / mean(E_{j,s}). Gravity fixes this
squeezed-bispectrum response; baryons, PSF and noise systematics change it in ways the 5 parameters do not.
"""
import math

import torch

H, W = 1424, 176
SMALL = (0.0, 1.0, 2.0, 4.0, 8.0)
LARGE = (4.0, 8.0, 16.0)
_cache = {}


def _gauss(s, r2):
    return torch.exp(-2 * math.pi ** 2 * s * s * r2)


def _setup(mask):
    dev = mask.device
    if dev in _cache:
        return _cache[dev]
    m = mask.float()
    fy = torch.fft.fftfreq(H, device=dev)[:, None]
    fx = torch.fft.rfftfreq(W, device=dev)[None, :]
    r2 = fy ** 2 + fx ** 2
    mk = torch.fft.rfft2(m)
    ks = {s: _gauss(s, r2) for s in set(SMALL) | set(LARGE) if s > 0}
    sm = {s: torch.fft.irfft2(mk * k, s=(H, W)) for s, k in ks.items()}
    inside = (sm[16.0] > 0.85) & mask
    idx = inside.flatten().nonzero().squeeze(1)
    _cache[dev] = (m, ks, sm, idx)
    return _cache[dev]


def _smooth(x, s, ks, sm, m):
    return torch.fft.irfft2(torch.fft.rfft2(x * m) * ks[s], s=(H, W)) / sm[s].clamp(min=0.05)


def features(maps, mask):
    m, ks, sm, idx = _setup(mask)
    B = maps.shape[0]
    x = maps * m
    low = {s: _smooth(x, s, ks, sm, m) for s in SMALL if s > 0}
    low[0.0] = x
    bands = [(low[SMALL[j]] - low[SMALL[j + 1]]) * m for j in range(len(SMALL) - 1)]
    out = []
    for s in LARGE:
        L = _smooth(x, s, ks, sm, m).reshape(B, -1)[:, idx]
        L = L - L.mean(1, keepdim=True)
        L = L / L.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-12)
        for j, w in enumerate(bands):
            if SMALL[j + 1] > s / 2:
                continue
            E = _smooth(w * w, s, ks, sm, m).reshape(B, -1)[:, idx]
            Em = E.mean(1, keepdim=True)
            Ec = E - Em
            cov = (L * Ec).mean(1)
            c = (cov / Ec.pow(2).mean(1).sqrt().clamp(min=1e-20)).clamp(-0.999, 0.999)
            out += [torch.atanh(c), cov / Em[:, 0].clamp(min=1e-20)]
    return torch.stack(out, 1)
