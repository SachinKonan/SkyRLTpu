"""Phase-only (whitened) band fields: one-point and cross-band coupling statistics.

For annular bands of |k| (cycles/px) the footprint-masked map's Fourier modes are replaced by F/|F| (unit
amplitude, phase kept), transformed back and normalized to unit rms on the interior. A Gaussian random field
has independent uniform phases, so these fields are Gaussian; gravitational non-linearity couples phases
(skewness, kurtosis, and energy/third-moment coupling between bands). Any isotropic Fourier filter or rescaling
of the map leaves the phases, and so these features, unchanged; the footprint's own effect is constant.
"""
import math

import torch

H, W = 1424, 176
BANDS = ((0.004, 0.015), (0.015, 0.03), (0.03, 0.06), (0.06, 0.12))
PAIRS = ((0, 1), (1, 2), (2, 3))
_cache = {}


def _setup(mask):
    dev = mask.device
    if dev in _cache:
        return _cache[dev]
    m = mask.float()
    fy = torch.fft.fftfreq(H, device=dev)[:, None]
    fx = torch.fft.rfftfreq(W, device=dev)[None, :]
    k = torch.sqrt(fy ** 2 + fx ** 2)
    sel = [((k >= lo) & (k < hi)).float() for lo, hi in BANDS]
    g = torch.exp(-2 * math.pi ** 2 * 8.0 ** 2 * k ** 2)
    sm = torch.fft.irfft2(torch.fft.rfft2(m) * g, s=(H, W))
    inside = ((sm > 0.85) & mask).flatten().nonzero().squeeze(1)
    _cache[dev] = (m, sel, inside)
    return _cache[dev]


def features(maps, mask):
    m, sel, inside = _setup(mask)
    B = maps.shape[0]
    f = torch.fft.rfft2(maps * m)
    u = f / f.abs().clamp(min=1e-12)
    ys = []
    for s in sel:
        y = torch.fft.irfft2(u * s, s=(H, W)).reshape(B, -1)[:, inside]
        y = y - y.mean(1, keepdim=True)
        ys.append(y / y.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-12))
    out = []
    for y in ys:
        out += [y.pow(3).mean(1), torch.asinh(y.pow(4).mean(1) - 3)]
    for a, b in PAIRS:
        ya, yb = ys[a], ys[b]
        ea, eb = ya * ya, yb * yb
        ac, bc = ea - ea.mean(1, keepdim=True), eb - eb.mean(1, keepdim=True)
        out += [(ac * bc).mean(1) / (ac.pow(2).mean(1) * bc.pow(2).mean(1)).sqrt().clamp(min=1e-12),
                (ea * yb).mean(1), (ya * eb).mean(1)]
    return torch.stack(out, 1)
