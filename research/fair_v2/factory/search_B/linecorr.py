"""Line correlation function of the phase-only field (filamentary phase coupling, amplitude-free).

eps = irfft2(F/|F| * [|k| < kc]) of the footprint-masked map, unit rms on the interior. For displacement
vectors r (8 directions, |r| = 2..24 px) the three-point product eps(x - r) eps(x) eps(x + r) is averaged
over interior pixels and directions (Obreschkow et al. 2013 line correlation). Gaussian phases give 0;
gravity aligns phases along filaments and around halos, with a fixed scale dependence. Isotropic Fourier
filters and rescalings leave it unchanged.
"""
import math

import torch

H, W = 1424, 176
CUTS = (0.03, 0.08)
RADII = (2, 4, 8, 16, 24)
DIRS = ((1, 0), (0, 1), (1, 1), (1, -1))           # +-r covers the opposite directions
_cache = {}


def _setup(mask):
    dev = mask.device
    if dev in _cache:
        return _cache[dev]
    m = mask.float()
    fy = torch.fft.fftfreq(H, device=dev)[:, None]
    fx = torch.fft.rfftfreq(W, device=dev)[None, :]
    k = torch.sqrt(fy ** 2 + fx ** 2)
    sel = [(k < kc).float() for kc in CUTS]
    g = torch.exp(-2 * math.pi ** 2 * 8.0 ** 2 * k ** 2)
    inner = (torch.fft.irfft2(torch.fft.rfft2(m) * g, s=(H, W)) > 0.85) & mask
    shifts, valid = [], []
    for rad in RADII:
        for dy, dx in DIRS:
            n = math.hypot(dy, dx)
            sy, sx = int(round(rad * dy / n)), int(round(rad * dx / n))
            v = inner & torch.roll(inner, (sy, sx), (0, 1)) & torch.roll(inner, (-sy, -sx), (0, 1))
            shifts.append((sy, sx))
            valid.append(v.float() / v.float().sum().clamp(min=1.0))
    ii = inner.float()
    _cache[dev] = (m, sel, ii, shifts, valid)
    return _cache[dev]


def features(maps, mask):
    m, sel, ii, shifts, valid = _setup(mask)
    f = torch.fft.rfft2(maps * m)
    u = f / f.abs().clamp(min=1e-12)
    nd = len(DIRS)
    out = []
    for s in sel:
        e = torch.fft.irfft2(u * s, s=(H, W))
        mu = (e * ii).sum((1, 2), keepdim=True) / ii.sum()
        e = (e - mu) * ii
        e = e / ((e * e).sum((1, 2), keepdim=True) / ii.sum()).sqrt().clamp(min=1e-12)
        for j in range(len(RADII)):
            acc = 0.0
            for d in range(nd):
                sy, sx = shifts[j * nd + d]
                acc = acc + (e * torch.roll(e, (sy, sx), (1, 2)) * torch.roll(e, (-sy, -sx), (1, 2))
                             * valid[j * nd + d]).sum((1, 2))
            out.append(acc / nd)
    return torch.stack(out, 1)
