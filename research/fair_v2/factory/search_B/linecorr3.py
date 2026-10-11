"""Line correlation of the phase-only field, denser sampling (linecorr with 8 directions, 4 cutoffs, taper).

The footprint-masked map is tapered by its 4-px-smoothed footprint (softens the edge so its fixed phases weigh
less), whitened to eps = irfft2(F/|F| * [|k| < kc]) for kc = 0.03, 0.05, 0.08, 0.12 cycles/px and normalized to
unit rms on the interior. l(r) = mean over interior pixels and 8 directions of eps(x - r) eps(x) eps(x + r) at
r = 2, 4, 8, 16, 24 px. Gaussian phases give 0; gravity aligns phases along filaments and around halos.
Isotropic Fourier filters and rescalings of the map leave it unchanged.
"""
import math

import torch

H, W = 1424, 176
CUTS = (0.03, 0.05, 0.08, 0.12)
RADII = (2, 4, 8, 16, 24)
DIRS = ((1, 0), (0, 1), (1, 1), (1, -1), (2, 1), (1, 2), (2, -1), (1, -2))
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
    mk = torch.fft.rfft2(m)
    inner = (torch.fft.irfft2(mk * torch.exp(-2 * math.pi ** 2 * 64.0 * k ** 2), s=(H, W)) > 0.85) & mask
    taper = torch.fft.irfft2(mk * torch.exp(-2 * math.pi ** 2 * 16.0 * k ** 2), s=(H, W)).clamp(0, 1) * m
    geo = []
    for rad in RADII:
        lst = []
        for dy, dx in DIRS:
            n = math.hypot(dy, dx)
            sy, sx = int(round(rad * dy / n)), int(round(rad * dx / n))
            v = inner & torch.roll(inner, (sy, sx), (0, 1)) & torch.roll(inner, (-sy, -sx), (0, 1))
            lst.append((sy, sx, v.float() / v.float().sum().clamp(min=1.0)))
        geo.append(lst)
    _cache[dev] = (taper, sel, inner.float(), geo)
    return _cache[dev]


def features(maps, mask):
    taper, sel, ii, geo = _setup(mask)
    f = torch.fft.rfft2(maps * taper)
    u = f / f.abs().clamp(min=1e-12)
    n_in = ii.sum()
    out = []
    for s in sel:
        e = torch.fft.irfft2(u * s, s=(H, W))
        mu = (e * ii).sum((1, 2), keepdim=True) / n_in
        e = (e - mu) * ii
        e = e / ((e * e).sum((1, 2), keepdim=True) / n_in).sqrt().clamp(min=1e-12)
        for lst in geo:
            acc = 0.0
            for sy, sx, v in lst:
                acc = acc + (e * torch.roll(e, (sy, sx), (1, 2)) * torch.roll(e, (-sy, -sx), (1, 2)) * v).sum((1, 2))
            out.append(acc / len(lst))
    return torch.stack(out, 1)
