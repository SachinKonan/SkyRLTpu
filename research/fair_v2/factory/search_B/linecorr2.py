"""Line correlation of the phase-only field: scale-adaptive LCF plus fixed-cutoff lines (amplitude-free).

eps = irfft2(F/|F| * [|k| < kc]) of the footprint-masked map, unit rms on the interior; l(r) = mean over
interior pixels and 4 directions of eps(x - r) eps(x) eps(x + r).
- Scale-adaptive LCF (Obreschkow et al. 2013): kc = 1/r cycles/px for r = 2, 3, 4, 6, 8, 12, 16, 24, 32 px.
- Fixed cutoffs kc = 0.03, 0.08 at r = 2, 4, 8, 16, 24 px.
Gaussian phases give 0; gravity aligns phases along filaments and around halos with a fixed scale dependence.
Isotropic Fourier filters (baryonic or PSF-like power changes) and rescalings leave it unchanged.
"""
import math

import torch

H, W = 1424, 176
ADAPT = (2, 3, 4, 6, 8, 12, 16, 24, 32)
FIXED = ((0.03, (2, 4, 8, 16, 24)), (0.08, (2, 4, 8, 16, 24)))
DIRS = ((1, 0), (0, 1), (1, 1), (1, -1))
_cache = {}


def _setup(mask):
    dev = mask.device
    if dev in _cache:
        return _cache[dev]
    m = mask.float()
    fy = torch.fft.fftfreq(H, device=dev)[:, None]
    fx = torch.fft.rfftfreq(W, device=dev)[None, :]
    k = torch.sqrt(fy ** 2 + fx ** 2)
    g = torch.exp(-2 * math.pi ** 2 * 8.0 ** 2 * k ** 2)
    inner = (torch.fft.irfft2(torch.fft.rfft2(m) * g, s=(H, W)) > 0.85) & mask
    radii = sorted(set(ADAPT) | {r for _, rs in FIXED for r in rs})
    geo = {}
    for rad in radii:
        lst = []
        for dy, dx in DIRS:
            n = math.hypot(dy, dx)
            sy, sx = int(round(rad * dy / n)), int(round(rad * dx / n))
            v = inner & torch.roll(inner, (sy, sx), (0, 1)) & torch.roll(inner, (-sy, -sx), (0, 1))
            lst.append((sy, sx, v.float() / v.float().sum().clamp(min=1.0)))
        geo[rad] = lst
    jobs = [(1.0 / r, (r,)) for r in ADAPT] + list(FIXED)
    sels = [((k < kc).float(), rs) for kc, rs in jobs]
    _cache[dev] = (m, inner.float(), geo, sels)
    return _cache[dev]


def features(maps, mask):
    m, ii, geo, sels = _setup(mask)
    f = torch.fft.rfft2(maps * m)
    u = f / f.abs().clamp(min=1e-12)
    n_in = ii.sum()
    out = []
    for s, rs in sels:
        e = torch.fft.irfft2(u * s, s=(H, W))
        mu = (e * ii).sum((1, 2), keepdim=True) / n_in
        e = (e - mu) * ii
        e = e / ((e * e).sum((1, 2), keepdim=True) / n_in).sqrt().clamp(min=1e-12)
        for rad in rs:
            acc = 0.0
            for sy, sx, v in geo[rad]:
                acc = acc + (e * torch.roll(e, (sy, sx), (1, 2)) * torch.roll(e, (-sy, -sx), (1, 2)) * v).sum((1, 2))
            out.append(acc / len(DIRS))
    return torch.stack(out, 1)
