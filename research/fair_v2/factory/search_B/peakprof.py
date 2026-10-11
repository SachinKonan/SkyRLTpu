"""Shape of stacked peak and trough profiles (amplitude-free).

Peaks/troughs are local extrema (5x5) of the map smoothed at 2 px, above nu x its own rms on the deep interior.
Around them the 1-px-smoothed map is ring-averaged at radii 1..12 px; features are the stacked ring means
divided by the stacked central value, plus the stacked central height in rms units. Halo profiles (baryons),
PSF/resolution and additive patterns change profile shapes; a rescaling of the map does not.
"""
import math

import torch
import torch.nn.functional as F

H, W = 1424, 176
RADII = (1, 2, 3, 4, 6, 8, 12)
NU = ((1.0, 2.0), (1.0, 3.0), (-1.0, 1.5), (-1.0, 2.5))   # (sign, threshold in rms units)
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
    g1, g2 = _gauss(1.0, r2), _gauss(2.0, r2)
    s1 = torch.fft.irfft2(mk * g1, s=(H, W))
    s2 = torch.fft.irfft2(mk * g2, s=(H, W))
    g8 = _gauss(8.0, r2)
    s8 = torch.fft.irfft2(mk * g8, s=(H, W))
    inside = (s8 > 0.9) & mask
    yy = torch.minimum(torch.arange(H, device=dev), H - torch.arange(H, device=dev))[:, None].float()
    xx = torch.minimum(torch.arange(W, device=dev), W - torch.arange(W, device=dev))[None, :].float()
    rr = torch.sqrt(yy ** 2 + xx ** 2)
    rings = []
    for r in RADII:
        k = ((rr >= r - 0.5) & (rr < r + 0.5)).float()
        rings.append(torch.fft.rfft2(k / k.sum()))
    cover = [torch.fft.irfft2(mk * k, s=(H, W)).clamp(min=0.2) for k in rings]   # ring coverage of the footprint
    _cache[dev] = (m, g1, g2, s1, s2, inside, rings, cover)
    return _cache[dev]


def features(maps, mask):
    m, g1, g2, s1, s2, inside, rings, cover = _setup(mask)
    B = maps.shape[0]
    x = maps * m
    f = torch.fft.rfft2(x)
    sm2 = torch.fft.irfft2(f * g2, s=(H, W)) / s2.clamp(min=0.05) * m
    sm1 = torch.fft.irfft2(f * g1, s=(H, W)) / s1.clamp(min=0.05) * m
    f1 = torch.fft.rfft2(sm1)
    R = [sm1] + [torch.fft.irfft2(f1 * k, s=(H, W)) / c for k, c in zip(rings, cover)]   # footprint-normalized ring means
    vals = sm2[:, inside]
    mu = vals.mean(1, keepdim=True)
    rms = (vals - mu).pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-12)
    Rv = [r[:, inside] for r in R]
    out = []
    for sign, nu in NU:
        g = sign * sm2
        mx = F.max_pool2d(g[:, None], 5, 1, 2)[:, 0]
        ext = (g == mx)[:, inside] & ((sign * (vals - mu)) > nu * rms)
        w = ext.float()
        n = w.sum(1).clamp(min=1.0)
        prof = [(w * (sign * (r - mu))).sum(1) / n for r in Rv]
        c = prof[0].clamp(min=1e-6)
        out += [p / c for p in prof[1:]] + [torch.log(c / rms[:, 0])]
    return torch.stack(out, 1)
