"""Cross-scale coupling of amplitude-normalized band-passed fields (non-Gaussian, amplitude-free).

Bands are differences of masked Gaussian smoothings at 0,1,2,4,8,16 px. Each band is normalized to unit rms
on the deep-interior pixels, so a pure rescaling or an isotropic Fourier filter of the map leaves the features
unchanged (deep interior = 16-px smoothed footprint > 0.85). For pairs of bands: correlation of local energies corr(y_a^2, y_b^2) (squeezed-trispectrum-like
amplitude coupling) and the cross third moments <y_a^2 y_b>, <y_a y_b^2> (band-space reduced bispectrum).
A Gaussian field gives ~0 for all; nonlinear structure fixes them as a function of cosmology.
"""
import math

import torch

H, W = 1424, 176
SCALES = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0)
PAIRS = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 2), (1, 3), (2, 4))
_cache = {}


def _setup(mask):
    dev = mask.device
    if dev in _cache:
        return _cache[dev]
    m = mask.float()
    fy = torch.fft.fftfreq(H, device=dev)[:, None]
    fx = torch.fft.rfftfreq(W, device=dev)[None, :]
    r2 = fy ** 2 + fx ** 2
    kern = [torch.exp(-2 * math.pi ** 2 * s * s * r2) for s in SCALES]
    mk = torch.fft.rfft2(m)
    smask = [torch.fft.irfft2(mk * g, s=(H, W)) for g in kern]
    inside = (smask[-1] > 0.85) & mask
    idx = inside.flatten().nonzero().squeeze(1)
    _cache[dev] = (m, kern, smask, idx)
    return _cache[dev]


def features(maps, mask):
    m, kern, smask, idx = _setup(mask)
    B = maps.shape[0]
    x = maps * m
    f = torch.fft.rfft2(x)
    fields = []
    for i, s in enumerate(SCALES):
        g = x if s == 0 else torch.fft.irfft2(f * kern[i], s=(H, W)) / smask[i].clamp(min=0.05)
        fields.append(g.reshape(B, -1)[:, idx])
    ys = []
    for j in range(len(SCALES) - 1):
        w = fields[j] - fields[j + 1]
        w = w - w.mean(1, keepdim=True)
        ys.append(w / w.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-12))
    out = []
    for a, b in PAIRS:
        ya, yb = ys[a], ys[b]
        ea, eb = ya * ya, yb * yb
        ac, bc = ea - ea.mean(1, keepdim=True), eb - eb.mean(1, keepdim=True)
        c2 = (ac * bc).mean(1) / (ac.pow(2).mean(1) * bc.pow(2).mean(1)).sqrt().clamp(min=1e-12)
        out += [c2, (ea * yb).mean(1), (ya * eb).mean(1)]
    return torch.stack(out, 1)
