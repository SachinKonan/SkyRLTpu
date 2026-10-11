"""Gaussian-mode statistics (modestat) + phase-only line correlation (linecorr), one family.

modestat helped as a misfit (tight on normal maps), linecorr as a parameter vote; both are amplitude-free.
"""
import math

import torch

H, W = 1424, 176



modestat_EDGES = torch.logspace(math.log10(0.003), math.log10(0.5), 13).tolist()
modestat__cache = {}


def modestat__setup(mask):
    dev = mask.device
    if dev in modestat__cache:
        return modestat__cache[dev]
    m = mask.float()
    fy = torch.fft.fftfreq(H, device=dev)[:, None]
    fx = torch.fft.rfftfreq(W, device=dev)[None, :]
    k = torch.sqrt(fy ** 2 + fx ** 2).flatten()
    nb = len(modestat_EDGES) - 1
    band = torch.bucketize(k, torch.as_tensor(modestat_EDGES, device=dev)) - 1
    keep = ((band >= 0) & (band < nb)).nonzero().squeeze(1)
    band = band[keep]
    cnt = torch.bincount(band, minlength=nb).float()
    modestat__cache[dev] = (m, keep, band, cnt, nb)
    return modestat__cache[dev]


def modestat_features(maps, mask):
    m, keep, band, cnt, nb = modestat__setup(mask)
    B = maps.shape[0]
    f = torch.fft.rfft2(maps * m)
    p = (f.real ** 2 + f.imag ** 2).reshape(B, -1)[:, keep]
    idx = band[None, :].expand(B, -1)
    def bmean(v):
        return torch.zeros((B, nb), device=maps.device).scatter_add_(1, idx, v) / cnt
    mu = bmean(p).clamp(min=1e-30)
    q = p / mu.gather(1, idx)
    return torch.cat([torch.log(bmean(q * q)), torch.log(bmean(q ** 3)), bmean((q > 6).float())], 1)




linecorr_CUTS = (0.03, 0.08)
linecorr_RADII = (2, 4, 8, 16, 24)
linecorr_DIRS = ((1, 0), (0, 1), (1, 1), (1, -1))           # +-r covers the opposite directions
linecorr__cache = {}


def linecorr__setup(mask):
    dev = mask.device
    if dev in linecorr__cache:
        return linecorr__cache[dev]
    m = mask.float()
    fy = torch.fft.fftfreq(H, device=dev)[:, None]
    fx = torch.fft.rfftfreq(W, device=dev)[None, :]
    k = torch.sqrt(fy ** 2 + fx ** 2)
    sel = [(k < kc).float() for kc in linecorr_CUTS]
    g = torch.exp(-2 * math.pi ** 2 * 8.0 ** 2 * k ** 2)
    inner = (torch.fft.irfft2(torch.fft.rfft2(m) * g, s=(H, W)) > 0.85) & mask
    shifts, valid = [], []
    for rad in linecorr_RADII:
        for dy, dx in linecorr_DIRS:
            n = math.hypot(dy, dx)
            sy, sx = int(round(rad * dy / n)), int(round(rad * dx / n))
            v = inner & torch.roll(inner, (sy, sx), (0, 1)) & torch.roll(inner, (-sy, -sx), (0, 1))
            shifts.append((sy, sx))
            valid.append(v.float() / v.float().sum().clamp(min=1.0))
    ii = inner.float()
    linecorr__cache[dev] = (m, sel, ii, shifts, valid)
    return linecorr__cache[dev]


def linecorr_features(maps, mask):
    m, sel, ii, shifts, valid = linecorr__setup(mask)
    f = torch.fft.rfft2(maps * m)
    u = f / f.abs().clamp(min=1e-12)
    nd = len(linecorr_DIRS)
    out = []
    for s in sel:
        e = torch.fft.irfft2(u * s, s=(H, W))
        mu = (e * ii).sum((1, 2), keepdim=True) / ii.sum()
        e = (e - mu) * ii
        e = e / ((e * e).sum((1, 2), keepdim=True) / ii.sum()).sqrt().clamp(min=1e-12)
        for j in range(len(linecorr_RADII)):
            acc = 0.0
            for d in range(nd):
                sy, sx = shifts[j * nd + d]
                acc = acc + (e * torch.roll(e, (sy, sx), (1, 2)) * torch.roll(e, (-sy, -sx), (1, 2))
                             * valid[j * nd + d]).sum((1, 2))
            out.append(acc / nd)
    return torch.stack(out, 1)


def features(maps, mask):
    return torch.cat([modestat_features(maps, mask), linecorr_features(maps, mask)], 1)
