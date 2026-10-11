"""Phase-only family: line correlation (linecorr2) + phase-only band statistics (phasew), one joint vote.

Both are invariant to isotropic Fourier filters and rescalings of the map. See the single files.
"""
import math

import torch

H, W = 1424, 176



linecorr2_ADAPT = (2, 3, 4, 6, 8, 12, 16, 24, 32)
linecorr2_FIXED = ((0.03, (2, 4, 8, 16, 24)), (0.08, (2, 4, 8, 16, 24)))
linecorr2_DIRS = ((1, 0), (0, 1), (1, 1), (1, -1))
linecorr2__cache = {}


def linecorr2__setup(mask):
    dev = mask.device
    if dev in linecorr2__cache:
        return linecorr2__cache[dev]
    m = mask.float()
    fy = torch.fft.fftfreq(H, device=dev)[:, None]
    fx = torch.fft.rfftfreq(W, device=dev)[None, :]
    k = torch.sqrt(fy ** 2 + fx ** 2)
    g = torch.exp(-2 * math.pi ** 2 * 8.0 ** 2 * k ** 2)
    inner = (torch.fft.irfft2(torch.fft.rfft2(m) * g, s=(H, W)) > 0.85) & mask
    radii = sorted(set(linecorr2_ADAPT) | {r for _, rs in linecorr2_FIXED for r in rs})
    geo = {}
    for rad in radii:
        lst = []
        for dy, dx in linecorr2_DIRS:
            n = math.hypot(dy, dx)
            sy, sx = int(round(rad * dy / n)), int(round(rad * dx / n))
            v = inner & torch.roll(inner, (sy, sx), (0, 1)) & torch.roll(inner, (-sy, -sx), (0, 1))
            lst.append((sy, sx, v.float() / v.float().sum().clamp(min=1.0)))
        geo[rad] = lst
    jobs = [(1.0 / r, (r,)) for r in linecorr2_ADAPT] + list(linecorr2_FIXED)
    sels = [((k < kc).float(), rs) for kc, rs in jobs]
    linecorr2__cache[dev] = (m, inner.float(), geo, sels)
    return linecorr2__cache[dev]


def linecorr2_features(maps, mask):
    m, ii, geo, sels = linecorr2__setup(mask)
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
            out.append(acc / len(linecorr2_DIRS))
    return torch.stack(out, 1)




phasew_BANDS = ((0.004, 0.015), (0.015, 0.03), (0.03, 0.06), (0.06, 0.12))
phasew_PAIRS = ((0, 1), (1, 2), (2, 3))
phasew__cache = {}


def phasew__setup(mask):
    dev = mask.device
    if dev in phasew__cache:
        return phasew__cache[dev]
    m = mask.float()
    fy = torch.fft.fftfreq(H, device=dev)[:, None]
    fx = torch.fft.rfftfreq(W, device=dev)[None, :]
    k = torch.sqrt(fy ** 2 + fx ** 2)
    sel = [((k >= lo) & (k < hi)).float() for lo, hi in phasew_BANDS]
    g = torch.exp(-2 * math.pi ** 2 * 8.0 ** 2 * k ** 2)
    sm = torch.fft.irfft2(torch.fft.rfft2(m) * g, s=(H, W))
    inside = ((sm > 0.85) & mask).flatten().nonzero().squeeze(1)
    phasew__cache[dev] = (m, sel, inside)
    return phasew__cache[dev]


def phasew_features(maps, mask):
    m, sel, inside = phasew__setup(mask)
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
    for a, b in phasew_PAIRS:
        ya, yb = ys[a], ys[b]
        ea, eb = ya * ya, yb * yb
        ac, bc = ea - ea.mean(1, keepdim=True), eb - eb.mean(1, keepdim=True)
        out += [(ac * bc).mean(1) / (ac.pow(2).mean(1) * bc.pow(2).mean(1)).sqrt().clamp(min=1e-12),
                (ea * yb).mean(1), (ya * eb).mean(1)]
    return torch.stack(out, 1)


def features(maps, mask):
    return torch.cat([linecorr2_features(maps, mask), phasew_features(maps, mask)], 1)
