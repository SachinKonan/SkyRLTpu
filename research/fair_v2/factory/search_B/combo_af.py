"""All amplitude-free views in one family (one joint parameter vote): modul, phasew, xcoup, hier, gaussz.

Each alone gave a ridge vote that gains on bary9.0 / supp0.15 / psfpat (power-shape shifts move the power-based
votes but not these); together the vote should be sharper. See the single-family files for definitions.
"""
import math

import torch
import torch.nn.functional as F

H, W = 1424, 176



modul_SMALL = (0.0, 1.0, 2.0, 4.0, 8.0)
modul_LARGE = (8.0, 16.0)
modul__cache = {}


def modul__gauss(s, r2):
    return torch.exp(-2 * math.pi ** 2 * s * s * r2)


def modul__setup(mask):
    dev = mask.device
    if dev in modul__cache:
        return modul__cache[dev]
    m = mask.float()
    fy = torch.fft.fftfreq(H, device=dev)[:, None]
    fx = torch.fft.rfftfreq(W, device=dev)[None, :]
    r2 = fy ** 2 + fx ** 2
    mk = torch.fft.rfft2(m)
    ks = {s: modul__gauss(s, r2) for s in set(modul_SMALL) | set(modul_LARGE) if s > 0}
    sm = {s: torch.fft.irfft2(mk * k, s=(H, W)) for s, k in ks.items()}
    inside = (sm[16.0] > 0.85) & mask
    idx = inside.flatten().nonzero().squeeze(1)
    modul__cache[dev] = (m, ks, sm, idx)
    return modul__cache[dev]


def modul__smooth(x, s, ks, sm, m):
    return torch.fft.irfft2(torch.fft.rfft2(x * m) * ks[s], s=(H, W)) / sm[s].clamp(min=0.05)


def modul_features(maps, mask):
    m, ks, sm, idx = modul__setup(mask)
    B = maps.shape[0]
    x = maps * m
    low = {s: modul__smooth(x, s, ks, sm, m) for s in modul_SMALL if s > 0}
    low[0.0] = x
    bands = [(low[modul_SMALL[j]] - low[modul_SMALL[j + 1]]) * m for j in range(len(modul_SMALL) - 1)]
    out = []
    for s in modul_LARGE:
        L = modul__smooth(x, s, ks, sm, m).reshape(B, -1)[:, idx]
        L = L - L.mean(1, keepdim=True)
        L = L / L.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-12)
        for w in bands:
            E = modul__smooth(w * w, s, ks, sm, m).reshape(B, -1)[:, idx]
            Em = E.mean(1, keepdim=True)
            Ec = E - Em
            cov = (L * Ec).mean(1)
            out += [cov / Ec.pow(2).mean(1).sqrt().clamp(min=1e-20), cov / Em[:, 0].clamp(min=1e-20)]
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




xcoup_SCALES = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0)
xcoup_PAIRS = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 2), (1, 3), (2, 4))
xcoup__cache = {}


def xcoup__setup(mask):
    dev = mask.device
    if dev in xcoup__cache:
        return xcoup__cache[dev]
    m = mask.float()
    fy = torch.fft.fftfreq(H, device=dev)[:, None]
    fx = torch.fft.rfftfreq(W, device=dev)[None, :]
    r2 = fy ** 2 + fx ** 2
    kern = [torch.exp(-2 * math.pi ** 2 * s * s * r2) for s in xcoup_SCALES]
    mk = torch.fft.rfft2(m)
    smask = [torch.fft.irfft2(mk * g, s=(H, W)) for g in kern]
    inside = (smask[-1] > 0.85) & mask
    idx = inside.flatten().nonzero().squeeze(1)
    xcoup__cache[dev] = (m, kern, smask, idx)
    return xcoup__cache[dev]


def xcoup_features(maps, mask):
    m, kern, smask, idx = xcoup__setup(mask)
    B = maps.shape[0]
    x = maps * m
    f = torch.fft.rfft2(x)
    fields = []
    for i, s in enumerate(xcoup_SCALES):
        g = x if s == 0 else torch.fft.irfft2(f * kern[i], s=(H, W)) / smask[i].clamp(min=0.05)
        fields.append(g.reshape(B, -1)[:, idx])
    ys = []
    for j in range(len(xcoup_SCALES) - 1):
        w = fields[j] - fields[j + 1]
        w = w - w.mean(1, keepdim=True)
        ys.append(w / w.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-12))
    out = []
    for a, b in xcoup_PAIRS:
        ya, yb = ys[a], ys[b]
        ea, eb = ya * ya, yb * yb
        ac, bc = ea - ea.mean(1, keepdim=True), eb - eb.mean(1, keepdim=True)
        c2 = (ac * bc).mean(1) / (ac.pow(2).mean(1) * bc.pow(2).mean(1)).sqrt().clamp(min=1e-12)
        out += [c2, (ea * yb).mean(1), (ya * eb).mean(1)]
    return torch.stack(out, 1)




hier_SCALES = (1.0, 2.0, 4.0, 8.0, 16.0)
hier_SIGMA_N = 0.4 / math.sqrt(240.0)
hier__cache = {}


def hier__setup(mask):
    dev = mask.device
    if dev in hier__cache:
        return hier__cache[dev]
    m = mask.float()
    fy = torch.fft.fftfreq(H, device=dev)[:, None]
    fx = torch.fft.rfftfreq(W, device=dev)[None, :]
    r2 = fy ** 2 + fx ** 2
    fxf = torch.fft.fftfreq(W, device=dev)[None, :]
    r2f = fy ** 2 + fxf ** 2
    mk = torch.fft.rfft2(m)
    ks, sms, nv = [], [], []
    for s in hier_SCALES:
        k = torch.exp(-2 * math.pi ** 2 * s * s * r2)
        ks.append(k)
        sms.append(torch.fft.irfft2(mk * k, s=(H, W)))
        nv.append(float(hier_SIGMA_N ** 2 * torch.exp(-4 * math.pi ** 2 * s * s * r2f).mean()))
    inside = [(sm > 0.9) & mask for sm in sms]
    hier__cache[dev] = (m, ks, sms, nv, inside)
    return hier__cache[dev]


def hier_features(maps, mask):
    m, ks, sms, nv, inside = hier__setup(mask)
    f = torch.fft.rfft2(maps * m)
    out = []
    for k, sm, n2, ins in zip(ks, sms, nv, inside):
        d = (torch.fft.irfft2(f * k, s=(H, W)) / sm.clamp(min=0.05))[:, ins]
        d = d - d.mean(1, keepdim=True)
        v = d.pow(2).mean(1)
        k3 = d.pow(3).mean(1)
        k4 = d.pow(4).mean(1) - 3 * v * v
        s2 = torch.maximum(v - n2, 0.1 * v)
        out += [k3 / s2 ** 1.5, torch.asinh(k4 / s2 ** 2)]
    return torch.stack(out, 1)




gaussz_EDGES = torch.logspace(math.log10(0.004), math.log10(0.25), 11).tolist()
gaussz__cache = {}


def gaussz__setup(mask):
    dev = mask.device
    if dev in gaussz__cache:
        return gaussz__cache[dev]
    m = mask.float()
    fy = torch.fft.fftfreq(H, device=dev)[:, None]
    fx = torch.fft.rfftfreq(W, device=dev)[None, :]
    k = torch.sqrt(fy ** 2 + fx ** 2)
    g = torch.exp(-2 * math.pi ** 2 * 1.5 ** 2 * k ** 2)
    g2 = torch.exp(-2 * math.pi ** 2 * 2.0 ** 2 * k ** 2)
    mk = torch.fft.rfft2(m)
    sm = torch.fft.irfft2(mk * g, s=(H, W))
    sm2 = torch.fft.irfft2(mk * g2, s=(H, W))
    inside = ((sm2 > 0.9) & mask).flatten().nonzero().squeeze(1)
    nb = len(gaussz_EDGES) - 1
    band = torch.bucketize(k.flatten(), torch.as_tensor(gaussz_EDGES, device=dev)) - 1
    ok = (band >= 0) & (band < nb)
    onehot = torch.zeros((k.numel(), nb), device=dev)
    onehot[ok.nonzero().squeeze(1), band[ok]] = 1.0
    onehot = onehot / onehot.sum(0, keepdim=True)
    gaussz__cache[dev] = (m, g, sm, inside, onehot)
    return gaussz__cache[dev]


def gaussz_features(maps, mask):
    m, g, sm, inside, onehot = gaussz__setup(mask)
    B = maps.shape[0]
    y = (torch.fft.irfft2(torch.fft.rfft2(maps * m) * g, s=(H, W)) / sm.clamp(min=0.05)).reshape(B, -1)[:, inside]
    y = y - y.mean(1, keepdim=True)
    y = y / y.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-12)
    n = y.shape[1]
    rank = torch.argsort(torch.argsort(y, dim=1), dim=1).float()
    z = math.sqrt(2.0) * torch.erfinv((2 * rank + 1) / n - 1)
    def power(v):
        full = torch.zeros((B, H * W), device=maps.device)
        full[:, inside] = v
        f = torch.fft.rfft2(full.reshape(B, H, W))
        return (f.real ** 2 + f.imag ** 2).reshape(B, -1) @ onehot
    return torch.log(power(z) + 1e-30) - torch.log(power(y) + 1e-30)


def features(maps, mask):
    return torch.cat([modul_features(maps, mask), phasew_features(maps, mask), xcoup_features(maps, mask), hier_features(maps, mask), gaussz_features(maps, mask)], 1)
