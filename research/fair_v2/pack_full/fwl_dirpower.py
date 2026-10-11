"""New-idea example: anisotropy of the power spectrum (PSF / shear-systematics probe).

For 12 log |k| bands: log power in |kx| > |ky| wedges minus log power in |ky| > |kx| wedges, plus the
log power in thin wedges along each axis relative to the band. A statistically isotropic map gives ~0 up to
the mask's own anisotropy, which is fixed and absorbed by the normal model.
"""
import numpy as np
import torch

H, W = 1424, 176
EDGES = np.geomspace(0.006, 0.7, 13)
_cache = {}


def _setup(mask):
    if mask.device in _cache:
        return _cache[mask.device]
    dev = mask.device
    fy = torch.fft.fftfreq(H, device=dev)[:, None].expand(H, W // 2 + 1)
    fx = torch.fft.rfftfreq(W, device=dev)[None, :].expand(H, W // 2 + 1)
    r = torch.sqrt(fy ** 2 + fx ** 2)
    nb = len(EDGES) - 1
    band = torch.bucketize(r, torch.as_tensor(EDGES, device=dev, dtype=torch.float32)) - 1
    ok = (band >= 0) & (band < nb)
    ang = torch.atan2(fy.abs(), fx.abs())                 # 0 = along x, pi/2 = along y
    groups = [ang < np.pi / 4, ang >= np.pi / 4, ang < np.pi / 12, ang > 5 * np.pi / 12]
    sel = []
    for g in [torch.ones_like(ok)] + groups:
        idx = torch.where(ok & g, band, torch.full_like(band, nb))
        cnt = torch.bincount(idx.flatten(), minlength=nb + 1)[:nb].clamp(min=1).float()
        sel.append((idx.flatten(), cnt))
    _cache[dev] = (mask.float(), sel, nb)
    return _cache[dev]


def features(maps, mask):
    m, sel, nb = _setup(mask)
    f = torch.fft.rfft2(maps * m)
    p = ((f.real ** 2 + f.imag ** 2)).reshape(len(maps), -1)
    lp = []
    for idx, cnt in sel:
        s = torch.zeros((len(maps), nb + 1), device=maps.device).index_add_(1, idx, p)[:, :nb] / cnt
        lp.append(torch.log(s + 1e-30))
    tot, xw, yw, xa, ya = lp
    return torch.cat([xw - yw, xa - tot, ya - tot], 1)
