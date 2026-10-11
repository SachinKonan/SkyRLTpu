"""Power redistribution under Gaussianization (non-Gaussian structure vs scale, amplitude-free).

The 1.5-px smoothed map (mask-normalized) is rank-transformed to a standard Gaussian on the interior pixels.
Features are, in 10 log bands of |k|, log power of the Gaussianized map minus log power of the standardized
map. A Gaussian field gives ~0; peaks and voids of non-linear structure move power between scales in a way
fixed by gravity. Monotone rescalings of the map leave it unchanged.
"""
import math

import torch

H, W = 1424, 176
EDGES = torch.logspace(math.log10(0.004), math.log10(0.25), 11).tolist()
_cache = {}


def _setup(mask):
    dev = mask.device
    if dev in _cache:
        return _cache[dev]
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
    nb = len(EDGES) - 1
    band = torch.bucketize(k.flatten(), torch.as_tensor(EDGES, device=dev)) - 1
    ok = (band >= 0) & (band < nb)
    onehot = torch.zeros((k.numel(), nb), device=dev)
    onehot[ok.nonzero().squeeze(1), band[ok]] = 1.0
    onehot = onehot / onehot.sum(0, keepdim=True)
    _cache[dev] = (m, g, sm, inside, onehot)
    return _cache[dev]


def features(maps, mask):
    m, g, sm, inside, onehot = _setup(mask)
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
