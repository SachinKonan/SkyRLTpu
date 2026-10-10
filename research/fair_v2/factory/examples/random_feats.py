"""Deliberately useless: 16 Gaussian numbers seeded by a hash of the map (per-map, deterministic, no signal)."""
import torch


def features(maps, mask):
    out = []
    for x in maps:
        seed = int(x[mask].double().abs().sum().item() * 1e6) % (2 ** 31)
        g = torch.Generator()
        g.manual_seed(seed)
        out.append(torch.randn(16, generator=g))
    return torch.stack(out)
