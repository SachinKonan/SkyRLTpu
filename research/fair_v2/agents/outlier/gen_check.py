"""Sanity check of the generator library: signal change per family, edge vs interior, timing."""
import sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
import numpy as np, torch
import data as D
from gens import Library, ALL_FAMILIES

dev = torch.device('cuda')
mask = D.mask(); m = torch.as_tensor(mask, device=dev, dtype=torch.float32)
lab = D.labels(); tr = np.load(D.TRAIN, mmap_mode='r')
idx = D.index('fit3')[:32]
full = torch.zeros((32, 1424, 176), device=dev)
full[:, m > 0] = torch.as_tensor(np.asarray(tr[idx[:, 0], idx[:, 1]]), device=dev).float()
labs = lab[idx[:, 0], idx[:, 1]]
lib = Library(m, dev)
# edge band: observed pixels within 3 px of the mask boundary
er = torch.nn.functional.max_pool2d((1 - m)[None, None], 7, 1, 3)[0, 0] * m
inner = m * (1 - er)
g = torch.Generator(device=dev); g.manual_seed(0)
# identity check: filter with T=1 must reproduce the map
y = lib.filt(full, torch.ones_like(lib.r)[None])
print('identity filt max err', float((y - full).abs().max()))
for fam in ALL_FAMILIES:
    torch.cuda.synchronize(); t = time.time()
    o = lib.fam[fam](full, labs, g)
    torch.cuda.synchronize(); dt = time.time() - t
    d = (o['sig'] - full)
    rel = (d.pow(2) * m).sum((1, 2)).sqrt() / (full.pow(2) * m).sum((1, 2)).sqrt()
    de = (d.pow(2) * er).sum() / er.sum(); di = (d.pow(2) * inner).sum() / inner.sum()
    sd_ratio = lib.std(o['sig']) / lib.std(full)
    dm = (lib.mean(o['sig']) - lib.mean(full)).abs().mean()
    nz = 'sigma' if o['sigma'] is not None else ''
    print(f'{fam:7s} {dt*1000:6.1f}ms rel-change med {rel.median():.3f} [{rel.min():.3f},{rel.max():.3f}] '
          f'edge/inner rms ratio {float((de/di.clamp(min=1e-20)).sqrt()):.2f} sd ratio [{sd_ratio.min():.3f},{sd_ratio.max():.3f}] '
          f'|dmean| {dm:.2e} {nz} finite {bool(torch.isfinite(o["sig"]).all())}', flush=True)
    x, s = lib.make(fam, full, labs, g)
    assert torch.isfinite(x).all() and float((x * (1 - m)).abs().max()) == 0.0
print('ok')
