import sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import data as D
from famfeats import Fam, FAMILIES, TRI
dev = torch.device('cuda')
mask = D.mask(); m = torch.as_tensor(mask, device=dev, dtype=torch.float32)
train = np.load(D.TRAIN, mmap_mode='r')
idx = D.index('fit3')[:256]
g = torch.Generator(device=dev); g.manual_seed(5)
full = torch.zeros((256, 1424, 176), device=dev)
full[:, m > 0] = torch.as_tensor(np.asarray(train[idx[:, 0], idx[:, 1]]), device=dev).float()
x = D.add_noise(full, m, g)
fam = Fam(mask, dev)
print('scat interiors', [int(i.sum()) for i in fam.scat_int], 'det', [int(i.sum()) for i in fam.det_int], 'ip', [int(i.sum()) for i in fam.ip_int], 'ntri', len(TRI))
e = fam.fit_edges(x); print({k: np.round(v, 4) for k, v in e.items()})
for f in FAMILIES:
    torch.cuda.synchronize(); t = time.time()
    o = fam(x[:64], (f,))[f]
    torch.cuda.synchronize(); print(f, tuple(o.shape), f'{time.time()-t:.2f}s', 'nonfinite', int((~torch.isfinite(o)).sum()), 'peak mem GB', torch.cuda.max_memory_allocated()/1e9, flush=True)
    o2 = fam(x[:64], (f,))[f]
    torch.cuda.synchronize(); t = time.time(); o3 = fam(x[:64], (f,))[f]; torch.cuda.synchronize(); print('  2nd', f'{time.time()-t:.2f}s')
    oo = o.cpu().numpy(); print('  mean', np.round(oo.mean(0)[:12], 4), '\n  sd', np.round(oo.std(0)[:12], 4))
# batch independence: map 0 alone padded vs in batch
o_a = fam(x[:64])
xx = torch.zeros_like(x[:64]); xx[0] = x[0]
o_b = fam(xx)
for f in FAMILIES:
    print('batch-indep', f, float((o_a[f][0] - o_b[f][0]).abs().max()))
