import sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import data as D
from famfeats import NoiseFam
dev = torch.device('cuda')
mask = D.mask(); m = torch.as_tensor(mask, device=dev, dtype=torch.float32)
train = np.load(D.TRAIN, mmap_mode='r')
idx = D.index('fit3')[:128]
g = torch.Generator(device=dev); g.manual_seed(5)
full = torch.zeros((128, 1424, 176), device=dev)
full[:, m > 0] = torch.as_tensor(np.asarray(train[idx[:, 0], idx[:, 1]]), device=dev).float()
x = D.add_noise(full, m, g)
fam = NoiseFam(mask, dev)
print('tiles', [int(t[1].sum()) for t in fam.tiles])
print(fam.fit_nedges(x))
for _ in range(2):
    torch.cuda.synchronize(); t = time.time(); o = fam(x[:64]); torch.cuda.synchronize(); print(f'{time.time()-t:.2f}s')
for k, v in o.items():
    vv = v.cpu().numpy(); print(k, vv.shape, 'nonfinite', int((~np.isfinite(vv)).sum())); print(np.round(vv.mean(0), 3)); print(np.round(vv.std(0), 3))
xs = D.noisy_shift('srcclust0.15', full[:64], m, g)
o2 = fam(xs)
for k in o: print(k, 'srcclust z', np.round(((o2[k]-o[k].mean(0))/o[k].std(0).clamp(min=1e-9)).mean(0).cpu().numpy(), 1))
