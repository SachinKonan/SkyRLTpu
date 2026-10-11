"""Mean 2D power (rfft2 of masked map) of normal fit maps, radially binned on a fine log grid -> constants."""
import sys, math, numpy as np, torch
sys.path.insert(0, '.')
import data as D
dev = 'cuda'
mask = torch.as_tensor(D.mask(), device=dev)
H, W = 1424, 176
arr = np.load('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/diffusion/noisy/fit_v0.npy', mmap_mode='r')
rows = np.arange(0, len(arr), 10)[:2000]
acc = torch.zeros((H, W // 2 + 1), device=dev, dtype=torch.float64)
for lo in range(0, len(rows), 64):
    x = torch.zeros((len(rows[lo:lo+64]), H, W), device=dev)
    x[:, mask] = torch.as_tensor(np.asarray(arr[rows[lo:lo+64]]), device=dev).float()
    f = torch.fft.rfft2(x)
    acc += (f.real ** 2 + f.imag ** 2).double().sum(0)
acc /= len(rows)
r = torch.sqrt(torch.fft.fftfreq(H, device=dev)[:, None] ** 2 + torch.fft.rfftfreq(W, device=dev)[None, :] ** 2)
edges = np.concatenate([[0], np.geomspace(0.7 / H, 0.71, 40)])
b = torch.bucketize(r, torch.as_tensor(edges, device=dev, dtype=torch.float32)) - 1
vals = []
for i in range(len(edges) - 1):
    s = b == i
    vals.append(float(acc[s].mean()) if s.any() else float('nan'))
vals = np.array(vals)
print('EDGES =', np.round(edges, 6).tolist())
print('LOGP =', np.round(np.log(vals), 4).tolist())
