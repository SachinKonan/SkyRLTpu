"""Within-cosmology spread of noise-kurtosis variants on normal audit maps (and a self-made Poisson-mixture
noise test on clean maps, implemented from the one-line description only)."""
import sys, math, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '.')
import data as D
dev = 'cuda'
mask = torch.as_tensor(D.mask(), device=dev)
H, W = 1424, 176
SIG = 0.4 / math.sqrt(240)
idx = D.index('audit3')[:768]
lab = D.labels()
clean = np.load(D.TRAIN if hasattr(D, 'TRAIN') else '/scratch/gpfs/ZHUANGL/sk7524/fair-data/WIDE12H_bin2_2arcmin_kappa_newrealization.npy', mmap_mode='r')
cos = idx[:, 0]
m = mask.float()
k3 = torch.ones(1, 1, 3, 3, device=dev); full = (F.conv2d(m[None, None], k3, padding=1)[0, 0] >= 8.999)
nb = torch.ones(1, 1, 3, 3, device=dev) / 8; nb[0, 0, 1, 1] = 0
fy = torch.fft.fftfreq(H, device=dev)[:, None]; fx = torch.fft.rfftfreq(W, device=dev)[None, :]
r2 = fy ** 2 + fx ** 2
def smooth(x, s):
    return torch.fft.irfft2(torch.fft.rfft2(x) * torch.exp(-2 * math.pi ** 2 * s * s * r2), s=(H, W))
g = torch.Generator(device=dev); g.manual_seed(7)
def stats(x):
    d = x - F.conv2d(x[:, None], nb, padding=1)[:, 0]
    xs = smooth(x, 2.0)
    out = {}
    for name, t in [('all', None), ('t2.5', 2.5), ('t2', 2.0), ('t1.5', 1.5), ('t1', 1.0)]:
        res = []
        for i in range(len(x)):
            sel = full.clone()
            if t is not None:
                v = xs[i][full]; s = v.std(); mu = v.mean()
                sel = sel & ((xs[i] - mu).abs() < t * s)
            di = d[i][sel].double(); m2 = (di ** 2).mean(); z = di / m2.sqrt()
            res.append([((z ** 4).mean() - 3).item(), (z.abs().mean() / math.sqrt(2 / math.pi)).item(), sel.sum().item()])
        out[name] = np.array(res)
    return out
def run(mode):
    allr = {}
    for lo in range(0, len(idx), 64):
        sl = idx[lo:lo + 64]
        x = torch.zeros((len(sl), H, W), device=dev)
        x[:, mask] = torch.as_tensor(np.asarray(clean[sl[:, 0], sl[:, 1]]), device=dev).float()
        noise = torch.randn(x.shape, generator=g, device=dev) * SIG
        if mode == 'poisson':
            n = torch.poisson(torch.full(x.shape, 120.0, device=dev), generator=g).clamp(min=1)
            noise = noise * torch.sqrt(120.0 / n)
        x = ((x + noise) * m).half().float() * m
        for k, v in stats(x).items():
            allr.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in allr.items()}
A = run('normal'); B = run('poisson')
def within(v):
    res = v - np.array([v[cos == c].mean() for c in cos])
    return res.std() * math.sqrt(np.mean([ (cos == c).sum() / max(1, (cos == c).sum() - 1) for c in cos]))
for k in A:
    for j, nm in enumerate(['k4', 'l1']):
        a, b = A[k][:, j], B[k][:, j]
        print(f'{k:5s} {nm}: normal mean {a.mean():.4f} sd {a.std():.4f} within-cos sd {within(a):.4f} | poisson shift {b.mean()-a.mean():+.4f} -> z {(b.mean()-a.mean())/within(a):+.2f}  pix {A[k][:,2].mean():.0f}')
