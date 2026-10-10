"""Does the conditional model use theta? EDM loss on 128 calib3 maps: true theta / CNN theta_hat / dropped / shuffled."""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import data as D
from unet import UNet, denoise, SIGMA_DATA
dev = torch.device('cuda')
m = torch.as_tensor(D.mask(), device=dev, dtype=torch.float32)
st = torch.load(sys.argv[1], map_location=dev, weights_only=False)
net = UNet(**st['cfg']).to(dev); net.load_state_dict(st['ema']); net.eval()
lab = D.labels(); fi = D.index('fit3'); lf = lab[fi[:, 0], fi[:, 1]]; tm, ts = lf.mean(0), lf.std(0)
idx = D.index('calib3')[:128]
tr = np.load(D.TRAIN, mmap_mode='r')
kap = torch.zeros((128, 1424, 176), device=dev); kap[:, m > 0] = torch.as_tensor(np.asarray(tr[idx[:, 0], idx[:, 1]]), device=dev).float()
th = torch.as_tensor((lab[idx[:, 0], idx[:, 1]] - tm) / ts, dtype=torch.float32, device=dev)
pred = torch.as_tensor(np.load('/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3/cnn0/calib.npz')['feats'][:128, 0, -5:], device=dev)
one = torch.ones((128, 1), device=dev)
conds = {'true': torch.cat([th, one], 1), 'cnn': torch.cat([pred, one], 1), 'drop': torch.zeros((128, 6), device=dev),
         'shuffled': torch.cat([th[torch.randperm(128, device=dev)], one], 1)}
g = torch.Generator(device=dev).manual_seed(1)
for s in [0.0258, 0.05, 0.1, 0.2, 0.5]:
    eps = torch.randn(kap.shape, generator=g, device=dev)
    y = (kap + s * eps) * m
    row = []
    for k, c in conds.items():
        with torch.no_grad():
            L = 0
            for lo in range(0, 128, 32):
                Dk = denoise(net, y[lo:lo+32], torch.full((32,), s, device=dev), m, cond=c[lo:lo+32])
                L += (((Dk - kap[lo:lo+32]) ** 2) * m).sum().item()
        row.append(f'{k}={L / (128 * m.sum().item()) / SIGMA_DATA**2:.5f}')
    print(f's={s}: ' + ' '.join(row), flush=True)
