import sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import data as D
from stats import Stats
from unet import UNet, pool_bottleneck
import extract_den as E
dev = torch.device('cuda')
mode = sys.argv[1]
torch.backends.cudnn.benchmark = mode == 'bench'
torch.backends.cudnn.deterministic = mode != 'bench'
mask = D.mask(); m = torch.as_tensor(mask, device=dev, dtype=torch.float32)
st = torch.load('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/denoiser/unet/net_ep00.pt', map_location=dev, weights_only=False)
net = UNet(tuple(st['widths'])).to(dev).to(memory_format=torch.channels_last); net.load_state_dict(st['net']); net.eval()
train = np.load(D.TRAIN, mmap_mode='r'); idx = D.index('audit3')[:64]
gen = torch.Generator(device=dev); gen.manual_seed(1)
t=time.time(); full = torch.zeros((64,1424,176),device=dev); full[:, m>0] = torch.as_tensor(np.asarray(train[idx[:,0], idx[:,1]]), device=dev).float(); torch.cuda.synchronize(); print('load', time.time()-t)
x = D.add_noise(full, m, gen)
stats = Stats(mask, dev, edges=dict(np.load('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/denoiser/tmpv3/den/edges_denoised.npz')))
rf = E.ResidualFeats(m, dev)
def tm(f, n=5):
    f(); torch.cuda.synchronize(); t=time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time()-t)/n
with torch.no_grad():
    def fwd():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return net(x, m)
    y, b = fwd(); y = y.float()
    print(mode, 'unet', tm(fwd), 'stats', tm(lambda: stats(y)), 'resid', tm(lambda: rf(x, y)), 'pool', tm(lambda: pool_bottleneck(b, m)))
    y2, _ = fwd()
    print('repeat-diff', float((y2.float()-y).abs().max()))
