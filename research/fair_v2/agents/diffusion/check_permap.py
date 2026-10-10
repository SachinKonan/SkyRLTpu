"""Per-map independence check: features of test maps 0..7 computed in two different batch contexts."""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D
from extract import Featurizer
dev = torch.device('cuda'); torch.backends.cudnn.benchmark = True
m = torch.as_tensor(D.mask(), device=dev, dtype=torch.float32)
fz = Featurizer('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/diffusion/feats_r1/model.pt', dev, m, bf16=True)
test = np.load(D.TEST, mmap_mode='r')
def run(rows):
    x = torch.zeros((64, 1424, 176), device=dev)
    x[:len(rows), m > 0] = torch.as_tensor(np.asarray(test[rows]), device=dev).float()
    return fz(x)
a = run(list(range(64)))[:8]
b = run(list(range(8)) + list(range(5000, 5056)))[:8]
c = run(list(range(8)))[:8]
print('max abs diff ctx A vs B', np.abs(a - b).max(), ' A vs zero-padded', np.abs(a - c).max())
