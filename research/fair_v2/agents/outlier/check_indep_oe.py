"""Per-map independence: rescore selected test maps ONE AT A TIME (slot 0 of a zero
batch of 64, eval mode) and compare with the batch-produced oe test features."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
import numpy as np, torch
import data as D
from train_oe import load_net, setup

ckpt, src = sys.argv[1], sys.argv[2]
dev, mask, m = setup()
net, st = load_net(ckpt, dev)
test = np.load(D.TEST, mmap_mode='r')
ref = np.load(f'/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3/{src}/test.npz')['feats'][:, 0]
for i in [0, 1, 17, 101, 1007, 4095, 7001, 9999]:
    full = torch.zeros((64, 1424, 176), device=dev)
    full[:1, m > 0] = torch.as_tensor(np.asarray(test[i:i + 1]), device=dev).float()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        pred, logit, flog, e = net(full, m)
    f = torch.cat([logit.float()[:, None], flog.float(), pred.float(), e.float()], 1)[0].cpu().numpy()
    print(i, 'logit single %.5f batch %.5f max|diff| %.2e' % (f[0], ref[i, 0], np.abs(f - ref[i]).max()), flush=True)
