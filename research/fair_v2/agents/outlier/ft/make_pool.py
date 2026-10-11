"""Embedding pool for low-FPR fine-tuning of the OE OoD head (frozen run_all14 backbone).

  python ft/make_pool.py --chunk K [--budget-s 480] [--normals]

Positives: synthetic maps from the run's own training families (gens.Library.make on random fit3 clean maps,
noise + float16 quantization after the transformation, exactly as train_oe.py). --normals instead draws fit3
maps with fresh noise (data.add_noise). Saves the pooled embedding e (input of the ood head), the current OoD
logit, the family index and the amplitude draw to fair-data/agents/outlier/ft_pool/{pos,neg}_K.npz.
"""
import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

import numpy as np
import torch

import data as D
from gens import Library
from train_oe import load_net, setup

CK = '/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/outlier/run_all14/ckpt.pt'
OUT = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/outlier/ft_pool')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chunk', type=int, required=True)
    ap.add_argument('--budget-s', type=float, default=480)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--normals', action='store_true')
    args = ap.parse_args()
    t0 = time.time()
    dev, mask, m = setup()
    net, st = load_net(CK, dev)
    fams = st['fams']
    lab = D.labels()
    idx = D.index('fit3')
    train_np = np.load(D.TRAIN, mmap_mode='r')
    xfit = torch.empty((len(idx), int(mask.sum())), dtype=torch.float16, device=dev)
    for lo in range(0, len(idx), 2048):
        sl = idx[lo:lo + 2048]
        xfit[lo:lo + len(sl)] = torch.as_tensor(np.asarray(train_np[sl[:, 0], sl[:, 1]]), device=dev)
    theta = lab[idx[:, 0], idx[:, 1]]
    print(f'loaded {len(idx)} fit3 maps in {time.time()-t0:.0f}s; families {fams}', flush=True)
    lib = Library(m, dev)
    g = torch.Generator(device=dev)
    g.manual_seed(7000 + 2 * args.chunk + int(args.normals))
    E, L, Fm, S = [], [], [], []
    N, B = len(idx), args.batch
    with torch.no_grad():
        while time.time() - t0 < args.budget_s:
            rows = torch.randint(0, N, (B,), generator=g, device=dev)
            full = torch.zeros((B, 1424, 176), device=dev)
            full[:, m > 0] = xfit[rows].float()
            if args.normals:
                x, fi, s = D.add_noise(full, m, g), -1, torch.zeros(B, device=dev)
            else:
                fi = int(torch.randint(0, len(fams), (1,), generator=g, device=dev))
                x, s = lib.make(fams[fi], full, theta[rows.cpu().numpy()], g)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, logit, _, e = net(x, m)
            E.append(e.float().cpu().numpy().astype(np.float16))
            L.append(logit.float().cpu().numpy())
            Fm.append(np.full(B, fi, np.int16))
            S.append(s.float().cpu().numpy())
    out = OUT / f'{"neg" if args.normals else "pos"}_{args.chunk}.npz'
    np.savez(out.with_suffix('.tmp.npz'), e=np.concatenate(E), logit=np.concatenate(L), fam=np.concatenate(Fm),
             s=np.concatenate(S), fams=np.array(fams))
    out.with_suffix('.tmp.npz').replace(out)
    print(f'wrote {out} n={sum(len(x) for x in E)} in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
