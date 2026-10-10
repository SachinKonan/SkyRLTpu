"""Parameter predictions of several CNNs on the exact dumped noisy maps -> source <name> (N, V, 5*K).

Columns are the K nets' normalized 5-parameter predictions (each net normalizes with the fit3 label
mean/std, the same for all nets). Test maps come from the test file in fixed-shape batches.
"""
import argparse
import fnmatch
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
import data as D  # noqa: E402
from cnn import Net  # noqa: E402

NOISY = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/diffusion/noisy')
VIEWS = {'fit': ('fit3', 2), 'calib': ('calib3', 2), 'audit': ('audit3', 1)}
BATCH = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs='+', required=True)
    ap.add_argument('--root', type=Path, default=Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3'))
    ap.add_argument('--name', required=True)
    ap.add_argument('--only', default='')
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    t0 = time.time()
    dev = torch.device('cuda')
    mask = D.mask(); m = torch.as_tensor(mask, device=dev, dtype=torch.float32); sel = m > 0
    lab = D.labels()
    nets = []
    for ck in args.ckpts:
        st = torch.load(ck, map_location=dev, weights_only=False)
        net = Net(widths=tuple(st['widths'])).to(dev).to(memory_format=torch.channels_last)
        net.load_state_dict(st['net']); net.eval(); nets.append(net)
    out = args.root / args.name; out.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def run_array(arr):
        ps = []
        for lo in range(0, len(arr), BATCH):
            n = min(BATCH, len(arr) - lo)
            full = torch.zeros((BATCH, 1424, 176), device=dev)
            full[:n, sel] = torch.as_tensor(np.asarray(arr[lo:lo + n]), device=dev).float()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pr = [net(full, m)[0].float() for net in nets]
            ps.append(torch.cat(pr, 1)[:n].cpu().numpy())
        return np.concatenate(ps).astype(np.float32)

    if args.check:
        idx = D.index('calib3')[:512]
        theta = lab[idx[:, 0], idx[:, 1]]
        fit = lab[tuple(D.index('fit3').T)]
        tz = (theta - fit.mean(0)) / fit.std(0)
        p = run_array(np.load(NOISY / 'calib_v0.npy', mmap_mode='r')[:512]).reshape(512, len(nets), 5)
        mse = ((p - tz[:, None]) ** 2).mean(0)
        print('per-net normalized MSE per param:\n', mse.round(3), '\nmean over nets', mse.mean(0).round(3),
              '\nensemble-mean pred MSE', ((p.mean(1) - tz) ** 2).mean(0).round(3), flush=True)
        ref = np.load(args.root / 'cnn0' / 'calib.npz')['feats'][:512, 0, -5:]
        print('cnn0 (torch-trained) MSE', ((ref - tz) ** 2).mean(0).round(3), flush=True)
        return
    only = list(filter(None, args.only.split(',')))
    want = lambda n: not only or any(fnmatch.fnmatch(n, p) for p in only)
    for name in list(VIEWS) + ['shift_' + s for s in D.SHIFTS] + ['test']:
        if not want(name) or (out / f'{name}.npz').exists():
            continue
        t = time.time()
        if name == 'test':
            np.savez(out / 'test.npz', feats=run_array(np.load(D.TEST, mmap_mode='r'))[:, None])
        else:
            part, views = VIEWS.get(name, ('audit3', 1))
            idx = D.index(part)
            f = np.stack([run_array(np.load(NOISY / f'{name}_v{v}.npy', mmap_mode='r')) for v in range(views)], 1)
            np.savez(out / f'{name}.npz', feats=f, index=idx, labels=lab[idx[:, 0], idx[:, 1]])
        print(f'{name} {time.time()-t:.0f}s', flush=True)
    print(f'done {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
