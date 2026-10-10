"""Feature source v3/<name>/ from a factory candidate (same layout as every other source).

  python factory/make_source.py <candidate.py> --name dirpower [--reuse <grade run>/feats_full]

Writes ROOT/<name>/{fit,calib,audit,shift_<s>,test}.npz: feats (N, V, d) float32 (+ index, labels for labelled
sets). Inputs are the exact frozen noisy dumps (agents/diffusion/noisy) every other source used; the public test
file is only featurized, in fixed batches of 64 (padded), with the frozen candidate. Feature arrays from a
factory grade run are reused when they cover all rows of a set (fit/calib/audit in full mode). Resumable per
set; della-gh safe with --budget.
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import data as D  # noqa: E402

NOISY = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/diffusion/noisy')
ROOT = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3')
VIEWS = {'fit': ('fit3', 2), 'calib': ('calib3', 2), 'audit': ('audit3', 1)}
BATCH = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('candidate', type=Path)
    ap.add_argument('--name', required=True)
    ap.add_argument('--reuse', type=Path, default=None)
    ap.add_argument('--budget', type=float, default=1e9)
    args = ap.parse_args()
    t0 = time.time()
    spec = importlib.util.spec_from_file_location('candidate', args.candidate)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    dev = torch.device('cuda')
    mask = torch.as_tensor(D.mask(), device=dev, dtype=torch.bool)
    lab = D.labels()
    out = ROOT / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / 'candidate.py').write_text(args.candidate.read_text())

    @torch.no_grad()
    def run_array(arr):
        res = []
        for lo in range(0, len(arr), BATCH):
            n = min(BATCH, len(arr) - lo)
            x = torch.zeros((BATCH, 1424, 176), device=dev)
            x[:n, mask] = torch.as_tensor(np.asarray(arr[lo:lo + n]), device=dev).float()
            f = mod.features(x, mask)
            f = f.detach().float().cpu().numpy() if isinstance(f, torch.Tensor) else np.asarray(f, np.float32)
            res.append(f[:n])
        return np.concatenate(res).astype(np.float32)

    for name in list(VIEWS) + ['shift_' + s for s in D.SHIFTS] + ['test']:
        p = out / f'{name}.npz'
        if p.exists():
            continue
        if time.time() - t0 > args.budget:
            print('budget reached', flush=True)
            return
        t = time.time()
        if name == 'test':
            f = run_array(np.load(D.TEST, mmap_mode='r'))[:, None]
            assert f.shape[0] == 10000 and np.isfinite(f).all()
            np.savez(out / 'test.tmp.npz', feats=f)
        else:
            part, views = VIEWS.get(name, ('audit3', 1))
            idx = D.index(part)
            r = args.reuse / f'{name}.npy' if args.reuse else None
            if r is not None and r.exists() and np.load(r, mmap_mode='r').shape[:2] == (len(idx), views):
                f = np.load(r).astype(np.float32)
                how = 'reused'
            else:
                f = np.stack([run_array(np.load(NOISY / f'{name}_v{v}.npy', mmap_mode='r')) for v in range(views)], 1)
                how = 'computed'
            assert np.isfinite(f).all()
            np.savez(out / f'{name}.tmp.npz', feats=f, index=idx, labels=lab[idx[:, 0], idx[:, 1]])
        (out / f'{name}.tmp.npz').replace(p)
        print(f'{name} {f.shape} {how if name != "test" else "computed"} {time.time()-t:.0f}s', flush=True)
    print(f'done {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
