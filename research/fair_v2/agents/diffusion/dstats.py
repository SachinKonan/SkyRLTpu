"""Summary statistics of the learned posterior-mean (denoised) map and of the residual.

kh0 = D(x, sigma_n) is the diffusion model's estimate E[kappa | x] for the observed
noisy map x (no added diffusion noise, deterministic, per map). r0 = x - kh0 is the
model's estimate of the shape noise. The full 466-dim Stats vector (power spectrum,
PDF, peaks, minima, Minkowski functionals, moments, DoG wavelets) is computed on
kh0 (source `dst`) and on r0 (source `dsr`), with bin edges frozen from fit3 maps.
Inputs are the dumped bit-identical noisy maps; test uses fixed-shape batches.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import data as D  # noqa: E402
from stats import Stats  # noqa: E402
from unet import UNet, denoise  # noqa: E402

SUB = 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--root', type=Path, default=Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3'))
    ap.add_argument('--noisy', type=Path, default=Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/diffusion/noisy'))
    ap.add_argument('--budget', type=float, default=45.0)
    ap.add_argument('--bf16', action='store_true')
    ap.add_argument('--limit', type=int, default=0, help='debug: maps per set')
    ap.add_argument('--suffix', default='')
    ap.add_argument('--edge-maps', type=int, default=1024)
    args = ap.parse_args()
    t0 = time.time()
    dev = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    mask = D.mask()
    m = torch.as_tensor(mask, device=dev, dtype=torch.float32)
    lab = D.labels()
    st = torch.load(args.ckpt, map_location=dev, weights_only=False)
    net = UNet(**st.get('cfg', {})).to(dev)
    net.load_state_dict(st['ema']); net.eval()
    if args.bf16:
        net = net.to(torch.bfloat16)
    sn = D.NOISE_SIGMA
    args.work.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def kh0(x):
        out = torch.empty_like(x)
        for lo in range(0, len(x), SUB):
            xb = torch.zeros((SUB,) + x.shape[1:], device=dev)
            k = min(SUB, len(x) - lo)
            xb[:k] = x[lo:lo + k]
            out[lo:lo + k] = denoise(net, xb, torch.full((SUB,), sn, device=dev), m, bf16=args.bf16)[:k]
        return out

    def load(src, lo, n):
        x = torch.zeros((n, 1424, 176), device=dev)
        x[:, m > 0] = torch.as_tensor(np.asarray(src[lo:lo + n]), device=dev).float()
        return x

    # ---- frozen bin edges from fit3 (view 0, first 2048 maps)
    ef = args.work / 'edges.npz'
    if not ef.exists():
        src = np.load(args.noisy / 'fit_v0.npy', mmap_mode='r')
        ks, rs = [], []
        torch.manual_seed(0)
        for lo in range(0, args.edge_maps, 64):
            x = load(src, lo, 64)
            k = kh0(x)
            ks.append(k); rs.append((x - k) * m)
        sk, sr = Stats(mask, dev), Stats(mask, dev)
        ek = sk.fit_edges(torch.cat(ks)); er = sr.fit_edges(torch.cat(rs))
        del ks, rs
        np.savez(ef, **{'k_' + a: b for a, b in ek.items()}, **{'r_' + a: b for a, b in er.items()})
        print(f'edges fitted {time.time()-t0:.0f}s', flush=True)
    E = np.load(ef)
    SK = Stats(mask, dev, edges={a[2:]: E[a] for a in E.files if a.startswith('k_')})
    SR = Stats(mask, dev, edges={a[2:]: E[a] for a in E.files if a.startswith('r_')})

    def feats(x):
        k = kh0(x)
        return SK(k).cpu().numpy(), SR((x - k) * m).cpu().numpy()

    outk = args.root / f'dst{args.suffix}'; outr = args.root / f'dsr{args.suffix}'
    outk.mkdir(parents=True, exist_ok=True); outr.mkdir(parents=True, exist_ok=True)
    jobs = [('calib', 'calib3', 2), ('audit', 'audit3', 1)]
    jobs += [('shift_' + s, 'audit3', 1) for s in D.SHIFTS]
    jobs += [('fit', 'fit3', 2)]
    for name, part, views in jobs:
        if (outr / f'{name}.npz').exists():
            continue
        t = time.time()
        idx = D.index(part)
        n = args.limit or len(idx)
        FK, FR = [], []
        for v in range(views):
            src = np.load(args.noisy / f'{name}_v{v}.npy', mmap_mode='r')
            fk, fr = [], []
            for lo in range(0, n, 64):
                a, b = feats(load(src, lo, min(64, n - lo)))
                fk.append(a); fr.append(b)
            FK.append(np.concatenate(fk)); FR.append(np.concatenate(fr))
        extra = dict(index=idx[:n], labels=lab[idx[:n, 0], idx[:n, 1]])
        np.savez(outk / f'{name}.npz', feats=np.stack(FK, 1).astype(np.float32), **extra)
        np.savez(outr / f'{name}.npz', feats=np.stack(FR, 1).astype(np.float32), **extra)
        print(f'{name} {n}x{views} {time.time()-t:.0f}s', flush=True)
        if (time.time() - t0) / 60 > args.budget:
            print('budget reached', flush=True)
            return
    if not (outr / 'test.npz').exists():
        t = time.time()
        test = np.load(D.TEST, mmap_mode='r')
        n = args.limit or len(test)
        fk, fr = [], []
        for lo in range(0, n, 64):
            k = min(64, n - lo)
            x = torch.zeros((64, 1424, 176), device=dev)   # fixed batch shape, zero padding
            x[:k, m > 0] = torch.as_tensor(np.asarray(test[lo:lo + k]), device=dev).float()
            a, b = feats(x)
            fk.append(a[:k]); fr.append(b[:k])
        np.savez(outk / 'test.npz', feats=np.concatenate(fk)[:, None].astype(np.float32))
        np.savez(outr / 'test.npz', feats=np.concatenate(fr)[:, None].astype(np.float32))
        print(f'test {n} {time.time()-t:.0f}s', flush=True)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
