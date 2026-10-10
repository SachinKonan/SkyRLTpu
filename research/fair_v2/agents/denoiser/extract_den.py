"""Denoiser feature source `den` (+ sub-sources den_r, den_s, den_b) on the SAME noisy maps as `stats`/`cnn0`.

Per noisy map x (one map at a time in effect; fixed batch shapes, eval mode):
  y = D(x) = U-Net estimate of E[kappa | x], r = (x - y) on observed pixels.
  den_r (residual block, 80 dims):
    log residual pseudo power spectrum in 40 log bands; log var / skew / kurt of the
    mask-normalized smoothed residual at 4 scales; residual-x-denoised spectral
    correlation in 20 bands; pixel correlation corr(r_s, y_s) and corr(r_s^2, y) at 4 scales.
  den_s: stats.Stats (466 dims) on y with bin edges frozen from denoised fit3 maps.
  den_b: masked mean/std of every U-Net bottleneck channel.
  den  : concatenation [den_r | den_s | den_b].

Noise seeds/loop copy pipeline_v3.embed.run exactly (batch 64, gen.manual_seed(seed*1000+v)).
Resumable: each (set, view) is saved as a partial file; whole sets are skipped when done.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import data as D  # noqa: E402
from stats import Stats, H, W  # noqa: E402
from unet import UNet, pool_bottleneck  # noqa: E402

RES_EDGES = np.geomspace(0.004, 0.72, 41)
X_EDGES = np.geomspace(0.004, 0.72, 21)
RSCALES = (0.0, 1.0, 2.0, 4.0)
BATCH = 64


class ResidualFeats:
    def __init__(self, m, dev):
        self.m = m
        self.dev = dev
        self.npix = float(m.sum())
        fy = torch.fft.fftfreq(H, device=dev)[:, None]
        fx = torch.fft.rfftfreq(W, device=dev)[None, :]
        r = torch.sqrt(fy ** 2 + fx ** 2)
        self.bins = []
        for e in (RES_EDGES, X_EDGES):
            nb = len(e) - 1
            b = torch.bucketize(r, torch.as_tensor(e, device=dev, dtype=torch.float32)) - 1
            b = torch.where((b >= 0) & (b < nb), b, torch.full_like(b, nb))
            cnt = torch.bincount(b.flatten(), minlength=nb + 1)[:nb].clamp(min=1).float()
            self.bins.append((b.flatten(), nb, cnt))
        self.kernels = [torch.exp(-2 * math.pi ** 2 * s * s * r * r) for s in RSCALES]
        mk = torch.fft.rfft2(m)
        self.smask = [torch.fft.irfft2(mk * g, s=(H, W)) for g in self.kernels]
        self.inter = [(m > 0) if s == 0 else ((sm > 0.95) & (m > 0)) for s, sm in zip(RSCALES, self.smask)]

    def binned(self, p, which):
        b, nb, cnt = self.bins[which]
        B = p.shape[0]
        return torch.zeros((B, nb + 1), device=self.dev).index_add_(1, b, p.reshape(B, -1))[:, :nb] / cnt

    def smooth(self, f, i):
        if RSCALES[i] == 0:
            return None
        return torch.fft.irfft2(f * self.kernels[i], s=(H, W)) / self.smask[i].clamp(min=0.05) * self.m

    @torch.no_grad()
    def __call__(self, x, y):
        m = self.m
        r = (x - y) * m
        fr = torch.fft.rfft2(r)
        fy = torch.fft.rfft2(y * m)
        pr = (fr.real ** 2 + fr.imag ** 2) / self.npix
        out = [torch.log(self.binned(pr, 0) + 1e-30)]
        py = (fy.real ** 2 + fy.imag ** 2) / self.npix
        cross = (fr.real * fy.real + fr.imag * fy.imag) / self.npix
        c = self.binned(cross, 1) / torch.sqrt(self.binned(pr, 1) * self.binned(py, 1) + 1e-30)
        mom, cor = [], []
        for i in range(len(RSCALES)):
            rs = r if RSCALES[i] == 0 else self.smooth(fr, i)
            ys = y * m if RSCALES[i] == 0 else self.smooth(fy, i)
            inter = self.inter[i]
            rv = rs[:, inter]; yv = ys[:, inter]
            rc = rv - rv.mean(1, keepdim=True)
            var = (rc * rc).mean(1)
            mom.append(torch.stack([torch.log(var), (rc ** 3).mean(1) / var ** 1.5, (rc ** 4).mean(1) / var ** 2], 1))
            yc = yv - yv.mean(1, keepdim=True)
            corr = (rc * yc).mean(1) / torch.sqrt(var * (yc * yc).mean(1))
            r2 = rc * rc
            r2c = r2 - r2.mean(1, keepdim=True)
            y0 = y[:, inter]; y0c = y0 - y0.mean(1, keepdim=True)
            corr2 = (r2c * y0c).mean(1) / torch.sqrt((r2c * r2c).mean(1) * (y0c * y0c).mean(1))
            cor.append(torch.stack([corr, corr2], 1))
        out += [torch.cat(mom, 1), c, torch.cat(cor, 1)]
        return torch.cat(out, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--root', type=Path, default=Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3'))
    ap.add_argument('--name', default='den')
    ap.add_argument('--budget', type=float, default=480.0)
    ap.add_argument('--check', action='store_true', help='verify noise is bit-identical to the stats source')
    ap.add_argument('--only', default='', help='comma list of set names to run (default all)')
    args = ap.parse_args()
    t0 = time.time()
    dev = torch.device('cuda')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    mask = D.mask(); m = torch.as_tensor(mask, device=dev, dtype=torch.float32)
    lab = D.labels()
    train = np.load(D.TRAIN, mmap_mode='r')
    st = torch.load(args.ckpt, map_location=dev, weights_only=False)
    net = UNet(tuple(st['widths'])).to(dev).to(memory_format=torch.channels_last)
    net.load_state_dict(st['net']); net.eval()
    print(f'ckpt epoch {st.get("epoch")} val {st.get("val")}', flush=True)
    out = args.root / args.name
    part_dir = out / '_partial'
    part_dir.mkdir(parents=True, exist_ok=True)
    subs = {f'{args.name}_r': None, f'{args.name}_s': None, f'{args.name}_b': None}
    for s in subs:
        (args.root / s).mkdir(parents=True, exist_ok=True)
    gen = torch.Generator(device=dev)
    rf = ResidualFeats(m, dev)

    def load_clean(sl):
        full = torch.zeros((len(sl), 1424, 176), device=dev)
        full[:, m > 0] = torch.as_tensor(np.asarray(train[sl[:, 0], sl[:, 1]]), device=dev).float()
        return full

    @torch.no_grad()
    def denoise(x):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            y, bott = net(x, m)
        return y.float() * m, pool_bottleneck(bott, m)

    # ---- frozen bin edges for Stats on the denoised field (fit3 maps, separate noise seed)
    edges_path = out / 'edges_denoised.npz'
    if not edges_path.exists():
        idx = D.index('fit3')[::40][:512]
        gen.manual_seed(777)
        ys = []
        for lo in range(0, len(idx), BATCH):
            x = D.add_noise(load_clean(idx[lo:lo + BATCH]), m, gen)
            ys.append(denoise(x)[0])
        torch.manual_seed(0)
        e = Stats(mask, dev).fit_edges(torch.cat(ys))
        np.savez(edges_path, **e)
        del ys
        torch.cuda.empty_cache()
        print(f'edges fitted on {len(idx)} denoised fit3 maps', flush=True)
    stats = Stats(mask, dev, edges=dict(np.load(edges_path)))

    @torch.no_grad()
    def features(x):
        y, b = denoise(x)
        return torch.cat([rf(x, y), stats(y), b], 1).cpu().numpy()

    if args.check:
        ref_stats = Stats(mask, dev, edges=dict(np.load('/scratch/gpfs/ZHUANGL/sk7524/fair-data/feats-v2/edges.npz')))
        idx = D.index('audit3')
        gen.manual_seed(33 * 1000 + 0)
        x = D.add_noise(load_clean(idx[:BATCH]), m, gen)
        mine = ref_stats(x).cpu().numpy()
        ref = np.load(args.root / 'stats' / 'audit.npz')['feats'][:BATCH, 0]
        print('check audit batch0 vs stats source: max abs diff', float(np.abs(mine - ref).max()),
              'identical', bool(np.array_equal(mine, ref)), flush=True)
        f = features(x)
        print('feature dims', f.shape, 'nonfinite', int((~np.isfinite(f)).sum()), flush=True)
        return

    dims = None

    def save(name, feats, idx=None):
        nonlocal dims
        extra = {} if idx is None else dict(index=idx, labels=lab[idx[:, 0], idx[:, 1]])
        nr = rf_dims
        ns = 466
        np.savez(out / f'{name}.npz', feats=feats, **extra)
        np.savez(args.root / f'{args.name}_r' / f'{name}.npz', feats=feats[..., :nr], **extra)
        np.savez(args.root / f'{args.name}_s' / f'{name}.npz', feats=feats[..., nr:nr + ns], **extra)
        np.savez(args.root / f'{args.name}_b' / f'{name}.npz', feats=feats[..., nr + ns:], **extra)

    rf_dims = len(RES_EDGES) - 1 + 3 * len(RSCALES) + len(X_EDGES) - 1 + 2 * len(RSCALES)
    view_time = {}
    only = set(filter(None, args.only.split(',')))

    def run(name, part, views, seed, shift=None):
        if (out / f'{name}.npz').exists() or (only and name not in only):
            return True
        idx = D.index(part)
        V = []
        for v in range(views):
            pf = part_dir / f'{name}_v{v}.npy'
            if pf.exists():
                V.append(np.load(pf)); continue
            est = view_time.get('per_map', 0.004) * len(idx)
            if time.time() - t0 + est > args.budget:
                print(f'budget: stop before {name} view {v} (est {est:.0f}s)', flush=True)
                return False
            t = time.time()
            gen.manual_seed(seed * 1000 + v)
            ps = []
            for lo in range(0, len(idx), BATCH):
                sl = idx[lo:lo + BATCH]
                full = load_clean(sl)
                x = (D.noisy_shift(shift, full, m, gen, lab[sl[:, 0], sl[:, 1]]) if shift
                     else D.add_noise(full, m, gen))
                ps.append(features(x))
            a = np.concatenate(ps).astype(np.float32)
            np.save(pf, a)
            V.append(a)
            view_time['per_map'] = (time.time() - t) / len(idx)
            print(f'{name} view {v}: {len(idx)} maps in {time.time()-t:.0f}s', flush=True)
        save(name, np.stack(V, 1).astype(np.float32), idx)
        print(f'{name} saved  {time.time()-t0:.0f}s', flush=True)
        return True

    ok = run('fit', 'fit3', 2, 31)
    ok = ok and run('calib', 'calib3', 2, 32)
    ok = ok and run('audit', 'audit3', 1, 33)
    for i, s in enumerate(D.SHIFTS):
        ok = ok and run('shift_' + s, 'audit3', 1, 300 + i, shift=s)
    if ok and not (out / 'test.npz').exists() and (not only or 'test' in only):
        test = np.load(D.TEST, mmap_mode='r')
        est = view_time.get('per_map', 0.004) * len(test)
        if time.time() - t0 + est > args.budget:
            print('budget: stop before test', flush=True); return
        t = time.time()
        ps = []
        for lo in range(0, len(test), BATCH):
            n = min(BATCH, len(test) - lo)
            # Fixed batch shape (zero padding) so a map's features never depend on others.
            full = torch.zeros((BATCH, 1424, 176), device=dev)
            full[:n, m > 0] = torch.as_tensor(np.asarray(test[lo:lo + n]), device=dev).float()
            ps.append(features(full)[:n])
        save('test', np.concatenate(ps)[:, None].astype(np.float32))
        print(f'test in {time.time()-t:.0f}s', flush=True)
    print(f'all done={ok} {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
