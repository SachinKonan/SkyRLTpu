"""Denoiser feature sources den_r / den_s / den_b for inference (copied from agents/denoiser/extract_den2.py).

Per map x: y = D(x) (U-Net, bf16 autocast on GPU), r = x - y.
  den_r (112) = [ResidualFeats(x, y) (80) | SURE block (6) | tile block (26)]
  den_s (466) = Stats(y) with edges frozen from denoised fit3 maps
  den_b (512) = masked mean/std of bottleneck channels
The SURE probe is a frozen Rademacher field shipped with the model (identical for every map).
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from fwl_stats import Stats, H, W
from fwl_unet import UNet, pool_bottleneck

RES_EDGES = np.geomspace(0.004, 0.72, 41)
X_EDGES = np.geomspace(0.004, 0.72, 21)
RSCALES = (0.0, 1.0, 2.0, 4.0)
NOISE_SIGMA = 0.4 / math.sqrt(2 * 30 * 4)


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


class DenFeats:
    def __init__(self, ckpt, edges, probe, mask, dev, amp=torch.bfloat16):
        self.dev = dev
        self.amp = amp
        m = torch.as_tensor(np.asarray(mask), device=dev, dtype=torch.float32)
        self.m = m
        net = UNet(tuple(ckpt['widths'])).to(dev).to(memory_format=torch.channels_last)
        net.load_state_dict(ckpt['net']); net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self.net = net
        self.rf = ResidualFeats(m, dev)
        self.stats = Stats(mask, dev, edges=edges)
        self.probe = torch.as_tensor(np.asarray(probe), device=dev, dtype=torch.float32) * m
        self.sig = NOISE_SIGMA
        ysd = float(np.asarray(edges['sd0']).reshape(-1)[0])
        self.ybins = torch.as_tensor([-1.0, 1.0, 2.0], device=dev) * ysd
        self.tiles = []
        for t in (8, 16):
            cov = F.avg_pool2d(m[None, None], t)[0, 0]
            vt = cov >= 0.5
            th, tw = cov.shape
            fy = torch.fft.fftfreq(th, device=dev)[:, None]
            fx = torch.fft.rfftfreq(tw, device=dev)[None, :]
            rr = torch.sqrt(fy ** 2 + fx ** 2)
            bands = [(rr > lo) & (rr <= hi) for lo, hi in ((0.0, 0.03), (0.03, 0.08), (0.08, 0.2))]
            self.tiles.append((t, cov, vt, bands))
        self.tq = torch.as_tensor([0.05, 0.5, 0.95], device=dev)
        self.tails = torch.as_tensor([2.5, 3.0, 4.0], device=dev)

    @torch.no_grad()
    def denoise(self, x):
        if self.amp is not None:
            with torch.autocast(self.dev.type, dtype=self.amp):
                y, bott = self.net(x, self.m)
        else:
            y, bott = self.net(x, self.m)
        return y.float() * self.m, bott

    @torch.no_grad()
    def sure(self, x, y):
        m, sig = self.m, self.sig
        y2, _ = self.denoise(x + sig * self.probe)
        g = self.probe * (y2 - y) / sig
        sel = m > 0
        gv = g[:, sel]; yv = y[:, sel]; rv = (x - y)[:, sel]
        div = gv.mean(1)
        bidx = torch.bucketize(yv, self.ybins)
        cnt = torch.zeros((len(x), 4), device=self.dev).scatter_add_(1, bidx, torch.ones_like(yv)).clamp(min=1)
        dbin = torch.zeros((len(x), 4), device=self.dev).scatter_add_(1, bidx, gv) / cnt
        r2 = (rv * rv).mean(1) / sig ** 2
        return torch.cat([torch.stack([r2, div], 1), dbin], 1)

    @torch.no_grad()
    def tile_feats(self, x, y):
        m, sig = self.m, self.sig
        r = (x - y) * m
        out = []
        for t, cov, vt, bands in self.tiles:
            pool = lambda a: F.avg_pool2d(a[:, None], t)[:, 0] / cov.clamp(min=1e-6)
            v = pool(r * r) / sig ** 2
            ym = pool(y * m)
            ys = (pool(y * y * m) - ym ** 2).clamp(min=0).sqrt()
            vv = v[:, vt]; ymv = ym[:, vt]; ysv = ys[:, vt]
            mu = vv.mean(1, keepdim=True)
            vc = vv - mu
            sd = vc.pow(2).mean(1).sqrt()

            def corr(a):
                ac = a - a.mean(1, keepdim=True)
                return (vc * ac).mean(1) / (sd * ac.pow(2).mean(1).sqrt() + 1e-12)
            q = torch.quantile(vv, self.tq, dim=1).T
            vm = torch.where(vt, v / mu[:, :, None] - 1, torch.zeros_like(v))
            f = torch.fft.rfft2(vm)
            p = f.real ** 2 + f.imag ** 2
            lowk = torch.stack([p[:, bd].mean(1) for bd in bands], 1) / float(vt.sum())
            out += [torch.log(mu), (sd / mu[:, 0])[:, None], torch.log(q), torch.stack([corr(ymv), corr(ysv)], 1),
                    torch.log(lowk + 1e-12)]
        z = r[:, m > 0] / sig
        pos = (z[:, :, None] > self.tails).float().sum(1)
        neg = (z[:, :, None] < -self.tails).float().sum(1)
        out += [torch.log1p(pos), torch.log1p(neg)]
        return torch.cat(out, 1)

    @torch.no_grad()
    def __call__(self, x, want):
        """x: (B, H, W) masked noisy maps. want: subset of {'den_r', 'den_s', 'den_b'}. Returns dict of tensors."""
        y, bott = self.denoise(x)
        out = {}
        if 'den_r' in want:
            out['den_r'] = torch.cat([self.rf(x, y), self.sure(x, y), self.tile_feats(x, y)], 1)
        if 'den_s' in want:
            out['den_s'] = self.stats(y)
        if 'den_b' in want:
            out['den_b'] = pool_bottleneck(bott, self.m)
        return out
