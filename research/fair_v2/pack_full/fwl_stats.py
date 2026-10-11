"""Rich per-map weak-lensing summary statistics on GPU (torch).

Every statistic is a fixed function of ONE map: no batch statistics, no test
aggregates. Bin edges are frozen from normal training maps (`fit_edges`) and
saved with the model.

Statistics (on the masked footprint, at several Gaussian smoothing scales):
  * pseudo power spectrum in log bands, plus a directional (x vs y) ratio
  * one-point PDF (counts in fixed bins)
  * peak and minimum counts in fixed height bins
  * Minkowski functionals V1 (contour length) and V2 (genus) in fixed bins
  * moments (mean, log variance, skewness, kurtosis, log gradient variance)
  * difference-of-Gaussian wavelet l1 norms in fixed SNR bins
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

H, W = 1424, 176
NOISE_SIGMA = 0.4 / math.sqrt(2 * 30 * 4)
SCALES = (0.0, 1.0, 2.0, 4.0, 8.0)      # Gaussian sigma in pixels (2 arcmin)
DOG = ((0, 1), (1, 2), (2, 3), (3, 4))  # wavelet bands between SCALES entries
PS_EDGES = np.geomspace(0.004, 0.72, 29)
PDF_Q = (0.001, 0.005, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.995, 0.999)
PEAK_Q = (0.02, 0.06, 0.15, 0.3, 0.45, 0.6, 0.75, 0.85, 0.92, 0.96, 0.985, 0.995, 0.999)
MF_Q = (0.01, 0.05, 0.15, 0.3, 0.45, 0.6, 0.75, 0.88, 0.95, 0.985, 0.997)
DOG_SNR = (-4., -3., -2., -1., 0., 1., 2., 3., 4., 5.)


class Stats:
    def __init__(self, mask, device, edges=None):
        self.device = device
        m = torch.as_tensor(np.asarray(mask), device=device, dtype=torch.float32)
        self.mask = m
        self.npix = float(m.sum())
        fy = torch.fft.fftfreq(H, device=device)[:, None]
        fx = torch.fft.rfftfreq(W, device=device)[None, :]
        r = torch.sqrt(fy ** 2 + fx ** 2)
        self.ps_bin = torch.bucketize(r, torch.as_tensor(PS_EDGES, device=device, dtype=torch.float32)) - 1
        nb = len(PS_EDGES) - 1
        valid = (self.ps_bin >= 0) & (self.ps_bin < nb)
        self.ps_bin = torch.where(valid, self.ps_bin, torch.full_like(self.ps_bin, nb))
        self.ps_count = torch.bincount(self.ps_bin.flatten(), minlength=nb + 1)[:nb].clamp(min=1).float()
        xdom = (fx.abs() > fy.abs()).expand(H, -1)
        self.dir_bin = torch.where(xdom, self.ps_bin, torch.full_like(self.ps_bin, nb))
        self.dir_count = torch.bincount(self.dir_bin.flatten(), minlength=nb + 1)[:nb].clamp(min=1).float()
        self.kernels = [torch.exp(-2 * math.pi ** 2 * s * s * r * r) for s in SCALES]
        mk = torch.fft.rfft2(m)
        self.smask = [torch.fft.irfft2(mk * g, s=(H, W)) for g in self.kernels]
        # Interior: footprint pixels whose smoothing window is mostly observed,
        # eroded by one pixel so 8-neighbour tests and gradients stay inside.
        self.interior = []
        for i, s in enumerate(SCALES):
            inside = (m > 0) if s == 0 else ((self.smask[i] > 0.95) & (m > 0))
            # Zero padding erodes the image border too (no wrap-around neighbours).
            padded = F.pad(inside.float()[None, None], (1, 1, 1, 1), value=0.)
            eroded = -F.max_pool2d(-padded, 3, 1, 0)[0, 0] > 0.5
            self.interior.append(eroded)
        self.ninterior = [float(i.sum()) for i in self.interior]
        self.edges = None if edges is None else {k: torch.as_tensor(v, device=device, dtype=torch.float32)
                                                 for k, v in edges.items()}

    # ------------------------------------------------------------------ fields
    def full(self, x):
        """Valid-pixel vectors (B, 132019) -> masked maps (B, H, W) float32."""
        x = torch.as_tensor(x, device=self.device).float()
        out = torch.zeros((x.shape[0], H, W), device=self.device)
        out[:, self.mask > 0] = x
        return out

    def smooth(self, maps):
        f = torch.fft.rfft2(maps * self.mask)
        fields = []
        for i, s in enumerate(SCALES):
            if s == 0:
                fields.append(maps * self.mask)
            else:
                k = torch.fft.irfft2(f * self.kernels[i], s=(H, W)) / self.smask[i].clamp(min=0.05)
                fields.append(k * self.mask)
        return f, fields

    # ------------------------------------------------------------------ edges
    @torch.no_grad()
    def fit_edges(self, maps):
        """Freeze bin edges from normal noisy maps (pooled quantiles)."""
        _, fields = self.smooth(maps)
        edges = {}
        q = lambda v, qs: torch.quantile(v.flatten()[torch.randperm(v.numel(), device=v.device)[:4_000_000]].float(),
                                         torch.as_tensor(qs, device=v.device))
        for i, k in enumerate(fields):
            vals = k[:, self.interior[i]]
            edges[f'pdf{i}'] = q(vals, PDF_Q)
            edges[f'mf{i}'] = q(vals, MF_Q)
            edges[f'sd{i}'] = vals.std().reshape(1)
            pk = self._extrema(k, i, +1)
            edges[f'peak{i}'] = q(k[pk], PEAK_Q)
            mn = self._extrema(k, i, -1)
            edges[f'min{i}'] = -q(-k[mn], PEAK_Q)
        for j, (a, b) in enumerate(DOG):
            w = (fields[a] - fields[b])[:, self.interior[b]]
            edges[f'dog{j}'] = w.std().reshape(1)
        self.edges = edges
        return {k: v.cpu().numpy() for k, v in edges.items()}

    def _extrema(self, k, i, sign):
        z = sign * k
        z = torch.where(self.interior[i][None], z, torch.full_like(z, -1e9))
        mx = F.max_pool2d(z[:, None], 3, 1, 1)[:, 0]
        return (z >= mx) & self.interior[i][None]

    # ------------------------------------------------------------- features
    @torch.no_grad()
    def __call__(self, maps):
        assert self.edges is not None, 'fit_edges first'
        B = maps.shape[0]
        e = self.edges
        f, fields = self.smooth(maps)
        feats = []
        power = (f.real ** 2 + f.imag ** 2) / self.npix
        nb = len(PS_EDGES) - 1
        ps = torch.zeros((B, nb + 1), device=self.device).index_add_(
            1, self.ps_bin.flatten(), power.reshape(B, -1))[:, :nb] / self.ps_count
        dirp = torch.zeros((B, nb + 1), device=self.device).index_add_(
            1, self.dir_bin.flatten(), power.reshape(B, -1))[:, :nb] / self.dir_count
        feats.append(torch.log(ps + 1e-30))
        # Directional ratio over 7 coarse bands (bands of 4 fine bins).
        num = dirp.reshape(B, 7, 4).sum(2); den = ps.reshape(B, 7, 4).sum(2)
        feats.append(torch.log(num + 1e-30) - torch.log(den + 1e-30))
        for i, k in enumerate(fields):
            inter = self.interior[i]
            n = self.ninterior[i]
            vals = k[:, inter]
            mu = vals.mean(1, keepdim=True)
            c = vals - mu
            var = (c * c).mean(1)
            sd = var.sqrt()
            skew = (c ** 3).mean(1) / sd ** 3
            kurt = (c ** 4).mean(1) / var ** 2
            # PDF counts in fixed bins
            pdf = self._hist(vals, e[f'pdf{i}'])
            feats.append(torch.log1p(pdf))
            # Peaks and minima
            pk = self._extrema(k, i, +1)
            feats.append(torch.log1p(self._hist_masked(k, pk, e[f'peak{i}'])))
            mn = self._extrema(k, i, -1)
            feats.append(torch.log1p(self._hist_masked(k, mn, e[f'min{i}'])))
            # Minkowski functionals V1, V2 via central differences
            kx = (torch.roll(k, -1, 2) - torch.roll(k, 1, 2)) / 2
            ky = (torch.roll(k, -1, 1) - torch.roll(k, 1, 1)) / 2
            kxx = torch.roll(k, -1, 2) - 2 * k + torch.roll(k, 1, 2)
            kyy = torch.roll(k, -1, 1) - 2 * k + torch.roll(k, 1, 1)
            kxy = (torch.roll(torch.roll(k, -1, 1), -1, 2) - torch.roll(torch.roll(k, -1, 1), 1, 2)
                   - torch.roll(torch.roll(k, 1, 1), -1, 2) + torch.roll(torch.roll(k, 1, 1), 1, 2)) / 4
            g2 = kx * kx + ky * ky
            gmag = g2.sqrt()
            curv = (2 * kx * ky * kxy - kx * kx * kyy - ky * ky * kxx) / (g2 + 1e-20)
            idx = torch.bucketize(vals, e[f'mf{i}'])
            nmf = len(MF_Q) + 1
            v1 = torch.zeros((B, nmf), device=self.device).scatter_add_(1, idx, gmag[:, inter]) / n
            v2 = torch.zeros((B, nmf), device=self.device).scatter_add_(1, idx, curv[:, inter]) / n
            scale = e[f'sd{i}']
            feats.append(v1 / scale)
            feats.append(v2 / scale)
            gvar = g2[:, inter].mean(1)
            feats.append(torch.stack([mu[:, 0] / scale[0], torch.log(var), skew, kurt, torch.log(gvar)], 1))
        for j, (a, b) in enumerate(DOG):
            inter = self.interior[b]
            w = (fields[a] - fields[b])[:, inter]
            s = e[f'dog{j}']
            snr = w / s
            idx = torch.bucketize(snr, torch.as_tensor(DOG_SNR, device=self.device))
            l1 = torch.zeros((B, len(DOG_SNR) + 1), device=self.device).scatter_add_(1, idx, w.abs() / s)
            feats.append(torch.log1p(l1))
            feats.append(torch.stack([torch.log((w * w).mean(1) / s[0] ** 2),
                                      (w ** 3).mean(1) / (w * w).mean(1) ** 1.5,
                                      (w ** 4).mean(1) / (w * w).mean(1) ** 2], 1))
        return torch.cat(feats, 1)

    def _hist(self, vals, edges):
        idx = torch.bucketize(vals, edges)
        out = torch.zeros((vals.shape[0], len(edges) + 1), device=self.device)
        return out.scatter_add_(1, idx, torch.ones_like(vals))

    def _hist_masked(self, k, sel, edges):
        idx = torch.bucketize(k, edges)
        out = torch.zeros((k.shape[0], len(edges) + 1), device=self.device)
        return out.scatter_add_(1, idx.reshape(k.shape[0], -1), sel.reshape(k.shape[0], -1).float())
