"""Synthetic alternative-physics generator library for outlier exposure.

Every generator acts on CLEAN training maps (full 1424x176 frame, zeros outside
the mask) BEFORE noise. The caller then adds shape noise and float16
quantization exactly as data.add_noise does, so InD and synthetic OoD maps
differ only in the injected physics (or in the noise model for the 'noise'
family). Amplitudes are drawn per sample, log-uniform from barely detectable
to well beyond plausible priors.

Fourier-space generators work on a mean-subtracted, edge-inpainted copy of the
map (normalized-convolution fill outside the mask) so that filtering does not
create an artificial step at the mask boundary; the output is re-masked.

Each family returns dict(sig=(B,H,W), sigma=None|(B,H,W), eps=None|(B,H,W),
s=(B,) strength in [0,1] of the main amplitude draw).
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

import data as D

H, W = 1424, 176
PIX_RAD = math.radians(2.0 / 60.0)
# Order is fixed (family-head class = index + 1); new families are appended only.
ALL_FAMILIES = ['tell', 'nz', 'shear', 'psf', 'noise', 'mask', 'nonlin', 'hmcode', 'halo', 'ia',
                'srcclust', 'poisson', 'psfpat', 'holes']
NFAM_HEAD = 16
HELDOUT_DEFAULT = ['hmcode', 'halo', 'ia', 'srcclust']


class Library:
    def __init__(self, m, dev):
        self.m = m
        self.dev = dev
        self.msum = m.sum()
        fy = torch.fft.fftfreq(H, device=dev)[:, None]
        fx = torch.fft.rfftfreq(W, device=dev)[None, :]
        self.fy = fy.expand(H, fx.shape[1]).contiguous()
        self.fx = fx.expand(H, fx.shape[1]).contiguous()
        self.r2 = self.fy ** 2 + self.fx ** 2
        self.r = torch.sqrt(self.r2)
        self.lr = torch.log(self.r.clamp(min=1e-4))
        self.Y = torch.linspace(-1, 1, H, device=dev)[:, None].expand(H, W)
        self.X = torch.linspace(-1, 1, W, device=dev)[None, :].expand(H, W) * (W / H)
        self.Yi = torch.arange(H, device=dev)[:, None]
        self.Xi = torch.arange(W, device=dev)[None, :]
        self.obs = torch.nonzero(m > 0)                       # (Nobs, 2)
        grid = np.load(D.TK_GRID)
        keys = sorted((float(n[2:]), n) for n in grid.files if n.startswith('lT'))
        self.lT = torch.tensor([p[0] for p in keys], device=dev, dtype=torch.float32)
        kk = (2 * math.pi * self.r / PIX_RAD / D.CHI_EFF).cpu().numpy()
        self.Ttab = torch.tensor(np.stack([np.interp(kk, grid['k'], grid[n]) for _, n in keys]),
                                 dtype=torch.float32, device=dev)          # (nT, H, Wr)
        self.fam = {n: getattr(self, 'f_' + n) for n in ALL_FAMILIES}

    # ------------------------------------------------------------ random
    def u(self, B, lo, hi, g):
        return torch.rand(B, generator=g, device=self.dev) * (hi - lo) + lo

    def lu(self, B, lo, hi, g):
        s = torch.rand(B, generator=g, device=self.dev)
        return torch.exp(math.log(lo) + s * (math.log(hi) - math.log(lo))), s

    def sign(self, B, g, p=0.5):
        return torch.where(torch.rand(B, generator=g, device=self.dev) < p, 1.0, -1.0)

    def choice(self, B, k, g):
        return torch.randint(0, k, (B,), generator=g, device=self.dev)

    # ------------------------------------------------------------ helpers
    def mean(self, x):
        return (x * self.m).sum((1, 2)) / self.msum

    def std(self, x):
        mu = self.mean(x)
        return torch.sqrt((((x - mu[:, None, None]) * self.m) ** 2).sum((1, 2)) / self.msum).clamp(min=1e-8)

    def gauss_T(self, sy, sx=None, theta=None):
        """Per-sample Gaussian transfer (B,H,Wr); sy, sx in pixels, theta rotation."""
        sx = sy if sx is None else sx
        if theta is None:
            q = sy[:, None, None] ** 2 * self.fy ** 2 + sx[:, None, None] ** 2 * self.fx ** 2
        else:
            c, s = torch.cos(theta)[:, None, None], torch.sin(theta)[:, None, None]
            a = c * self.fx + s * self.fy
            b = -s * self.fx + c * self.fy
            q = sx[:, None, None] ** 2 * a ** 2 + sy[:, None, None] ** 2 * b ** 2
        return torch.exp(-2 * math.pi ** 2 * q)

    def nsmooth(self, x, T):
        """Normalized (masked) convolution with transfer T; values on the mask."""
        num = torch.fft.irfft2(torch.fft.rfft2(x * self.m) * T, s=(H, W))
        den = torch.fft.irfft2(torch.fft.rfft2(self.m.expand_as(x)) * T, s=(H, W))
        return num / den.clamp(min=0.05)

    def inpaint(self, x):
        """Fill outside the mask with a normalized 3-px smooth continuation."""
        B = len(x)
        T = self.gauss_T(torch.full((B,), 3.0, device=self.dev))
        fill = self.nsmooth(x, T)
        den = torch.fft.irfft2(torch.fft.rfft2(self.m.expand_as(x)) * T, s=(H, W))
        taper = (den / 0.3).clamp(0, 1)
        return x * self.m + (1 - self.m) * fill * taper

    def filt(self, x, T):
        mu = self.mean(x)[:, None, None]
        xf = self.inpaint((x - mu) * self.m)
        y = torch.fft.irfft2(torch.fft.rfft2(xf) * T, s=(H, W)) + mu
        return y * self.m

    def smooth(self, x, sig):
        """Masked Gaussian smoothing with per-sample width (pixels)."""
        return self.nsmooth(x, self.gauss_T(sig)) * self.m

    def grf(self, B, rc, g):
        """Unit-std (over mask) smooth Gaussian random field with cutoff rc (cycles/px)."""
        z = torch.randn((B, H, W), generator=g, device=self.dev)
        T = torch.exp(-(self.r2[None] / rc[:, None, None] ** 2))
        f = torch.fft.irfft2(torch.fft.rfft2(z) * T, s=(H, W))
        f = f - self.mean(f)[:, None, None]
        return f / self.std(f)[:, None, None]

    def rect(self, B, hlo, hhi, wlo, whi, g):
        """Per-sample rectangle masks centred on random observed pixels."""
        c = self.obs[torch.randint(0, len(self.obs), (B,), generator=g, device=self.dev)]
        h = torch.randint(hlo, hhi + 1, (B,), generator=g, device=self.dev)
        w = torch.randint(wlo, whi + 1, (B,), generator=g, device=self.dev)
        y0 = (c[:, 0] - h // 2)[:, None, None]
        x0 = (c[:, 1] - w // 2)[:, None, None]
        return ((self.Yi >= y0) & (self.Yi < y0 + h[:, None, None]) &
                (self.Xi >= x0) & (self.Xi < x0 + w[:, None, None])).float()

    @staticmethod
    def pick(sel, *opts):
        out = opts[-1]
        for i in range(len(opts) - 2, -1, -1):
            shape = (-1,) + (1,) * (out.dim() - 1)
            out = torch.where((sel == i).view(shape), opts[i], out)
        return out

    def out(self, sig, s, sigma=None, eps=None):
        return dict(sig=sig * self.m, s=s, sigma=sigma, eps=eps)

    # ------------------------------------------------------------ TRAIN families
    def f_tell(self, x, lab, g):
        """Random smooth scale-dependent transfer T(ell): bumps, steps (cutoffs), tilts."""
        B = len(x)
        logT = torch.zeros((B,) + self.r.shape, device=self.dev)
        s_main = None
        for comp in range(2):
            r0, _ = self.lu(B, 0.01, 0.4, g)
            w = self.u(B, 0.25, 1.0, g)
            A, s = self.lu(B, 0.02, 0.5, g)
            A = A * self.sign(B, g)
            z = (self.lr[None] - torch.log(r0)[:, None, None]) / w[:, None, None]
            shape = self.choice(B, 3, g)
            bump = torch.exp(-0.5 * z ** 2)
            step = torch.sigmoid(z)
            tilt = 0.5 * F.softplus(z) * w[:, None, None]
            on = torch.ones(B, device=self.dev) if comp == 0 else (torch.rand(B, generator=g, device=self.dev) < 0.4).float()
            logT = logT + on[:, None, None] * A[:, None, None] * self.pick(shape, bump, step, tilt)
            s_main = s if s_main is None else s_main
        return self.out(self.filt(x, torch.exp(logT.clamp(-1.5, 1.0))), s_main)

    def f_nz(self, x, lab, g):
        """Lensing-kernel / n(z)-like change: overall amplitude plus a smooth scale tilt."""
        B = len(x)
        a, s = self.lu(B, 0.02, 0.15, g)
        a = a * self.sign(B, g)
        b, _ = self.lu(B, 0.01, 0.15, g)
        b = b * self.sign(B, g)
        rp, _ = self.lu(B, 0.02, 0.2, g)
        logT = torch.log1p(a)[:, None, None] + b[:, None, None] * torch.tanh(self.lr[None] - torch.log(rp)[:, None, None])
        return self.out(self.filt(x, torch.exp(logT)), s)

    def f_shear(self, x, lab, g):
        """Multiplicative and additive shear-like biases (constant, gradient, large-scale field)."""
        B = len(x)
        mode = self.choice(B, 4, g)
        a, s1 = self.lu(B, 0.02, 0.25, g)
        mult = x * (1 + a * self.sign(B, g))[:, None, None]
        c, s2 = self.lu(B, 0.0005, 0.005, g)
        const = x + (c * self.sign(B, g))[:, None, None]
        th = self.u(B, 0, 2 * math.pi, g)
        ramp = torch.cos(th)[:, None, None] * self.Y + torch.sin(th)[:, None, None] * self.X
        cg, s3 = self.lu(B, 0.001, 0.008, g)
        grad = x + cg[:, None, None] * ramp
        rc, _ = self.lu(B, 0.003, 0.02, g)
        cf, s4 = self.lu(B, 0.0005, 0.005, g)
        field = x + cf[:, None, None] * self.grf(B, rc, g)
        return self.out(self.pick(mode, mult, const, grad, field), self.pick(mode, s1, s2, s3, s4))

    def f_psf(self, x, lab, g):
        """PSF-like smoothing (isotropic/anisotropic, rotated) or over-deconvolution (sharpening)."""
        B = len(x)
        mode = self.choice(B, 3, g)
        s1, st = self.lu(B, 0.15, 1.2, g)
        e = self.u(B, 0.0, 0.85, g) * (torch.rand(B, generator=g, device=self.dev) < 0.6).float()
        th = self.u(B, 0, math.pi, g)
        smooth = self.gauss_T(s1 * (1 - e), s1, th)
        sh, ss = self.lu(B, 0.15, 0.6, g)
        sharp = torch.exp(2 * math.pi ** 2 * sh[:, None, None] ** 2 * self.r2[None]).clamp(max=3.0)
        # smoothing along one axis, sharpening along the other
        c, s = torch.cos(th)[:, None, None], torch.sin(th)[:, None, None]
        aa = c * self.fx + s * self.fy
        bb = -s * self.fx + c * self.fy
        mixed = torch.exp(-2 * math.pi ** 2 * (s1[:, None, None] ** 2 * aa ** 2
                                               - (0.6 * sh)[:, None, None] ** 2 * bb ** 2)).clamp(max=3.0)
        T = self.pick(mode, smooth, sharp, mixed)
        return self.out(self.filt(x, T), self.pick(mode, st, ss, st))

    def f_noise(self, x, lab, g):
        """Noise mis-modelling: level, spatially varying level, non-Gaussian tails, correlated noise."""
        B = len(x)
        mode = self.choice(B, 4, g)
        sig0 = D.NOISE_SIGMA
        a, s1 = self.lu(B, 0.01, 0.1, g)
        level = torch.ones((B, H, W), device=self.dev) * (sig0 * (1 + a * self.sign(B, g)))[:, None, None]
        rc, _ = self.lu(B, 0.002, 0.03, g)
        av, s2 = self.lu(B, 0.03, 0.3, g)
        th = self.u(B, 0, 2 * math.pi, g)
        ramp = torch.cos(th)[:, None, None] * self.Y + torch.sin(th)[:, None, None] * self.X
        fld = torch.where((torch.rand(B, generator=g, device=self.dev) < 0.5)[:, None, None],
                          self.grf(B, rc, g), ramp / 0.6)
        spatial = sig0 * (1 + av[:, None, None] * fld).clamp(min=0.3)
        sigma = self.pick(mode, level, spatial, torch.full_like(level, sig0), torch.full_like(level, sig0))
        z1 = torch.randn((B, H, W), generator=g, device=self.dev)
        z2 = torch.randn((B, H, W), generator=g, device=self.dev)
        # heavy tails: Gaussian scale mixture with log-normal variance (unit mean variance)
        sv, s3 = self.lu(B, 0.1, 1.0, g)
        v = torch.exp(sv[:, None, None] * z2 - 0.5 * sv[:, None, None] ** 2)
        heavy = z1 * torch.sqrt(v)
        # correlated noise at fixed per-pixel variance
        rho, s4 = self.lu(B, 0.03, 0.4, g)
        sc = self.u(B, 0.5, 2.0, g)
        zs = torch.fft.irfft2(torch.fft.rfft2(z2) * self.gauss_T(sc), s=(H, W))
        zs = zs / zs.std((1, 2), keepdim=True)
        corr = torch.sqrt(1 - rho)[:, None, None] * z1 + torch.sqrt(rho)[:, None, None] * zs
        eps = self.pick(mode, z1, z1, heavy, corr)
        return self.out(x, self.pick(mode, s1, s2, s3, s4), sigma=sigma, eps=eps)

    def f_mask(self, x, lab, g):
        """Patch artifacts: signal-free patches, rescaled patches, stitched patches, stripes."""
        B = len(x)
        mode = self.choice(B, 4, g)
        P = torch.zeros_like(x)
        npatch = torch.randint(1, 5, (B,), generator=g, device=self.dev)
        for k in range(4):
            on = (npatch > k).float()[:, None, None]
            P = torch.maximum(P, on * self.rect(B, 8, 120, 8, 60, g))
        P = P * self.m
        hole = x * (1 - P)
        f = self.u(B, 0.3, 2.0, g)
        scaled = x * (1 + (f[:, None, None] - 1) * P)
        stitched = x * (1 - P) + torch.roll(x, 1, 0) * P
        S = torch.zeros_like(x)
        nst = torch.randint(1, 4, (B,), generator=g, device=self.dev)
        for k in range(3):
            on = (nst > k).float()[:, None, None]
            horiz = (torch.rand(B, generator=g, device=self.dev) < 0.7)[:, None, None]
            band = torch.where(horiz, self.rect(B, 3, 30, 2 * W, 2 * W, g), self.rect(B, H // 2, H, 3, 20, g))
            amp, _ = self.lu(B, 0.002, 0.01, g)
            S = S + on * band * (amp * self.sign(B, g))[:, None, None]
        stripes = x + S
        s = torch.rand(B, device=self.dev, generator=g)
        return self.out(self.pick(mode, hole, scaled, stitched, stripes), s)

    def f_nonlin(self, x, lab, g):
        """Local non-Gaussianity (f_NL / modified-gravity-like) at roughly fixed power."""
        B = len(x)
        sig = torch.where(torch.rand(B, generator=g, device=self.dev) < 0.3, 0.0, self.u(B, 0.5, 2.0, g))
        xs = torch.where((sig > 0)[:, None, None], self.smooth(x, sig.clamp(min=0.1)), x)
        mu = self.mean(xs)[:, None, None]
        q = (xs - mu) / self.std(xs)[:, None, None]
        a, s = self.lu(B, 0.01, 0.25, g)
        a = a * self.sign(B, g)
        b, _ = self.lu(B, 0.005, 0.08, g)
        b = b * self.sign(B, g) * (torch.rand(B, generator=g, device=self.dev) < 0.5).float()
        sd = self.std(x)[:, None, None]
        y = x + sd * (a[:, None, None] * (q ** 2 - 1) + b[:, None, None] * (q ** 3 - 3 * q)) * self.m
        keep = (torch.rand(B, generator=g, device=self.dev) < 0.7)[:, None, None]
        mx = self.mean(x)[:, None, None]
        my = self.mean(y)[:, None, None]
        y2 = mx + (y - my) * sd / self.std(y)[:, None, None]
        return self.out(torch.where(keep, y2, y), s)

    # ------------------------------------------------------------ HELD-OUT families
    def T_at(self, lt):
        lt = lt.clamp(float(self.lT[0]), float(self.lT[-1]))
        i = torch.searchsorted(self.lT, lt.contiguous()).clamp(1, len(self.lT) - 1)
        w = ((lt - self.lT[i - 1]) / (self.lT[i] - self.lT[i - 1]))[:, None, None]
        return self.Ttab[i - 1] * (1 - w) + self.Ttab[i] * w

    def f_hmcode(self, x, lab, g):
        """HMcode-2020 baryon transfer extrapolated beyond the prior (logT_AGN<7.1, >8.6) or DMO."""
        B = len(x)
        mode = self.choice(B, 3, g)
        lt_new = self.pick(mode, self.u(B, 6.0, 7.1, g), self.u(B, 8.6, 9.5, g), torch.zeros(B, device=self.dev))
        told = self.T_at(torch.as_tensor(lab[:, 2], dtype=torch.float32, device=self.dev))
        tnew = torch.where((mode == 2)[:, None, None], torch.ones_like(told), self.T_at(lt_new))
        p = self.u(B, 0.5, 1.0, g)
        ratio = (tnew / told) ** p[:, None, None]
        return self.out(self.filt(x, ratio), (p - 0.5) / 0.5)

    def f_halo(self, x, lab, g):
        """Mass-dependent halo-profile change (BCM-like): flatten/steepen peak cores or
        redistribute mass from peaks to surroundings; optionally at conserved variance."""
        B = len(x)
        s1 = self.u(B, 0.5, 1.5, g)
        xs = self.smooth(x, s1)
        sd = self.std(xs)[:, None, None]
        t = self.u(B, 1.0, 3.5, g)[:, None, None]
        E = torch.relu(xs - self.mean(xs)[:, None, None] - t * sd) * self.m
        mode = self.choice(B, 2, g)
        a, s = self.lu(B, 0.1, 0.8, g)
        a = (a * self.sign(B, g))[:, None, None]
        core = x - a * self.smooth(E, s1)
        s2 = self.u(B, 2.0, 6.0, g)
        redist = x - a * E + a * self.smooth(E, s2)
        y = self.pick(mode, core, redist)
        keep = (torch.rand(B, generator=g, device=self.dev) < 0.5)[:, None, None]
        mx, my = self.mean(x)[:, None, None], self.mean(y)[:, None, None]
        y2 = mx + (y - my) * self.std(x)[:, None, None] / self.std(y)[:, None, None]
        return self.out(torch.where(keep, y2, y), s)

    def f_ia(self, x, lab, g):
        """Intrinsic-alignment-like: subtract a fraction of smoothed kappa (NLA-like) or a
        tidal-shear-squared term (TATT-like quadratic)."""
        B = len(x)
        mode = self.choice(B, 2, g)
        a, s = self.lu(B, 0.03, 0.3, g)
        a = (a * self.sign(B, g, p=0.8))[:, None, None]
        sg, _ = self.lu(B, 1.0, 8.0, g)
        nla = x - a * self.smooth(x, sg)
        st = self.u(B, 1.0, 4.0, g)
        mu = self.mean(x)[:, None, None]
        kf = torch.fft.rfft2(self.inpaint((x - mu) * self.m)) * self.gauss_T(st)
        r2 = self.r2.clamp(min=1e-12)
        g1 = torch.fft.irfft2(kf * ((self.fx ** 2 - self.fy ** 2) / r2), s=(H, W))
        g2 = torch.fft.irfft2(kf * (2 * self.fx * self.fy / r2), s=(H, W))
        q = (g1 ** 2 + g2 ** 2) * self.m
        q = (q - self.mean(q)[:, None, None]) / self.std(q)[:, None, None]
        tatt = x - 0.5 * a * self.std(x)[:, None, None] * q
        # partial replacement of the lensing signal by a phase-randomized (Gaussian) copy
        fa = torch.fft.rfft2(self.inpaint((x - mu) * self.m))
        ph = torch.exp(2j * math.pi * torch.rand(fa.shape, generator=g, device=self.dev))
        ph[:, 0, 0] = 1.0
        G = torch.fft.irfft2(fa.abs() * ph, s=(H, W)) * self.m
        G = G - self.mean(G)[:, None, None]
        G = G / self.std(G)[:, None, None] * self.std(x)[:, None, None]
        fr, _ = self.lu(B, 0.03, 0.4, g)
        lin = torch.rand(B, generator=g, device=self.dev) < 0.5
        c1 = torch.where(lin, 1 - fr, torch.sqrt(1 - fr))[:, None, None]
        c2 = torch.where(lin, fr, torch.sqrt(fr))[:, None, None]
        gauss = mu + c1 * (x - mu) + c2 * G
        mode = torch.where(torch.rand(B, generator=g, device=self.dev) < 0.4, 2, mode)
        return self.out(self.pick(mode, nla, tatt, gauss), s)

    # ------------------------------------------------------------ coordinator families
    def f_srcclust(self, x, lab, g):
        """Source clustering: galaxy counts (noise level) modulated by local foreground
        structure traced by smoothed kappa, optionally with Poisson count scatter."""
        B = len(x)
        sg = self.u(B, 1.0, 6.0, g)
        q = self.smooth(x, sg)
        q = (q - self.mean(q)[:, None, None]) / self.std(q)[:, None, None]
        a, s = self.lu(B, 0.03, 0.4, g)
        a = (a * self.sign(B, g, p=0.75))[:, None, None]
        nbar = 120.0 * (1 + a * q).clamp(0.2, 3.0)          # more sources behind overdensities
        pois = torch.rand(B, generator=g, device=self.dev) < 0.5
        cnt = torch.poisson(nbar, generator=g).clamp(min=1.0)
        n = torch.where(pois[:, None, None], cnt, nbar)
        return self.out(x, s, sigma=D.NOISE_SIGMA * torch.sqrt(120.0 / n))

    def f_poisson(self, x, lab, g):
        """Poisson galaxy counts per pixel: sigma_pix = sigma0 sqrt(nbar_ref / N), N ~ Poisson(nbar),
        with nbar from 15 to 600 (level kept, or level following nbar)."""
        B = len(x)
        nb, s = self.lu(B, 15.0, 600.0, g)
        cnt = torch.poisson(nb[:, None, None].expand(B, H, W).contiguous(), generator=g).clamp(min=1.0)
        keep = torch.rand(B, generator=g, device=self.dev) < 0.7
        ref = torch.where(keep, nb, torch.full_like(nb, 120.0))[:, None, None]
        return self.out(x, 1 - s, sigma=D.NOISE_SIGMA * torch.sqrt(ref / cnt))

    def f_psfpat(self, x, lab, g):
        """PSF-residual additive coherent anisotropic pattern, correlation lengths 0.1-1.5 deg
        (3-45 px), random orientation and anisotropy."""
        B = len(x)
        L1, _ = self.lu(B, 3.0, 45.0, g)
        ar = self.u(B, 1.0, 6.0, g)
        th = self.u(B, 0, math.pi, g)
        T = self.gauss_T(L1, L1 / ar, th)
        z = torch.randn((B, H, W), generator=g, device=self.dev)
        pat = torch.fft.irfft2(torch.fft.rfft2(z) * T, s=(H, W))
        pat = (pat - self.mean(pat)[:, None, None]) / self.std(pat)[:, None, None]
        rms, s = self.lu(B, 0.0005, 0.006, g)
        return self.out(x + rms[:, None, None] * pat, s)

    def f_holes(self, x, lab, g):
        """Extra survey masks: circular star holes and rectangular gaps set to 0 after noise."""
        B = len(x)
        nh = torch.randint(1, 40, (B,), generator=g, device=self.dev)
        keep = torch.ones_like(x)
        for k in range(40):
            on = (nh > k)[:, None, None]
            c = self.obs[torch.randint(0, len(self.obs), (B,), generator=g, device=self.dev)]
            rad, _ = self.lu(B, 1.0, 8.0, g)
            d2 = (self.Yi - c[:, 0, None, None]) ** 2 + (self.Xi - c[:, 1, None, None]) ** 2
            keep = keep * torch.where(on & (d2 <= rad[:, None, None] ** 2), 0.0, 1.0)
        rect = (torch.rand(B, generator=g, device=self.dev) < 0.3)[:, None, None]
        keep = keep * torch.where(rect, 1 - self.rect(B, 4, 60, 4, 30, g), torch.ones_like(x))
        o = self.out(x, nh.float() / 40)
        o['post'] = keep
        return o

    # ------------------------------------------------------------ apply + noise
    def make(self, fam, clean, lab, g, gnoise=None):
        """Noisy synthetic OoD maps from clean maps (noise drawn from gnoise or g)."""
        gn = g if gnoise is None else gnoise
        o = self.fam[fam](clean, lab, g)
        sig = o['sig']
        sigma = D.NOISE_SIGMA if o['sigma'] is None else o['sigma']
        eps = o['eps'] if o['eps'] is not None else torch.randn(sig.shape, generator=gn, device=self.dev)
        x = ((sig + sigma * eps * self.m).half().float()) * self.m
        if o.get('post') is not None:
            x = x * o['post']
        return x, o['s']
