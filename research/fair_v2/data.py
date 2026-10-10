"""Data access, fixed realization split, noise and synthetic diagnostic shifts.

Split: the prior team's seed-2026 realization split (manifest in the original
worktree's .fair/data-v1/manifest.json), same realizations for every cosmology:
  fit   = train (160) + val (32) realizations   -> model fitting
  calib = development (32)                       -> normal score standardization
  audit = locked (32)                            -> normal tail + synthetic shifts
"""
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get('FAIR_DATA', '/scratch/gpfs/ZHUANGL/sk7524/fair-data'))
TRAIN = ROOT / 'WIDE12H_bin2_2arcmin_kappa_newrealization.npy'
LABELS = ROOT / 'label_newrealization.npy'
MASK = ROOT / 'WIDE12H_bin2_2arcmin_mask.npy'
TEST = ROOT / 'WIDE12H_bin2_2arcmin_kappa_test_phase2_new_v2.npy'
MANIFEST = Path('/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-fair-universe/.fair/data-v1/manifest.json')
NOISE_SIGMA = 0.4 / math.sqrt(2 * 30 * 4)


def split_ids():
    ids = json.loads(MANIFEST.read_text())['realization_ids']
    return dict(fit=sorted(ids['train'] + ids['val']), calib=sorted(ids['development']),
                audit=sorted(ids['locked']))


def load_train(ram=True):
    x = np.load(TRAIN, mmap_mode='r')
    return np.ascontiguousarray(x) if ram else x


def labels():
    return np.load(LABELS).astype(np.float64)


def mask():
    return np.load(MASK)


def index(part):
    """(cosmology, realization) pairs of a split, cosmology-major.

    v3 splits ('fit3', 'calib3', 'audit3') are MAP-level: every realization
    index carries its own fixed baryon model (T_AGN, f0) in all 101
    cosmologies, so a realization-level split would hold out whole baryon
    models. Map-level splits keep all 256 baryon models in the fit set.
    """
    if part.endswith('3'):
        u = np.random.default_rng(2027).random((101, 256))
        sel = {'fit3': u < 0.8, 'calib3': (u >= 0.8) & (u < 0.9), 'audit3': u >= 0.9}[part]
        c, r = np.nonzero(sel)
        return np.stack([c, r], 1).astype(np.int64)
    r = split_ids()[part]
    return np.array([(c, j) for c in range(101) for j in r], dtype=np.int64)


def add_noise(clean_full, mask_t, gen, sigma=NOISE_SIGMA, quantize=True):
    """Shape noise on observed pixels; quantize like the float16 test files."""
    noisy = clean_full + sigma * torch.randn(clean_full.shape, generator=gen, device=clean_full.device) * mask_t
    if quantize:
        noisy = noisy.half().float()
    return noisy * mask_t


# --------------------------------------------------------------------------
# Synthetic diagnostic shifts (applied to clean maps before noise). These are
# physically motivated stand-ins for unknown alternative physics; they are
# diagnostics only and never used for fitting.
# --------------------------------------------------------------------------
def _filter(k, mask_t, transfer):
    f = torch.fft.rfft2(k * mask_t)
    return torch.fft.irfft2(f * transfer, s=k.shape[-2:]) * mask_t


def _smooth(k, mask_t, sy, sx):
    H, W = k.shape[-2:]
    fy = torch.fft.fftfreq(H, device=k.device)[:, None]
    fx = torch.fft.rfftfreq(W, device=k.device)[None, :]
    g = torch.exp(-2 * math.pi ** 2 * (sy * sy * fy * fy + sx * sx * fx * fx))
    num = torch.fft.irfft2(torch.fft.rfft2(k * mask_t) * g, s=(H, W))
    den = torch.fft.irfft2(torch.fft.rfft2(mask_t) * g, s=(H, W)).clamp(min=0.05)
    return num / den * mask_t


def radial(k):
    H, W = k.shape[-2:]
    fy = torch.fft.fftfreq(H, device=k.device)[:, None]
    fx = torch.fft.rfftfreq(W, device=k.device)[None, :]
    return torch.sqrt(fy ** 2 + fx ** 2)


SHIFTS = {
    # multiplicative shear calibration bias
    'mult+0.05': dict(kind='mult', a=1.05), 'mult-0.05': dict(kind='mult', a=0.95),
    # resolution / PSF-like smoothing of the signal
    'smooth0.5': dict(kind='smooth', sy=0.5, sx=0.5), 'aniso0.7': dict(kind='smooth', sy=0.0, sx=0.7),
    # misestimated shape noise
    'noise+0.05': dict(kind='noise', a=1.05), 'noise-0.05': dict(kind='noise', a=0.95),
    'noisegrad0.15': dict(kind='noisegrad', a=0.15), 'noise_t5': dict(kind='noise_t', df=5),
    # scale-dependent suppression/enhancement beyond a smooth baryon transfer
    'supp0.15': dict(kind='transfer', a=0.15, r0=0.08), 'boost0.15': dict(kind='transfer', a=-0.15, r0=0.08),
    # mass-dependent peak flattening (halo-profile change, BCM-like)
    'peakcut0.3': dict(kind='peakcut', a=0.3, t=2.5),
    # large-scale additive mode (e.g. residual systematics)
    'gradient': dict(kind='gradient', a=0.004),
    # baryonic feedback outside the training prior (HMcode-2020 T(k) ratio,
    # single effective lensing distance; grid from the public phase-2 toolkit)
    'bary9.0': dict(kind='baryon', logT=9.0), 'bary6.5': dict(kind='baryon', logT=6.5),
    'baryDMO': dict(kind='baryon', logT=None),
    # Systematics planned in the organizers' first README but absent from training (appended
    # at the end so earlier shift seeds 300+i are unchanged):
    # source clustering: noise level anti-correlated with local foreground structure
    'srcclust0.15': dict(kind='srcclust', a=0.15),
    # Poisson galaxy counts per pixel (n_g=30/arcmin^2, 4 arcmin^2 -> mean 120 galaxies)
    'poisson120': dict(kind='poisson', n=120.0),
    # intrinsic-alignment-like: part of the signal replaced by a Gaussianized (phase-randomized) copy
    'ia0.2': dict(kind='ia', a=0.2),
    # PSF-residual-like additive coherent anisotropic pattern
    'psfpat': dict(kind='psfpat', rms=0.003),
}

TK_GRID = ROOT / 'tk_grid.npz'
CHI_EFF = 1105.0   # h^-1 Mpc, comoving distance to z~0.4 (Om~0.3)


def _baryon(clean, mask_t, labels, logT_new):
    grid = np.load(TK_GRID)
    k = grid['k']
    keys = sorted((float(n[2:]), n) for n in grid.files if n.startswith('lT'))
    def T(lt):
        return grid[min(keys, key=lambda p: abs(p[0] - float(lt)))[1]]
    ell = 2 * math.pi * radial(clean) / math.radians(2.0 / 60.0)
    kk = (ell / CHI_EFF).cpu().numpy()
    tnew = np.ones_like(kk) if logT_new is None else np.interp(kk, k, T(logT_new))
    out = torch.empty_like(clean)
    for i in range(len(clean)):
        told = np.interp(kk, k, T(labels[i, 2]))
        ratio = torch.as_tensor(tnew / told, device=clean.device, dtype=torch.float32)
        img = clean[i] * mask_t
        mean = img[mask_t > 0].mean()
        f = torch.fft.rfft2((img - mean) * mask_t)
        out[i] = (torch.fft.irfft2(f * ratio, s=img.shape) + mean) * mask_t
    return out


def apply_shift(name, clean, mask_t, gen, labels=None):
    """Returns (signal_full, noise_sigma_map_or_None, noise_kind)."""
    s = SHIFTS[name]
    kind = s['kind']
    if kind == 'baryon':
        return _baryon(clean, mask_t, labels, s['logT']), None, 'gauss'
    if kind == 'mult':
        return clean * s['a'], None, 'gauss'
    if kind == 'smooth':
        return _smooth(clean, mask_t, s['sy'], s['sx']), None, 'gauss'
    if kind == 'noise':
        return clean, NOISE_SIGMA * s['a'], 'gauss'
    if kind == 'noisegrad':
        H = clean.shape[-2]
        ramp = torch.linspace(1 - s['a'], 1 + s['a'], H, device=clean.device)[:, None]
        return clean, NOISE_SIGMA * ramp, 'gauss'
    if kind == 'noise_t':
        return clean, NOISE_SIGMA, ('t', s['df'])
    if kind == 'transfer':
        r = radial(clean)
        x2 = (r / s['r0']) ** 2
        return _filter(clean, mask_t, 1 - s['a'] * x2 / (1 + x2)), None, 'gauss'
    if kind == 'peakcut':
        sm = _smooth(clean, mask_t, 1.0, 1.0)
        sd = sm[:, mask_t > 0].std(1, keepdim=True)[:, :, None]
        excess = torch.relu(sm - s['t'] * sd)
        return clean - s['a'] * _smooth(excess, mask_t, 1.0, 1.0) * mask_t, None, 'gauss'
    if kind == 'srcclust':
        g = _smooth(clean, mask_t, 2.0, 2.0)
        g = g / g[:, mask_t > 0].std(1)[:, None, None]
        return clean, NOISE_SIGMA * (1 - s['a'] * g).clamp(0.3, 2.0), 'gauss'
    if kind == 'poisson':
        lam = torch.full(clean.shape, s['n'], device=clean.device)
        counts = torch.poisson(lam, generator=gen).clamp(min=1.0)
        return clean, NOISE_SIGMA * torch.sqrt(s['n'] / counts), 'gauss'
    if kind == 'ia':
        f = torch.fft.rfft2(clean * mask_t)
        phase = torch.exp(2j * math.pi * torch.rand(f.shape, generator=gen, device=clean.device))
        grf = torch.fft.irfft2(f.abs() * phase, s=clean.shape[-2:]) * mask_t
        return (1 - s['a']) * clean + s['a'] * grf, None, 'gauss'
    if kind == 'psfpat':
        H, W = clean.shape[-2:]
        fy = torch.fft.fftfreq(H, device=clean.device)[:, None]
        fx = torch.fft.rfftfreq(W, device=clean.device)[None, :]
        filt = torch.exp(-((fy / 0.01) ** 2 + (fx / 0.04) ** 2))
        white = torch.randn(clean.shape, generator=gen, device=clean.device)
        pat = torch.fft.irfft2(torch.fft.rfft2(white) * filt, s=(H, W))
        pat = pat / pat.flatten(1).std(1)[:, None, None] * s['rms']
        return clean + pat * mask_t, None, 'gauss'
    if kind == 'gradient':
        H = clean.shape[-2]
        ramp = torch.linspace(-1, 1, H, device=clean.device)[:, None]
        return clean + s['a'] * ramp * mask_t, None, 'gauss'
    raise KeyError(name)


def noisy_shift(name, clean, mask_t, gen, labels=None):
    sig, sigma, kind = apply_shift(name, clean, mask_t, gen, labels)
    sigma = NOISE_SIGMA if sigma is None else sigma
    if kind == 'gauss':
        eps = torch.randn(sig.shape, generator=gen, device=sig.device)
    else:
        df = kind[1]
        z = torch.randn(sig.shape, generator=gen, device=sig.device)
        chi2 = torch.distributions.Chi2(torch.tensor(float(df), device=sig.device)).sample(sig.shape)
        eps = z / torch.sqrt(chi2 / df) * math.sqrt((df - 2) / df)
    return ((sig + sigma * eps * mask_t).half().float()) * mask_t
