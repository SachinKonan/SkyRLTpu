"""Synthetic, *not official*, physical proxy shifts for local development rewards.

All shifts act on NOISELESS maps; identical observational noise is then added to
InD and OoD. The proxy is not a substitute for the hidden competition test.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter

FAMILIES = ("amplitude", "spectral_tilt", "nonlinear_peak", "smoothing")


def _variance_match(reference, shifted, mask):
    r = reference[mask]
    s = shifted[mask]
    sr = max(np.std(r), 1e-8)
    ss = max(np.std(s), 1e-8)
    return (shifted - np.mean(s)) * (sr / ss) + np.mean(r)


def proxy_shift(clean: np.ndarray, mask: np.ndarray, family: str,
                rng: np.random.Generator, strength=1.0) -> np.ndarray:
    """Deterministic for a supplied RNG, 2D clean field -> shifted 2D field."""
    if family not in FAMILIES:
        raise ValueError(f"Unknown proxy family {family}")
    x = np.asarray(clean, dtype=np.float32) * mask
    sign = -1 if rng.integers(2) else 1
    v = float(strength)
    if family == "amplitude":
        y = x * (1 + sign * .12 * v)
    elif family == "nonlinear_peak":
        obs = x[mask]
        sd = max(float(np.std(obs)), 1.e-6)
        u = x / sd
        quadratic = u * u - float(np.mean(u[mask] * u[mask]))
        y = x + sign * (.14 * v * sd) * quadratic
        y = _variance_match(x, y, mask)
    elif family == "smoothing":
        # Smooth the signal only, before noise. Normalize masked, edge-aware.
        blur_mask = gaussian_filter(mask.astype(np.float32), sigma=(2.0 * v, 1.5 * v))
        smoothed = gaussian_filter(x, sigma=(2.0 * v, 1.5 * v))
        y = smoothed / np.maximum(blur_mask, 1e-3)
        y = _variance_match(x, y, mask)
    else:  # spectral_tilt
        freq = np.fft.rfft2(x)
        kh = np.fft.fftfreq(x.shape[0])[:, None]
        kw = np.fft.rfftfreq(x.shape[1])[None, :]
        k = np.sqrt(kh * kh + kw * kw)
        # Change relative scale power while keeping roughly the same RMS.
        tilt = np.clip((np.maximum(k, 0.015) / .09) ** (sign * .24 * v), .55, 1.8)
        y = np.fft.irfft2(freq * tilt, s=x.shape).real
        y = _variance_match(x, y, mask)
    return (np.asarray(y, dtype=np.float32) * mask)
