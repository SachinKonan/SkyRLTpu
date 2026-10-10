"""Additional trainable-statistic *baseline* using JAX FFT + shrinkage Gaussian.

Not a reproduction of the competition's conditional power-spectrum/MCMC
baseline. Included as an inexpensive second search starting point.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from sklearn.covariance import LedoitWolf

BANDS = 20


@lru_cache(maxsize=10)
def radial_bin_map(height, width):
    ky = np.fft.fftfreq(height)[:, None]
    kx = np.fft.rfftfreq(width)[None, :]
    r = np.sqrt(ky*ky + kx*kx)
    edges = np.geomspace(.004, .72, BANDS + 1)
    bins = np.clip(np.digitize(r, edges)-1, 0, BANDS-1)
    return jnp.asarray(bins.reshape(-1), dtype=jnp.int32)


def spectral_features(maps, mask):
    x = jnp.asarray(maps, jnp.float32)
    footprint = jnp.asarray(mask, jnp.float32)
    x = x * footprint
    b, h, w = x.shape
    power = jnp.abs(jnp.fft.rfft2(x)) ** 2
    index = radial_bin_map(h, w)
    def band_summary(p):
        vec = p.reshape(-1)
        sums = jnp.bincount(index, weights=vec, length=BANDS)
        cnt = jnp.bincount(index, length=BANDS)
        return jnp.log1p(sums / jnp.maximum(cnt, 1))
    bands = jax.vmap(band_summary)(power)
    nobs = jnp.maximum(jnp.sum(footprint), 1.)
    mean = jnp.sum(x, axis=(1,2)) / nobs
    cen = (x - mean[:, None, None]) * footprint
    var = jnp.sum(cen**2, axis=(1,2)) / nobs
    skew = jnp.sum(cen**3, axis=(1,2)) / nobs / (var + 1e-8) ** 1.5
    kurt = jnp.sum(cen**4, axis=(1,2)) / nobs / (var + 1e-8) ** 2
    extras = jnp.stack((mean, jnp.sqrt(var+1e-9), skew, kurt), axis=1)
    return np.asarray(jnp.concatenate([bands, extras], axis=1))


def fit(train, val, config, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bs = int(config['batch_size'])
    # Interprets steps as the feature-extraction work budget.
    target = min(len(train), max(bs, int(config['steps']) * bs))
    feats = []
    used = 0
    for batch in train.iter_batches(batch_size=bs, epochs=1, shuffle=True,
                                    seed=int(config['seed']), limit=target):
        feat = spectral_features(batch['maps'], batch['mask'])
        feats.append(feat)
        used += len(feat)
    z = np.concatenate(feats, axis=0).astype(np.float64)
    center = z.mean(axis=0)
    scale = np.maximum(z.std(axis=0), 1e-5)
    whitened = (z-center) / scale
    covariance = LedoitWolf().fit(whitened)
    np.savez_compressed(out/'model.npz', center=center, scale=scale,
                        mean=covariance.location_, precision=covariance.precision_)
    return {'feature_train_examples': int(used), 'num_features': z.shape[1]}


def load_model(out_dir, config):
    with np.load(Path(out_dir)/'model.npz') as f:
        return {key: f[key] for key in f.files}


def predict(model, maps, mask, config):
    z = spectral_features(maps, mask)
    residual = (z - model['center']) / model['scale'] - model['mean']
    sq = np.sum((residual @ model['precision']) * residual, axis=1)
    return np.sqrt(np.maximum(sq, 0.))
