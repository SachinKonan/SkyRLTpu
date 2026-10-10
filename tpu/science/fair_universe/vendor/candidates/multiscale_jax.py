"""Multiscale JAX anomaly detector with in-distribution-only calibration.

The expensive image operations and distance scoring are JAX/XLA operations.
Small covariance fits use NumPy/scikit-learn on the host. This is a new local
experimental candidate, NOT a reproduction of a published or winning solution.

modes: global, conditional, neighbor, ensemble, max_ensemble
The candidate never synthesizes anomalies or sees reward/holdout labels.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from sklearn.covariance import LedoitWolf


@lru_cache(maxsize=8)
def _extractor(height: int, width: int):
    ky = np.fft.fftfreq(height)[:, None]
    kx = np.fft.rfftfreq(width)[None, :]
    radius = np.sqrt(ky * ky + kx * kx)
    edges = np.geomspace(max(1. / max(height, width), .003), .71, 13)
    bins = jnp.asarray(np.clip(np.digitize(radius.ravel(), edges)-1, 0, 11))
    counts = jnp.maximum(jnp.bincount(bins, length=12), 1)
    kernels = jnp.asarray(np.stack([
        np.exp(-2. * np.pi**2 * sigma**2 * radius**2)
        for sigma in (.8, 1.6, 3.2, 6.4)]), dtype=jnp.float32)

    @jax.jit
    def extract(maps, footprint):
        x = jnp.asarray(maps, jnp.float32) * footprint
        count = jnp.maximum(footprint.sum(), 1.)
        fft = jnp.fft.rfft2(x)
        power = jnp.abs(fft)**2 / count
        def summarize(p):
            return jnp.log(jnp.bincount(bins, weights=p.ravel(), length=12)
                           / counts + 1.e-9)
        features = [jax.vmap(summarize)(power)]
        smask = jnp.fft.irfft2(jnp.fft.rfft2(footprint)[None] * kernels,
                              s=(height, width))
        smoothed = jnp.fft.irfft2(fft[:, None] * kernels[None],
                                 s=(height, width)) / jnp.maximum(smask[None], .1)
        fields = [x] + [smoothed[:, s] for s in range(4)]
        standardized = []
        for field in fields:
            mean = (field * footprint).sum(axis=(1, 2)) / count
            centered = (field - mean[:, None, None]) * footprint
            variance = (centered**2).sum(axis=(1, 2)) / count
            sd = jnp.sqrt(variance + 1.e-10)
            z = centered / sd[:, None, None]
            standardized.append(z)
            skew = (z**3 * footprint).sum(axis=(1, 2)) / count
            kurt = (z**4 * footprint).sum(axis=(1, 2)) / count
            positive = ((z > 2.) * footprint).sum(axis=(1, 2)) / count
            negative = ((z < -2.) * footprint).sum(axis=(1, 2)) / count
            features.append(jnp.stack([mean, jnp.log(sd), skew, kurt,
                                        positive, negative], axis=1))
        # Cross-scale interactions, not just a larger radial power spectrum.
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 4)]:
            za, zb = standardized[a], standardized[b]
            features.append(jnp.stack([
                (za * zb * footprint).sum(axis=(1, 2)) / count,
                (za**2 * zb * footprint).sum(axis=(1, 2)) / count,
                (za**2 * zb**2 * footprint).sum(axis=(1, 2)) / count], axis=1))
        return jnp.concatenate(features, axis=1)
    return extract


def features(maps, mask):
    if np.ndim(maps) != 3 or tuple(np.shape(maps)[1:]) != tuple(np.shape(mask)):
        raise ValueError('Expected maps[B,H,W] and mask[H,W]')
    return np.asarray(_extractor(*mask.shape)(jnp.asarray(maps), jnp.asarray(mask, jnp.float32)))


def _read(view, config, seed, limit):
    values, labels = [], []
    for batch in view.iter_batches(batch_size=int(config['batch_size']), epochs=1,
                                   shuffle=True, seed=seed, limit=limit):
        values.append(features(batch['maps'], batch['mask']))
        labels.append(batch['labels'][:, :2])
    return np.concatenate(values).astype(np.float64), np.concatenate(labels)


def _whitener(z, ridge):
    cov = LedoitWolf().fit(z).covariance_
    eig, vectors = np.linalg.eigh(cov + float(ridge) * np.eye(cov.shape[0]))
    return (vectors / np.sqrt(np.maximum(eig, 1.e-7))) @ vectors.T


@jax.jit
def _distances(z, mean, global_whiten, centers, conditional_whiten, references):
    a = (z - mean) @ global_whiten
    global_score = jnp.sum(a*a, axis=1)
    b = z @ conditional_whiten
    prototypes = centers @ conditional_whiten
    d = jnp.maximum(jnp.sum(b*b, axis=1)[:, None]
                    + jnp.sum(prototypes*prototypes, axis=1)[None]
                    - 2. * b @ prototypes.T, 0.)
    conditional_score = jnp.min(d, axis=1)
    dknn = jnp.maximum(jnp.sum(a*a, axis=1)[:, None]
                       + jnp.sum(references*references, axis=1)[None]
                       - 2. * a @ references.T, 0.)
    # Average five nearest distances: less noisy than distance to one map.
    nearest, _ = jax.lax.top_k(-dknn, min(5, references.shape[0]))
    neighbor_score = -nearest.mean(axis=1)
    return jnp.stack([global_score, conditional_score, neighbor_score], axis=1)


def _raw(model, f):
    z = (np.asarray(f, np.float64) - model['center']) / model['scale']
    keys = ['mean', 'global_whiten', 'centers', 'conditional_whiten', 'references']
    return np.asarray(_distances(jnp.asarray(z, jnp.float32), *[
        jnp.asarray(model[k], jnp.float32) for k in keys]))


def fit(train, val, config, out_dir):
    target = min(len(train), max(int(config['batch_size']),
                                int(config['steps']) * int(config['batch_size'])))
    x, theta = _read(train, config, int(config['seed']), target)
    center = np.mean(x, axis=0)
    scale = np.maximum(np.std(x, axis=0), 1.e-5)
    z = (x-center) / scale
    mean = z.mean(axis=0)
    global_whiten = _whitener(z, config.get('evolved_ridge', .05))
    # Training labels specify known cosmologies. Fit a tied-noise model around
    # their feature means; no cosmology labels are required at prediction time.
    groups, inverse = np.unique(np.round(theta, 6), axis=0, return_inverse=True)
    if len(groups) > 512 or min(np.bincount(inverse)) < 4:
        from sklearn.cluster import KMeans
        fit = KMeans(n_clusters=min(16, max(1, len(z)//32)), n_init=3,
                     random_state=int(config['seed'])).fit(z)
        inverse = fit.labels_
    centers = np.stack([z[inverse == i].mean(axis=0) for i in range(inverse.max()+1)])
    residual = z - centers[inverse]
    conditional_whiten = _whitener(residual, config.get('evolved_ridge', .05))
    rng = np.random.default_rng(int(config['seed']))
    reference_ids = rng.choice(len(z), min(len(z), 2048), replace=False)
    references = ((z-mean) @ global_whiten)[reference_ids]
    model = dict(center=center, scale=scale, mean=mean, global_whiten=global_whiten,
                 centers=centers, conditional_whiten=conditional_whiten,
                 references=references)
    calibration, _ = _read(val, config, int(config['seed'])+111,
                            min(len(val), int(config.get('calibration_max', 2048))))
    model['calibration'] = np.sort(_raw(model, calibration), axis=0)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/'model.npz', **model)
    return {'train_examples': len(z), 'features': int(z.shape[1]),
            'centers': len(centers), 'calibration_examples': len(calibration),
            'uses_proxy_labels': False}


def load_model(out_dir, config):
    with np.load(Path(out_dir)/'model.npz') as file:
        return {k: file[k] for k in file.files}


def all_scores(model, maps, mask, config):
    raw = _raw(model, features(maps, mask))
    cal = model['calibration']
    # InD validation only. Rank calibration is not calibrated P(OoD | x).
    unusual = np.stack([-np.log((len(cal) + 1 - np.searchsorted(cal[:, i], raw[:, i],
                               side='right')) / (len(cal)+1.)) for i in range(3)], axis=1)
    return dict(global_score=raw[:, 0], conditional=raw[:, 1], neighbor=raw[:, 2],
                ensemble=unusual.mean(axis=1), max_ensemble=unusual.max(axis=1))


def predict(model, maps, mask, config):
    mode = str(config.get('evolved_mode', 'conditional'))
    mode = 'global_score' if mode == 'global' else mode
    scores = all_scores(model, maps, mask, config)
    if mode not in scores:
        raise ValueError(f'Unknown evolved_mode {mode}')
    return scores[mode]
