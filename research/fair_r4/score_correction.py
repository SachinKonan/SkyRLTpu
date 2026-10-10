"""Read-only polynomial score correction, fitted on normal validation maps."""
import numpy as np


def design(x, powers):
    return np.prod(np.asarray(x, np.float64)[:, None, :] ** powers[None, :, :], axis=2)


def predict(x, model):
    return (design(x, model['powers'])-model['center'])/model['scale']@model['weights']+model['bias']


def predict_sd(x, model):
    variance = model['variance_factor']*np.exp(np.clip(predict(x, model), -20, 20))
    return np.sqrt(np.maximum(variance, model['sd_floor']**2))


def load(path):
    with np.load(path, allow_pickle=False) as f:
        return tuple({k.removeprefix(prefix): f[k] for k in f.files if k.startswith(prefix)}
                     for prefix in ('location_', 'variance_'))


def score(energy, coordinates, models):
    energy = np.asarray(energy, np.float64)
    if np.any(energy <= 0) or not np.isfinite(energy).all():
        raise ValueError('Expected strictly positive finite profile energy')
    location, variance = models
    residual = np.log(energy)-predict(coordinates, location)
    return dict(centered=residual, scaled=residual/predict_sd(coordinates, variance))
