"""New-idea example (JAX port of replay variant r000c00: 8 bands): anisotropy of the power spectrum.

For 8 log |k| bands: log power in |kx| > |ky| wedges minus log power in |ky| > |kx| wedges, plus the
log power in thin wedges along each axis relative to the band. A statistically isotropic map gives ~0 up to
the mask's own anisotropy, which is fixed and absorbed by the normal model.
"""
import numpy as np
import jax.numpy as jnp

H, W = 1424, 176
EDGES = np.geomspace(0.006, 0.7, 9)
WEDGE = np.pi / 12           # thin axis wedges: ang < WEDGE and ang > pi/2 - WEDGE


def _band_matrix():
    fy = np.broadcast_to(np.fft.fftfreq(H).astype(np.float32)[:, None], (H, W // 2 + 1))
    fx = np.broadcast_to(np.fft.rfftfreq(W).astype(np.float32)[None, :], (H, W // 2 + 1))
    r = np.sqrt(fy ** 2 + fx ** 2)
    nb = len(EDGES) - 1
    band = np.searchsorted(EDGES.astype(np.float32), r, side='left') - 1
    ok = (band >= 0) & (band < nb)
    ang = np.arctan2(np.abs(fy), np.abs(fx))
    groups = [np.ones_like(ok), ang < np.pi / 4, ang >= np.pi / 4, ang < WEDGE, ang > np.pi / 2 - WEDGE]
    mats = []
    for g in groups:
        sel = (ok & g).ravel()
        M = np.zeros((sel.size, nb), np.float32)
        M[np.nonzero(sel)[0], band.ravel()[sel]] = 1.0
        mats.append(M / np.maximum(M.sum(0), 1.0))
    return np.concatenate(mats, 1), nb


_M, _NB = _band_matrix()


def features(maps, mask):
    f = jnp.fft.rfft2(maps * mask)
    p = (f.real ** 2 + f.imag ** 2).reshape(maps.shape[0], -1)
    lp = jnp.log(p @ jnp.asarray(_M) + 1e-30)
    tot, xw, yw, xa, ya = (lp[:, i * _NB:(i + 1) * _NB] for i in range(5))
    return jnp.concatenate([xw - yw, xa - tot, ya - tot], 1)
