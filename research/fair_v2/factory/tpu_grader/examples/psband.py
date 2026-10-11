"""Existing family re-expressed (JAX): pseudo power spectrum of the masked map in 28 log-spaced |k| bands."""
import numpy as np
import jax.numpy as jnp

H, W = 1424, 176
PS_EDGES = np.geomspace(0.004, 0.72, 29)


def _band_matrix():
    fy = np.fft.fftfreq(H).astype(np.float32)[:, None]
    fx = np.fft.rfftfreq(W).astype(np.float32)[None, :]
    r = np.sqrt(fy ** 2 + fx ** 2)
    nb = len(PS_EDGES) - 1
    b = (np.searchsorted(PS_EDGES.astype(np.float32), r, side='left') - 1).ravel()
    sel = (b >= 0) & (b < nb)
    M = np.zeros((b.size, nb), np.float32)
    M[np.nonzero(sel)[0], b[sel]] = 1.0
    return M / np.maximum(M.sum(0), 1.0)


_M = _band_matrix()


def features(maps, mask):
    m = mask.astype(jnp.float32)
    f = jnp.fft.rfft2(maps * m)
    p = (f.real ** 2 + f.imag ** 2).reshape(maps.shape[0], -1) / jnp.sum(m)
    return jnp.log(p @ jnp.asarray(_M) + 1e-30)
