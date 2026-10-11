"""Deliberately invalid (JAX): log power in 8 bands minus the BATCH mean (depends on the other maps in the batch)."""
import jax.numpy as jnp


def features(maps, mask):
    f = jnp.fft.rfft2(maps * mask)
    p = (f.real ** 2 + f.imag ** 2).reshape(maps.shape[0], 8, -1).mean(2)
    lp = jnp.log(p)
    return lp - lp.mean(0, keepdims=True)
