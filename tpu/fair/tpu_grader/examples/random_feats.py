"""Deliberately useless (JAX): 16 Gaussian numbers keyed by a hash of the map (per-map, deterministic, no signal)."""
import jax
import jax.numpy as jnp


def features(maps, mask):
    s = jnp.sum(jnp.abs(maps) * mask, axis=(1, 2))
    bits = jax.lax.bitcast_convert_type(s, jnp.uint32)
    keys = jax.vmap(lambda b: jax.random.fold_in(jax.random.PRNGKey(0), b))(bits)
    return jax.vmap(lambda k: jax.random.normal(k, (16,)))(keys)
