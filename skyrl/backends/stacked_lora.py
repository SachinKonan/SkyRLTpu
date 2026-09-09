"""Adapter-axis batching with one shared frozen NNX base.

The adapter axis is replicated; existing TP/FSDP axes stay on each factor.
NNX variable metadata describes the *unbatched* model and is consumed inside
vmap. Do not apply model partitioning helpers to the temporary stacked state.
"""
from functools import lru_cache

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec as P


@lru_cache(maxsize=256)
def _stack_fn(sharding):
    out = NamedSharding(sharding.mesh, P(None, *sharding.spec)) if isinstance(sharding, NamedSharding) else sharding
    return jax.jit(lambda *xs: jnp.stack(xs), out_shardings=out)


def stack_states(states):
    """Stack only adapter-sized trees, preserving factor placement."""
    return jax.tree.map(lambda *xs: _stack_fn(xs[0].sharding)(*xs), *states)


@lru_cache(maxsize=256)
def _slice_fn(index, sharding):
    return jax.jit(lambda x: x[index], out_shardings=sharding)


def unstack_state(state, index, reference):
    return jax.tree.map(lambda x, ref: _slice_fn(index, ref.sharding)(x), state, reference)


@jax.jit
def gradient_relative_error(actual, expected):
    pairs = zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True)
    errors, norms = [], []
    for a, b in pairs:
        a, b = a.astype(jnp.float32), b.astype(jnp.float32)
        errors.append(jnp.sum(jnp.square(a - b)))
        norms.append(jnp.sum(jnp.square(b)))
    return jnp.sqrt(sum(errors) / jnp.maximum(sum(norms), 1e-30))


def stacked_model(template, states):
    # merge creates independent NNX objects but reuses the frozen JAX buffers.
    graph, _, rest = nnx.split(template, nnx.LoRAParam, ...)
    return nnx.merge(graph, stack_states(states), rest)


def vectorized_backward(single_backward, *, eager=False):
    """Batch independent losses/gradients; inputs and behavior logps broadcast.

    Intermediates are per-call scratch and must be removed *inside* vmap:
    otherwise NNX tries to return adapter-dependent mutations of shared state.
    """
    def one(model, accum, args):
        result = single_backward(model, accum, *args)
        nnx.pop(model, nnx.Intermediate)
        return result

    mapped = nnx.vmap(one, in_axes=(nnx.StateAxes({nnx.LoRAParam: 0, ...: None}), 0, None))
    return mapped if eager else nnx.jit(mapped, donate_argnums=1)
