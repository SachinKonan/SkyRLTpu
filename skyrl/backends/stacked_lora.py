"""Adapter-axis batching with one shared frozen NNX base.

The adapter axis is replicated; existing TP/FSDP axes stay on each factor.
NNX variable metadata describes the *unbatched* model and is consumed inside
vmap. Do not apply model partitioning helpers to the temporary stacked state.
"""
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
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


@jax.jit
def _gradient_moments(actual, expected):
    rows = []
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        a, b = a.astype(jnp.float32), b.astype(jnp.float32)
        rows.append(jnp.stack((jnp.sum((a - b) ** 2), jnp.sum(a ** 2),
                               jnp.sum(b ** 2), jnp.sum(a * b), jnp.max(jnp.abs(a - b)))))
    return jnp.stack(rows)


def gradient_comparison(actual, expected):
    """Small global reductions for diagnosing replay failures on every host."""
    moments = np.asarray(_gradient_moments(actual, expected))
    error2, actual2, expected2, dot = moments[:, :4].sum(axis=0, dtype=np.float64)
    denominator = np.sqrt(actual2 * expected2)
    paths = [jax.tree_util.keystr(path) for path, _ in jax.tree_util.tree_flatten_with_path(actual)[0]]
    return dict(
        actual_norm=float(np.sqrt(actual2)), expected_norm=float(np.sqrt(expected2)),
        error_norm=float(np.sqrt(error2)),
        cosine=float(dot / denominator) if denominator else float(actual2 == expected2),
        largest_error_leaves=[dict(path=paths[i], error_norm=float(np.sqrt(moments[i, 0])),
                                  actual_norm=float(np.sqrt(moments[i, 1])),
                                  expected_norm=float(np.sqrt(moments[i, 2])),
                                  max_abs=float(moments[i, 4]))
                              for i in np.argsort(-moments[:, 0])[:10]],
    )


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
