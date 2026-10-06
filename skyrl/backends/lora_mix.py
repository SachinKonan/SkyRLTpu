"""Learnable per-adapter mix of a carried (frozen) LoRA and a fresh LoRA.

Motivation (Erdős gen-1 arms, 2026-09-07): "fresh" (gamma = 1) and "carry"
(gamma = 0) weights have both been run.  This makes the interpolation itself a
learnable parameter, one scalar per adapter (per projection, per layer):

    W_eff = W_base + (alpha / r) * [ gamma * B_new A_new + (1 - gamma) * B_old A_old ]

Representation.  One qwix LoRA state of rank 2r holds both halves along the
rank axis: A = [A_new | A_old] (rank is the LAST axis of lora_a) and
B = [B_new ; B_old] (rank is the FIRST axis of lora_b).  The template is built
with (rank = 2r, alpha = 2 * alpha) so qwix's alpha / rank scale is unchanged.
gamma multiplies the B factor only:

    B_eff = [ gamma * B_new ; (1 - gamma) * B_old ]

which makes the effective product exactly the convex mix above, and keeps the
export to vLLM a single ordinary rank-2r PEFT adapter.

Training.  The backend runs its usual forward/backward on the EFFECTIVE state
and accumulates gradients in that space.  At the optimizer step this module
maps them back by the chain rule:

    dL/dA_new = dL/dA_eff[..., :r]          dL/dA_old = 0
    dL/dB_new = gamma * dL/dB_eff[:r]       dL/dB_old = 0
    dL/dgamma = <dL/dB_eff[:r], B_new> - <dL/dB_eff[r:], B_old>

The old half never receives a gradient and is written back verbatim after the
optimizer step, so weight decay cannot erode it either.  gamma has its own
Adam (learning rate ``TUNIX_LORA_MIX_GAMMA_LR``) and is clamped to [0, 1].

Everything here is pure pytree arithmetic; the backend owns state, jit and
sharding.  gamma leaves are keyed by the lora_b leaf's keystr and shaped so
they broadcast over that leaf with the rank and output axes reduced
(``(1, L, 1)`` for MaxText's layer-stacked ``(2r, L, out)``; ``(1, 1)`` for a
plain ``(2r, out)``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

GAMMA_ENV = "TUNIX_LORA_MIX_GAMMA"
GAMMA_LR_ENV = "TUNIX_LORA_MIX_GAMMA_LR"
DEFAULT_GAMMA_LR = 0.02


def mix_settings_from_env(environ=None) -> tuple[float, float] | None:
    """(gamma_init, gamma_lr) when the mix is enabled, else None."""
    environ = os.environ if environ is None else environ
    raw = environ.get(GAMMA_ENV, "").strip()
    if not raw:
        return None
    gamma = float(raw)
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"{GAMMA_ENV} must be in [0, 1], got {raw}")
    lr = float(environ.get(GAMMA_LR_ENV, str(DEFAULT_GAMMA_LR)))
    if lr < 0:
        raise ValueError(f"{GAMMA_LR_ENV} must be >= 0, got {lr}")
    return gamma, lr


def is_lora_b(keystr: str) -> bool:
    return "lora_b" in keystr


def is_lora_a(keystr: str) -> bool:
    return "lora_a" in keystr


def rank_axis(keystr: str) -> int:
    """qwix lora_a: (..., in, r) -> last axis; lora_b: (r, ...) -> first axis."""
    if is_lora_a(keystr):
        return -1
    if is_lora_b(keystr):
        return 0
    raise ValueError(f"not a qwix LoRA leaf: {keystr}")


def gamma_shape_for(b_shape: tuple[int, ...]) -> tuple[int, ...]:
    """Broadcast shape of one adapter's gamma over its lora_b leaf."""
    return (1,) + tuple(b_shape[1:-1]) + (1,)


@dataclass
class MixState:
    """Per-model_id bookkeeping for the mix (kept on the backend's ModelSlot)."""

    rank: int  # r: rank of each half (the client-facing LoRA rank)
    gamma: dict[str, Any]  # lora_b keystr -> gamma array (float32, broadcastable)
    gamma_lr: float
    gamma_init: float
    opt_state: Any = None  # optax state for gamma
    old_loaded: bool = False  # True once a carried checkpoint filled the old half
    tx: Any = field(default=None, repr=False)

    def optimizer(self):
        if self.tx is None:
            self.tx = optax.adam(self.gamma_lr)
        return self.tx

    def gamma_values(self) -> np.ndarray:
        if not self.gamma:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate([np.asarray(v, dtype=np.float32).ravel() for v in self.gamma.values()])

    def metrics(self, prefix: str = "lora_mix/") -> dict[str, float]:
        g = self.gamma_values()
        if g.size == 0:
            return {}
        return {
            prefix + "gamma_mean": float(g.mean()),
            prefix + "gamma_min": float(g.min()),
            prefix + "gamma_max": float(g.max()),
            prefix + "gamma_std": float(g.std()),
        }


def _flat_with_keys(tree):
    leaves, treedef = jax.tree_util.tree_flatten_with_path(tree)
    keys = [jax.tree_util.keystr(p) for p, _ in leaves]
    return keys, [v for _, v in leaves], treedef


def init_gamma(lora_state, gamma_init: float) -> dict[str, jnp.ndarray]:
    """One gamma per lora_b leaf, initialised to gamma_init."""
    keys, leaves, _ = _flat_with_keys(lora_state)
    out = {}
    for key, leaf in zip(keys, leaves):
        if is_lora_b(key):
            out[key] = jnp.full(gamma_shape_for(tuple(leaf.shape)), gamma_init, dtype=jnp.float32)
    return out


def effective_state(lora_state, gamma: dict[str, Any], rank: int):
    """State whose lora_b halves are scaled by gamma / (1 - gamma). lora_a untouched."""
    keys, leaves, treedef = _flat_with_keys(lora_state)
    out = []
    for key, leaf in zip(keys, leaves):
        if is_lora_b(key):
            g = gamma[key].astype(leaf.dtype)
            out.append(jnp.concatenate([g * leaf[:rank], (1.0 - g) * leaf[rank:]], axis=0))
        else:
            out.append(leaf)
    return jax.tree_util.tree_unflatten(treedef, out)


def split_gradients(eff_grads, lora_state, gamma: dict[str, Any], rank: int):
    """Chain rule from effective-space gradients to (lora grads, gamma grads).

    Returns a pytree matching ``lora_state`` (old halves zero, B_new scaled by
    gamma) and a dict of gamma gradients shaped like the gamma leaves.
    """
    keys, g_leaves, treedef = _flat_with_keys(eff_grads)
    _, s_leaves, _ = _flat_with_keys(lora_state)
    lora_out, gamma_out = [], {}
    for key, g, s in zip(keys, g_leaves, s_leaves):
        if is_lora_b(key):
            gam = gamma[key].astype(g.dtype)
            b_new, b_old = s[:rank], s[rank:]
            g_new, g_old = g[:rank], g[rank:]
            lora_out.append(jnp.concatenate([gam * g_new, jnp.zeros_like(g_old)], axis=0))
            reduce_axes = (0, g.ndim - 1)
            gamma_out[key] = (
                jnp.sum(g_new * b_new, axis=reduce_axes, keepdims=True)
                - jnp.sum(g_old * b_old, axis=reduce_axes, keepdims=True)
            ).astype(jnp.float32)
        elif is_lora_a(key):
            lora_out.append(jnp.concatenate([g[..., :rank], jnp.zeros_like(g[..., rank:])], axis=-1))
        else:
            lora_out.append(g)
    return jax.tree_util.tree_unflatten(treedef, lora_out), gamma_out


def restore_old_half(updated_state, previous_state, rank: int):
    """Write the frozen old half of ``previous_state`` back into ``updated_state``."""
    keys, u_leaves, treedef = _flat_with_keys(updated_state)
    _, p_leaves, _ = _flat_with_keys(previous_state)
    out = []
    for key, u, p in zip(keys, u_leaves, p_leaves):
        if is_lora_b(key):
            out.append(jnp.concatenate([u[:rank], p[rank:]], axis=0))
        elif is_lora_a(key):
            out.append(jnp.concatenate([u[..., :rank], p[..., rank:]], axis=-1))
        else:
            out.append(u)
    return jax.tree_util.tree_unflatten(treedef, out)


def gamma_step(mix: MixState, gamma_grads: dict[str, Any]) -> None:
    """One Adam step on gamma (in place), clamped to [0, 1]."""
    if not mix.gamma:
        return
    tx = mix.optimizer()
    grads = {k: jnp.asarray(gamma_grads[k], dtype=jnp.float32) for k in mix.gamma}
    if mix.opt_state is None:
        mix.opt_state = tx.init(mix.gamma)
    updates, mix.opt_state = tx.update(grads, mix.opt_state, mix.gamma)
    new = optax.apply_updates(mix.gamma, updates)
    mix.gamma = {k: jnp.clip(v, 0.0, 1.0) for k, v in new.items()}


def merge_plain_checkpoint(
    template_flat: dict[str, np.ndarray],
    plain_flat: dict[str, np.ndarray],
    rank: int,
) -> dict[str, np.ndarray]:
    """Fill the OLD half from a rank-r checkpoint; the new half stays as the template.

    ``template_flat`` are the rank-2r arrays (fresh init: random A, zero B);
    ``plain_flat`` the rank-r arrays of the carried checkpoint, same keys.
    """
    missing = set(template_flat) - set(plain_flat)
    extra = set(plain_flat) - set(template_flat)
    if missing or extra:
        raise ValueError(
            f"carried checkpoint does not match the adapter layout: missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}"
        )
    out = {}
    for key, tmpl in template_flat.items():
        old = np.asarray(plain_flat[key])
        axis = rank_axis(key)
        if tmpl.shape[axis] != 2 * rank or old.shape[axis] != rank:
            raise ValueError(
                f"rank layout mismatch for {key}: template {tmpl.shape}, checkpoint {old.shape}, rank {rank}"
            )
        old = old.astype(tmpl.dtype, copy=False)
        if axis == 0:
            new_half = np.zeros_like(tmpl[:rank]) if is_lora_b(key) else tmpl[:rank]
            out[key] = np.concatenate([new_half, old], axis=0)
        else:
            out[key] = np.concatenate([tmpl[..., :rank], old], axis=-1)
    return out


def gamma_to_flat(gamma: dict[str, Any]) -> dict[str, np.ndarray]:
    return {k: np.asarray(v, dtype=np.float32) for k, v in gamma.items()}


def gamma_from_flat(flat: dict[str, np.ndarray]) -> dict[str, jnp.ndarray]:
    return {k: jnp.asarray(v, dtype=jnp.float32) for k, v in flat.items()}
