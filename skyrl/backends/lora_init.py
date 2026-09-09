"""Independent dense and sparse expert LoRA initialization without constructing another base model."""
import hashlib
from dataclasses import dataclass
from functools import lru_cache, partial
import re

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class RepeatedKVHeads:
    """Keep MaxText's repeat-interleaved GQA heads equivalent to native HF GQA.

    Only the output axis of K/V LoRA B is replicated. A is already shared.
    The derivative of a shared B is the SUM over its logical copies; each
    copy must receive that same derivative before Adam updates its moments.
    """

    native_heads: int
    logical_heads: int
    head_dim: int

    def __post_init__(self):
        if (min(self.native_heads, self.logical_heads, self.head_dim) <= 0
                or self.logical_heads % self.native_heads):
            raise ValueError("logical K/V heads must be a positive multiple of native heads")

    @staticmethod
    def is_kv_b(path):
        name = path if isinstance(path, str) else jax.tree_util.keystr(path)
        return bool(re.search(r"\['(?:key|value)'\]\['kernel_lora_b'\]", name))

    def grouped(self, value):
        if value.ndim not in (2, 3) or value.shape[-1] != self.logical_heads * self.head_dim:
            raise ValueError(f"unexpected repeated K/V LoRA B shape: {value.shape}")
        return value.reshape(*value.shape[:-1], self.native_heads,
                             self.logical_heads // self.native_heads, self.head_dim)

    def collapse_numpy(self, value):
        grouped = self.grouped(np.asarray(value))
        canonical = grouped[..., :1, :]
        if not np.all(grouped == canonical):
            raise ValueError("K/V LoRA replicas diverged; cannot export the trained policy as native HF GQA")
        return np.ascontiguousarray(canonical.reshape(
            *value.shape[:-1], self.native_heads * self.head_dim))

    @partial(jax.jit, static_argnums=0)
    def replicas_equal(self, tree):
        checks = [jnp.all(self.grouped(value) == self.grouped(value)[..., :1, :])
                  for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]
                  if self.is_kv_b(path)]
        return jnp.all(jnp.stack(checks)) if checks else jnp.array(True)

    def validate(self, tree):
        if not bool(self.replicas_equal(tree)):
            raise ValueError("K/V LoRA weights or optimizer moments have divergent replicas; "
                             "this checkpoint cannot preserve the native HF policy")

    def tie_gradients_and_norm(self, gradients):
        leaves, treedef = jax.tree.flatten(gradients)
        return self._gradient_transform(treedef, tuple(x.sharding for x in leaves))(gradients)

    @lru_cache(maxsize=32)
    def _gradient_transform(self, treedef, shardings):
        # Summing replicated heads can make XLA choose a fully replicated
        # output. Keep the original placement for optimizer/checkpoint state.
        return jax.jit(self._tie_gradients_and_norm,
                       out_shardings=(jax.tree.unflatten(treedef, shardings), None))

    def _tie_gradients_and_norm(self, gradients):
        canonical = jax.tree_util.tree_map_with_path(
            lambda path, value: self.grouped(value).sum(axis=-2)
            if self.is_kv_b(path) else value, gradients)
        norm = jnp.sqrt(sum(jnp.sum(jnp.square(x.astype(jnp.float32)))
                            for x in jax.tree.leaves(canonical)))
        tied = jax.tree_util.tree_map_with_path(
            lambda path, value: jnp.repeat(value[..., :, None, :],
                self.logical_heads // self.native_heads, axis=-2).reshape(
                    *value.shape[:-2], self.logical_heads * self.head_dim)
            if self.is_kv_b(path) else value, canonical)
        return tied, norm


def initialize_lora(reference, seed):
    """Seed only adapter factors, preserving placement and the frozen base.

    Dense/scanned A is [in, rank] or [in, layer, rank]. GPT-OSS expert
    down-projection A is [expert, layer, in, rank]; its fan-in is the input
    width, not the expert count. B starts at zero for every projection.
    Initialization runs with explicit output sharding, without constructing
    a full host copy of each expert tensor or another base model.
    """
    def initialize(path, value):
        name = jax.tree_util.keystr(path)
        if "_lora_a" in name:
            if value.ndim in (2, 3):
                fan_in = value.shape[0]
            elif value.ndim == 4 and "wo_lora_a" in name:
                fan_in = value.shape[-2]
            else:
                raise ValueError(f"unsupported LoRA A layout: {name} {value.shape}")
            digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
            key = jax.random.key(int.from_bytes(digest[:4], "little"))
            init = lambda key: (jax.random.normal(key, value.shape, dtype=jnp.float32)
                                / np.sqrt(fan_in)).astype(value.dtype)
        elif "_lora_b" in name:
            key = jax.random.key(0)
            init = lambda key: jnp.zeros(value.shape, value.dtype)
        else:
            raise ValueError(f"unexpected LoRA parameter: {name}")
        return jax.jit(init, out_shardings=value.sharding)(key)

    return jax.tree_util.tree_map_with_path(initialize, reference)


# Compatibility with the original Qwen validation harness.
initialize_dense_lora = initialize_lora
