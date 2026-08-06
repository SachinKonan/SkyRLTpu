"""Task 3 — Megablox grouped matmul (MoE), uniform + Zipf-skewed groups.

Baseline: the tuned Pallas gmm that MaxText vendors
(jax.experimental.pallas.ops.tpu.megablox.gmm at our jax pin — same kernel
family; the TPU judge binds it directly). `jax.lax.ragged_dot` is the legal
floor and the CPU stand-in baseline for the test battery. `group_sizes` are
freshly sampled per grading from BOTH uniform and Zipf-skewed distributions
so imbalance-tuned kernels can't fake wins.

kernel(lhs, rhs, group_sizes) -> out
  lhs:         [m, k]              bfloat16 (tokens, grouped contiguously)
  rhs:         [num_groups, k, n]  bfloat16 (per-expert weights)
  group_sizes: [num_groups]        int32, sum == m (zeros allowed!)
  out:         [m, n]              float32, out[rows of group g] = lhs @ rhs[g]
fp32 accumulation. Contract: fwd only.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from pallas_arena.judge.problems.base import (
    AdversarialCase,
    Problem,
    ShapeCase,
)


def gmm_reference(lhs, rhs, group_sizes):
    """fp32 closed form via lax.ragged_dot (the design floor) — itself
    cross-checked against a python loop in the CPU battery."""
    return jax.lax.ragged_dot(lhs.astype(jnp.float32), rhs.astype(jnp.float32), group_sizes)


def gmm_loop_reference(lhs, rhs, group_sizes):
    """Independent numpy loop formulation (test-battery cross-check only)."""
    lhs = np.asarray(lhs, np.float32)
    rhs = np.asarray(rhs, np.float32)
    sizes = np.asarray(group_sizes)
    out = np.zeros((lhs.shape[0], rhs.shape[2]), np.float32)
    row = 0
    for g, size in enumerate(sizes):
        if size:
            out[row : row + size] = lhs[row : row + size] @ rhs[g]
        row += size
    return out


def _sample_group_sizes(key, num_groups, m, dist):
    if dist == "uniform":
        w = jnp.ones((num_groups,))
    elif dist == "zipf":
        w = 1.0 / jnp.arange(1, num_groups + 1, dtype=jnp.float32)
        w = jax.random.permutation(key, w)
    else:
        raise ValueError(dist)
    # multinomial split of m tokens by weights
    probs = w / jnp.sum(w)
    counts = jnp.floor(probs * m).astype(jnp.int32)
    # hand the remainder to the heaviest group (keeps sum == m, determinism)
    rem = m - jnp.sum(counts)
    counts = counts.at[jnp.argmax(probs)].add(rem)
    return counts.astype(jnp.int32)


class MegabloxGmmProblem(Problem):
    name = "megablox_gmm"
    version = "1"
    has_bwd = False
    require_pallas = True
    memory_bound = False

    def shape_cases(self):
        return [
            # (tokens=32k, experts=8 and 64, k=4096, n=14336)
            ShapeCase("32k-e8-uniform", {"m": 32768, "g": 8, "k": 4096, "n": 14336, "dist": "uniform"}),
            ShapeCase("32k-e8-zipf", {"m": 32768, "g": 8, "k": 4096, "n": 14336, "dist": "zipf"}),
            ShapeCase("32k-e64-uniform", {"m": 32768, "g": 64, "k": 4096, "n": 14336, "dist": "uniform"}),
            ShapeCase("32k-e64-zipf", {"m": 32768, "g": 64, "k": 4096, "n": 14336, "dist": "zipf"}),
            ShapeCase(
                "holdout-16k-e32-zipf", {"m": 16384, "g": 32, "k": 2048, "n": 7168, "dist": "zipf"}, holdout=True
            ),
            # CPU battery
            ShapeCase("tiny", {"m": 64, "g": 4, "k": 16, "n": 24, "dist": "uniform"}, smoke=True),
            ShapeCase("tiny-zipf", {"m": 64, "g": 4, "k": 16, "n": 24, "dist": "zipf"}, smoke=True),
            ShapeCase("tiny-holdout", {"m": 32, "g": 4, "k": 8, "n": 16, "dist": "zipf"}, smoke=True, holdout=True),
        ]

    def make_inputs(self, key, case):
        kl, kr, kg = jax.random.split(key, 3)
        dm = case.dims
        lhs = jax.random.normal(kl, (dm["m"], dm["k"]), jnp.float32).astype(jnp.bfloat16)
        rhs = (jax.random.normal(kr, (dm["g"], dm["k"], dm["n"]), jnp.float32) / np.sqrt(dm["k"])).astype(jnp.bfloat16)
        sizes = _sample_group_sizes(kg, dm["g"], dm["m"], dm["dist"])
        return (lhs, rhs, sizes)

    def reference(self, lhs, rhs, group_sizes):
        return gmm_reference(lhs, rhs, group_sizes)

    def baseline(self, lhs, rhs, group_sizes):
        if jax.default_backend() != "tpu":
            # CPU battery: the design floor is the stand-in baseline
            return jax.lax.ragged_dot(lhs, rhs, group_sizes).astype(jnp.float32)
        from jax.experimental.pallas.ops.tpu.megablox import gmm

        return gmm(lhs, rhs, group_sizes).astype(jnp.float32)

    def adversarial_cases(self):
        tiny = self.case_by_name("tiny")

        def empty_group(key):
            lhs, rhs, sizes = self.make_inputs(key, tiny)
            m = lhs.shape[0]
            sizes = jnp.array([0, m // 2, 0, m - m // 2], jnp.int32)
            return (lhs, rhs, sizes)

        def max_skew(key):
            lhs, rhs, sizes = self.make_inputs(key, tiny)
            m = lhs.shape[0]
            sizes = jnp.array([m, 0, 0, 0], jnp.int32)  # single-expert pileup
            return (lhs, rhs, sizes)

        def expect_finite(ref, inputs):
            assert np.isfinite(np.asarray(ref, np.float64)).all()

        def expect_skew_exact(ref, inputs):
            lhs, rhs, sizes = inputs
            want = np.asarray(lhs, np.float32) @ np.asarray(rhs, np.float32)[0]
            np.testing.assert_allclose(np.asarray(ref), want, rtol=2e-4, atol=2e-4)

        return [
            AdversarialCase("empty-expert-groups", empty_group, expect_finite),
            AdversarialCase("max-skew-single-expert", max_skew, expect_skew_exact),
        ]


PROBLEM = MegabloxGmmProblem()
