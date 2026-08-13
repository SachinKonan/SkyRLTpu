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


def _honest_faithful_bf16(lhs, rhs, group_sizes):
    """bf16 operands into the MXU, fp32 accumulator -- the production path.

    NOTE: on CPU this is indistinguishable from the fp32 reference, because
    ``preferred_element_type`` is a no-op inside a fused op there; it only
    separates from ``reference_bf16`` on a real TPU. It is included so the TPU
    calibration is right, and the CPU battery simply sees it as exact.
    """
    return jax.lax.ragged_dot(lhs, rhs, group_sizes, preferred_element_type=jnp.float32)


def _honest_group_loop_bf16(lhs, rhs, group_sizes):
    """Same math with a per-group reduction ORDER (one dot per expert instead
    of one fused ragged dot), fp32 accumulators. A legal candidate strategy, so
    it must not be punished for a different-but-valid accumulation order."""
    m = lhs.shape[0]
    n = rhs.shape[2]
    rows = jnp.arange(m)
    starts = jnp.concatenate([jnp.zeros((1,), group_sizes.dtype), jnp.cumsum(group_sizes)[:-1]])

    def body(acc, g):
        lo = starts[g]
        hi = lo + group_sizes[g]
        sel = (rows >= lo) & (rows < hi)
        part = jnp.dot(lhs, rhs[g], preferred_element_type=jnp.float32)
        return acc + jnp.where(sel[:, None], part, 0.0), None

    acc, _ = jax.lax.scan(body, jnp.zeros((m, n), jnp.float32), jnp.arange(rhs.shape[0]))
    return acc


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
            # PROBE set: one-chip sizes at the UNCHANGED expert widths
            # (k=4096, n=14336). Only the token count and the expert COUNT
            # shrink -- 4 experts instead of 8/64 -- because `rhs` is the whole
            # memory budget here: [g, 4096, 14336] bf16 is 940 MB at g=8, and
            # the worker holds `#scored x correctness_seeds` full input tuples
            # live at once. At g=4 that is 470 MB a fixture. Zipf is kept so
            # the imbalance (and the zero-size-group trap) survives the shrink,
            # and the holdout m=3000 is deliberately not a multiple of any
            # reasonable row tile.
            ShapeCase("probe-m4096-e4-uniform", {"m": 4096, "g": 4, "k": 4096, "n": 14336, "dist": "uniform"}, probe=True),
            ShapeCase("probe-m2048-e4-zipf", {"m": 2048, "g": 4, "k": 4096, "n": 14336, "dist": "zipf"}, probe=True),
            ShapeCase(
                "probe-holdout-m3000-e4-zipf",
                {"m": 3000, "g": 4, "k": 4096, "n": 14336, "dist": "zipf"},
                holdout=True,
                probe=True,
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

    # Which baseline the last `baseline()` call actually used. Recorded (not
    # asserted) so a boot report says plainly what the score denominator IS.
    baseline_impl: str = "?"

    def baseline(self, lhs, rhs, group_sizes):
        """The tuned Pallas megablox gmm, with `lax.ragged_dot` as the honest
        fallback (the design floor, and the CPU stand-in).

        megablox has its own tiling constraints and refuses some (m, k, n); a
        judge that dies at BOOT on that has graded nothing at all. Which one
        ran is recorded, because a score against `ragged_dot` is not a score
        against the production kernel.
        """
        if jax.default_backend() != "tpu":
            type(self).baseline_impl = "lax-ragged-dot"
            return jax.lax.ragged_dot(lhs, rhs, group_sizes).astype(jnp.float32)
        try:
            from jax.experimental.pallas.ops.tpu.megablox import gmm

            out = gmm(lhs, rhs, group_sizes).astype(jnp.float32)
            type(self).baseline_impl = "pallas-megablox-gmm"
            return out
        except Exception:
            type(self).baseline_impl = "lax-ragged-dot-fallback"
            return jax.lax.ragged_dot(lhs, rhs, group_sizes).astype(jnp.float32)

    def honest_variants(self):
        return [_honest_faithful_bf16, _honest_group_loop_bf16]

    def adversarial_cases(self):
        # settable base case: a judge grading a non-default (e.g. probe) case
        # set must not silently force candidates to also trace at the tiny
        # CPU-battery shapes, which no prompt declares
        tiny = self.case_by_name(self.adversarial_case_name)

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
