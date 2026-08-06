"""Task 5 — RG-LRU gated diagonal linear scan (SSM).

    h_t = a_t * h_{t-1} + sqrt(1 - a_t^2) * x_t

with gates a_t precomputed as inputs (per DESIGN.md task scope) and reset
boundaries (h resets to 0 where reset_t = True; a_t is forced to 0 there so
no state crosses a segment boundary). Kernel-vs-kernel against DeepMind's
recurrentgemma Pallas scan on the TPU judge (importing recurrentgemma is
banned for candidates); `lax.associative_scan` is a LEGAL candidate
strategy, and doubles as the CPU-available stand-in baseline for the test
battery. Tolerance is calibrated vs the reference's own bf16 drift at long
T, never a fixed atol.

kernel(x, a, reset) -> h
  x:     [b, t, d] bfloat16
  a:     [b, t, d] float32 in [0, 1)
  reset: [b, t]    bool
  h:     [b, t, d] float32
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from pallas_arena.judge.problems.base import (
    AdversarialCase,
    BaselineUnavailable,
    Problem,
    ShapeCase,
)


def _apply_reset(a, reset):
    return a * (1.0 - reset[..., None].astype(a.dtype))


def rg_lru_scan_reference(x, a, reset):
    """fp32 sequential lax.scan — the correctness oracle."""
    x32 = x.astype(jnp.float32)
    a32 = _apply_reset(a.astype(jnp.float32), reset)
    gx = jnp.sqrt(jnp.maximum(1.0 - jnp.square(a32), 0.0)) * x32

    def step(h, xs):
        a_t, gx_t = xs
        h = a_t * h + gx_t
        return h, h

    h0 = jnp.zeros((x32.shape[0], x32.shape[2]), jnp.float32)
    _, hs = jax.lax.scan(step, h0, (jnp.moveaxis(a32, 1, 0), jnp.moveaxis(gx, 1, 0)))
    return jnp.moveaxis(hs, 0, 1)


def rg_lru_associative(x, a, reset):
    """lax.associative_scan formulation — legal candidate strategy and the
    CPU stand-in baseline (the TPU judge binds recurrentgemma's Pallas scan)."""
    x32 = x.astype(jnp.float32)
    a32 = _apply_reset(a.astype(jnp.float32), reset)
    gx = jnp.sqrt(jnp.maximum(1.0 - jnp.square(a32), 0.0)) * x32

    def combine(left, right):
        a_l, b_l = left
        a_r, b_r = right
        return a_l * a_r, b_l * a_r + b_r

    _, hs = jax.lax.associative_scan(combine, (a32, gx), axis=1)
    return hs


class RGLRUProblem(Problem):
    name = "rg_lru"
    version = "1"
    has_bwd = False  # kernel-vs-kernel forward scan
    require_pallas = False  # associative_scan is explicitly legal
    memory_bound = True

    def shape_cases(self):
        return [
            # RecurrentGemma-2B width
            ShapeCase("8x4096x2560", {"b": 8, "t": 4096, "d": 2560}),
            ShapeCase("1x32768x2560", {"b": 1, "t": 32768, "d": 2560}),
            ShapeCase("holdout-4x8192x2560", {"b": 4, "t": 8192, "d": 2560}, holdout=True),
            # CPU battery (tiny-ragged: non-block-divisible T)
            ShapeCase("tiny", {"b": 2, "t": 64, "d": 16}, smoke=True),
            ShapeCase("tiny-ragged", {"b": 2, "t": 100, "d": 16}, smoke=True),
            ShapeCase("tiny-holdout", {"b": 1, "t": 128, "d": 8}, smoke=True, holdout=True),
        ]

    def make_inputs(self, key, case):
        kx, ka, kr = jax.random.split(key, 3)
        b, t, d = case.dims["b"], case.dims["t"], case.dims["d"]
        x = jax.random.normal(kx, (b, t, d), jnp.float32).astype(jnp.bfloat16)
        # gates concentrated near 1 (long memory), as the trained model has
        a = jax.nn.sigmoid(jax.random.normal(ka, (b, t, d), jnp.float32) * 2.0 + 3.0)
        reset = jax.random.bernoulli(kr, 0.02, (b, t))
        reset = reset.at[:, 0].set(True)
        return (x, a, reset)

    def reference(self, x, a, reset):
        return rg_lru_scan_reference(x, a, reset)

    def reference_bf16(self, x, a, reset):
        # bf16 OUTPUT precision through the same sequential scan. The gates
        # stay fp32 — they are fp32 INPUTS by contract, and quantizing them
        # here would destroy the a->1 adversarial vector's calibration
        # (bf16 rounds 1-1e-6 to exactly 1.0, zeroing sqrt(1-a^2)).
        h = rg_lru_scan_reference(x, a, reset)
        return h.astype(jnp.bfloat16).astype(jnp.float32)

    def baseline(self, x, a, reset):
        try:
            from recurrentgemma.jax import scan as rg_scan  # noqa: F401

            # Production judge path (TPU host with recurrentgemma installed):
            # bound during Phase-2 bring-up; shapes/layout adapted there.
            raise BaselineUnavailable("recurrentgemma pallas scan binding is a Phase-2 (TPU judge) step")
        except ImportError:
            if jax.default_backend() == "cpu":
                return rg_lru_associative(x, a, reset)
            raise BaselineUnavailable("recurrentgemma not installed on this judge host")

    def adversarial_cases(self):
        tiny = self.case_by_name("tiny")

        def a_to_one(key):
            x, a, reset = self.make_inputs(key, tiny)
            a = jnp.full_like(a, 1.0 - 1e-6)  # near-perfect memory: drift test
            return (x, a, reset)

        def dense_resets(key):
            x, a, reset = self.make_inputs(key, tiny)
            reset = (jnp.arange(x.shape[1]) % 7 == 0)[None, :].repeat(x.shape[0], axis=0)
            return (x, a, reset)

        def a_zero(key):
            x, a, reset = self.make_inputs(key, tiny)
            return (x, jnp.zeros_like(a), reset)

        def expect_finite(ref, inputs):
            assert np.isfinite(np.asarray(ref, np.float64)).all()

        def expect_passthrough(ref, inputs):
            # a == 0 -> h_t = x_t exactly
            x = np.asarray(inputs[0], np.float32)
            np.testing.assert_allclose(np.asarray(ref), x, rtol=1e-5, atol=1e-5)

        def expect_reset_rows(ref, inputs):
            # at reset positions h_t == sqrt(1-a^2)*x_t with a forced to 0
            reset = np.asarray(inputs[2])
            x = np.asarray(inputs[0], np.float32)
            h = np.asarray(ref)
            np.testing.assert_allclose(h[reset], x[reset], rtol=1e-5, atol=1e-5)

        return [
            AdversarialCase("a-to-one-long-memory", a_to_one, expect_finite),
            AdversarialCase("dense-reset-boundaries", dense_resets, expect_reset_rows),
            AdversarialCase("a-zero-passthrough", a_zero, expect_passthrough),
        ]

    def bytes_moved(self, case):
        b, t, d = case.dims["b"], case.dims["t"], case.dims["d"]
        return b * t * d * (2 + 4 + 4) + b * t  # x + a read, h write, reset


PROBLEM = RGLRUProblem()
