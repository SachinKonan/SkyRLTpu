"""Task 0 — RMSNorm fwd+bwd (judge shakedown warm-up).

Baseline: the XLA fusion of the closed form (jit), scored additionally as a
fraction of speed-of-light HBM bandwidth (memory-bound). Contract: fwd+bwd
(gradient wrt x and gamma must match the fp32 reference).

kernel(x, gamma) -> y
  x:     [rows, d]  bfloat16
  gamma: [d]        float32
  y:     [rows, d]  bfloat16,  y = x * rsqrt(mean(x^2, -1) + eps) * gamma
  eps = 1e-6, mean/rsqrt computed in fp32.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from pallas_arena.judge.problems.base import (
    AdversarialCase,
    Problem,
    ShapeCase,
)

EPS = 1e-6


def _rmsnorm_f32(x, gamma):
    x32 = x.astype(jnp.float32)
    var = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
    return x32 * jax.lax.rsqrt(var + EPS) * gamma.astype(jnp.float32)


@functools.partial(jax.jit)
def _baseline_jit(x, gamma):
    return _rmsnorm_f32(x, gamma).astype(x.dtype)


class RMSNormProblem(Problem):
    name = "rmsnorm"
    version = "1"
    has_bwd = True
    require_pallas = False       # the baseline IS XLA; shakedown task
    memory_bound = True
    banned_call_names = ()       # nothing to wrap: closed form is public

    def shape_cases(self):
        return [
            ShapeCase("8192x4096", {"rows": 8192, "d": 4096}),
            ShapeCase("32768x8192", {"rows": 32768, "d": 8192}),
            ShapeCase("73728x2880", {"rows": 73728, "d": 2880}),  # our fb tokens x gpt-oss width
            ShapeCase("holdout-16384x6144", {"rows": 16384, "d": 6144},
                      holdout=True),
            # CPU battery cases
            ShapeCase("tiny", {"rows": 32, "d": 64}, smoke=True),
            ShapeCase("tiny-odd", {"rows": 17, "d": 96}, smoke=True),
            ShapeCase("tiny-holdout", {"rows": 24, "d": 128}, smoke=True,
                      holdout=True),
        ]

    def make_inputs(self, key, case):
        kx, kg = jax.random.split(key)
        rows, d = case.dims["rows"], case.dims["d"]
        x = jax.random.normal(kx, (rows, d), jnp.float32).astype(jnp.bfloat16)
        gamma = 1.0 + 0.1 * jax.random.normal(kg, (d,), jnp.float32)
        return (x, gamma)

    def reference(self, x, gamma):
        return _rmsnorm_f32(x, gamma)

    def baseline(self, x, gamma):
        return _baseline_jit(x, gamma)

    def grad_outputs(self, kernel_fn, x, gamma):
        """Gradient contract: d/d(x32), d/d(gamma) of a fixed scalar probe
        sum(out * cos(iota)) — a non-trivial cotangent so wrong backward
        rules can't hide behind symmetric reductions."""
        probe = jnp.cos(jnp.arange(x.size, dtype=jnp.float32)).reshape(x.shape)

        def scalar(x32, g):
            out = kernel_fn(x32.astype(x.dtype), g)
            return jnp.sum(out.astype(jnp.float32) * probe)

        return jax.grad(scalar, argnums=(0, 1))(x.astype(jnp.float32), gamma)

    def adversarial_cases(self):
        def outlier_rows(key):
            x, g = self.make_inputs(key, self.case_by_name("tiny"))
            x = x.at[3].set(jnp.full((x.shape[1],), 1e18, jnp.bfloat16))
            x = x.at[7].set(jnp.full((x.shape[1],), 1e-20, jnp.bfloat16))
            return (x, g)

        def zero_row(key):
            x, g = self.make_inputs(key, self.case_by_name("tiny"))
            x = x.at[0].set(jnp.zeros((x.shape[1],), jnp.bfloat16))
            return (x, g)

        def near_overflow_bf16(key):
            # values whose square is near the fp32 max: forces fp32 (not
            # bf16) accumulation in the mean; naive bf16 accumulation infs
            x, g = self.make_inputs(key, self.case_by_name("tiny"))
            x = x.at[1].set(jnp.full((x.shape[1],), 1.5e19, jnp.bfloat16))
            return (x, g)

        def expect_finite(ref, inputs):
            assert np.isfinite(np.asarray(ref, np.float64)).all(), \
                "reference produced non-finite values"

        def expect_zero_row(ref, inputs):
            row = np.asarray(ref)[0]
            assert np.isfinite(row).all() and np.abs(row).max() == 0.0, \
                "all-zero input row must normalize to exactly 0, not NaN"

        return [
            AdversarialCase("outlier-rows", outlier_rows, expect_finite),
            AdversarialCase("zero-row", zero_row, expect_zero_row),
            AdversarialCase("near-overflow-bf16", near_overflow_bf16,
                            expect_finite),
        ]

    def bytes_moved(self, case):
        rows, d = case.dims["rows"], case.dims["d"]
        return rows * d * 2 + d * 4 + rows * d * 2  # read x + gamma, write y


PROBLEM = RMSNormProblem()
