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
    # BACKWARD IS PART OF THE CONTRACT: recurrentgemma ships _lru_fwd AND
    # _lru_bwd, so the production scan is differentiable and a forward-only
    # candidate is not a replacement for it.
    has_bwd = True
    # FLIPPED 2026-08-17 (was False, "associative_scan is explicitly legal").
    # As a kernel-writing RL env the old setting made the task winnable
    # without ever writing a kernel: all 16 sd-run winners were plain-XLA
    # scans (pid=False, custom_vjp=False across the board -- the backward
    # survey), which means the arena was paying kernel rewards for
    # formulation choice. associative_scan remains the honest CALIBRATION
    # floor (it stays a baseline candidate and the election still uses it);
    # it is no longer an admissible CANDIDATE. Historical rg_lru winners are
    # invalidated as candidates by this flip -- accepted, that is the point.
    require_pallas = True
    general_mode = True  # score the holdout; denominator = fastest honest impl per shape
    memory_bound = True

    def shape_cases(self):
        return [
            # RecurrentGemma-2B width
            ShapeCase("8x4096x2560", {"b": 8, "t": 4096, "d": 2560}),
            ShapeCase("1x32768x2560", {"b": 1, "t": 32768, "d": 2560}),
            ShapeCase("holdout-4x8192x2560", {"b": 4, "t": 8192, "d": 2560}, holdout=True),
            # PROBE set: one-chip sizes at the UNCHANGED RecurrentGemma width
            # (d=2560); only the batch and time axes shrink, so the long-memory
            # fp32-drift difficulty the task exists for is intact. The holdout
            # T is deliberately not a multiple of any reasonable chunk length.
            ShapeCase("probe-4x2048x2560", {"b": 4, "t": 2048, "d": 2560}, probe=True),
            ShapeCase("probe-2x1024x2560", {"b": 2, "t": 1024, "d": 2560}, probe=True),
            # GENERAL sweep: sequence length is the axis a scan kernel must
            # block over; width varies too since lane tiling depends on d.
            ShapeCase("probe-8x512x2560", {"b": 8, "t": 512, "d": 2560}, probe=True),
            ShapeCase("probe-2x4096x2560", {"b": 2, "t": 4096, "d": 2560}, probe=True),
            ShapeCase("probe-4x2048x1024", {"b": 4, "t": 2048, "d": 1024}, probe=True),
            # TENSOR PARALLEL (v6e-8): the feature axis d sharded 8 ways.
            # The recurrence is independent per feature, so each device
            # runs a complete scan over d/8 with no collective.
            ShapeCase("tp8-4x2048x2560", {"b": 4, "t": 2048, "d": 2560}, probe=True, tp=8),
            ShapeCase("tp8-holdout-2x1500x2560", {"b": 2, "t": 1500, "d": 2560}, probe=True, tp=8, holdout=True),
            ShapeCase(
                "probe-holdout-2x1500x2560", {"b": 2, "t": 1500, "d": 2560}, holdout=True, probe=True
            ),
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

    # Which baseline the last `baseline()` call actually used. Recorded (not
    # asserted) so a boot report says plainly what the score denominator IS.
    baseline_impl: str = "?"

    def baseline(self, x, a, reset):
        """DeepMind's recurrentgemma Pallas scan when the judge host has it;
        otherwise `lax.associative_scan`, recorded as such.

        The fallback is honest and strong rather than a straw man: the parallel
        scan is the formulation you would actually reach for on TPU, it is an
        explicitly LEGAL candidate strategy for this task, and the design
        document already names it the legal floor. What it is NOT is
        recurrentgemma's kernel, so a score against it must be reported as
        "versus lax.associative_scan", never as beating the production scan.

        A judge that refuses to boot has graded nothing at all, which is
        strictly worse than scoring against a slower-but-real denominator --
        the same trade splash_attention already makes.
        """
        try:
            if jax.default_backend() != "tpu":
                raise BaselineUnavailable("recurrentgemma pallas scan requires TPU")
            from recurrentgemma.jax.pallas import lru_pallas_scan

            # Layout adapter. `lru_pallas_scan(x, a, h0)` runs the PLAIN LRU
            # recurrence h_t = a_t * h_{t-1} + x_t on [b, t, d] -- the gating
            # sqrt(1 - a^2) and the reset live OUTSIDE the kernel in Griffin
            # (RGLRU module), exactly as they live outside here. Both wrapper
            # steps are elementwise O(btd) prep against an O(btd)-serial scan,
            # and the timed candidate does the same prep itself, so the
            # comparison stays like-for-like.
            #
            # Tunables max_seq_block_size=256 / min_seq_block_size=16 are the
            # LIBRARY DEFAULTS. Audited before trusting (the splash/megablox
            # lesson): unlike those two, this default is a real recommendation
            # -- Griffin's own training config uses the module as shipped --
            # but the block sweep in verify/ still covers it before any reward
            # against this denominator is reported.
            x32 = x.astype(jnp.float32)
            a32 = _apply_reset(a.astype(jnp.float32), reset)
            gx = jnp.sqrt(jnp.maximum(1.0 - jnp.square(a32), 0.0)) * x32
            h, _last_carry = lru_pallas_scan(gx, a32)
            type(self).baseline_impl = "recurrentgemma-pallas-lru"
            return h.astype(jnp.float32)
        except BaselineUnavailable:
            type(self).baseline_impl = "lax-associative-scan"
            return rg_lru_associative(x, a, reset)
        except ImportError:
            type(self).baseline_impl = "lax-associative-scan (recurrentgemma not installed)"
            return rg_lru_associative(x, a, reset)
        except Exception:
            type(self).baseline_impl = "lax-associative-scan-fallback"
            return rg_lru_associative(x, a, reset)

    def baseline_candidates(self):
        """DeepMind's Pallas LRU scan vs `lax.associative_scan`. The parallel
        scan is an explicitly legal candidate strategy, so it is also a fair
        denominator wherever it happens to be the faster of the two."""
        return {"production": self.baseline, "lax-associative-scan": rg_lru_associative}

    def tp_specs(self, case=None):
        """Shard the FEATURE axis d. The recurrence is independent per feature,
        so each device runs a complete scan over its own slice with no
        collective; `reset` is replicated because it is per (batch, time)."""
        from jax.sharding import PartitionSpec as P

        return ((P(None, None, "tp"), P(None, None, "tp"), P()), P(None, None, "tp"))

    def grad_outputs(self, kernel_fn, x, a, reset):
        """d/d(x, a) of a fixed scalar functional of the hidden states.

        Both x and the gate a are differentiated: a scan's backward has to
        carry the gradient BACK THROUGH THE RECURRENCE, and a kernel that
        handles d/dx but drops the gate path would otherwise pass. The cos
        probe is deliberately non-uniform in time so an incorrect carry cannot
        cancel out across the sequence."""
        probe = jnp.cos(jnp.arange(x.size, dtype=jnp.float32)).reshape(x.shape)

        def scalar(x32, a32):
            h = kernel_fn(x32.astype(x.dtype), a32, reset)
            return jnp.sum(h.astype(jnp.float32) * probe)

        return jax.grad(scalar, argnums=(0, 1))(x.astype(jnp.float32), a.astype(jnp.float32))

    def honest_variants(self):
        """rg_lru's band was never too tight for a faithful kernel (the fp32
        associative scan measures ~0x against the sequential one). What matters
        here is reduction ORDER: the parallel scan and a chunked scan are both
        explicitly legal candidate strategies, so the band must span them."""

        def _chunked_scan(x, a, reset, chunk=64):
            """Sequential across chunks, associative within -- the blocking a
            real Pallas kernel uses, at fp32."""
            x32 = x.astype(jnp.float32)
            a32 = _apply_reset(a.astype(jnp.float32), reset)
            gx = jnp.sqrt(jnp.maximum(1.0 - jnp.square(a32), 0.0)) * x32
            b, t, d = x32.shape
            nb = -(-t // chunk)
            pad = nb * chunk - t
            if pad:
                a32 = jnp.pad(a32, ((0, 0), (0, pad), (0, 0)), constant_values=0.0)
                gx = jnp.pad(gx, ((0, 0), (0, pad), (0, 0)))
            ab = a32.reshape(b, nb, chunk, d)
            gb = gx.reshape(b, nb, chunk, d)

            def combine(left, right):
                a_l, b_l = left
                a_r, b_r = right
                return a_l * a_r, b_l * a_r + b_r

            def body(h, j):
                a_j, g_j = ab[:, j], gb[:, j]
                cum_a, cum_b = jax.lax.associative_scan(combine, (a_j, g_j), axis=1)
                out = cum_a * h[:, None, :] + cum_b
                return out[:, -1, :], out

            _, chunks = jax.lax.scan(body, jnp.zeros((b, d), jnp.float32), jnp.arange(nb))
            hs = jnp.moveaxis(chunks, 0, 1).reshape(b, nb * chunk, d)
            return hs[:, :t, :]

        return [rg_lru_associative, _chunked_scan]

    def adversarial_cases(self):
        # settable base case: a judge grading a non-default (e.g. probe) case
        # set must not silently force candidates to also trace at the tiny
        # CPU-battery shapes, which no prompt declares
        tiny = self.case_by_name(self.adversarial_case_name)

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
