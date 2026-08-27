"""Task 4 — FLCE (fused linear cross-entropy) @ 73728 tokens x ~150k vocab.

Baseline: OUR OWN custom_vjp tiled kernel (commits 198f41fa / 2e85086f in
skyrl/backends/tunix_backend.py), copied verbatim below so the judge never
imports skyrl (skyrl is a banned prefix for candidates, and the judge child
poisons it). A win here cuts every train step we run.

kernel(hidden, w, targets) -> logprobs
  hidden:  [n, h]  bfloat16   (flattened B*T token axis)
  w:       [h, v]  bfloat16   (frozen output head)
  targets: [n]     int32
  logprobs:[n]     float32,   logprobs[i] = logits[i, targets[i]] - LSE(logits[i])
  where logits = hidden @ w computed at fp32 accumulation.
Contract: fwd+bwd — d/d(hidden) only (LoRA-only training; w is frozen).
The full [n, v] logits must never be materialized (that's the entire point);
the judge additionally reports peak memory, which exposes violators.
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


def flce_target_logprobs(project_fn, hidden, target_ids, tile_size):
    """Verbatim copy of TunixBackend._flce_target_logprobs (198f41fa/2e85086f):
    custom_vjp over token tiles; fwd discards per-tile logits (residual =
    hidden tiles only), bwd recomputes one tile's logits at a time."""
    *lead, hdim = hidden.shape
    n = 1
    for d in lead:
        n *= d
    flat = hidden.reshape(n, hdim)
    tgt = target_ids.reshape(n)
    num_tiles = -(-n // tile_size)  # ceil
    pad = num_tiles * tile_size - n
    if pad:
        flat = jnp.pad(flat, ((0, pad), (0, 0)))
        tgt = jnp.pad(tgt, ((0, pad),))
    hf = flat.reshape(num_tiles, tile_size, hdim)
    tf = tgt.reshape(num_tiles, tile_size)

    def _tile_lp(h_tile, t_tile):  # [tile, H], [tile] -> [tile]
        logits = project_fn(h_tile).astype(jnp.float32)  # [tile, V], transient
        tl = jnp.take_along_axis(logits, t_tile[:, None], axis=1)[:, 0]
        return tl - jax.nn.logsumexp(logits, axis=-1)

    @jax.custom_vjp
    def _tiled(hf):
        _, lp = jax.lax.scan(lambda c, x: (c, _tile_lp(x[0], x[1])), None, (hf, tf))
        return lp  # [num_tiles, tile]

    def _tiled_fwd(hf):
        _, lp = jax.lax.scan(lambda c, x: (c, _tile_lp(x[0], x[1])), None, (hf, tf))
        return lp, hf  # residual = hidden tiles only (never the logits)

    def _tiled_bwd(hf_res, cot):  # cot: [num_tiles, tile]
        def _bwd_body(_carry, xs):
            h_tile, t_tile, c_tile = xs
            _, vjp_fn = jax.vjp(lambda h: _tile_lp(h, t_tile), h_tile)
            (grad_h_tile,) = vjp_fn(c_tile)
            return None, grad_h_tile

        _, grad_hf = jax.lax.scan(_bwd_body, None, (hf_res, tf, cot))
        return (grad_hf,)

    _tiled.defvjp(_tiled_fwd, _tiled_bwd)
    return _tiled(hf).reshape(n + pad)[:n].reshape(*lead)


BASELINE_TILE_TOKENS = 2048  # the production flce_tile_size


# THE ORACLE MUST ACTUALLY BE FP32. `@` without a precision argument is
# Precision.DEFAULT, which on TPU multiplies f32 inputs through bf16 -- so
# this reference, documented as "logits = hidden @ w computed at fp32
# accumulation", was not. Measured on splash (v6e 2026-08-27): the oracle
# disagreed with a true f32 evaluation of itself by up to 3.7e-3, an EXACT
# candidate was rejected for errors the oracle committed, and the band hid it
# because the honest bf16 variant shared the defect (0.000e+00 measured
# error). reference_bf16 below deliberately KEEPS the bf16 path -- it models
# an honest bf16 kernel and defines the band; only the oracle changes.
_F32 = jax.lax.Precision.HIGHEST


class FLCEProblem(Problem):
    name = "flce"
    version = "1"
    has_bwd = True
    require_pallas = False  # baseline is a custom_vjp scan, not pallas;
    # candidates may use pallas but need not
    memory_bound = False

    def shape_cases(self):
        return [
            ShapeCase("73728x2880x151936", {"n": 73728, "h": 2880, "v": 151936, "tile": BASELINE_TILE_TOKENS}),
            ShapeCase("24576x2880x151936", {"n": 24576, "h": 2880, "v": 151936, "tile": BASELINE_TILE_TOKENS}),
            ShapeCase(
                "holdout-36864x4096x131072",
                {"n": 36864, "h": 4096, "v": 131072, "tile": BASELINE_TILE_TOKENS},
                holdout=True,
            ),
            # PROBE set: same op, one-chip size. The production reference
            # materializes [n, v] fp32 (44.8 GB at 73728x151936), which no
            # 32 GB judge can hold; at n=4096 it is 2.5 GB. Vocab and hidden
            # width are UNCHANGED -- only the token axis shrinks -- so the
            # LSE-at-150k-vocab difficulty the task exists for is intact.
            ShapeCase("probe-4096x2880x151936", {"n": 4096, "h": 2880, "v": 151936, "tile": 2048}, probe=True),
            ShapeCase("probe-2048x2880x151936", {"n": 2048, "h": 2880, "v": 151936, "tile": 2048}, probe=True),
            ShapeCase(
                "probe-holdout-3000x2880x151936",
                {"n": 3000, "h": 2880, "v": 151936, "tile": 2048},
                holdout=True,
                probe=True,
            ),
            # CPU battery cases (non-divisible n exercises the pad path)
            ShapeCase("tiny", {"n": 64, "h": 32, "v": 997, "tile": 16}, smoke=True),
            ShapeCase("tiny-ragged", {"n": 45, "h": 32, "v": 997, "tile": 16}, smoke=True),
            ShapeCase("tiny-holdout", {"n": 40, "h": 16, "v": 499, "tile": 16}, smoke=True, holdout=True),
        ]

    def make_inputs(self, key, case):
        kh, kw, kt = jax.random.split(key, 3)
        n, h, v = case.dims["n"], case.dims["h"], case.dims["v"]
        hidden = (jax.random.normal(kh, (n, h), jnp.float32) / np.sqrt(h)).astype(jnp.bfloat16)
        w = jax.random.normal(kw, (h, v), jnp.float32).astype(jnp.bfloat16)
        targets = jax.random.randint(kt, (n,), 0, v, jnp.int32)
        return (hidden, w, targets)

    def reference(self, hidden, w, targets):
        logits = jax.lax.dot_general(
            hidden.astype(jnp.float32), w.astype(jnp.float32),
            (((1,), (0,)), ((), ())), precision=_F32)
        tl = jnp.take_along_axis(logits, targets[:, None], axis=1)[:, 0]
        return tl - jax.nn.logsumexp(logits, axis=-1)

    def baseline(self, hidden, w, targets):
        tile = min(BASELINE_TILE_TOKENS, hidden.shape[0])
        return flce_target_logprobs(lambda ht: ht @ w, hidden, targets, tile)

    def reference_bf16(self, hidden, w, targets):
        """Calibration at the baseline's true precision: logits materialized
        at bf16 (the bf16 matmul path), LSE at fp32 — matches what our
        custom_vjp kernel (and any honest bf16 candidate) actually computes."""
        logits = hidden.astype(jnp.float32) @ w.astype(jnp.float32)
        logits = logits.astype(jnp.bfloat16).astype(jnp.float32)
        tl = jnp.take_along_axis(logits, targets[:, None], axis=1)[:, 0]
        return tl - jax.nn.logsumexp(logits, axis=-1)

    def grad_outputs(self, kernel_fn, hidden, w, targets):
        """d/d(hidden) of sum(logprobs * cos-probe); w frozen (LoRA-only)."""
        n = hidden.shape[0]
        probe = jnp.cos(jnp.arange(n, dtype=jnp.float32))

        def scalar(h32):
            lp = kernel_fn(h32.astype(hidden.dtype), w, targets)
            return jnp.sum(lp.astype(jnp.float32) * probe)

        return jax.grad(scalar)(hidden.astype(jnp.float32))

    def honest_variants(self):
        """FLCE's ``reference_bf16`` already models the bf16 matmul path (bf16
        logits, fp32 LSE), which is STRICTLY more error than the production
        fp32-accumulator path -- so unlike splash/RPA this band was never too
        tight for a faithful kernel (measured ~0x). What it did not span is
        reduction ORDER, which is the real design freedom here: tile width and
        one-pass-vs-two-pass LSE are exactly the knobs a candidate tunes."""

        def _tiled(tile):
            def variant(hidden, w, targets):
                return flce_target_logprobs(lambda ht: ht @ w, hidden, targets, min(tile, hidden.shape[0]))

            return variant

        def _online_lse(hidden, w, targets, chunk_v=256):
            """Streaming max/sum over the VOCAB axis (running rescale) instead
            of one logsumexp over the full row -- the online-softmax formulation
            a real fused kernel uses, at fp32."""
            logits = hidden.astype(jnp.float32) @ w.astype(jnp.float32)
            logits = logits.astype(jnp.bfloat16).astype(jnp.float32)
            v = logits.shape[1]
            nb = -(-v // chunk_v)
            pad = nb * chunk_v - v
            lg = jnp.pad(logits, ((0, 0), (0, pad)), constant_values=-jnp.inf) if pad else logits
            lg = lg.reshape(logits.shape[0], nb, chunk_v)

            def body(carry, j):
                run_m, run_l = carry
                blk = lg[:, j, :]
                new_m = jnp.maximum(run_m, jnp.max(blk, axis=-1))
                new_l = run_l * jnp.exp(run_m - new_m) + jnp.sum(jnp.exp(blk - new_m[:, None]), axis=-1)
                return (new_m, new_l), None

            init = (jnp.full((logits.shape[0],), -jnp.inf), jnp.zeros((logits.shape[0],)))
            (mx, ls), _ = jax.lax.scan(body, init, jnp.arange(nb))
            tl = jnp.take_along_axis(logits, targets[:, None], axis=1)[:, 0]
            return tl - (mx + jnp.log(ls))

        return [_tiled(BASELINE_TILE_TOKENS // 4), _tiled(BASELINE_TILE_TOKENS * 2), _online_lse]

    def adversarial_cases(self):
        tiny = self.case_by_name(self.adversarial_case_name)

        def label_in_tail(key):
            hidden, w, targets = self.make_inputs(key, tiny)
            v = w.shape[1]
            # every label in the last 8 vocab slots: a top-k / truncated-tail
            # LSE approximator scores these catastrophically wrong
            targets = (v - 1 - (jnp.arange(targets.shape[0]) % 8)).astype(jnp.int32)
            return (hidden, w, targets)

        def lse_saturation(key):
            hidden, w, targets = self.make_inputs(key, tiny)
            # +-80 logit offsets: exp() under/overflows without a max-shifted
            # (or online-softmax) LSE
            hidden = (hidden.astype(jnp.float32) * 80.0).astype(jnp.bfloat16)
            return (hidden, w, targets)

        def big_vocab_row(key):
            # 150k vocab at tiny n: LSE accumulation error across the full
            # production vocab width
            case = ShapeCase("adv-150k", {"n": 4, "h": 32, "v": 151936, "tile": 4})
            return self.make_inputs(key, case)

        def expect_finite(ref, inputs):
            assert np.isfinite(np.asarray(ref, np.float64)).all()
            assert (np.asarray(ref) <= 1e-6).all(), "log-probs must be <= 0"

        return [
            AdversarialCase("label-in-tail", label_in_tail, expect_finite),
            AdversarialCase("lse-saturation", lse_saturation, expect_finite),
            AdversarialCase("150k-vocab", big_vocab_row, expect_finite),
        ]


PROBLEM = FLCEProblem()
