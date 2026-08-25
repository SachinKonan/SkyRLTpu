# RG-LRU Pallas TPU kernel -- TIME-BLOCKED scan (the production structure).
#
# WORKING SEED. This mirrors how DeepMind's shipped recurrentgemma kernel is
# organized: the time axis is tiled into BLOCK_T-step chunks and the hidden
# state is CARRIED across chunks in VMEM scratch, so only one [BLOCK_T,
# BLOCK_D] tile lives on-chip at a time and HBM streaming overlaps compute.
# (A whole-sequence block at t=32768, d-tile 512 would need >250 MB of VMEM;
# the chip has ~32 MB. Time-blocking is not an optimization here, it is what
# makes long sequences POSSIBLE.)
#
# This is ONE valid approach. Faster kernels may restructure everything --
# e.g. chunked associative/parallel scan inside each block, wider time tiles,
# fusing the gate math differently -- as long as fwd AND bwd stay correct.
#
# Platform facts (each one is a measured failure class in prior attempts):
#   * Ref blocks arrive with a leading 1: [1, BLOCK_T, BLOCK_D]; reset is
#     [1, BLOCK_T, 1]. Index [0, i, :] for a [BLOCK_D] vector -- rank
#     mismatches here are the single most common bug.
#   * You cannot call jax.lax.scan inside a Pallas TPU kernel body -- it is
#     forbidden there (the reference uses it OUTSIDE the kernel; do not
#     imitate that in here). Use jax.lax.fori_loop for in-kernel loops.
#     Python for/if over traced values also fails.
#   * Arrays are immutable -- writes go through refs: h_ref[0, i, :] = v.
#   * Accumulate in float32 (inputs are bf16); half-precision accumulation
#     fails the correctness tolerance over long sequences.
#   * The grid's LAST dimension iterates sequentially on TPU; VMEM scratch
#     persists across those steps -- that is what makes the carry work.
#     dimension_semantics marks the parallel vs sequential ("arbitrary") dims.
#
# Scoring: reward = (prod fwd_i * prod bwd_i)^(1/2n), the geometric mean of
# per-shape speedups vs the production kernel, forward AND backward weighted
# equally. Backward speed is HALF the reward -- do not neglect it.

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BLOCK_T = 512   # timesteps per grid step (sequential axis)
BLOCK_D = 512   # feature lanes per grid step (multiple of 128)


def _fwd_body(x_ref, a_ref, reset_ref, h_ref, carry_ref):
    """One (batch, d-tile, t-block) grid cell of the forward scan."""
    # First t-block of each (batch, d-tile) cell: zero the carried state.
    @pl.when(pl.program_id(2) == 0)
    def _init():
        carry_ref[...] = jnp.zeros_like(carry_ref)

    a = a_ref[0].astype(jnp.float32)                 # [BLOCK_T, BLOCK_D]
    reset = reset_ref[0]                             # [BLOCK_T, 1] bool
    a = jnp.where(reset, 0.0, a)                     # a' = 0 where reset
    gx = jnp.sqrt(jnp.maximum(1.0 - a * a, 0.0)) * x_ref[0].astype(jnp.float32)

    def step(i, h):                                  # h: [1, BLOCK_D] f32
        h = a[i][None, :] * h + gx[i][None, :]
        h_ref[0, i, :] = h[0]
        return h

    carry_ref[...] = jax.lax.fori_loop(0, a.shape[0], step, carry_ref[...])


def _bwd_body(x_ref, a_ref, reset_ref, hprev_ref, g_ref, dx_ref, da_ref, dcarry_ref):
    """Reverse-time cotangent propagation, t-blocks visited LAST to FIRST
    (the index maps below reverse the sequential axis). The carried value is
    dL/dh flowing backward across the block boundary. h_{t-1} arrives as its
    own pre-shifted input (computed with one cheap XLA roll outside the
    kernel) so no cross-block h access is needed."""
    @pl.when(pl.program_id(2) == 0)
    def _init():
        dcarry_ref[...] = jnp.zeros_like(dcarry_ref)

    a = a_ref[0].astype(jnp.float32)
    reset = reset_ref[0]
    a_eff = jnp.where(reset, 0.0, a)
    g_val = jnp.sqrt(jnp.maximum(1.0 - a_eff * a_eff, 0.0))
    x = x_ref[0].astype(jnp.float32)
    h_prev = hprev_ref[0]
    g_cot = g_ref[0]
    bt = a.shape[0]

    def step(i, dh):                                 # dh: [1, BLOCK_D] f32
        idx = bt - 1 - i                             # reverse time in-block
        grad_h = g_cot[idx][None, :] + dh
        dx_ref[0, idx, :] = (grad_h * g_val[idx][None, :])[0]
        # dL/da_t = grad_h * (h_{t-1} - (a/g) * x_t); exactly 0 where reset.
        inv_g = 1.0 / jnp.maximum(g_val[idx], 1e-6)
        da = grad_h * (h_prev[idx][None, :] - (a_eff[idx] * inv_g)[None, :] * x[idx][None, :])
        da_ref[0, idx, :] = jnp.where(reset[idx][None, :], 0.0, da)[0]
        return grad_h * a_eff[idx][None, :]          # dL/dh_{t-1}

    dcarry_ref[...] = jax.lax.fori_loop(0, bt, step, dcarry_ref[...])


def _pad_axis(arr, axis, mult):
    pad = (-arr.shape[axis]) % mult
    if pad == 0:
        return arr
    widths = [(0, 0)] * arr.ndim
    widths[axis] = (0, pad)
    return jnp.pad(arr, widths)


def _scan_fwd(x, a, reset):
    b, t, d = x.shape
    # Pad t and d up to block multiples; padded a-tail is 0 so no state leaks
    # across the padded boundary (and padded cotangents are 0 in backward).
    xp = _pad_axis(_pad_axis(x, 1, BLOCK_T), 2, BLOCK_D)
    ap = _pad_axis(_pad_axis(a.astype(jnp.float32), 1, BLOCK_T), 2, BLOCK_D)
    rp = _pad_axis(reset[..., None], 1, BLOCK_T)
    tp, dp = xp.shape[1], xp.shape[2]
    grid = (b, dp // BLOCK_D, tp // BLOCK_T)
    h = pl.pallas_call(
        _fwd_body,
        grid=grid,
        in_specs=[
            pl.BlockSpec((1, BLOCK_T, BLOCK_D), lambda i, j, k: (i, k, j)),
            pl.BlockSpec((1, BLOCK_T, BLOCK_D), lambda i, j, k: (i, k, j)),
            pl.BlockSpec((1, BLOCK_T, 1), lambda i, j, k: (i, k, 0)),
        ],
        out_specs=pl.BlockSpec((1, BLOCK_T, BLOCK_D), lambda i, j, k: (i, k, j)),
        out_shape=jax.ShapeDtypeStruct((b, tp, dp), jnp.float32),
        scratch_shapes=[pltpu.VMEM((1, BLOCK_D), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
    )(xp, ap, rp)
    return h[:, :t, :d]


def _scan_bwd(x, a, reset, h, g):
    b, t, d = x.shape
    xp = _pad_axis(_pad_axis(x, 1, BLOCK_T), 2, BLOCK_D)
    ap = _pad_axis(_pad_axis(a.astype(jnp.float32), 1, BLOCK_T), 2, BLOCK_D)
    rp = _pad_axis(reset[..., None], 1, BLOCK_T)
    # h_{t-1}: one XLA shift outside the kernel (h_prev[0] = 0).
    h_prev = jnp.pad(h[:, :-1, :], ((0, 0), (1, 0), (0, 0)))
    hp = _pad_axis(_pad_axis(h_prev, 1, BLOCK_T), 2, BLOCK_D)
    gp = _pad_axis(_pad_axis(g.astype(jnp.float32), 1, BLOCK_T), 2, BLOCK_D)
    tp, dp = xp.shape[1], xp.shape[2]
    ntb = tp // BLOCK_T
    grid = (b, dp // BLOCK_D, ntb)
    # Reverse the sequential axis: grid step k visits t-block (ntb-1-k).
    rev = lambda i, j, k: (i, ntb - 1 - k, j)
    rev_r = lambda i, j, k: (i, ntb - 1 - k, 0)
    dx, da = pl.pallas_call(
        _bwd_body,
        grid=grid,
        in_specs=[
            pl.BlockSpec((1, BLOCK_T, BLOCK_D), rev),
            pl.BlockSpec((1, BLOCK_T, BLOCK_D), rev),
            pl.BlockSpec((1, BLOCK_T, 1), rev_r),
            pl.BlockSpec((1, BLOCK_T, BLOCK_D), rev),
            pl.BlockSpec((1, BLOCK_T, BLOCK_D), rev),
        ],
        out_specs=[
            pl.BlockSpec((1, BLOCK_T, BLOCK_D), rev),
            pl.BlockSpec((1, BLOCK_T, BLOCK_D), rev),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((b, tp, dp), jnp.float32),
            jax.ShapeDtypeStruct((b, tp, dp), jnp.float32),
        ],
        scratch_shapes=[pltpu.VMEM((1, BLOCK_D), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
    )(xp, ap, rp, hp, gp)
    return dx[:, :t, :d], da[:, :t, :d]


def kernel(x, a, reset):
    @jax.custom_vjp
    def lru(x, a, reset):
        return _scan_fwd(x, a, reset)

    def fwd(x, a, reset):
        h = _scan_fwd(x, a, reset)
        return h, (x, a, reset, h)

    def bwd(res, g):
        x, a, reset, h = res
        dx, da = _scan_bwd(x, a, reset, h, g)
        return (dx.astype(x.dtype), da.astype(a.dtype), None)

    lru.defvjp(fwd, bwd)
    return lru(x, a, reset)
