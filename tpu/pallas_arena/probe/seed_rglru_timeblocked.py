"""RG-LRU Pallas TPU kernel -- time-blocked scan (the production structure).

The time axis is tiled into BLOCK_T chunks and the hidden state is CARRIED
across chunks in VMEM scratch: the grid's last dimension iterates
sequentially on TPU (dimension_semantics "arbitrary"), which is what makes
the carry work. Only one [BLOCK_T, BLOCK_D] tile is on-chip at a time -- a
whole-sequence block at t=32768 would need >250 MB VMEM vs the chip's
~32 MB. This is ONE valid approach; faster kernels may restructure
everything (chunked/associative scan per block, wider tiles, different
gate fusion) as long as forward AND backward stay correct.

Platform facts (each one is a measured failure class in prior attempts):
  * Ref blocks arrive with a leading 1: [1, BLOCK_T, BLOCK_D]; reset is
    [1, BLOCK_T, 1]. Index [0, i, :] for a [BLOCK_D] vector.
  * jax.lax.scan is FORBIDDEN inside a kernel body (the reference uses it
    outside); in-kernel loops are jax.lax.fori_loop. Python for/if over
    traced values also fails.
  * Read refs AT the traced index (a_ref[0, i, :]); indexing a pre-loaded
    array at a traced i emits dynamic_slice, which is forbidden on TPU.
  * Arrays are immutable; writes go through refs: h_ref[0, i, :] = v.
  * Accumulate in float32 (inputs bf16); bf16 accumulation fails tolerance
    over long sequences.
"""

import functools
import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

INTERPRET = bool(int(os.environ.get("PALLAS_INTERPRET", "0")))
BLOCK_T = 512
BLOCK_D = 512


def _fwd_body(x_ref, a_ref, reset_ref, h_ref, carry_ref):
    @pl.when(pl.program_id(2) == 0)
    def _init():
        carry_ref[...] = jnp.zeros_like(carry_ref)

    def step(i, h):
        a_i = a_ref[0, i, :].astype(jnp.float32)
        r_i = reset_ref[0, i, 0]
        a_eff = jnp.where(r_i, 0.0, a_i)
        gx_i = (jnp.sqrt(jnp.maximum(1.0 - a_eff * a_eff, 0.0))
                * x_ref[0, i, :].astype(jnp.float32))
        h = a_eff[None, :] * h + gx_i[None, :]
        h_ref[0, i, :] = h[0]
        return h

    carry_ref[...] = jax.lax.fori_loop(0, a_ref.shape[1], step, carry_ref[...])


def _bwd_body(x_ref, a_ref, reset_ref, hprev_ref, g_ref, dx_ref, da_ref, dcarry_ref):
    @pl.when(pl.program_id(2) == 0)
    def _init():
        dcarry_ref[...] = jnp.zeros_like(dcarry_ref)

    bt = a_ref.shape[1]

    def step(i, dh):
        idx = bt - 1 - i
        a_i = a_ref[0, idx, :].astype(jnp.float32)
        r_i = reset_ref[0, idx, 0]
        a_eff = jnp.where(r_i, 0.0, a_i)
        g_val = jnp.sqrt(jnp.maximum(1.0 - a_eff * a_eff, 0.0))
        grad_h = g_ref[0, idx, :][None, :] + dh
        dx_ref[0, idx, :] = (grad_h * g_val[None, :])[0]
        inv_g = 1.0 / jnp.maximum(g_val, 1e-6)
        da = grad_h * (hprev_ref[0, idx, :][None, :]
                       - (a_eff * inv_g)[None, :] * x_ref[0, idx, :].astype(jnp.float32)[None, :])
        da_ref[0, idx, :] = jnp.where(r_i, 0.0, da)[0]
        return grad_h * a_eff[None, :]

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
        interpret=INTERPRET,
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
    h_prev = jnp.pad(h[:, :-1, :], ((0, 0), (1, 0), (0, 0)))
    hp = _pad_axis(_pad_axis(h_prev, 1, BLOCK_T), 2, BLOCK_D)
    gp = _pad_axis(_pad_axis(g.astype(jnp.float32), 1, BLOCK_T), 2, BLOCK_D)
    tp, dp = xp.shape[1], xp.shape[2]
    ntb = tp // BLOCK_T
    grid = (b, dp // BLOCK_D, ntb)
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
        interpret=INTERPRET,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
    )(xp, ap, rp, hp, gp)
    return dx[:, :t, :d], da[:, :t, :d]


def kernel(x, a, reset):
    """GRADED ENTRYPOINT -- the tests call kernel(x, a, reset) -> h at every
    shape. x: [b, t, d] bfloat16; a: [b, t, d] float32 in [0, 1);
    reset: [b, t] bool; h: [b, t, d] float32. The judge also differentiates
    through this function (d/dx, d/da) via the custom_vjp below."""
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
