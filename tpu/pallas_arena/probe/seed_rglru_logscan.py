
"""RG-LRU Pallas TPU kernel -- time-blocked scan with an in-tile parallel scan.

The time axis is tiled into BLOCK_T chunks and the hidden state is CARRIED
across chunks in VMEM scratch (the grid's last dimension iterates
sequentially on TPU, which is what makes the carry work). Only one
[BLOCK_T, BLOCK_D] tile is on-chip at a time -- a whole-sequence block at
t=32768 would need >250 MB of VMEM against the chip's ~32 MB.

Within a tile the recurrence is solved by an ASSOCIATIVE SCAN over whole
vectors, never by walking timesteps: h_t = A_t*h_init + B_t, where (A, B)
is the running composition of the affine maps (a_t, gx_t). This is ONE valid
approach; faster kernels may restructure everything (different tile shapes,
chunked two-level scans, different gate fusion) as long as forward AND
backward stay correct.

Platform facts (each one is a measured failure class in prior attempts):
  * NEVER index a ref at a traced index: `ref[0, i, :]` inside a fori_loop
    fails to compile with
      E2003 CompileTimeMosaicUnprovenMemoryAccessAlignment: cannot
      statically prove that index in dimension 1 is a multiple of 8
    because VMEM is (8,128)-tiled. Load whole tiles (`ref[...]`) and operate
    on them with vector ops; slice only at STATIC offsets.
  * jax.lax.scan is FORBIDDEN inside a kernel body, and
    jax.lax.associative_scan does not lower there either. Solve a linear
    recurrence with an explicit log-step scan over STATIC shifts (below).
    Python for/if over traced values also fails.
  * Arrays are immutable; writes go through refs: h_ref[...] = value.
  * Accumulate in float32 (inputs are bf16); bf16 accumulation fails the
    correctness tolerance over long sequences.
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


def _scan_affine(a, b):
    """Inclusive scan of the affine maps h -> a_t*h + b_t along axis 0.

    Hillis-Steele with STATIC power-of-two shifts. jax.lax.associative_scan
    does not lower inside a Pallas body (its odd/even slicing degenerates:
    "vector types must have positive dimensions"), and a fori_loop over
    timesteps cannot lower either (E2003 alignment). Static concatenate +
    slice is what Mosaic can prove.
    """
    t = a.shape[0]
    k = 1
    while k < t:
        ones = jnp.ones((k, a.shape[1]), a.dtype)
        zeros = jnp.zeros((k, b.shape[1]), b.dtype)
        a_prev = jnp.concatenate([ones, a[:-k]], axis=0)
        b_prev = jnp.concatenate([zeros, b[:-k]], axis=0)
        b = a * b_prev + b          # uses the CURRENT a; order matters
        a = a_prev * a
        k *= 2
    return a, b


def _fwd_body(x_ref, a_ref, reset_ref, h_ref, carry_ref):
    @pl.when(pl.program_id(2) == 0)
    def _init():
        carry_ref[...] = jnp.zeros_like(carry_ref)

    a = a_ref[...][0].astype(jnp.float32)          # [BLOCK_T, BLOCK_D]
    x = x_ref[...][0].astype(jnp.float32)
    reset = reset_ref[...][0]                      # [BLOCK_T, 1]
    a_eff = jnp.where(reset, 0.0, a)
    gx = jnp.sqrt(jnp.maximum(1.0 - a_eff * a_eff, 0.0)) * x

    A, B = _scan_affine(a_eff, gx)
    h = A * carry_ref[...] + B                     # carry: [1, BLOCK_D]
    h_ref[0] = h
    carry_ref[...] = h[-1:]                        # STATIC slice


def _bwd_body(x_ref, a_ref, reset_ref, hprev_ref, anext_ref, g_ref,
              dx_ref, da_ref, dcarry_ref):
    """Reverse-time cotangent propagation, t-blocks visited last-to-first.

    G_t = g_t + a_{t+1} * G_{t+1} is the same linear recurrence run
    backwards, so it is the same associative scan on flipped arrays.
    a_{t+1} and h_{t-1} arrive as their own pre-shifted inputs (one cheap
    XLA roll each, outside the kernel) so no cross-block element access is
    needed inside.
    """
    @pl.when(pl.program_id(2) == 0)
    def _init():
        dcarry_ref[...] = jnp.zeros_like(dcarry_ref)

    a = a_ref[...][0].astype(jnp.float32)
    reset = reset_ref[...][0]
    a_eff = jnp.where(reset, 0.0, a)
    g_val = jnp.sqrt(jnp.maximum(1.0 - a_eff * a_eff, 0.0))
    x = x_ref[...][0].astype(jnp.float32)
    h_prev = hprev_ref[...][0]
    a_next = anext_ref[...][0].astype(jnp.float32)
    g_cot = g_ref[...][0]

    # Flip time (static reverse), scan, flip back.
    coef_r = jnp.flip(a_next, axis=0)
    src_r = jnp.flip(g_cot, axis=0)
    Ar, Br = _scan_affine(coef_r, src_r)
    G = jnp.flip(Ar * dcarry_ref[...] + Br, axis=0)

    dx_ref[0] = G * g_val
    inv_g = 1.0 / jnp.maximum(g_val, 1e-6)
    da = G * (h_prev - (a_eff * inv_g) * x)
    da_ref[0] = jnp.where(reset, 0.0, da)
    dcarry_ref[...] = G[:1]                        # STATIC slice


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
    a32 = a.astype(jnp.float32)
    a_eff_full = a32 * (1.0 - reset[..., None].astype(jnp.float32))
    a_next = jnp.pad(a_eff_full[:, 1:, :], ((0, 0), (0, 1), (0, 0)))
    h_prev = jnp.pad(h[:, :-1, :], ((0, 0), (1, 0), (0, 0)))

    xp = _pad_axis(_pad_axis(x, 1, BLOCK_T), 2, BLOCK_D)
    ap = _pad_axis(_pad_axis(a32, 1, BLOCK_T), 2, BLOCK_D)
    rp = _pad_axis(reset[..., None], 1, BLOCK_T)
    hp = _pad_axis(_pad_axis(h_prev, 1, BLOCK_T), 2, BLOCK_D)
    anp = _pad_axis(_pad_axis(a_next, 1, BLOCK_T), 2, BLOCK_D)
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
    )(xp, ap, rp, hp, anp, gp)
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

