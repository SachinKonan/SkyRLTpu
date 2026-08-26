"""RG-LRU Pallas TPU kernel -- the production structure, stripped to basics.

This is a barebones transcription of how DeepMind's recurrentgemma Pallas
scan is organised (its complex-number, sharding and multi-shard machinery
removed). Two things carry all of the structure:

  1. LAYOUT. The feature dim is split as d -> (d // 128, 128) so a block is
     4-D: (batch, seq_tile, dim_tile, 128). The last two dims form the
     (8,128) VMEM tile, which leaves the SEQUENCE axis outside the tiled
     dims -- and that is what makes a per-timestep `ref[:, i]` legal: it
     selects whole tiles, so no alignment has to be proven.
     Putting time in the second-minor position instead fails to compile:
       E2003 CompileTimeMosaicUnprovenMemoryAccessAlignment: cannot
       statically prove that index in dimension 1 is a multiple of 8.
  2. CARRY. The grid's last dimension iterates sequentially on TPU, and
     VMEM scratch persists across those steps, so the hidden state is
     carried from one seq tile to the next instead of materialising the
     whole sequence (t=32768 at once would need >250 MB against ~32 MB).

The gate math (reset, sqrt(1-a^2)) is elementwise, so it runs OUTSIDE the
kernel in XLA and the kernel stays a pure linear recurrence h = a*h + gx --
the same division of labour the production kernel uses.

This is ONE valid approach; faster kernels may restructure anything (tile
sizes, two-level chunked scans, fusing the gate work in, a different
backward decomposition) as long as forward AND backward stay correct.

Platform facts (each is a measured failure class):
  * jax.lax.scan is FORBIDDEN inside a kernel body, and
    jax.lax.associative_scan does not lower there either ("vector types
    must have positive dimensions"). Use jax.lax.fori_loop.
  * Python for/if over traced values fails; use jnp.where / lax.cond.
  * Arrays are immutable; writes go through refs.
  * Accumulate in float32 (inputs are bf16); bf16 accumulation fails
    tolerance over long sequences.
"""

import functools
import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

INTERPRET = bool(int(os.environ.get("PALLAS_INTERPRET", "0")))
LANE = 128          # last dim of the VMEM tile
DIM_TILE = 8        # sublanes: dim chunks per block
SEQ_TILE = 256      # timesteps per grid step (sequential axis)


def _rnn_body(a_ref, x_ref, h_ref, carry_ref, *, reverse: bool):
    """h_t = a_t * h_{t-1} + x_t over one seq tile.

    Refs are (1, seq_tile, dim_tile, LANE); `[:, i]` indexes the SEQUENCE,
    which is outside the tiled last-two dims, so the dynamic index is a
    whole-tile offset.
    """
    @pl.when(pl.program_id(2) == 0)
    def _init():
        carry_ref[...] = jnp.zeros_like(carry_ref)

    n = a_ref.shape[1]

    def step(i, _):
        idx = (n - 1 - i) if reverse else i
        h = carry_ref[...]
        h_next = a_ref[:, idx].astype(jnp.float32) * h + x_ref[:, idx].astype(jnp.float32)
        carry_ref[...] = h_next
        h_ref[:, idx] = h_next
        return 0

    jax.lax.fori_loop(0, n, step, 0)


def _lin_rnn(a, x, *, reverse: bool):
    """Run h = a*h + x along the sequence. a, x: [b, t, dt, LANE] f32."""
    b, t, dt, lane = a.shape
    nseq = t // SEQ_TILE
    ndim = dt // DIM_TILE
    grid = (b, ndim, nseq)
    if reverse:
        seq_idx = lambda i, j, k: (i, nseq - 1 - k, j, 0)
    else:
        seq_idx = lambda i, j, k: (i, k, j, 0)
    blk = (1, SEQ_TILE, DIM_TILE, lane)
    return pl.pallas_call(
        functools.partial(_rnn_body, reverse=reverse),
        grid=grid,
        in_specs=[pl.BlockSpec(blk, seq_idx), pl.BlockSpec(blk, seq_idx)],
        out_specs=pl.BlockSpec(blk, seq_idx),
        out_shape=jax.ShapeDtypeStruct((b, t, dt, lane), jnp.float32),
        # Carry has the block shape WITHOUT the sequence axis -- production's
        # h_shape = x_shape[:1] + x_shape[2:] -- so it matches ref[:, i].
        scratch_shapes=[pltpu.VMEM((1, DIM_TILE, lane), jnp.float32)],
        interpret=INTERPRET,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
    )(a, x)


def _pad_t(arr, mult):
    pad = (-arr.shape[1]) % mult
    return arr if pad == 0 else jnp.pad(arr, ((0, 0), (0, pad), (0, 0)))


def _to_tiles(arr):
    """[b, t, d] -> [b, t, d//LANE, LANE] (d padded up to a LANE*DIM_TILE
    multiple so both tiled dims divide)."""
    b, t, d = arr.shape
    mult = LANE * DIM_TILE
    padd = (-d) % mult
    if padd:
        arr = jnp.pad(arr, ((0, 0), (0, 0), (0, padd)))
    return arr.reshape(b, arr.shape[1], arr.shape[2] // LANE, LANE)


def _from_tiles(arr, d):
    b, t = arr.shape[0], arr.shape[1]
    return arr.reshape(b, t, -1)[:, :, :d]


def _prep(x, a, reset):
    """Elementwise gate work, outside the kernel (as production does)."""
    a32 = a.astype(jnp.float32) * (1.0 - reset[..., None].astype(jnp.float32))
    gx = jnp.sqrt(jnp.maximum(1.0 - a32 * a32, 0.0)) * x.astype(jnp.float32)
    return a32, gx


def _fwd(x, a, reset):
    b, t, d = x.shape
    a32, gx = _prep(x, a, reset)
    at, gt = _to_tiles(_pad_t(a32, SEQ_TILE)), _to_tiles(_pad_t(gx, SEQ_TILE))
    h = _lin_rnn(at, gt, reverse=False)
    return _from_tiles(h, d)[:, :t, :]


def _bwd(x, a, reset, h, g):
    """G_t = g_t + a_{t+1} G_{t+1}: the same recurrence, reversed, with the
    coefficient shifted by one (computed outside the kernel)."""
    b, t, d = x.shape
    a32, _ = _prep(x, a, reset)
    g_val = jnp.sqrt(jnp.maximum(1.0 - a32 * a32, 0.0))
    a_next = jnp.pad(a32[:, 1:, :], ((0, 0), (0, 1), (0, 0)))
    h_prev = jnp.pad(h[:, :-1, :], ((0, 0), (1, 0), (0, 0)))

    ant, gct = _to_tiles(_pad_t(a_next, SEQ_TILE)), _to_tiles(_pad_t(g.astype(jnp.float32), SEQ_TILE))
    G = _from_tiles(_lin_rnn(ant, gct, reverse=True), d)[:, :t, :]

    dx = G * g_val
    inv_g = 1.0 / jnp.maximum(g_val, 1e-6)
    da = jnp.where(reset[..., None], 0.0, G * (h_prev - (a32 * inv_g) * x.astype(jnp.float32)))
    return dx, da


def kernel(x, a, reset):
    """GRADED ENTRYPOINT -- the tests call kernel(x, a, reset) -> h at every
    shape. x: [b, t, d] bfloat16; a: [b, t, d] float32 in [0, 1);
    reset: [b, t] bool; h: [b, t, d] float32. The judge also differentiates
    through this function (d/dx, d/da) via the custom_vjp below."""
    @jax.custom_vjp
    def lru(x, a, reset):
        return _fwd(x, a, reset)

    def fwd_rule(x, a, reset):
        h = _fwd(x, a, reset)
        return h, (x, a, reset, h)

    def bwd_rule(res, g):
        x, a, reset, h = res
        dx, da = _bwd(x, a, reset, h, g)
        return (dx.astype(x.dtype), da.astype(a.dtype), None)

    lru.defvjp(fwd_rule, bwd_rule)
    return lru(x, a, reset)
