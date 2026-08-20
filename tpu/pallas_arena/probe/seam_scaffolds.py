"""rf3s SEAM SCAFFOLDS: the harness owns the wiring, the model owns the math.

Measured poles this design sits between (PROBE-REPORT):
  * whole-program splash: 0/96 exports (gemma) -- the pallas_call plumbing
    (grid, BlockSpecs, index maps) kills everyone before algorithm quality
    ever gets measured;
  * the `tailored` fill-in-the-blank scaffold: 16/16 pass and within-group
    spread 0.0042 -- BELOW the judge noise floor -- because every structural
    decision was made for the model. Perfect validity, zero trainability.

The split: the scaffold owns what fails in correlated, uninformative ways
(pallas_call wiring, BlockSpec index maps incl. the GQA h//group mapping,
padding for non-divisible sequences, and the ENTIRE jax.custom_vjp wiring
incl. the integer-input None-cotangent rule). The model owns everything
where candidates can differ meaningfully: both kernel BODIES, the tile
sizes, and what to stash as residuals.

The scaffolds are shipped inside the rf3s prompt as an EDITABLE starting
program (the TriMul starter-code principle applied to plumbing): bodies are
`raise NotImplementedError` and the model outputs the COMPLETE program with
them filled -- it may restructure anything, the scaffold is a working
skeleton, not a cage. NAIVE_FILLS below are reference fills used ONLY by our
own tests to prove the wiring exports and is numerically sound; they are
never shown to a model.
"""

SPLASH_SCAFFOLD = '''import functools
import os
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# CPU-validation hook -- leave as is. 0 on the judge (real Mosaic kernel);
# our CPU harness sets it to 1 to run the kernel in interpret mode.
INTERPRET = bool(int(os.environ.get("PALLAS_INTERPRET", "0")))

# ------------------------- YOUR TILE CHOICES (tune these) -------------------
BLOCK_Q = 512          # query rows per grid step (multiple of 8)
NEG_INF = -1e30

# ---------------------------------------------------------------- forward ---
def _fwd_body(q_ref, k_ref, v_ref, seg_ref, o_ref, *, seq, block_q,
              window, soft_cap, sinks_val):
    """One grid step = ONE query block of ONE query head.

    Ref shapes handed to you (already sliced by the BlockSpecs below):
      q_ref  : [block_q, d]     bf16   this head's query block
      k_ref  : [seq_pad, d]     bf16   the matching KV head, full (padded) seq
      v_ref  : [seq_pad, d_v]   bf16
      seg_ref: [seq_pad]        int32  segment ids (0 = padding)
      o_ref  : [block_q, d_v]   f32    write your output block here
    Grid position: pl.program_id(0) = query head h, pl.program_id(1) = block i.
    The first block_q*i rows of seg belong to positions before your block.
    sinks_val is a float for THIS head, or None.
    """
    raise NotImplementedError("YOUR FORWARD BODY")


def _forward(q, k, v, segment_ids, window, soft_cap, sinks):
    qh, seq, d = q.shape
    kvh = k.shape[0]
    dv = v.shape[-1]
    group = qh // kvh
    block_q = min(BLOCK_Q, seq)
    seq_pad = -(-seq // block_q) * block_q                     # pad to multiple
    pad = seq_pad - seq
    qp = jnp.pad(q, ((0, 0), (0, pad), (0, 0)))
    kp = jnp.pad(k, ((0, 0), (0, pad), (0, 0)))
    vp = jnp.pad(v, ((0, 0), (0, pad), (0, 0)))
    segp = jnp.pad(segment_ids, (0, pad))                      # pad rows = seg 0

    grid = (qh, seq_pad // block_q)
    out = pl.pallas_call(
        functools.partial(_fwd_body, seq=seq, block_q=block_q,
                          window=window, soft_cap=soft_cap,
                          sinks_val=None),
        grid=grid,
        in_specs=[
            pl.BlockSpec((1, block_q, d), lambda h, i: (h, i, 0)),
            pl.BlockSpec((1, seq_pad, d), lambda h, i: (h // group, 0, 0)),
            pl.BlockSpec((1, seq_pad, dv), lambda h, i: (h // group, 0, 0)),
            pl.BlockSpec((seq_pad,), lambda h, i: (0,)),
        ],
        out_specs=pl.BlockSpec((1, block_q, dv), lambda h, i: (h, i, 0)),
        out_shape=jax.ShapeDtypeStruct((qh, seq_pad, dv), jnp.float32),
        interpret=INTERPRET,
    )(qp, kp, vp, segp)
    return out[:, :seq, :]


# ------------------------------------------------- backward (custom_vjp) ---
# The wiring below is COMPLETE and correct -- segment_ids is an integer input
# so its cotangent slot must be None; window/soft_cap/sinks are static.
# You fill the two bwd bodies (or replace the whole strategy). Until they are
# filled the kernel still grades: forward reward is kept, backward is
# forfeited (it is scored, not gated).

def _bwd_dq_body(q_ref, k_ref, v_ref, seg_ref, do_ref, dq_ref, *, seq, block_q,
                 window, soft_cap):
    """Same grid as forward. do_ref: [block_q, d_v] f32 incoming cotangent;
    write dq for your block into dq_ref [block_q, d] f32."""
    raise NotImplementedError("YOUR dQ BODY")


def _bwd_dkv_body(q_ref, k_ref, v_ref, seg_ref, do_ref, dk_ref, dv_ref, *,
                  seq, window, soft_cap):
    """Grid over KV heads only: one step owns ONE kv head's FULL dk/dv.
    q_ref/do_ref carry all `group` query heads for this kv head:
      q_ref  [group, seq_pad, d], do_ref [group, seq_pad, d_v]
      dk_ref [seq_pad, d] f32, dv_ref [seq_pad, d_v] f32."""
    raise NotImplementedError("YOUR dK/dV BODY")


def _make_bwd(window, soft_cap, sinks):
    def bwd(res, g):
        q, k, v, segment_ids = res
        qh, seq, d = q.shape
        kvh = k.shape[0]
        dv = v.shape[-1]
        group = qh // kvh
        block_q = min(BLOCK_Q, seq)
        seq_pad = -(-seq // block_q) * block_q
        pad = seq_pad - seq
        qp = jnp.pad(q, ((0, 0), (0, pad), (0, 0)))
        kp = jnp.pad(k, ((0, 0), (0, pad), (0, 0)))
        vp = jnp.pad(v, ((0, 0), (0, pad), (0, 0)))
        gp = jnp.pad(g.astype(jnp.float32), ((0, 0), (0, pad), (0, 0)))
        segp = jnp.pad(segment_ids, (0, pad))

        dq = pl.pallas_call(
            functools.partial(_bwd_dq_body, seq=seq, block_q=block_q,
                              window=window, soft_cap=soft_cap),
            grid=(qh, seq_pad // block_q),
            in_specs=[
                pl.BlockSpec((1, block_q, d), lambda h, i: (h, i, 0)),
                pl.BlockSpec((1, seq_pad, d), lambda h, i: (h // group, 0, 0)),
                pl.BlockSpec((1, seq_pad, dv), lambda h, i: (h // group, 0, 0)),
                pl.BlockSpec((seq_pad,), lambda h, i: (0,)),
                pl.BlockSpec((1, block_q, dv), lambda h, i: (h, i, 0)),
            ],
            out_specs=pl.BlockSpec((1, block_q, d), lambda h, i: (h, i, 0)),
            out_shape=jax.ShapeDtypeStruct((qh, seq_pad, d), jnp.float32),
            interpret=INTERPRET,
        )(qp, kp, vp, segp, gp)[:, :seq, :]

        qg = qp.reshape(kvh, group, seq_pad, d)
        gg = gp.reshape(kvh, group, seq_pad, dv)
        dk, dvv = pl.pallas_call(
            functools.partial(_bwd_dkv_body, seq=seq,
                              window=window, soft_cap=soft_cap),
            grid=(kvh,),
            in_specs=[
                pl.BlockSpec((1, group, seq_pad, d), lambda h: (h, 0, 0, 0)),
                pl.BlockSpec((1, seq_pad, d), lambda h: (h, 0, 0)),
                pl.BlockSpec((1, seq_pad, dv), lambda h: (h, 0, 0)),
                pl.BlockSpec((seq_pad,), lambda h: (0,)),
                pl.BlockSpec((1, group, seq_pad, dv), lambda h: (h, 0, 0, 0)),
            ],
            out_specs=[
                pl.BlockSpec((1, seq_pad, d), lambda h: (h, 0, 0)),
                pl.BlockSpec((1, seq_pad, dv), lambda h: (h, 0, 0)),
            ],
            out_shape=[
                jax.ShapeDtypeStruct((kvh, seq_pad, d), jnp.float32),
                jax.ShapeDtypeStruct((kvh, seq_pad, dv), jnp.float32),
            ],
            interpret=INTERPRET,
        )(qg, kp, vp, segp, gg)
        dk = dk[:, :seq, :]
        dvv = dvv[:, :seq, :]
        # segment_ids is integer-typed: its cotangent slot is None.
        return (dq.astype(q.dtype), dk.astype(k.dtype), dvv.astype(v.dtype), None)
    return bwd


def kernel(q, k, v, segment_ids, *, window=None, soft_cap=None, sinks=None):
    @jax.custom_vjp
    def attn(q, k, v, segment_ids):
        return _forward(q, k, v, segment_ids, window, soft_cap, sinks)

    def fwd(q, k, v, segment_ids):
        out = _forward(q, k, v, segment_ids, window, soft_cap, sinks)
        # residuals: YOUR choice -- store more (e.g. row max / denominator)
        # to avoid recomputing the softmax statistics in the backward.
        return out, (q, k, v, segment_ids)

    attn.defvjp(fwd, _make_bwd(window, soft_cap, sinks))
    return attn(q, k, v, segment_ids)
'''


RGLRU_SCAFFOLD = '''import functools
import os
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# CPU-validation hook -- leave as is. 0 on the judge (real Mosaic kernel);
# our CPU harness sets it to 1 to run the kernel in interpret mode.
INTERPRET = bool(int(os.environ.get("PALLAS_INTERPRET", "0")))

# ------------------------- YOUR TILE CHOICES (tune these) -------------------
BLOCK_D = 512          # feature columns per grid step (multiple of 128)

# ---------------------------------------------------------------- forward ---
def _fwd_body(x_ref, a_ref, reset_ref, h_ref, *, t):
    """One grid step = ONE batch row x ONE feature tile, FULL time axis.

    Ref shapes (sliced by the BlockSpecs below):
      x_ref    : [t, BLOCK_D] bf16
      a_ref    : [t, BLOCK_D] f32   gates in [0, 1)
      reset_ref: [t]          bool  (True -> state resets to 0 at this step)
      h_ref    : [t, BLOCK_D] f32   write ALL hidden states here
    Recurrence: h_t = a_t * h_{t-1} + sqrt(1 - a_t^2) * x_t, with a_t forced
    to 0 where reset_t (so no state crosses the boundary).
    """
    raise NotImplementedError("YOUR FORWARD BODY")


def _forward(x, a, reset):
    b, t, d = x.shape
    block_d = min(BLOCK_D, d)
    grid = (b, d // block_d)
    return pl.pallas_call(
        functools.partial(_fwd_body, t=t),
        grid=grid,
        in_specs=[
            pl.BlockSpec((1, t, block_d), lambda i, j: (i, 0, j)),
            pl.BlockSpec((1, t, block_d), lambda i, j: (i, 0, j)),
            pl.BlockSpec((1, t), lambda i, j: (i, 0)),
        ],
        out_specs=pl.BlockSpec((1, t, block_d), lambda i, j: (i, 0, j)),
        out_shape=jax.ShapeDtypeStruct((b, t, d), jnp.float32),
        interpret=INTERPRET,
    )(x, a, reset)


# ------------------------------------------------- backward (custom_vjp) ---
def _bwd_body(x_ref, a_ref, reset_ref, h_ref, g_ref, dx_ref, da_ref, *, t):
    """Same grid as forward, REVERSE-time recurrence.
    g_ref: [t, BLOCK_D] f32 incoming cotangent dL/dh.
    Write dx into dx_ref [t, BLOCK_D] and da into da_ref [t, BLOCK_D].
    Remember the gate path: h_t depends on a_t both through a_t*h_{t-1} AND
    through sqrt(1-a_t^2)*x_t."""
    raise NotImplementedError("YOUR BACKWARD BODY")


def kernel(x, a, reset):
    @jax.custom_vjp
    def lru(x, a, reset):
        return _forward(x, a, reset)

    def fwd(x, a, reset):
        h = _forward(x, a, reset)
        return h, (x, a, reset, h)

    def bwd(res, g):
        x, a, reset, h = res
        b, t, d = x.shape
        block_d = min(BLOCK_D, d)
        dx, da = pl.pallas_call(
            functools.partial(_bwd_body, t=t),
            grid=(b, d // block_d),
            in_specs=[
                pl.BlockSpec((1, t, block_d), lambda i, j: (i, 0, j)),
                pl.BlockSpec((1, t, block_d), lambda i, j: (i, 0, j)),
                pl.BlockSpec((1, t), lambda i, j: (i, 0)),
                pl.BlockSpec((1, t, block_d), lambda i, j: (i, 0, j)),
                pl.BlockSpec((1, t, block_d), lambda i, j: (i, 0, j)),
            ],
            out_specs=[
                pl.BlockSpec((1, t, block_d), lambda i, j: (i, 0, j)),
                pl.BlockSpec((1, t, block_d), lambda i, j: (i, 0, j)),
            ],
            out_shape=[
                jax.ShapeDtypeStruct((b, t, d), jnp.float32),
                jax.ShapeDtypeStruct((b, t, d), jnp.float32),
            ],
            interpret=INTERPRET,
        )(x, a, reset, h, g.astype(jnp.float32))
        # reset is boolean: cotangent slot None.
        return (dx.astype(x.dtype), da.astype(a.dtype), None)

    lru.defvjp(fwd, bwd)
    return lru(x, a, reset)
'''

# --------------------------------------------------------------------------
# NAIVE FILLS: test-only bodies proving the scaffold WIRING (BlockSpecs,
# grids, custom_vjp plumbing) exports and is numerically sound. They read the
# whole (padded) sequence per block -- fine for CPU tiny-case validation,
# NOT TPU-viable at production shapes, and never shown to a model.

SPLASH_NAIVE_FWD = '''
    del sinks_val
    i = pl.program_id(1)
    q = q_ref[0].astype(jnp.float32)          # [block_q, d]
    k = k_ref[0].astype(jnp.float32)          # [seq_pad, d]
    v = v_ref[0].astype(jnp.float32)          # [seq_pad, d_v]
    seg = seg_ref[...]                        # [seq_pad]
    seq_pad = k.shape[0]
    logits = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())),
                                 preferred_element_type=jnp.float32)
    if soft_cap is not None:
        logits = soft_cap * jnp.tanh(logits / soft_cap)
    row = jax.lax.broadcasted_iota(jnp.int32, (block_q, seq_pad), 0) + i * block_q
    col = jax.lax.broadcasted_iota(jnp.int32, (block_q, seq_pad), 1)
    seg_q = jnp.take(seg, jnp.minimum(row[:, 0], seq_pad - 1), axis=0)[:, None]
    mask = (row >= col) & (seg_q == seg[None, :]) & (seg_q != 0) & (seg[None, :] != 0)
    if window is not None:
        mask = mask & (row - col < window)
    logits = jnp.where(mask, logits, NEG_INF)
    row_live = mask.any(axis=1)
    m = jnp.max(logits, axis=1, keepdims=True)
    p = jnp.where(mask, jnp.exp(logits - m), 0.0)
    denom = jnp.maximum(jnp.sum(p, axis=1, keepdims=True), 1e-30)
    p = jnp.where(row_live[:, None], p / denom, 0.0)
    o_ref[0] = jax.lax.dot_general(p, v, (((1,), (0,)), ((), ())),
                                   preferred_element_type=jnp.float32)
'''

RGLRU_NAIVE_FWD = '''
    x = x_ref[0].astype(jnp.float32)          # [t, block_d]
    a = a_ref[0]                              # [t, block_d] f32
    reset = reset_ref[0]                      # [t]
    a = a * (1.0 - reset[:, None].astype(jnp.float32))
    gx = jnp.sqrt(jnp.maximum(1.0 - a * a, 0.0)) * x

    def step(carry, idx):
        h = a[idx] * carry + gx[idx]
        return h, h

    _, hs = jax.lax.scan(step, jnp.zeros((x.shape[1],), jnp.float32),
                         jnp.arange(t))
    h_ref[0] = hs
'''

RGLRU_NAIVE_BWD = '''
    x = x_ref[0].astype(jnp.float32)
    a_raw = a_ref[0]
    reset = reset_ref[0]
    h = h_ref[0]
    g = g_ref[0]
    live = 1.0 - reset[:, None].astype(jnp.float32)
    a = a_raw * live
    root = jnp.sqrt(jnp.maximum(1.0 - a * a, 0.0))
    h_prev = jnp.concatenate([jnp.zeros((1, x.shape[1]), jnp.float32), h[:-1]], axis=0)

    def step(lam_next, idx):
        i = t - 1 - idx
        lam = g[i] + lam_next            # dL/dh_i including path through h_{i+1}
        dx_i = lam * root[i]
        da_eff = lam * (h_prev[i] - (a[i] / jnp.maximum(root[i], 1e-20)) * x[i])
        da_i = da_eff * live[i]          # gate was zeroed at resets
        lam_next = lam * a[i]
        return lam_next, (dx_i, da_i)

    _, (dxs, das) = jax.lax.scan(step, jnp.zeros((x.shape[1],), jnp.float32),
                                 jnp.arange(t))
    dx_ref[0] = dxs[::-1]
    da_ref[0] = das[::-1]
'''


def compose(scaffold: str, fills: dict) -> str:
    """Replace each body's `raise NotImplementedError(...)` LINE with a fill.

    Whole-line replacement, because the fills are already written at
    function-body indent -- substring replacement would stack the needle
    line's indent on top of the fill's first line (measured: 8-space
    `del sinks_val`, SyntaxError)."""
    lines = scaffold.splitlines()
    for marker, body in fills.items():
        needle = f'raise NotImplementedError("{marker}")'
        hits = [i for i, l in enumerate(lines) if needle in l]
        assert len(hits) == 1, (marker, hits)
        lines[hits[0] : hits[0] + 1] = body.strip("\n").splitlines()
    return "\n".join(lines) + "\n"
