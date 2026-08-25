"""Splash attention (causal, segment-gated, GQA) -- Pallas TPU kernel.

GRADED ENTRYPOINT: the tests call kernel(q, k, v, segment_ids, *, window,
soft_cap, sinks) -> o at every shape (q: [q_heads, seq, d] bf16 pre-scaled;
k/v: [kv_heads, seq, d|d_v] bf16; segment_ids: [seq] int32, 0 = padding;
o: [q_heads, seq, d_v] f32). The judge differentiates through it (d/dq,
d/dk, d/dv) via the custom_vjp wiring below.

Structure: the flash-attention pattern -- grid tiles (query head, BLOCK_Q
query rows); K/V arrive whole per (grouped) head and are walked in blocks
inside the body with an ONLINE SOFTMAX (running max + running sum, output
rescaled as the max moves; statistics kept float32). Causality and segment
ids gate each tile. Fully-masked KV tiles still cost compute here -- the
production kernel SKIPS them, a large part of its speed. This is ONE valid
approach; faster kernels may restructure everything.

Platform facts: refs have their BLOCK shape with leading 1s; jax.lax.scan
is FORBIDDEN inside a kernel body (fori_loop only; python if/for on traced
values fails -- use jnp.where/lax.cond); read refs at traced indices, not
pre-loaded arrays (forbidden dynamic_slice); writes go through refs;
matmuls need preferred_element_type=jnp.float32.
"""

import functools
import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

INTERPRET = bool(int(os.environ.get("PALLAS_INTERPRET", "0")))
BLOCK_Q = 512
NEG_INF = -1e30

# ---------------------------------------------------------------- forward ---
def _fwd_body(q_ref, k_ref, v_ref, seg_ref, o_ref, *, seq, block_q, window, soft_cap, sinks_val):
    """
    Computes causal segment-masked attention for one query head and block.
    Grid: (q_heads, num_blocks)
    q_ref: [1, block_q, d]
    k_ref: [1, seq_pad, d]
    v_ref: [1, seq_pad, dv]
    seg_ref: [seq_pad]
    o_ref: [1, block_q, dv]
    """
    # Load inputs
    q_block = q_ref[0].astype(jnp.float32)  # [block_q, d]
    k_full = k_ref[0].astype(jnp.float32)   # [seq_pad, d]
    v_full = v_ref[0].astype(jnp.float32)   # [seq_pad, dv]
    seg_vals = seg_ref[...]                 # [seq_pad,] int32
    
    # Indices
    q_head_idx = pl.program_id(0)
    q_block_idx = pl.program_id(1)
    q_idx = jnp.arange(block_q, dtype=jnp.int32) + (q_block_idx * block_q)
    k_idx = jnp.arange(seg_vals.shape[0], dtype=jnp.int32)  # [seq_pad]
    
    # Q @ K^T
    logits = jnp.matmul(q_block, k_full.T, preferred_element_type=jnp.float32)  # [block_q, seq_pad]
    
    # Soft cap
    if soft_cap is not None:
        logits = soft_cap * jnp.tanh(logits / soft_cap)
    
    # Masks
    # Causal
    causal = k_idx[None, :] <= q_idx[:, None]
    if window is not None:
        causal = causal & ((q_idx[:, None] - k_idx[None, :]) < window)
    
    # Segment
    seg_q = seg_vals[q_idx] # [block_q]
    seg_k = seg_vals        # [seq_pad]
    same_seg = (seg_q[:, None] == seg_k[None, :])
    
    # Live (non-padding)
    live_q = (seg_q != 0)[:, None]
    live_k = (seg_k != 0)[None, :]
    live = live_q & live_k
    
    mask = causal & same_seg & live
    
    # Apply mask
    logits_masked = jnp.where(mask, logits, NEG_INF)
    
    # Row max
    m = jnp.max(logits_masked, axis=-1, keepdims=True)
    
    # Sinks
    if sinks_val is not None:
        sk = float(sinks_val)
        m = jnp.maximum(m, sk)
    
    # Exponentials
    exp_logits = jnp.exp(logits_masked - m)
    
    # Denominator
    denom = jnp.sum(exp_logits, axis=-1, keepdims=True)
    if sinks_val is not None:
        sk = float(sinks_val)
        denom = denom + jnp.exp(sk - m)
    
    # Attention Probabilities
    row_live = mask.any(axis=-1)[:, None]
    p = jnp.where(row_live, exp_logits / jnp.maximum(denom, 1e-30), 0.0)
    
    # Output
    out_block = jnp.matmul(p, v_full, preferred_element_type=jnp.float32)
    o_ref[0, :, :] = out_block


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

def _bwd_dq_body(q_ref, k_ref, v_ref, seg_ref, do_ref, dq_ref, *, seq, block_q, window, soft_cap):
    """
    Computes gradient w.r.t Q.
    Same grid as forward.
    """
    q_block = q_ref[0].astype(jnp.float32)
    k_full = k_ref[0].astype(jnp.float32)
    v_full = v_ref[0].astype(jnp.float32)
    do_block = do_ref[0].astype(jnp.float32)
    seg_vals = seg_ref[...]
    
    q_head_idx = pl.program_id(0)
    q_block_idx = pl.program_id(1)
    q_idx = jnp.arange(block_q, dtype=jnp.int32) + (q_block_idx * block_q)
    k_idx = jnp.arange(seg_vals.shape[0], dtype=jnp.int32)
    
    logits = jnp.matmul(q_block, k_full.T, preferred_element_type=jnp.float32)
    if soft_cap is not None:
        logits = soft_cap * jnp.tanh(logits / soft_cap)
    
    causal = k_idx[None, :] <= q_idx[:, None]
    if window is not None:
        causal = causal & ((q_idx[:, None] - k_idx[None, :]) < window)
    
    seg_q = seg_vals[q_idx]
    seg_k = seg_vals
    same_seg = (seg_q[:, None] == seg_k[None, :])
    live = (seg_q[:, None] != 0) & (seg_k[None, :] != 0)
    mask = causal & same_seg & live
    
    logits_masked = jnp.where(mask, logits, NEG_INF)
    m = jnp.max(logits_masked, axis=-1, keepdims=True)
    
    exp_logits = jnp.exp(logits_masked - m)
    denom = jnp.sum(exp_logits, axis=-1, keepdims=True)
    row_live = mask.any(axis=-1)[:, None]
    p = jnp.where(row_live, exp_logits / jnp.maximum(denom, 1e-30), 0.0)
    
    # G = do @ v^T
    G = jnp.matmul(do_block, v_full.T, preferred_element_type=jnp.float32) # [block_q, seq_pad]
    
    # d_logits
    sum_pG = jnp.sum(p * G, axis=-1, keepdims=True)
    d_logits = p * (G - sum_pG)
    
    # dQ
    dq_block = jnp.matmul(d_logits, k_full, preferred_element_type=jnp.float32)
    dq_ref[0, :, :] = dq_block


def _bwd_dkv_body(q_ref, k_ref, v_ref, seg_ref, do_ref, dk_ref, dv_ref, *, seq, window, soft_cap):
    """
    Gradient w.r.t K, V. Iterates over all query heads/blocks for this KV head.
    Grid: (kv_heads,)
    Inputs contain all query heads for this KV head.
    """
    k_full = k_ref[0].astype(jnp.float32)  # [seq_pad, d]
    v_full = v_ref[0].astype(jnp.float32)  # [seq_pad, dv]
    seg_vals = seg_ref[...]                # [seq_pad]
    
    q_heads = q_ref.shape[1]               # group
    seq_pad = seg_vals.shape[0]
    num_blocks = (seq_pad + BLOCK_Q - 1) // BLOCK_Q
    
    # Initialize accumulators
    dk_acc = jnp.zeros((seq_pad, k_full.shape[1]), dtype=jnp.float32)
    dv_acc = jnp.zeros((seq_pad, v_full.shape[1]), dtype=jnp.float32)
    
    # Helper for block loops
    def scan_fn(acc, state):
        dk_curr, dv_curr = acc
        h_idx = state
        h = h_idx // num_blocks
        b = h_idx % num_blocks
        
        # Slice q and do for this head and block
        # q_ref: [1, group, seq_pad, d]
        start = b * BLOCK_Q
        end = min(start + BLOCK_Q, seq_pad)
        
        # Extract slices
        # Note: jnp.take or slicing
        q_tile = jnp.take(q_ref[0, h, :, :], jnp.arange(start, end), axis=0) # [block_q, d]
        do_tile = jnp.take(do_ref[0, h, :, :], jnp.arange(start, end), axis=0) # [block_q, dv]
        
        # Current sequence indices for the block
        q_idx = jnp.arange(start, end, dtype=jnp.int32)
        k_idx = jnp.arange(seq_pad, dtype=jnp.int32)
        
        # Logits
        logits = jnp.matmul(q_tile, k_full.T, preferred_element_type=jnp.float32)
        if soft_cap is not None:
            logits = soft_cap * jnp.tanh(logits / soft_cap)
            
        causal = k_idx[None, :] <= q_idx[:, None]
        if window is not None:
            causal = causal & ((q_idx[:, None] - k_idx[None, :]) < window)
            
        seg_q = seg_vals[q_idx]
        seg_k = seg_vals
        same_seg = (seg_q[:, None] == seg_k[None, :])
        live = (seg_q[:, None] != 0) & (seg_k[None, :] != 0)
        mask = causal & same_seg & live
        
        logits_masked = jnp.where(mask, logits, NEG_INF)
        m = jnp.max(logits_masked, axis=-1, keepdims=True)
        
        exp_logits = jnp.exp(logits_masked - m)
        denom = jnp.sum(exp_logits, axis=-1, keepdims=True)
        row_live = mask.any(axis=-1)[:, None]
        p = jnp.where(row_live, exp_logits / jnp.maximum(denom, 1e-30), 0.0) # [block_q, seq_pad]
        
        # dV update: p^T @ do_tile
        # p.T is [seq_pad, block_q]
        dv_update = jnp.matmul(p.T, do_tile, preferred_element_type=jnp.float32)
        
        # dK update:
        # d_logits = p * (G - sum(p*G))
        # G = do @ v^T
        G = jnp.matmul(do_tile, v_full.T, preferred_element_type=jnp.float32)
        sum_pG = jnp.sum(p * G, axis=-1, keepdims=True)
        d_logits = p * (G - sum_pG)
        
        dk_update = jnp.matmul(d_logits.T, q_tile, preferred_element_type=jnp.float32)
        
        return ((dk_curr + dk_update, dv_curr + dv_update), None)
    
    # Run scan
    (dk_final, dv_final), _ = jax.lax.scan(scan_fn, (dk_acc, dv_acc), jnp.arange(q_heads * num_blocks))
    
    dk_ref[0, :, :] = dk_final
    dv_ref[0, :, :] = dv_final


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
