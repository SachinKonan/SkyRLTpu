"""Splash attention (causal, segment-gated, GQA) -- flash kernel, the
production structure stripped to basics.

GRADED ENTRYPOINT: kernel(q, k, v, segment_ids, *, window=None,
soft_cap=None, sinks=None) -> o, with q: [q_heads, seq, d] bf16
(pre-scaled), k/v: [kv_heads, seq, d|d_v] bf16 (q_heads = kv_heads * group),
segment_ids: [seq] int32 where 0 marks padding, o: [q_heads, seq, d_v] f32
with padding (segment-0) query rows exactly zero. The judge differentiates
through kernel() (d/dq, d/dk, d/dv) via the custom_vjp below.

STRUCTURE (the same shape as the production splash kernel):
  * Grid (q_head, q_block, kv_block) with the KV axis LAST: the last grid
    dimension iterates sequentially on TPU and VMEM scratch persists across
    its steps -- that is what makes the ONLINE SOFTMAX carry (running max,
    running sum, output accumulator) work. Only one [B, B] logits tile ever
    exists; materialising full [block_q, seq] logits at the declared
    s=18432 shapes would need ~37 MB against ~32 MB of VMEM.
  * EVERY input arrives pre-sliced by a BlockSpec, including the segment
    ids (a q-block ref and a kv-block ref). Nothing indexes a loaded array
    at a traced index -- that is a dynamic gather/slice and Mosaic rejects
    it (E2003 alignment); it killed every previously-"correct" kernel.
  * Masks come from 2-D broadcasted_iota + program_id arithmetic (1-D iota
    does not lower on TPU), never from gathers.
  * The backward saves per-row LOGSUMEXP as a residual and recomputes
    tiles (the production bwd's approach). dK/dV walk (group-head, q-block)
    pairs on the sequential axis, accumulating in scratch; the GQA head
    mapping happens in BlockSpec index maps, not by indexing inside.

This is ONE valid approach; faster kernels may restructure everything
(block sizes, SKIPPING fully-masked KV tiles -- the production kernel does,
a large part of its speed -- a fused backward, different residuals) as long
as forward AND backward stay correct.
"""

import functools
import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

INTERPRET = bool(int(os.environ.get("PALLAS_INTERPRET", "0")))
B = 512
NEG_INF = -1e30


def _tile_mask(qi, ki, qseg_ref, kseg_ref, window):
    """Causal + same-segment + both-live mask for one [B, B] tile. qi/ki are
    BLOCK indices (program_id scalars are fine in arithmetic); positions come
    from 2-D iota, segments from the pre-sliced refs."""
    qpos = qi * B + jax.lax.broadcasted_iota(jnp.int32, (B, 1), 0)
    kpos = ki * B + jax.lax.broadcasted_iota(jnp.int32, (1, B), 1)
    m = kpos <= qpos
    if window is not None:
        m = m & ((qpos - kpos) < window)
    qs = qseg_ref[...].reshape(B, 1)
    ks = kseg_ref[...].reshape(1, B)
    return m & (qs == ks) & (qs != 0) & (ks != 0)


def _tile_logits(q_ref, k_ref, soft_cap):
    z = jnp.dot(q_ref[0].astype(jnp.float32), k_ref[0].astype(jnp.float32).T,
                preferred_element_type=jnp.float32)
    if soft_cap is not None:
        z = soft_cap * jnp.tanh(z / soft_cap)
    return z


def _fwd_body(q_ref, k_ref, v_ref, qseg_ref, kseg_ref, o_ref, lse_ref,
              m_ref, l_ref, acc_ref, *, nkv, window, soft_cap):
    ki = pl.program_id(2)

    @pl.when(ki == 0)
    def _init():
        m_ref[...] = jnp.full_like(m_ref, NEG_INF)
        l_ref[...] = jnp.zeros_like(l_ref)
        acc_ref[...] = jnp.zeros_like(acc_ref)

    z = _tile_logits(q_ref, k_ref, soft_cap)
    mask = _tile_mask(pl.program_id(1), ki, qseg_ref, kseg_ref, window)
    z = jnp.where(mask, z, NEG_INF)

    m_prev = m_ref[...]
    m_new = jnp.maximum(m_prev, jnp.max(z, axis=1, keepdims=True))
    alpha = jnp.exp(m_prev - m_new)
    p = jnp.where(mask, jnp.exp(z - m_new), 0.0)
    l_ref[...] = l_ref[...] * alpha + jnp.sum(p, axis=1, keepdims=True)
    acc_ref[...] = acc_ref[...] * alpha + jnp.dot(
        p, v_ref[0].astype(jnp.float32), preferred_element_type=jnp.float32)
    m_ref[...] = m_new

    @pl.when(ki == nkv - 1)
    def _flush():
        l = l_ref[...]
        live = l > 0.0
        safe = jnp.where(live, l, 1.0)
        o_ref[0] = jnp.where(live, acc_ref[...] / safe, 0.0)
        lse_ref[0] = jnp.where(live, m_ref[...] + jnp.log(safe), NEG_INF)[:, 0]


def _dq_body(q_ref, k_ref, v_ref, qseg_ref, kseg_ref, do_ref, lse_ref,
             dcoef_ref, dq_ref, acc_ref, *, nkv, window, soft_cap):
    ki = pl.program_id(2)

    @pl.when(ki == 0)
    def _init():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    zc = _tile_logits(q_ref, k_ref, soft_cap)
    mask = _tile_mask(pl.program_id(1), ki, qseg_ref, kseg_ref, window)
    lse = lse_ref[0].reshape(B, 1)
    p = jnp.where(mask, jnp.exp(jnp.where(mask, zc, NEG_INF) - lse), 0.0)
    dp = jnp.dot(do_ref[0].astype(jnp.float32), v_ref[0].astype(jnp.float32).T,
                 preferred_element_type=jnp.float32)
    ds = p * (dp - dcoef_ref[0].reshape(B, 1))
    if soft_cap is not None:
        ds = ds * (1.0 - (zc / soft_cap) ** 2)     # d(capped)/d(pre-cap)
    acc_ref[...] += jnp.dot(ds, k_ref[0].astype(jnp.float32),
                            preferred_element_type=jnp.float32)

    @pl.when(ki == nkv - 1)
    def _flush():
        dq_ref[0] = acc_ref[...]


def _dkv_body(q_ref, k_ref, v_ref, qseg_ref, kseg_ref, do_ref, lse_ref,
              dcoef_ref, dk_ref, dv_ref, dk_acc, dv_acc, *, nq, group,
              window, soft_cap):
    si = pl.program_id(2)          # walks group*nq (head, q-block) pairs

    @pl.when(si == 0)
    def _init():
        dk_acc[...] = jnp.zeros_like(dk_acc)
        dv_acc[...] = jnp.zeros_like(dv_acc)

    qi = si % nq                   # q-block index (the head arrives via BlockSpec)
    zc = _tile_logits(q_ref, k_ref, soft_cap)
    mask = _tile_mask(qi, pl.program_id(1), qseg_ref, kseg_ref, window)
    lse = lse_ref[0].reshape(B, 1)
    p = jnp.where(mask, jnp.exp(jnp.where(mask, zc, NEG_INF) - lse), 0.0)
    do = do_ref[0].astype(jnp.float32)
    dv_acc[...] += jnp.dot(p.T, do, preferred_element_type=jnp.float32)
    dp = jnp.dot(do, v_ref[0].astype(jnp.float32).T,
                 preferred_element_type=jnp.float32)
    ds = p * (dp - dcoef_ref[0].reshape(B, 1))
    if soft_cap is not None:
        ds = ds * (1.0 - (zc / soft_cap) ** 2)
    dk_acc[...] += jnp.dot(ds.T, q_ref[0].astype(jnp.float32),
                           preferred_element_type=jnp.float32)

    @pl.when(si == group * nq - 1)
    def _flush():
        dk_ref[0] = dk_acc[...]
        dv_ref[0] = dv_acc[...]


def _pad_seq(a, s_pad, axis=1, fill=0):
    pad = s_pad - a.shape[axis]
    if pad == 0:
        return a
    widths = [(0, 0)] * a.ndim
    widths[axis] = (0, pad)
    return jnp.pad(a, widths, constant_values=fill)


def _pad_lane(a):
    """Pad the feature dim to a 128 multiple (d=192 exists in the graded
    set; zero columns change neither the q.k dots nor sliced-off outputs)."""
    pad = (-a.shape[-1]) % 128
    if pad == 0:
        return a
    return jnp.pad(a, [(0, 0)] * (a.ndim - 1) + [(0, pad)])


def _forward(q, k, v, seg, window, soft_cap):
    qh = q.shape[0]
    kvh = k.shape[0]
    group = qh // kvh
    s_pad = -(-q.shape[1] // B) * B
    qp = _pad_lane(_pad_seq(q, s_pad))
    kp = _pad_lane(_pad_seq(k, s_pad))
    vp = _pad_lane(_pad_seq(v, s_pad))
    segp = _pad_seq(seg, s_pad, axis=0)
    dp_, dvp = qp.shape[-1], vp.shape[-1]
    nq = s_pad // B

    o, lse = pl.pallas_call(
        functools.partial(_fwd_body, nkv=nq, window=window, soft_cap=soft_cap),
        grid=(qh, nq, nq),
        in_specs=[
            pl.BlockSpec((1, B, dp_), lambda h, i, j: (h, i, 0)),
            pl.BlockSpec((1, B, dp_), lambda h, i, j: (h // group, j, 0)),
            pl.BlockSpec((1, B, dvp), lambda h, i, j: (h // group, j, 0)),
            pl.BlockSpec((B,), lambda h, i, j: (i,)),
            pl.BlockSpec((B,), lambda h, i, j: (j,)),
        ],
        out_specs=[
            pl.BlockSpec((1, B, dvp), lambda h, i, j: (h, i, 0)),
            pl.BlockSpec((1, B), lambda h, i, j: (h, i)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((qh, s_pad, dvp), jnp.float32),
            jax.ShapeDtypeStruct((qh, s_pad), jnp.float32),
        ],
        scratch_shapes=[
            pltpu.VMEM((B, 1), jnp.float32),
            pltpu.VMEM((B, 1), jnp.float32),
            pltpu.VMEM((B, dvp), jnp.float32),
        ],
        interpret=INTERPRET,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
    )(qp, kp, vp, segp, segp)
    return o, lse, (qp, kp, vp, segp, s_pad)


def _backward(res, lse_pad, dcoef_pad, g, dims, window, soft_cap):
    qp, kp, vp, segp, s_pad = res
    qh, seq, d, dv, kvh = dims
    group = qh // kvh
    dp_, dvp = qp.shape[-1], vp.shape[-1]
    nq = s_pad // B
    gp = _pad_lane(_pad_seq(g.astype(jnp.float32), s_pad))

    dq = pl.pallas_call(
        functools.partial(_dq_body, nkv=nq, window=window, soft_cap=soft_cap),
        grid=(qh, nq, nq),
        in_specs=[
            pl.BlockSpec((1, B, dp_), lambda h, i, j: (h, i, 0)),
            pl.BlockSpec((1, B, dp_), lambda h, i, j: (h // group, j, 0)),
            pl.BlockSpec((1, B, dvp), lambda h, i, j: (h // group, j, 0)),
            pl.BlockSpec((B,), lambda h, i, j: (i,)),
            pl.BlockSpec((B,), lambda h, i, j: (j,)),
            pl.BlockSpec((1, B, dvp), lambda h, i, j: (h, i, 0)),
            pl.BlockSpec((1, B), lambda h, i, j: (h, i)),
            pl.BlockSpec((1, B), lambda h, i, j: (h, i)),
        ],
        out_specs=pl.BlockSpec((1, B, dp_), lambda h, i, j: (h, i, 0)),
        out_shape=jax.ShapeDtypeStruct((qh, s_pad, dp_), jnp.float32),
        scratch_shapes=[pltpu.VMEM((B, dp_), jnp.float32)],
        interpret=INTERPRET,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
    )(qp, kp, vp, segp, segp, gp, lse_pad, dcoef_pad)

    # GQA head mapping lives in the INDEX MAPS: q-head = h*group + s//nq.
    dk, dv_out = pl.pallas_call(
        functools.partial(_dkv_body, nq=nq, group=group,
                          window=window, soft_cap=soft_cap),
        grid=(kvh, nq, group * nq),
        in_specs=[
            pl.BlockSpec((1, B, dp_), lambda h, j, s: (h * group + s // nq, s % nq, 0)),
            pl.BlockSpec((1, B, dp_), lambda h, j, s: (h, j, 0)),
            pl.BlockSpec((1, B, dvp), lambda h, j, s: (h, j, 0)),
            pl.BlockSpec((B,), lambda h, j, s: (s % nq,)),
            pl.BlockSpec((B,), lambda h, j, s: (j,)),
            pl.BlockSpec((1, B, dvp), lambda h, j, s: (h * group + s // nq, s % nq, 0)),
            pl.BlockSpec((1, B), lambda h, j, s: (h * group + s // nq, s % nq)),
            pl.BlockSpec((1, B), lambda h, j, s: (h * group + s // nq, s % nq)),
        ],
        out_specs=[
            pl.BlockSpec((1, B, dp_), lambda h, j, s: (h, j, 0)),
            pl.BlockSpec((1, B, dvp), lambda h, j, s: (h, j, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((kvh, s_pad, dp_), jnp.float32),
            jax.ShapeDtypeStruct((kvh, s_pad, dvp), jnp.float32),
        ],
        scratch_shapes=[
            pltpu.VMEM((B, dp_), jnp.float32),
            pltpu.VMEM((B, dvp), jnp.float32),
        ],
        interpret=INTERPRET,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
    )(qp, kp, vp, segp, segp, gp, lse_pad, dcoef_pad)
    return dq, dk, dv_out


def kernel(q, k, v, segment_ids, *, window=None, soft_cap=None, sinks=None):
    """See module docstring: graded entrypoint, differentiated by the judge."""
    if sinks is not None:
        raise NotImplementedError("sinks not supported by this seed")
    qh, seq, d = q.shape
    kvh, dv = k.shape[0], v.shape[2]
    dims = (qh, seq, d, dv, kvh)

    @jax.custom_vjp
    def attn(q, k, v, seg):
        o, _lse, _res = _forward(q, k, v, seg, window, soft_cap)
        return o[:, :seq, :dv]

    def fwd_rule(q, k, v, seg):
        o, lse, res = _forward(q, k, v, seg, window, soft_cap)
        return o[:, :seq, :dv], (res, lse, o)

    def bwd_rule(saved, g):
        res, lse_pad, o_pad = saved
        s_pad = res[-1]
        gp = jnp.pad(g.astype(jnp.float32),
                     ((0, 0), (0, s_pad - seq), (0, o_pad.shape[-1] - g.shape[-1])))
        dcoef_pad = jnp.sum(gp * o_pad, axis=-1)      # [qh, s_pad]
        dq, dk, dv_ = _backward(res, lse_pad, dcoef_pad, g, dims,
                                window, soft_cap)
        return (dq[:, :seq, :d].astype(q.dtype),
                dk[:, :seq, :d].astype(k.dtype),
                dv_[:, :seq, :dv].astype(v.dtype),
                None)

    attn.defvjp(fwd_rule, bwd_rule)
    return attn(q, k, v, segment_ids)
