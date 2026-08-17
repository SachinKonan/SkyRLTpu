"""Task 1 — Splash attention @ [4, 18432] causal (train flagship).

Baseline: Google's production Pallas splash-attention kernel
(jax.experimental.pallas.ops.tpu.splash_attention) at our real fb shard
shapes [4, 18432, heads, 128] PLUS one deliberately non-block-divisible
length (the case splash itself rejects — bit us at TP=4). Candidates are
banned from importing any jax pallas ops (they write their own kernel) and
from jax.nn attention entry points.

kernel(q, k, v, segment_ids) -> o
  q, k, v:     [heads, seq, head_dim] bfloat16  (per-shard layout, MHA)
  segment_ids: [seq] int32; 0 = padding. Attention is causal AND restricted
               to equal segment ids; queries in padding (segment 0) must
               produce EXACTLY 0 (not NaN).
  o:           [heads, seq, head_dim] float32
Softmax at fp32. Contract: fwd only (bwd is phase 2 of this task).
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

NEG_INF = -0.7 * float(np.finfo(np.float32).max)


def causal_segment_attention(
    q, k, v, segment_ids, *, window=None, soft_cap=None, sinks=None
):
    """fp32 masked-softmax closed form; fully-masked rows -> exactly 0.

    ``window`` (int or None): SLIDING-WINDOW attention. A query at position i
    may attend to keys in (i - window, i] -- i.e. the `window` most recent
    positions including itself, which is the convention splash's
    ``LocalMask(window_size=(window - 1, 0))`` implements for a causal left
    window. Mistral/Ministral/Gemma-2 all ship this.

    ``soft_cap`` (float or None): LOGIT SOFT-CAP, ``cap * tanh(logits / cap)``,
    applied BEFORE masking and the softmax max-shift. Gemma-2/3 use it, and
    splash takes it as ``attn_logits_soft_cap``. Order matters: capping after
    the mask would squash the -inf sentinel into +-cap and silently unmask
    every forbidden position.

    ``sinks`` (tuple of one float per q-head, or None): ATTENTION SINKS,
    gpt-oss / streaming-LLM style, exactly tokamax's reference semantics
    (experimental splash base.py): the sink joins the softmax max
    (``m = max(row_max, sink)``) and adds ``exp(sink - m)`` to the
    denominator, but contributes NO value row -- it absorbs probability mass.
    Static per-case values (a learned parameter at train time; fixed for
    grading, which changes neither the algorithm a kernel must implement nor
    its cost). A fully-masked row keeps output exactly 0: every p_i is 0
    regardless of the denominator.

    GENERALIZED to the attention shapes real models actually use, because
    MHA-with-equal-head-dims was a contract gap: tokamax benchmarks mixtral
    8x7b as GQA (32 query heads over 8 KV heads) and deepseek2-16b with a
    VALUE head dim that differs from q/k (192 vs 128). A kernel that only
    handles square MHA is not a replacement for either.

      q: [q_heads, seq, d]      k: [kv_heads, seq, d]     v: [kv_heads, seq, d_v]
      out: [q_heads, seq, d_v]

    MHA is exactly the group == 1 case and d_v == d, so every existing shape
    keeps its meaning and its numbers.
    """
    q32 = q.astype(jnp.float32)
    k32 = k.astype(jnp.float32)
    v32 = v.astype(jnp.float32)
    qh, seq, d = q32.shape
    kvh = k32.shape[0]
    dv = v32.shape[-1]
    group = qh // kvh
    qg = q32.reshape(kvh, group, seq, d)

    logits = jnp.einsum("hgqd,hkd->hgqk", qg, k32)
    if soft_cap is not None:
        # BEFORE masking: see the docstring -- capping afterwards would pull
        # the -inf sentinel back into [-cap, cap] and unmask everything.
        logits = soft_cap * jnp.tanh(logits / soft_cap)
    idx = jnp.arange(seq)
    causal = idx[:, None] >= idx[None, :]
    if window is not None:
        causal = causal & (idx[:, None] - idx[None, :] < window)
    same_seg = segment_ids[:, None] == segment_ids[None, :]
    live = segment_ids != 0
    mask = causal & same_seg & live[:, None] & live[None, :]
    mask4 = mask[None, None, :, :]
    logits = jnp.where(mask4, logits, NEG_INF)
    row_live = mask.any(axis=-1)  # [q] rows with at least one visible key
    # max-shifted softmax; fully-masked rows produce 0, never NaN
    m = jnp.max(logits, axis=-1, keepdims=True)
    if sinks is not None:
        # [q_heads] -> [kvh, group, 1, 1], joining the max-shift per q-head
        sk = jnp.asarray(sinks, jnp.float32).reshape(kvh, group)[:, :, None, None]
        m = jnp.maximum(m, sk)
    p = jnp.exp(logits - m)
    p = jnp.where(mask4, p, 0.0)
    denom = jnp.sum(p, axis=-1, keepdims=True)
    if sinks is not None:
        denom = denom + jnp.exp(sk - m)
    p = jnp.where(row_live[None, None, :, None], p / jnp.maximum(denom, 1e-30), 0.0)
    out = jnp.einsum("hgqk,hkv->hgqv", p, v32)
    return out.reshape(qh, seq, dv)


def _match_kv(q, k, v):
    """Repeat KV heads up to the query head count for implementations written
    against square MHA.

    Deliberately a REPEAT and not a regrouping: it is what a naive caller does,
    it multiplies KV traffic by the group size, and it is therefore an honest
    representation of "this implementation does not natively do GQA". A native
    grouped XLA denominator is a follow-up; claiming one before it is written
    would be the kind of unverified assertion this arena keeps getting caught
    by. Also rejects the deepseek2-style asymmetric value head dim, which no
    square-MHA formulation can express.
    """
    if v.shape[-1] != q.shape[-1]:
        raise ValueError("d_v != d is not expressible in a square-MHA implementation")
    if k.shape[0] != q.shape[0]:
        rep = q.shape[0] // k.shape[0]
        k = jnp.repeat(k, rep, axis=0)
        v = jnp.repeat(v, rep, axis=0)
    return k, v


def _splash_block_sizes(sak, seq: int):
    """Block sizes for the splash baseline. LOAD-BEARING -- see below.

    `make_splash_mha(..., block_sizes=None)` falls back to
    `BlockSizes.get_default()`, which is all-128 and carries JAX's own
    admission that they are placeholders:

        # TODO(apaszke,sharadmv): Select better parameters based on a heuristic.

    This arena passed no `block_sizes` at all. Measured on v6e-1 (job 3691558)
    across tokamax's canonical attention sweep and our own shapes, the default
    makes splash 3.3x to 6.0x SLOWER THAN XLA at every shape, while 1024x1024
    makes it FASTER than XLA at every shape (0.25x-0.88x). Tuning is worth
    3.5x-14.7x; at our own probe-h8-s4096 it is 9.9x.

    So every splash verdict this arena has recorded -- including the 3 passes in
    sd-results-3687904 -- was scored against a bar roughly 10x too slow.

    1024x1024 won UNANIMOUSLY across all 12 measured shapes (unlike megablox,
    whose best tiling varied), so a fixed choice is right here; it is only
    clamped down when the sequence cannot hold it. A shape splash then refuses
    still falls back, and the fallback is still recorded.
    """
    b = min(1024, max(128, 1 << (int(seq).bit_length() - 1)))
    while b > 128 and seq % b:
        b //= 2
    return sak.BlockSizes(
        block_q=b, block_kv=b, block_kv_compute=b,
        block_q_dkv=b, block_kv_dkv=b, block_kv_dkv_compute=b,
        block_q_dq=b, block_kv_dq=b,
    )


def _honest_faithful_bf16(
    q, k, v, segment_ids, block_q: int = 512, *, window=None, soft_cap=None, sinks=None
):
    """The production numeric path, query-blocked: bf16 arrays enter the MXU as
    matmul INPUTS, every accumulator is fp32 (``preferred_element_type``), the
    softmax is fp32, and only the probabilities are re-narrowed to bf16 on the
    way into the PV matmul. This is what Google's splash kernel, vLLM-TPU and
    tokamax all do, so it defines the error an HONEST bf16 candidate incurs --
    strictly more than ``reference_bf16`` (which does every intermediate at
    fp32 and rounds only the output). Measured on the CPU battery it lands at
    0.4-1.15x of the reference-bf16-only band, i.e. it FAILS that band on some
    shapes; that is the phase-2 lesson, and why it belongs in the calibration.
    """
    h, s, d = q.shape
    k, v = _match_kv(q, k, v)
    pad = (-s) % block_q
    qp = jnp.pad(q, ((0, 0), (0, pad), (0, 0))) if pad else q
    idx_k = jnp.arange(s)
    seg_k = segment_ids
    live_k = seg_k != 0

    def block(_carry, i):
        start = i * block_q
        qb = jax.lax.dynamic_slice(qp, (0, start, 0), (h, block_q, d))
        pos_q = start + jnp.arange(block_q)
        seg_q = jnp.where(pos_q < s, seg_k[jnp.minimum(pos_q, s - 1)], 0)
        live_q = (seg_q != 0) & (pos_q < s)
        # bf16 inputs -> fp32 accumulator (the whole point)
        logits = jnp.einsum("hqd,hkd->hqk", qb, k, preferred_element_type=jnp.float32)
        if soft_cap is not None:
            logits = soft_cap * jnp.tanh(logits / soft_cap)
        causal = pos_q[:, None] >= idx_k[None, :]
        if window is not None:
            causal = causal & (pos_q[:, None] - idx_k[None, :] < window)
        m = causal & (seg_q[:, None] == seg_k[None, :]) & live_q[:, None] & live_k[None, :]
        logits = jnp.where(m[None], logits, NEG_INF)
        row_live = m.any(axis=-1)
        mx = jnp.max(logits, axis=-1, keepdims=True)
        if sinks is not None:
            mx = jnp.maximum(mx, jnp.asarray(sinks, jnp.float32)[:, None, None])
        p = jnp.where(m[None], jnp.exp(logits - mx), 0.0)
        denom = jnp.sum(p, axis=-1, keepdims=True)
        if sinks is not None:
            denom = denom + jnp.exp(jnp.asarray(sinks, jnp.float32)[:, None, None] - mx)
        p = jnp.where(row_live[None, :, None], p / jnp.maximum(denom, 1e-30), 0.0)
        out = jnp.einsum("hqk,hkd->hqd", p.astype(jnp.bfloat16), v, preferred_element_type=jnp.float32)
        return None, out

    _, blocks = jax.lax.scan(block, None, jnp.arange((s + pad) // block_q))
    out = jnp.transpose(blocks, (1, 0, 2, 3)).reshape(h, s + pad, d)
    return out[:, :s, :]


def _honest_online_softmax(
    q, k, v, segment_ids, block_k: int = 256, *, window=None, soft_cap=None, sinks=None
):
    """Flash-style streaming softmax: a DIFFERENT reduction order over the key
    axis (running max + rescaled running sum) at fp32. Legal, and the shape a
    real Pallas kernel takes; included so a candidate is never punished for
    accumulating keys in blocks rather than all at once."""
    h, s, d = q.shape
    k, v = _match_kv(q, k, v)
    idx_k = jnp.arange(s)
    live_k = segment_ids != 0
    pos_q = jnp.arange(s)
    live_q = segment_ids != 0
    nb = -(-s // block_k)
    pad = nb * block_k - s
    kp = jnp.pad(k, ((0, 0), (0, pad), (0, 0))) if pad else k
    vp = jnp.pad(v, ((0, 0), (0, pad), (0, 0))) if pad else v

    def body(carry, j):
        run_m, run_l, acc = carry
        start = j * block_k
        kb = jax.lax.dynamic_slice(kp, (0, start, 0), (h, block_k, d))
        vb = jax.lax.dynamic_slice(vp, (0, start, 0), (h, block_k, d))
        kpos = start + jnp.arange(block_k)
        in_range = kpos < s
        seg_kb = jnp.where(in_range, segment_ids[jnp.minimum(kpos, s - 1)], 0)
        live_kb = (seg_kb != 0) & in_range
        logits = jnp.einsum("hqd,hkd->hqk", q, kb, preferred_element_type=jnp.float32)
        if soft_cap is not None:
            logits = soft_cap * jnp.tanh(logits / soft_cap)
        causal = pos_q[:, None] >= kpos[None, :]
        if window is not None:
            causal = causal & (pos_q[:, None] - kpos[None, :] < window)
        m = (
            causal
            & (segment_ids[:, None] == seg_kb[None, :])
            & live_q[:, None]
            & live_kb[None, :]
        )
        logits = jnp.where(m[None], logits, NEG_INF)
        blk_m = jnp.max(logits, axis=-1, keepdims=True)
        new_m = jnp.maximum(run_m, blk_m)
        corr = jnp.exp(run_m - new_m)
        p = jnp.where(m[None], jnp.exp(logits - new_m), 0.0)
        new_l = run_l * corr + jnp.sum(p, axis=-1, keepdims=True)
        new_acc = acc * corr + jnp.einsum("hqk,hkd->hqd", p.astype(jnp.bfloat16), vb, preferred_element_type=jnp.float32)
        return (new_m, new_l, new_acc), None

    init = (
        jnp.full((h, s, 1), NEG_INF, jnp.float32),
        jnp.zeros((h, s, 1), jnp.float32),
        jnp.zeros((h, s, d), jnp.float32),
    )
    (run_m, run_l, acc), _ = jax.lax.scan(body, init, jnp.arange(nb))
    if sinks is not None:
        # The sink merges AFTER the streaming pass, exactly like folding in one
        # more block containing a single logit and no value: rescale the
        # running sum/accumulator to the new max and add the sink's mass.
        sk = jnp.asarray(sinks, jnp.float32)[:, None, None]
        new_m = jnp.maximum(run_m, sk)
        corr = jnp.exp(run_m - new_m)
        run_l = run_l * corr + jnp.exp(sk - new_m)
        acc = acc * corr
    row_live = ((idx_k[None, :] <= pos_q[:, None]) & (segment_ids[:, None] == segment_ids[None, :]) & live_q[:, None] & live_k[None, :]).any(-1)
    return jnp.where(row_live[None, :, None], acc / jnp.maximum(run_l, 1e-30), 0.0)


_FALLBACK_BLOCK_Q = 512


def _xla_masked_attention(
    q, k, v, segment_ids, block_q: int = _FALLBACK_BLOCK_Q, *, window=None, soft_cap=None, sinks=None
):
    """Query-blocked XLA attention: same math as ``causal_segment_attention``
    but never materializes more than [heads, block_q, seq] of logits, so it
    runs at shapes where the closed form would not fit.

    Serves GQA by REPEATING KV (see _match_kv), exactly as the splash baseline
    does, and cannot express d_v != d at all. ``_xla_grouped_attention`` is the
    native-grouped counterpart; both are registered as baseline candidates so
    the per-shape election picks whichever is actually faster.
    """
    h, s, d = q.shape
    k, v = _match_kv(q, k, v)
    pad = (-s) % block_q
    qp = jnp.pad(q, ((0, 0), (0, pad), (0, 0))) if pad else q
    idx_k = jnp.arange(s)
    seg_k = segment_ids
    live_k = seg_k != 0
    k32 = k.astype(jnp.float32)
    v32 = v.astype(jnp.float32)

    def block(_carry, i):
        start = i * block_q
        qb = jax.lax.dynamic_slice(qp, (0, start, 0), (h, block_q, d)).astype(jnp.float32)
        pos_q = start + jnp.arange(block_q)
        seg_q = jnp.where(pos_q < s, seg_k[jnp.minimum(pos_q, s - 1)], 0)
        live_q = (seg_q != 0) & (pos_q < s)
        logits = jnp.einsum("hqd,hkd->hqk", qb, k32)
        if soft_cap is not None:
            logits = soft_cap * jnp.tanh(logits / soft_cap)
        causal = pos_q[:, None] >= idx_k[None, :]
        if window is not None:
            causal = causal & (pos_q[:, None] - idx_k[None, :] < window)
        m = causal & (seg_q[:, None] == seg_k[None, :]) & live_q[:, None] & live_k[None, :]
        logits = jnp.where(m[None], logits, NEG_INF)
        row_live = m.any(axis=-1)
        mx = jnp.max(logits, axis=-1, keepdims=True)
        if sinks is not None:
            mx = jnp.maximum(mx, jnp.asarray(sinks, jnp.float32)[:, None, None])
        p = jnp.where(m[None], jnp.exp(logits - mx), 0.0)
        denom = jnp.sum(p, axis=-1, keepdims=True)
        if sinks is not None:
            denom = denom + jnp.exp(jnp.asarray(sinks, jnp.float32)[:, None, None] - mx)
        p = jnp.where(row_live[None, :, None], p / jnp.maximum(denom, 1e-30), 0.0)
        return None, jnp.einsum("hqk,hkd->hqd", p, v32)

    _, blocks = jax.lax.scan(block, None, jnp.arange((s + pad) // block_q))
    out = jnp.transpose(blocks, (1, 0, 2, 3)).reshape(h, s + pad, d)
    return out[:, :s, :]


def _xla_grouped_attention(
    q, k, v, segment_ids, block_q: int = _FALLBACK_BLOCK_Q, *, window=None, soft_cap=None, sinks=None
):
    """Query-blocked XLA attention that is NATIVELY GROUPED: KV is read once per
    KV head and shared across its query group, and the value head dim is free to
    differ from the q/k head dim.

    This exists because without it the GQA and deepseek2 shapes have no usable
    denominator: splash refuses d_v != d outright, and both splash and
    ``_xla_masked_attention`` serve GQA by repeating KV, which multiplies KV
    traffic by the group size. Grading a grouped kernel against a bar that pays
    a repeat the candidate is expected to avoid would hand out reward for
    clearing a deliberately weak bar. So the repeat-paying and repeat-free
    implementations are both registered, and the per-shape election takes the
    faster one -- on MHA (group == 1) the two are the same computation.
    """
    qh, s, d = q.shape
    kvh = k.shape[0]
    dv = v.shape[-1]
    if qh % kvh:
        raise ValueError(f"q_heads {qh} not divisible by kv_heads {kvh}")
    group = qh // kvh
    qg = q.reshape(kvh, group, s, d)
    pad = (-s) % block_q
    qp = jnp.pad(qg, ((0, 0), (0, 0), (0, pad), (0, 0))) if pad else qg
    idx_k = jnp.arange(s)
    seg_k = segment_ids
    live_k = seg_k != 0
    k32 = k.astype(jnp.float32)
    v32 = v.astype(jnp.float32)

    def block(_carry, i):
        start = i * block_q
        qb = jax.lax.dynamic_slice(
            qp, (0, 0, start, 0), (kvh, group, block_q, d)
        ).astype(jnp.float32)
        pos_q = start + jnp.arange(block_q)
        seg_q = jnp.where(pos_q < s, seg_k[jnp.minimum(pos_q, s - 1)], 0)
        live_q = (seg_q != 0) & (pos_q < s)
        logits = jnp.einsum("hgqd,hkd->hgqk", qb, k32)
        if soft_cap is not None:
            logits = soft_cap * jnp.tanh(logits / soft_cap)
        causal = pos_q[:, None] >= idx_k[None, :]
        if window is not None:
            causal = causal & (pos_q[:, None] - idx_k[None, :] < window)
        m = (
            causal
            & (seg_q[:, None] == seg_k[None, :])
            & live_q[:, None]
            & live_k[None, :]
        )
        m4 = m[None, None]
        logits = jnp.where(m4, logits, NEG_INF)
        row_live = m.any(axis=-1)
        mx = jnp.max(logits, axis=-1, keepdims=True)
        if sinks is not None:
            sk = jnp.asarray(sinks, jnp.float32).reshape(kvh, group)[:, :, None, None]
            mx = jnp.maximum(mx, sk)
        p = jnp.where(m4, jnp.exp(logits - mx), 0.0)
        denom = jnp.sum(p, axis=-1, keepdims=True)
        if sinks is not None:
            denom = denom + jnp.exp(sk - mx)
        p = jnp.where(
            row_live[None, None, :, None],
            p / jnp.maximum(denom, 1e-30),
            0.0,
        )
        return None, jnp.einsum("hgqk,hkv->hgqv", p, v32)

    _, blocks = jax.lax.scan(block, None, jnp.arange((s + pad) // block_q))
    # blocks: [nblocks, kv_heads, group, block_q, d_v]
    out = jnp.transpose(blocks, (1, 2, 0, 3, 4)).reshape(kvh, group, s + pad, dv)
    return out[:, :, :s, :].reshape(qh, s, dv)


class SplashAttentionProblem(Problem):
    name = "splash_attention"
    version = "1"
    # BACKWARD IS PART OF THE CONTRACT. Google's splash kernel has one --
    # BlockSizes carries block_q_dkv / block_kv_dkv / block_q_dq and a
    # use_fused_bwd_kernel switch -- and tokamax ships tuned vjp configs for
    # attention on tpu7x (dot_product_attention_vjp,
    # pallas_mosaic_tpu_flash_attention_vjp). A forward-only attention kernel
    # is not a candidate for upstreaming, so it must not be able to pass here.
    has_bwd = True
    require_pallas = True
    general_mode = True  # score the holdout; denominator = fastest honest impl per shape
    memory_bound = False
    banned_call_names = (
        "jax.nn.dot_product_attention",
        "jax.nn.scaled_dot_product_attention",
    )

    def shape_cases(self):
        return [
            # our real fb shard shapes: [batch 4 folded into heads] x 18432
            ShapeCase("h32-s18432", {"heads": 32, "seq": 18432, "d": 128}),
            ShapeCase("h8-s18432", {"heads": 8, "seq": 18432, "d": 128}),
            # the non-block-divisible length splash itself rejects (TP=4 bite)
            ShapeCase("h8-s18433-ragged", {"heads": 8, "seq": 18433, "d": 128}),
            ShapeCase("holdout-h16-s12288", {"heads": 16, "seq": 12288, "d": 128}, holdout=True),
            # PROBE set: the same op at a size ONE 32 GB chip can hold. The
            # production cases cannot be graded on a single judge at all --
            # the fp32 reference materializes [heads, seq, seq], i.e. 10.9 GB
            # at h8-s18432 and 43.5 GB at h32-s18432. The declared-set lesson
            # is kept intact: the probe holdout is deliberately NOT block
            # divisible, so a kernel that assumes seq % block == 0 still
            # fails to trace.
            # GENERAL sweep. tokamax's own attention arg_specs vary seq at a
            # FIXED TOKEN BUDGET (batch = 16384 // seq) precisely so total work
            # stays comparable while every blocking decision changes; heads
            # here play the batch role. head_dim 64 and 128 both appear because
            # a kernel tuned to one lane width often breaks at the other.
            ShapeCase("probe-h8-s4096", {"heads": 8, "seq": 4096, "d": 128}, probe=True),
            ShapeCase("probe-h4-s2048", {"heads": 4, "seq": 2048, "d": 128}, probe=True),
            ShapeCase("probe-h16-s1024", {"heads": 16, "seq": 1024, "d": 128}, probe=True),
            # DROPPED probe-h2-s8192: the fp32 closed-form reference is
            # [2, 8192, 8192] = 537 MB, and calibration runs it alongside two
            # honest variants -- ~2 GB live for ONE case, on top of five other
            # cases' warmed programs. Measured: the judge segfaults during boot
            # (rc=139, job 3699504) once the sweep includes it. Long sequences
            # are still swept via s4096 vs s1024 at fixed token budget; testing
            # 8192 needs a memory-lean reference, not a bigger sweep.
            ShapeCase("probe-h8-s4096-d64", {"heads": 8, "seq": 4096, "d": 64}, probe=True),
            # PROVENANCE, tokamax attention arg_specs. These are the shapes
            # real models run, and both were previously inexpressible:
            #   mixtral 8x7b -- GQA, 32 query heads over 8 KV heads
            #   deepseek2 16b -- value head dim 128 against q/k 192
            # MHA is the group==1, d_v==d special case, so every existing
            # shape keeps its meaning.
            ShapeCase("mixtral-8x7b-gqa32x8-s4096",
                      {"heads": 32, "kv_heads": 8, "seq": 4096, "d": 128}, probe=True),
            ShapeCase("deepseek2-16b-s1024-d192-dv128",
                      {"heads": 16, "kv_heads": 16, "seq": 1024, "d": 192, "d_v": 128}, probe=True),
            ShapeCase("mixtral-holdout-gqa32x8-s2049",
                      {"heads": 32, "kv_heads": 8, "seq": 2049, "d": 128}, probe=True, holdout=True),
            # MULTI-QUERY (kv_heads == 1) -- tokamax parameterizes its TPU
            # attention test over num_kv_heads=[1, 2, 4], so MQA is a first-
            # class case there, not an afterthought. It is the extreme of the
            # grouping axis: ONE KV head feeding every query head, where a
            # KV-repeat costs the full head count in wasted bandwidth and a
            # native grouped kernel wins biggest. Free under the generalized
            # contract (group == q_heads); verified against the grouped
            # baseline at 1.6e-7, including with d_v != d.
            ShapeCase("mqa-h32kv1-s4096",
                      {"heads": 32, "kv_heads": 1, "seq": 4096, "d": 128}, probe=True),
            # From tokamax's experimental splash property sweep, two structures
            # our set did not span:
            #   (64, 128): d_v LARGER than d_qk -- deepseek2 covers only the
            #   shrinking direction, and an output-tiling bug that only trips
            #   when the value dim GROWS would sail through it.
            #   (6, 2): non-power-of-two head counts with group 3 -- head-count
            #   assumptions (po2 grids, halving loops) break here, not at 32/8.
            ShapeCase("dvgt-h8-s2048-d64-dv128",
                      {"heads": 8, "seq": 2048, "d": 64, "d_v": 128}, probe=True),
            ShapeCase("h6kv2-holdout-s2048",
                      {"heads": 6, "kv_heads": 2, "seq": 2048, "d": 128},
                      probe=True, holdout=True),
            ShapeCase("mqa-holdout-h16kv1-s2049",
                      {"heads": 16, "kv_heads": 1, "seq": 2049, "d": 128}, probe=True, holdout=True),
            # FEATURE CASES (static, see ShapeCase.features). Until these
            # existed the arena graded only the plain causal path while using
            # denominators that support the whole feature surface -- so a
            # candidate got full credit for implementing strictly less than the
            # kernel it was measured against.
            #
            # Sliding window: Mistral-7B's 4096 over a 4096 context, and a
            # narrow 512 window where block-skipping is the dominant effect.
            # The denominator is splash's own LocalMask, which skips KV blocks,
            # so a candidate that merely masks a full attention loses on time
            # rather than being flattered by a full-attention bar.
            ShapeCase("mistral-7b-window4096-s4096",
                      {"heads": 32, "kv_heads": 8, "seq": 4096, "d": 128},
                      probe=True, features=(("window", 4096),)),
            ShapeCase("window512-h8-s4096",
                      {"heads": 8, "seq": 4096, "d": 128},
                      probe=True, features=(("window", 512),)),
            # Logit soft-cap: Gemma-2 caps attention logits at 50.0; tokamax
            # parameterizes its TPU attention test over soft_cap=[None, 3.4].
            ShapeCase("gemma2-softcap50-h8-s4096",
                      {"heads": 8, "seq": 4096, "d": 128},
                      probe=True, features=(("soft_cap", 50.0),)),
            # Both at once, on a non-power-of-two sequence, held out: the
            # combination is where an implementation that special-cases each
            # feature separately breaks.
            ShapeCase("holdout-window1024-softcap30-h8-s2049",
                      {"heads": 8, "seq": 2049, "d": 128},
                      probe=True, holdout=True,
                      features=(("window", 1024), ("soft_cap", 30.0))),
            # ATTENTION SINKS -- gpt-oss ships them, tokamax's experimental
            # splash tests them (use_sinks with dsinks gradients), and the
            # jax-pin splash cannot express them, so the elected denominator
            # here is the grouped XLA path (recorded via baseline_impl).
            # Values vary per head (sign and magnitude) so a kernel that
            # broadcasts one sink over all heads fails rather than passes.
            ShapeCase("gptoss-sinks-h8-s2048",
                      {"heads": 8, "seq": 2048, "d": 64}, probe=True,
                      features=(("sinks", (0.5, -0.7, 1.9, -1.3, 2.6, 0.1, -2.2, 3.4)),)),
            ShapeCase("sinks-holdout-window512-h8-s2049",
                      {"heads": 8, "seq": 2049, "d": 128}, probe=True, holdout=True,
                      features=(("window", 512),
                                ("sinks", (1.1, -0.4, 2.3, -1.8, 0.9, 3.1, -2.6, 0.2)))),
            # TENSOR PARALLEL (v6e-8): heads sharded 8 ways, which is how
            # splash is sharded in production (`head_shards`). Per shard the
            # kernel sees heads=4, i.e. an ordinary attention problem.
            ShapeCase("tp8-h32-s4096", {"heads": 32, "seq": 4096, "d": 128}, probe=True, tp=8),
            # GROUPED ATTENTION UNDER TP -- what mixtral on 8 chips actually
            # is, and previously untested: every tp8 case was MHA, so the KV
            # head axis was always 32 and the MHA-only sharding assumption
            # never showed. At kv_heads=8 the KV shards exactly one head per
            # device; at kv_heads=1 it cannot shard at all and is replicated
            # (tokamax: test_broadcasted_multi_query_attention).
            ShapeCase("tp8-gqa32x8-s4096",
                      {"heads": 32, "kv_heads": 8, "seq": 4096, "d": 128}, probe=True, tp=8),
            ShapeCase("tp8-mqa-h32kv1-s4096",
                      {"heads": 32, "kv_heads": 1, "seq": 4096, "d": 128}, probe=True, tp=8),
            ShapeCase("tp8-holdout-h32-s2049", {"heads": 32, "seq": 2049, "d": 128}, probe=True, tp=8, holdout=True),
            ShapeCase("probe-holdout-h4-s2049", {"heads": 4, "seq": 2049, "d": 128}, holdout=True, probe=True),
            # DROPPED: a non-divisible sequence AT d=64 segfaults the TPU
            # runtime (rc=139, jobs 3697756/3697854) -- and only in
            # combination: probe-h8-s4096-d64 (d=64, divisible) and
            # probe-holdout-h4-s2049 (non-divisible, d=128) both pass. Each
            # property is covered separately; the intersection is a runtime bug
            # we are not here to find, and a shape that kills the judge would
            # fail every candidate for reasons that are not about kernels.
            # CPU battery
            ShapeCase("tiny", {"heads": 2, "seq": 128, "d": 32}, smoke=True),
            ShapeCase("tiny-ragged", {"heads": 2, "seq": 67, "d": 32}, smoke=True),
            ShapeCase("tiny-holdout", {"heads": 1, "seq": 96, "d": 16}, smoke=True, holdout=True),
        ]

    def make_inputs(self, key, case):
        kq, kk, kv, ks = jax.random.split(key, 4)
        # `heads` is the QUERY head count; `kv_heads` defaults to it (MHA) and
        # `d_v` defaults to d (square). GQA and an asymmetric value head dim are
        # what mixtral 8x7b and deepseek2-16b actually run, so they are shape
        # dims here rather than a different task.
        h, s, d = case.dims["heads"], case.dims["seq"], case.dims["d"]
        kvh = case.dims.get("kv_heads", h)
        dv = case.dims.get("d_v", d)
        scale = 1.0 / np.sqrt(d)
        q = (jax.random.normal(kq, (h, s, d), jnp.float32) * scale).astype(jnp.bfloat16)
        k = jax.random.normal(kk, (kvh, s, d), jnp.float32).astype(jnp.bfloat16)
        v = jax.random.normal(kv, (kvh, s, dv), jnp.float32).astype(jnp.bfloat16)
        # two segments + a padded tail (~6%), mirroring packed fb batches
        n_pad = max(s // 16, 1)
        boundary = jax.random.randint(ks, (), s // 4, 3 * s // 4)
        pos = jnp.arange(s)
        segment_ids = jnp.where(pos < boundary, 1, 2).astype(jnp.int32)
        segment_ids = jnp.where(pos >= s - n_pad, 0, segment_ids)
        return (q, k, v, segment_ids)

    def reference(self, q, k, v, segment_ids, *, window=None, soft_cap=None, sinks=None):
        return causal_segment_attention(
            q, k, v, segment_ids, window=window, soft_cap=soft_cap, sinks=sinks
        )

    # Which baseline the last `baseline()` call actually used. Recorded (not
    # asserted) so a boot report says plainly whether the score denominator
    # is Google's production kernel or the XLA fallback.
    baseline_impl: str = "?"

    def baseline(self, q, k, v, segment_ids, *, window=None, soft_cap=None, sinks=None):
        """Production Pallas splash MHA, with an honest XLA fallback.

        Splash refuses shapes it cannot tile and its block sizes are not
        valid at every (seq, head_dim); a judge whose BOOT dies on that has
        graded nothing at all, which is strictly worse than scoring against
        a slower-but-real denominator. So: try splash, fall back to a fused
        XLA masked attention, and record which one ran.
        """
        if jax.default_backend() != "tpu":
            raise BaselineUnavailable("splash attention pallas kernel requires TPU")
        try:
            from jax.experimental.pallas.ops.tpu.splash_attention import (
                splash_attention_kernel as sak,
            )
            from jax.experimental.pallas.ops.tpu.splash_attention import (
                splash_attention_mask as sam,
            )

            # make_splash_mha wants matched head counts and a square head dim.
            # GQA is served by REPEATING the KV heads, which is what a naive
            # caller does and is deliberately not free -- it multiplies KV
            # traffic by the group size, so on a GQA shape this denominator is
            # weak and the per-shape election will usually hand the bar to XLA
            # instead. That is the honest outcome, and it is recorded rather
            # than hidden: a real GQA kernel avoids the repeat, which is
            # exactly the headroom the task is meant to expose.
            if k.shape[0] != q.shape[0]:
                rep = q.shape[0] // k.shape[0]
                k = jnp.repeat(k, rep, axis=0)
                v = jnp.repeat(v, rep, axis=0)
                type(self).baseline_impl = "pallas-splash-mha (kv-repeated for GQA)"
            if v.shape[-1] != q.shape[-1]:
                raise BaselineUnavailable("splash requires d_v == d; deepseek2-style asymmetry unsupported")
            if sinks is not None:
                # Our jax pin's splash has NO attention-sinks parameter
                # (tokamax's experimental fork does; the pin does not), so on
                # sinks cases the production denominator is genuinely
                # unavailable and the election falls to the grouped XLA path,
                # recorded as such -- an honest weaker bar, not a fake one.
                raise BaselineUnavailable("jax-pin splash has no attention sinks")

            h, seq, _d = q.shape
            # SLIDING WINDOW via splash's own LocalMask rather than a masked
            # full attention: LocalMask lets splash skip whole KV blocks, which
            # is the entire performance argument for a local kernel. Grading a
            # windowed candidate against a full-attention denominator would
            # hand it a free win for doing less work than the bar.
            #
            # window_size=(left, right) counts EXCLUSIVE neighbours, so our
            # inclusive `window` (i-window, i] maps to left=window-1, right=0.
            if window is not None:
                per_head = sam.LocalMask(
                    shape=(seq, seq), window_size=(window - 1, 0), offset=0
                )
            else:
                per_head = sam.CausalMask(shape=(seq, seq))
            mask = sam.MultiHeadMask([per_head for _ in range(h)])
            kernel = sak.make_splash_mha(
                mask=mask,
                block_sizes=_splash_block_sizes(sak, seq),
                head_shards=1,
                q_seq_shards=1,
                attn_logits_soft_cap=soft_cap,
            )
            segs = sak.SegmentIds(q=segment_ids, kv=segment_ids)
            out = kernel(q, k, v, segment_ids=segs)
            # Contract adapter: OUR contract zeroes padding (segment-0) query
            # rows exactly; splash treats 0 as an ordinary segment and computes
            # attention within it. Measured (job 3692058 agreement check):
            # max_err 3.38 on ~6% of rows -- a contract difference, not a
            # numerics one. Zeroing is O(bd) elementwise against O(b s^2 d)
            # attention, and is what a real serving stack does with padded
            # rows anyway, so the timed comparison stays honest.
            out = out * (segment_ids != 0)[None, :, None]
            type(self).baseline_impl = "pallas-splash-mha"
            return out.astype(jnp.float32)
        except Exception:
            type(self).baseline_impl = "xla-fallback"
            return _xla_grouped_attention(
                q, k, v, segment_ids, window=window, soft_cap=soft_cap, sinks=sinks
            )

    def baseline_candidates(self):
        """Production splash vs two query-blocked XLA attentions. At the smaller
        sweep shapes the XLA path is genuinely competitive, and GENERAL mode
        must grade against whichever is actually faster there.

        ``xla-grouped`` is the only candidate that survives d_v != d, and the
        only one that does not pay a KV repeat on GQA -- without it the mixtral
        and deepseek2 shapes would be graded against a bar handicapped by
        exactly the traffic a good grouped kernel eliminates.
        """
        return {
            "production": self.baseline,
            "xla-blocked": _xla_masked_attention,
            "xla-grouped": _xla_grouped_attention,
        }

    def tp_specs(self, case=None):
        """Shard HEADS -- exactly what splash's own `head_shards` argument
        exists for. Each device owns a head slice, attends within it, and needs
        no collective; segment_ids is replicated because every head reads it.

        KV IS REPLICATED WHEN IT CANNOT BE SHARDED. Under GQA/MQA the KV head
        count is smaller than the query head count, and at MQA it is 1 -- a
        size-1 axis cannot be split across 8 devices. tokamax tests exactly
        this (`test_broadcasted_multi_query_attention`: a single KV head run
        under partitioning along batch / seq_q / heads), and the name is the
        rule: the short axis BROADCASTS across the mesh rather than splitting.

        Sharding q while replicating kv is also what production does -- it is
        why GQA is cheap to serve tensor-parallel: each device holds all of a
        small KV cache and its own slice of the queries.
        """
        from jax.sharding import PartitionSpec as P

        shard, repl = P("tp", None, None), P(None, None, None)
        kv = shard
        if case is not None:
            heads = case.dims.get("heads")
            kv_heads = case.dims.get("kv_heads", heads)
            width = case.tp or 1
            if kv_heads is not None and kv_heads % width:
                kv = repl
        return ((shard, kv, kv, P()), shard)

    def grad_outputs(self, kernel_fn, q, k, v, segment_ids):
        """d/d(q, k, v) of a fixed scalar functional of the output.

        The probe is a deterministic non-symmetric cotangent (cos of a flat
        iota), NOT a plain sum: summing the output makes many wrong backward
        rules look right, because the errors cancel across the reduction. The
        reference is differentiated the same way, so a candidate whose forward
        is correct but whose backward is not is caught here rather than
        shipping a kernel nobody can train through.

        Cast to fp32 for the differentiated inputs: the gradient is compared
        against the fp32 reference at the same calibrated tolerance the forward
        uses, and differentiating through a bf16 cast adds a quantization step
        the contract does not ask about.
        """
        # Shaped from the OUTPUT, not from q: under the generalized contract the
        # output is [q_heads, seq, d_v] and d_v need not equal the q/k head dim
        # (deepseek2: 128 vs 192), so a q-shaped probe fails to broadcast there.
        out_shape = (q.shape[0], q.shape[1], v.shape[-1])
        probe = jnp.cos(
            jnp.arange(int(np.prod(out_shape)), dtype=jnp.float32)
        ).reshape(out_shape)

        def scalar(q32, k32, v32):
            out = kernel_fn(q32.astype(q.dtype), k32.astype(k.dtype), v32.astype(v.dtype), segment_ids)
            return jnp.sum(out.astype(jnp.float32) * probe)

        return jax.grad(scalar, argnums=(0, 1, 2))(
            q.astype(jnp.float32), k.astype(jnp.float32), v.astype(jnp.float32)
        )

    def honest_variants(self):
        """Calibrate against what an honest bf16 kernel actually computes, not
        against an fp32-intermediate idealization. See ``_honest_faithful_bf16``:
        the reference-bf16-only band rejects the production numeric path on some
        shapes (RPA's tiny-holdout measured 1.15x), which is precisely the
        phase-2 failure this hook exists to prevent. The end-to-end-bf16 path
        still fails by 1.7-2.2x with these in, so discrimination survives."""
        return [_honest_faithful_bf16, _honest_online_softmax]

    def adversarial_cases(self):
        tiny = self.case_by_name(self.adversarial_case_name)

        def saturating_logits(key):
            q, k, v, seg = self.make_inputs(key, tiny)
            q = (q.astype(jnp.float32) * 300.0).astype(jnp.bfloat16)
            return (q, k, v, seg)

        def fully_masked_rows(key):
            q, k, v, seg = self.make_inputs(key, tiny)
            s = seg.shape[0]
            seg = jnp.where(jnp.arange(s) % 5 == 0, 0, seg).astype(jnp.int32)
            return (q, k, v, seg)

        def outlier_rows(key):
            q, k, v, seg = self.make_inputs(key, tiny)
            v = v.at[:, 3, :].set(jnp.full((v.shape[0], v.shape[2]), 3e4, jnp.bfloat16))
            return (q, k, v, seg)

        def near_overflow_bf16(key):
            q, k, v, seg = self.make_inputs(key, tiny)
            k = k.at[:, 1, :].set(jnp.full((k.shape[0], k.shape[2]), 1e4, jnp.bfloat16))
            return (q, k, v, seg)

        def expect_finite(ref, inputs):
            assert np.isfinite(np.asarray(ref, np.float64)).all()

        def expect_masked_zero(ref, inputs):
            seg = np.asarray(inputs[3])
            out = np.asarray(ref)
            dead = seg == 0
            assert np.isfinite(out).all(), "NaN in attention output"
            assert np.abs(out[:, dead, :]).max() == 0.0, "fully-masked (padding) query rows must be exactly 0"

        return [
            AdversarialCase("softmax-saturating", saturating_logits, expect_finite),
            AdversarialCase("fully-masked-rows", fully_masked_rows, expect_masked_zero),
            AdversarialCase("outlier-value-rows", outlier_rows, expect_finite),
            AdversarialCase("near-overflow-bf16-keys", near_overflow_bf16, expect_finite),
        ]


PROBLEM = SplashAttentionProblem()
