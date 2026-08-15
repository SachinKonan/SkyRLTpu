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


def causal_segment_attention(q, k, v, segment_ids):
    """fp32 masked-softmax closed form; fully-masked rows -> exactly 0."""
    q32 = q.astype(jnp.float32)
    k32 = k.astype(jnp.float32)
    v32 = v.astype(jnp.float32)
    seq = q.shape[1]
    logits = jnp.einsum("hqd,hkd->hqk", q32, k32)
    idx = jnp.arange(seq)
    causal = idx[:, None] >= idx[None, :]
    same_seg = segment_ids[:, None] == segment_ids[None, :]
    live = segment_ids != 0
    mask = causal & same_seg & live[:, None] & live[None, :]
    logits = jnp.where(mask[None, :, :], logits, NEG_INF)
    row_live = mask.any(axis=-1)  # [q] rows with at least one visible key
    # max-shifted softmax; fully-masked rows produce 0, never NaN
    m = jnp.max(logits, axis=-1, keepdims=True)
    p = jnp.exp(logits - m)
    p = jnp.where(mask[None, :, :], p, 0.0)
    denom = jnp.sum(p, axis=-1, keepdims=True)
    p = jnp.where(row_live[None, :, None], p / jnp.maximum(denom, 1e-30), 0.0)
    return jnp.einsum("hqk,hkd->hqd", p, v32)


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


def _honest_faithful_bf16(q, k, v, segment_ids, block_q: int = 512):
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
        m = (pos_q[:, None] >= idx_k[None, :]) & (seg_q[:, None] == seg_k[None, :]) & live_q[:, None] & live_k[None, :]
        logits = jnp.where(m[None], logits, NEG_INF)
        row_live = m.any(axis=-1)
        mx = jnp.max(logits, axis=-1, keepdims=True)
        p = jnp.where(m[None], jnp.exp(logits - mx), 0.0)
        p = jnp.where(row_live[None, :, None], p / jnp.maximum(jnp.sum(p, axis=-1, keepdims=True), 1e-30), 0.0)
        out = jnp.einsum("hqk,hkd->hqd", p.astype(jnp.bfloat16), v, preferred_element_type=jnp.float32)
        return None, out

    _, blocks = jax.lax.scan(block, None, jnp.arange((s + pad) // block_q))
    out = jnp.transpose(blocks, (1, 0, 2, 3)).reshape(h, s + pad, d)
    return out[:, :s, :]


def _honest_online_softmax(q, k, v, segment_ids, block_k: int = 256):
    """Flash-style streaming softmax: a DIFFERENT reduction order over the key
    axis (running max + rescaled running sum) at fp32. Legal, and the shape a
    real Pallas kernel takes; included so a candidate is never punished for
    accumulating keys in blocks rather than all at once."""
    h, s, d = q.shape
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
        m = (
            (pos_q[:, None] >= kpos[None, :])
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
    (_, run_l, acc), _ = jax.lax.scan(body, init, jnp.arange(nb))
    row_live = ((idx_k[None, :] <= pos_q[:, None]) & (segment_ids[:, None] == segment_ids[None, :]) & live_q[:, None] & live_k[None, :]).any(-1)
    return jnp.where(row_live[None, :, None], acc / jnp.maximum(run_l, 1e-30), 0.0)


_FALLBACK_BLOCK_Q = 512


def _xla_masked_attention(q, k, v, segment_ids, block_q: int = _FALLBACK_BLOCK_Q):
    """Query-blocked XLA attention: same math as ``causal_segment_attention``
    but never materializes more than [heads, block_q, seq] of logits, so it
    runs at shapes where the closed form would not fit. Used only as the
    baseline fallback when the production splash kernel refuses a shape."""
    h, s, d = q.shape
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
        m = (pos_q[:, None] >= idx_k[None, :]) & (seg_q[:, None] == seg_k[None, :]) & live_q[:, None] & live_k[None, :]
        logits = jnp.where(m[None], logits, NEG_INF)
        row_live = m.any(axis=-1)
        mx = jnp.max(logits, axis=-1, keepdims=True)
        p = jnp.where(m[None], jnp.exp(logits - mx), 0.0)
        p = jnp.where(row_live[None, :, None], p / jnp.maximum(jnp.sum(p, axis=-1, keepdims=True), 1e-30), 0.0)
        return None, jnp.einsum("hqk,hkd->hqd", p, v32)

    _, blocks = jax.lax.scan(block, None, jnp.arange((s + pad) // block_q))
    out = jnp.transpose(blocks, (1, 0, 2, 3)).reshape(h, s + pad, d)
    return out[:, :s, :]


class SplashAttentionProblem(Problem):
    name = "splash_attention"
    version = "1"
    has_bwd = False  # fwd-only contract in phase 1 of the task
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
            ShapeCase("probe-h2-s8192", {"heads": 2, "seq": 8192, "d": 128}, probe=True),
            ShapeCase("probe-h8-s4096-d64", {"heads": 8, "seq": 4096, "d": 64}, probe=True),
            ShapeCase("probe-holdout-h4-s2049", {"heads": 4, "seq": 2049, "d": 128}, holdout=True, probe=True),
            ShapeCase("probe-holdout-h3-s1535-d64", {"heads": 3, "seq": 1535, "d": 64}, holdout=True, probe=True),
            # CPU battery
            ShapeCase("tiny", {"heads": 2, "seq": 128, "d": 32}, smoke=True),
            ShapeCase("tiny-ragged", {"heads": 2, "seq": 67, "d": 32}, smoke=True),
            ShapeCase("tiny-holdout", {"heads": 1, "seq": 96, "d": 16}, smoke=True, holdout=True),
        ]

    def make_inputs(self, key, case):
        kq, kk, kv, ks = jax.random.split(key, 4)
        h, s, d = case.dims["heads"], case.dims["seq"], case.dims["d"]
        scale = 1.0 / np.sqrt(d)
        q = (jax.random.normal(kq, (h, s, d), jnp.float32) * scale).astype(jnp.bfloat16)
        k = jax.random.normal(kk, (h, s, d), jnp.float32).astype(jnp.bfloat16)
        v = jax.random.normal(kv, (h, s, d), jnp.float32).astype(jnp.bfloat16)
        # two segments + a padded tail (~6%), mirroring packed fb batches
        n_pad = max(s // 16, 1)
        boundary = jax.random.randint(ks, (), s // 4, 3 * s // 4)
        pos = jnp.arange(s)
        segment_ids = jnp.where(pos < boundary, 1, 2).astype(jnp.int32)
        segment_ids = jnp.where(pos >= s - n_pad, 0, segment_ids)
        return (q, k, v, segment_ids)

    def reference(self, q, k, v, segment_ids):
        return causal_segment_attention(q, k, v, segment_ids)

    # Which baseline the last `baseline()` call actually used. Recorded (not
    # asserted) so a boot report says plainly whether the score denominator
    # is Google's production kernel or the XLA fallback.
    baseline_impl: str = "?"

    def baseline(self, q, k, v, segment_ids):
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

            h, seq, _d = q.shape
            mask = sam.MultiHeadMask([sam.CausalMask(shape=(seq, seq)) for _ in range(h)])
            kernel = sak.make_splash_mha(
                mask=mask, block_sizes=_splash_block_sizes(sak, seq), head_shards=1, q_seq_shards=1
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
            return _xla_masked_attention(q, k, v, segment_ids)

    def baseline_candidates(self):
        """Production splash vs a competent query-blocked XLA attention. At the
        smaller sweep shapes the XLA path is genuinely competitive, and GENERAL
        mode must grade against whichever is actually faster there."""
        return {"production": self.baseline, "xla-blocked": _xla_masked_attention}

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
