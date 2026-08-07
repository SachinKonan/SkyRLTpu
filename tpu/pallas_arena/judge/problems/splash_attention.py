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
            ShapeCase("probe-h8-s4096", {"heads": 8, "seq": 4096, "d": 128}, probe=True),
            ShapeCase("probe-h4-s2048", {"heads": 4, "seq": 2048, "d": 128}, probe=True),
            ShapeCase("probe-holdout-h4-s2049", {"heads": 4, "seq": 2049, "d": 128}, holdout=True, probe=True),
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
            kernel = sak.make_splash_mha(mask=mask, head_shards=1, q_seq_shards=1)
            segs = sak.SegmentIds(q=segment_ids, kv=segment_ids)
            out = kernel(q, k, v, segment_ids=segs)
            type(self).baseline_impl = "pallas-splash-mha"
            return out.astype(jnp.float32)
        except Exception:
            type(self).baseline_impl = "xla-fallback"
            return _xla_masked_attention(q, k, v, segment_ids)

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
