"""Task 2 — Ragged paged attention (decode, GQA 8 KV heads).

Baseline: vLLM-TPU's Pallas kernel
(third_party/tpu-inference tpu_inference/kernels/ragged_paged_attention/v3).
A win here speeds our own sampling fleet. Decode-only contract: one query
token per sequence, KV read from a paged cache.

kernel(q, k_pages, v_pages, page_tables, seq_lens) -> o
  q:           [batch, q_heads, d]                bfloat16 (1 token/seq)
  k_pages:     [num_pages, page_size, kv_heads, d] bfloat16
  v_pages:     [num_pages, page_size, kv_heads, d] bfloat16
  page_tables: [batch, max_pages_per_seq]          int32 (page ids; padded 0)
  seq_lens:    [batch]                             int32 (>= 1)
  o:           [batch, q_heads, d]                 float32
GQA: q_heads = kv_heads * group; head g of q attends kv head g // group.
Softmax fp32. Positions >= seq_len are masked. Contract: fwd only.
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


# THE ORACLE MUST ACTUALLY BE FP32. jnp.einsum/@ without precision= is
# Precision.DEFAULT, which on TPU multiplies f32 inputs through bf16, so a
# reference documented as fp32 is not. Measured on splash (v6e 2026-08-27):
# the oracle disagreed with a true f32 evaluation of itself by up to 3.7e-3
# and an EXACT candidate was rejected for errors the oracle committed, while
# the band was calibrated by honest variants that shared the defect. Same
# defect, same fix, here.
_F32 = jax.lax.Precision.HIGHEST

def paged_decode_attention_reference(q, k_pages, v_pages, page_tables, seq_lens):
    """fp32 closed form: gather pages -> dense masked attention per seq."""
    b, qh, d = q.shape
    _, page_size, kvh, _ = k_pages.shape
    group = qh // kvh
    max_pages = page_tables.shape[1]
    max_len = max_pages * page_size

    # [b, max_pages, page_size, kvh, d] -> [b, max_len, kvh, d]
    k = k_pages[page_tables].reshape(b, max_len, kvh, d).astype(jnp.float32)
    v = v_pages[page_tables].reshape(b, max_len, kvh, d).astype(jnp.float32)
    q32 = q.astype(jnp.float32).reshape(b, kvh, group, d)

    logits = jnp.einsum("bhgd,bthd->bhgt", q32, k, precision=_F32)  # [b, kvh, group, t]
    pos = jnp.arange(max_len)
    live = pos[None, :] < seq_lens[:, None]  # [b, t]
    logits = jnp.where(live[:, None, None, :], logits, -0.7 * float(np.finfo(np.float32).max))
    m = jnp.max(logits, axis=-1, keepdims=True)
    p = jnp.exp(logits - m)
    p = jnp.where(live[:, None, None, :], p, 0.0)
    p = p / jnp.maximum(jnp.sum(p, axis=-1, keepdims=True), 1e-30)
    o = jnp.einsum("bhgt,bthd->bhgd", p, v, precision=_F32)
    return o.reshape(b, qh, d)


def _honest_faithful_bf16(q, k_pages, v_pages, page_tables, seq_lens):
    """The production numeric path: bf16 arrays are matmul INPUTS, every
    accumulator is fp32 (``preferred_element_type``), softmax at fp32, probs
    re-narrowed to bf16 only on the way into the PV matmul.

    This variant is why the hook exists. On the ``tiny-holdout`` case it lands
    at 1.15x of the reference-bf16-only band -- i.e. the band as previously
    calibrated REJECTED the numeric path that vLLM-TPU itself uses. With it in
    the calibration the band widens 1.73x and the end-to-end-bf16 path still
    fails at 1.95x, so the sloppy/faithful discrimination survives.
    """
    b, qh, d = q.shape
    _, ps, kvh, _ = k_pages.shape
    group = qh // kvh
    ml = page_tables.shape[1] * ps
    k = k_pages[page_tables].reshape(b, ml, kvh, d)
    v = v_pages[page_tables].reshape(b, ml, kvh, d)
    qr = q.reshape(b, kvh, group, d)
    logits = jnp.einsum("bhgd,bthd->bhgt", qr, k, preferred_element_type=jnp.float32)
    live = jnp.arange(ml)[None, :] < seq_lens[:, None]
    logits = jnp.where(live[:, None, None, :], logits, -0.7 * float(np.finfo(np.float32).max))
    p = jnp.exp(logits - jnp.max(logits, axis=-1, keepdims=True))
    p = jnp.where(live[:, None, None, :], p, 0.0)
    p = p / jnp.maximum(jnp.sum(p, axis=-1, keepdims=True), 1e-30)
    o = jnp.einsum("bhgt,bthd->bhgd", p.astype(jnp.bfloat16), v, preferred_element_type=jnp.float32)
    return o.reshape(b, qh, d)


def _xla_paged_decode(q, k_pages, v_pages, page_tables, seq_lens, block_b: int = 16):
    """Batch-blocked XLA paged decode -- the honest fallback baseline.

    Same math as the closed form, but it gathers only ONE batch block's pages
    at a time, so it runs at shapes where materializing the whole gathered KV
    would not fit. Softmax at fp32. This is a competent XLA implementation, not
    a straw man -- but it is NOT vLLM-TPU's Pallas v3 kernel, and any score
    against it must be reported as such.
    """
    b, qh, d = q.shape
    _, ps, kvh, _ = k_pages.shape
    group = qh // kvh
    mp = page_tables.shape[1]
    ml = mp * ps
    nb = -(-b // block_b)
    pad = nb * block_b - b
    pt = jnp.pad(page_tables, ((0, pad), (0, 0))) if pad else page_tables
    sl = jnp.pad(seq_lens, ((0, pad),), constant_values=1) if pad else seq_lens
    qq = jnp.pad(q, ((0, pad), (0, 0), (0, 0))) if pad else q
    qq = qq.reshape(nb, block_b, kvh, group, d)
    ptb = pt.reshape(nb, block_b, mp)
    slb = sl.reshape(nb, block_b)
    pos = jnp.arange(ml)
    neg = -0.7 * float(np.finfo(np.float32).max)

    def blk(_carry, i):
        ptt = ptb[i]
        k = k_pages[ptt].reshape(block_b, ml, kvh, d).astype(jnp.float32)
        v = v_pages[ptt].reshape(block_b, ml, kvh, d).astype(jnp.float32)
        logits = jnp.einsum("bhgd,bthd->bhgt", qq[i].astype(jnp.float32), k)
        live = pos[None, :] < slb[i][:, None]
        logits = jnp.where(live[:, None, None, :], logits, neg)
        m = jnp.max(logits, axis=-1, keepdims=True)
        p = jnp.where(live[:, None, None, :], jnp.exp(logits - m), 0.0)
        p = p / jnp.maximum(jnp.sum(p, axis=-1, keepdims=True), 1e-30)
        return None, jnp.einsum("bhgt,bthd->bhgd", p, v)

    _, outs = jax.lax.scan(blk, None, jnp.arange(nb))
    return outs.reshape(nb * block_b, qh, d)[:b]


class RaggedPagedAttentionProblem(Problem):
    name = "ragged_paged_attention"
    version = "1"
    has_bwd = False
    require_pallas = True
    general_mode = True  # score the holdout; denominator = fastest honest impl per shape
    memory_bound = True  # decode attention is HBM-bandwidth bound
    banned_call_names = ("jax.nn.dot_product_attention",)
    # Google's own kernel misses the reference_bf16-calibrated band at 1.05x
    # (job 3692058), and it solves this EXACT contract -- so it participates in
    # calibration: a candidate is never held tighter than the production kernel.
    baseline_calibrates = True

    PAGE_SIZE = 64
    KV_HEADS = 8  # GQA 8 KV heads (our sampling config)
    Q_HEADS = 32
    HEAD_DIM = 128

    def shape_cases(self):
        d = dict(page_size=self.PAGE_SIZE, kv_heads=self.KV_HEADS, q_heads=self.Q_HEADS, head_dim=self.HEAD_DIM)
        return [
            ShapeCase("b64-len2048", {**d, "batch": 64, "max_len": 2048, "num_pages": 64 * 32 + 8}),
            ShapeCase("b256-len1024", {**d, "batch": 256, "max_len": 1024, "num_pages": 256 * 16 + 8}),
            ShapeCase(
                "holdout-b128-len4096", {**d, "batch": 128, "max_len": 4096, "num_pages": 128 * 64 + 8}, holdout=True
            ),
            # PROBE set: one-chip sizes at UNCHANGED page_size (64), kv_heads
            # (8), q_heads (32) and head_dim (128) -- only the batch and the
            # context length shrink, so the GQA grouping and the paged-gather
            # difficulty the task exists for are intact. The holdout batch is
            # deliberately 17, a prime, so a kernel that assumes the batch tiles
            # its block size still fails to trace.
            ShapeCase("probe-b16-len1024", {**d, "batch": 16, "max_len": 1024, "num_pages": 16 * 16 + 8}, probe=True),
            ShapeCase("probe-b8-len512", {**d, "batch": 8, "max_len": 512, "num_pages": 8 * 8 + 8}, probe=True),
            # GENERAL sweep: decode batch is THE axis for paged attention, and
            # tiny batches are exactly where the Pallas kernel loses to XLA.
            ShapeCase("probe-b64-len1024", {**d, "batch": 64, "max_len": 1024, "num_pages": 64 * 16 + 8}, probe=True),
            ShapeCase("probe-b32-len2048", {**d, "batch": 32, "max_len": 2048, "num_pages": 32 * 32 + 8}, probe=True),
            ShapeCase("probe-b128-len512", {**d, "batch": 128, "max_len": 512, "num_pages": 128 * 8 + 8}, probe=True),
            # REGIME-FAITHFUL set. The probe cases above are OUTSIDE the
            # regime paged attention exists for, and it shows: measured on
            # v6e-1, jax's Pallas kernel LOSES to a blocked XLA decode at every
            # one of them (0.50 vs 0.21 ms at b16-len1024), while both sit at
            # 6-20% of HBM bandwidth and ~0.1% of compute peak -- fixed
            # overhead dominates and the simpler implementation wins because it
            # has less machinery to amortize.
            #
            # tokamax's own paged-decode specs (experimental/mla/arg_specs.py,
            # project deepseek_v3) are 8192-token contexts at page_size 256,
            # with both a 3-sequence and a 128-sequence case. Ours were <=1024
            # at page_size 64: 8x short and 4x more page indirection per unit
            # of work. Context length is what decode attention actually walks,
            # so it moves first. (b64/b128 at 8192 next: the fp32 reference
            # gathers b*ml*kvh*d*4*2 = 4.3 GB at b64 before variants, and boot
            # has already segfaulted this judge at less.)
            ShapeCase("tok-b8-len8192-ps256",
                      {"page_size": 256, "kv_heads": self.KV_HEADS, "q_heads": self.Q_HEADS,
                       "head_dim": self.HEAD_DIM, "batch": 8, "max_len": 8192,
                       "num_pages": 8 * 32 + 8}, probe=True),
            ShapeCase("tok-b32-len8192-ps256",
                      {"page_size": 256, "kv_heads": self.KV_HEADS, "q_heads": self.Q_HEADS,
                       "head_dim": self.HEAD_DIM, "batch": 32, "max_len": 8192,
                       "num_pages": 32 * 32 + 8}, probe=True),
            ShapeCase("tok-holdout-b17-len8192-ps256",
                      {"page_size": 256, "kv_heads": self.KV_HEADS, "q_heads": self.Q_HEADS,
                       "head_dim": self.HEAD_DIM, "batch": 17, "max_len": 8192,
                       "num_pages": 17 * 32 + 8}, probe=True, holdout=True),
            # TENSOR PARALLEL (v6e-8): KV heads sharded 8 ways -- with
            # KV_HEADS=8 that is exactly one KV head (and its 4 query
            # heads) per device, the standard serving split.
            ShapeCase("tp8-b64-len1024", {**d, "batch": 64, "max_len": 1024, "num_pages": 64 * 16 + 8}, probe=True, tp=8),
            ShapeCase("tp8-holdout-b17-len512", {**d, "batch": 17, "max_len": 512, "num_pages": 17 * 8 + 8}, probe=True, tp=8, holdout=True),
            ShapeCase("tp4-b64-len1024", {**d, "batch": 64, "max_len": 1024, "num_pages": 64 * 16 + 8}, probe=True, tp=4),
            ShapeCase("tp4-holdout-b17-len512", {**d, "batch": 17, "max_len": 512, "num_pages": 17 * 8 + 8}, probe=True, tp=4, holdout=True),
            ShapeCase(
                "probe-holdout-b17-len512",
                {**d, "batch": 17, "max_len": 512, "num_pages": 17 * 8 + 8},
                holdout=True,
                probe=True,
            ),
            # CPU battery
            ShapeCase(
                "tiny",
                {
                    "page_size": 8,
                    "kv_heads": 2,
                    "q_heads": 4,
                    "head_dim": 16,
                    "batch": 3,
                    "max_len": 32,
                    "num_pages": 16,
                },
                smoke=True,
            ),
            ShapeCase(
                "tiny-holdout",
                {
                    "page_size": 8,
                    "kv_heads": 2,
                    "q_heads": 4,
                    "head_dim": 16,
                    "batch": 2,
                    "max_len": 16,
                    "num_pages": 8,
                },
                smoke=True,
                holdout=True,
            ),
        ]

    def make_inputs(self, key, case):
        kq, kk, kv, kl, kp = jax.random.split(key, 5)
        dm = case.dims
        b, qh, kvh, d = dm["batch"], dm["q_heads"], dm["kv_heads"], dm["head_dim"]
        ps, np_, ml = dm["page_size"], dm["num_pages"], dm["max_len"]
        mp = ml // ps
        scale = 1.0 / np.sqrt(d)
        q = (jax.random.normal(kq, (b, qh, d), jnp.float32) * scale).astype(jnp.bfloat16)
        k_pages = jax.random.normal(kk, (np_, ps, kvh, d), jnp.float32).astype(jnp.bfloat16)
        v_pages = jax.random.normal(kv, (np_, ps, kvh, d), jnp.float32).astype(jnp.bfloat16)
        seq_lens = jax.random.randint(kl, (b,), 1, ml + 1, jnp.int32)
        page_tables = jax.random.randint(kp, (b, mp), 0, np_, jnp.int32)
        return (q, k_pages, v_pages, page_tables, seq_lens)

    def reference(self, q, k_pages, v_pages, page_tables, seq_lens):
        return paged_decode_attention_reference(q, k_pages, v_pages, page_tables, seq_lens)

    # Which baseline the last `baseline()` call actually used. Recorded (not
    # asserted) so a boot report says plainly what the score denominator IS.
    baseline_impl: str = "?"

    def baseline(self, *inputs):
        """Google's Pallas ragged-paged-attention kernel (shipped IN JAX at the
        arena's own 0.10.2 pin -- `jax.experimental.pallas.ops.tpu.
        ragged_paged_attention`), otherwise a batch-blocked XLA paged decode,
        recorded, never hidden.

        HISTORY: this slot spent months as "vendor vLLM-TPU's v3 kernel +
        adapter, someday" and always fell through to the XLA fallback. The
        kernel was in our own jax install the whole time; nobody looked.

        Layout adapter (validated by `kernel.static_validate_inputs`):
          ours                          theirs
          q          [b, qh, d]     ->  q [total_q_tokens, qh, d]  (decode:
                                        1 token/seq, so total == b)
          k_pages,v_pages
            [np, ps, kvh, d] x2     ->  kv_pages [np, ps, 2*kvh, d], heads
                                        INTERLEAVED k0 v0 k1 v1 ...
          page_tables [b, maxp]     ->  page_indices (int32)
          seq_lens    [b]           ->  kv_lens (int32)
          (implicit)                ->  cu_q_lens = arange(b+1), num_seqs=[b]
        sm_scale=1.0 because our make_inputs already folds 1/sqrt(d) into q.

        The interleave order (k even, v odd) is asserted against the fp32
        reference by the boot-check job before any reward is reported -- a
        transposed adapter produces garbage agreement, not a subtle bias.

        `num_kv_pages_per_block` / `num_queries_per_block` are left None here,
        which selects the kernel's own shape-dependent choice, NOT a fixed
        placeholder (unlike splash's all-128 BlockSizes) -- verified in
        kernel.py, and covered by the design sweep before scores are trusted.
        """
        try:
            if jax.default_backend() != "tpu":
                raise BaselineUnavailable("pallas ragged_paged_attention requires TPU")
            from jax.experimental.pallas.ops.tpu.ragged_paged_attention import (
                ragged_paged_attention,
            )

            q, k_pages, v_pages, page_tables, seq_lens = inputs
            b = q.shape[0]
            np_, ps, kvh, d = k_pages.shape
            # [np, ps, kvh, 2, d] -> [np, ps, 2*kvh, d]: k_i at 2i, v_i at 2i+1
            kv_pages = jnp.stack([k_pages, v_pages], axis=3).reshape(np_, ps, 2 * kvh, d)
            out = ragged_paged_attention(
                q,
                kv_pages,
                seq_lens.astype(jnp.int32),
                page_tables.astype(jnp.int32),
                jnp.arange(b + 1, dtype=jnp.int32),
                jnp.array([b], jnp.int32),
                sm_scale=1.0,
            )
            type(self).baseline_impl = "pallas-ragged-paged-attention"
            return out.astype(jnp.float32)
        except BaselineUnavailable:
            type(self).baseline_impl = "xla-paged-decode-fallback"
            return _xla_paged_decode(*inputs)
        except Exception:
            type(self).baseline_impl = "xla-paged-decode-fallback (kernel raised)"
            return _xla_paged_decode(*inputs)

    def baseline_candidates(self):
        """jax's Pallas RPA vs the blocked XLA paged decode. MEASURED (job
        3692058): at probe batch sizes the XLA path is 0.21ms against the
        Pallas kernel's 0.51ms -- so grading against "the production kernel"
        alone hands out credit for beating the slower of the two."""
        return {"production": self.baseline, "xla-paged-decode": _xla_paged_decode}

    def tp_specs(self, case=None):
        """Shard KV HEADS, with the query heads that map to them -- how a
        paged-attention decode is sharded in a real serving stack. Page tables
        and sequence lengths replicate; each device owns whole KV heads, so
        there is no cross-device gather."""
        from jax.sharding import PartitionSpec as P

        return ((P(None, "tp", None), P(None, None, "tp", None), P(None, None, "tp", None), P(), P()),
                P(None, "tp", None))

    def honest_variants(self):
        """See ``_honest_faithful_bf16``: measured at 1.15x of the
        reference-bf16-only band on tiny-holdout, so that band was rejecting
        the production numeric path outright."""
        return [_honest_faithful_bf16]

    def adversarial_cases(self):
        # settable base case: a judge grading a non-default (e.g. probe) case
        # set must not silently force candidates to also trace at the tiny
        # CPU-battery shapes, which no prompt declares
        tiny = self.case_by_name(self.adversarial_case_name)

        def single_token(key):
            q, kp, vp, pt, sl = self.make_inputs(key, tiny)
            return (q, kp, vp, pt, jnp.ones_like(sl))

        def page_boundary(key):
            q, kp, vp, pt, sl = self.make_inputs(key, tiny)
            ps = kp.shape[1]
            # lengths exactly on page boundaries: off-by-one page loops bite
            sl = jnp.maximum((sl // ps) * ps, ps)
            return (q, kp, vp, pt, sl)

        def max_length(key):
            q, kp, vp, pt, sl = self.make_inputs(key, tiny)
            ml = pt.shape[1] * kp.shape[1]
            return (q, kp, vp, pt, jnp.full_like(sl, ml))

        def saturating(key):
            q, kp, vp, pt, sl = self.make_inputs(key, tiny)
            q = (q.astype(jnp.float32) * 300.0).astype(jnp.bfloat16)
            return (q, kp, vp, pt, sl)

        def expect_finite(ref, inputs):
            assert np.isfinite(np.asarray(ref, np.float64)).all()

        def expect_single_token(ref, inputs):
            # len==1: output must equal v of the first cached token exactly
            q, kp, vp, pt, sl = inputs
            b, qh, d = np.asarray(ref).shape
            kvh = np.asarray(kp).shape[2]
            group = qh // kvh
            first_page = np.asarray(pt)[:, 0]
            v0 = np.asarray(vp, np.float32)[first_page, 0]  # [b, kvh, d]
            want = np.repeat(v0, group, axis=1)
            np.testing.assert_allclose(np.asarray(ref), want, rtol=1e-4, atol=1e-4)

        return [
            AdversarialCase("single-token-decode", single_token, expect_single_token),
            AdversarialCase("page-boundary-lengths", page_boundary, expect_finite),
            AdversarialCase("max-length", max_length, expect_finite),
            AdversarialCase("softmax-saturating", saturating, expect_finite),
        ]

    def bytes_moved(self, case):
        dm = case.dims
        # dominant term: reading the live KV pages once
        avg_len = dm["max_len"] // 2
        return dm["batch"] * avg_len * dm["kv_heads"] * dm["head_dim"] * 2 * 2 + dm["batch"] * dm["q_heads"] * dm[
            "head_dim"
        ] * (2 + 4)


PROBLEM = RaggedPagedAttentionProblem()
