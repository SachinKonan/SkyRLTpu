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

    logits = jnp.einsum("bhgd,bthd->bhgt", q32, k)  # [b, kvh, group, t]
    pos = jnp.arange(max_len)
    live = pos[None, :] < seq_lens[:, None]         # [b, t]
    logits = jnp.where(live[:, None, None, :], logits,
                       -0.7 * float(np.finfo(np.float32).max))
    m = jnp.max(logits, axis=-1, keepdims=True)
    p = jnp.exp(logits - m)
    p = jnp.where(live[:, None, None, :], p, 0.0)
    p = p / jnp.maximum(jnp.sum(p, axis=-1, keepdims=True), 1e-30)
    o = jnp.einsum("bhgt,bthd->bhgd", p, v)
    return o.reshape(b, qh, d)


class RaggedPagedAttentionProblem(Problem):
    name = "ragged_paged_attention"
    version = "1"
    has_bwd = False
    require_pallas = True
    memory_bound = True          # decode attention is HBM-bandwidth bound
    banned_call_names = (
        "jax.nn.dot_product_attention",
    )

    PAGE_SIZE = 64
    KV_HEADS = 8                 # GQA 8 KV heads (our sampling config)
    Q_HEADS = 32
    HEAD_DIM = 128

    def shape_cases(self):
        d = dict(page_size=self.PAGE_SIZE, kv_heads=self.KV_HEADS,
                 q_heads=self.Q_HEADS, head_dim=self.HEAD_DIM)
        return [
            ShapeCase("b64-len2048", {**d, "batch": 64, "max_len": 2048,
                                      "num_pages": 64 * 32 + 8}),
            ShapeCase("b256-len1024", {**d, "batch": 256, "max_len": 1024,
                                       "num_pages": 256 * 16 + 8}),
            ShapeCase("holdout-b128-len4096", {**d, "batch": 128,
                                               "max_len": 4096,
                                               "num_pages": 128 * 64 + 8},
                      holdout=True),
            # CPU battery
            ShapeCase("tiny", {"page_size": 8, "kv_heads": 2, "q_heads": 4,
                               "head_dim": 16, "batch": 3, "max_len": 32,
                               "num_pages": 16}, smoke=True),
            ShapeCase("tiny-holdout", {"page_size": 8, "kv_heads": 2,
                                       "q_heads": 4, "head_dim": 16,
                                       "batch": 2, "max_len": 16,
                                       "num_pages": 8},
                      smoke=True, holdout=True),
        ]

    def make_inputs(self, key, case):
        kq, kk, kv, kl, kp = jax.random.split(key, 5)
        dm = case.dims
        b, qh, kvh, d = dm["batch"], dm["q_heads"], dm["kv_heads"], dm["head_dim"]
        ps, np_, ml = dm["page_size"], dm["num_pages"], dm["max_len"]
        mp = ml // ps
        scale = 1.0 / np.sqrt(d)
        q = (jax.random.normal(kq, (b, qh, d), jnp.float32) * scale
             ).astype(jnp.bfloat16)
        k_pages = jax.random.normal(kk, (np_, ps, kvh, d), jnp.float32
                                    ).astype(jnp.bfloat16)
        v_pages = jax.random.normal(kv, (np_, ps, kvh, d), jnp.float32
                                    ).astype(jnp.bfloat16)
        seq_lens = jax.random.randint(kl, (b,), 1, ml + 1, jnp.int32)
        page_tables = jax.random.randint(kp, (b, mp), 0, np_, jnp.int32)
        return (q, k_pages, v_pages, page_tables, seq_lens)

    def reference(self, q, k_pages, v_pages, page_tables, seq_lens):
        return paged_decode_attention_reference(
            q, k_pages, v_pages, page_tables, seq_lens)

    def baseline(self, *inputs):
        if jax.default_backend() != "tpu":
            raise BaselineUnavailable(
                "vLLM-TPU ragged_paged_attention v3 requires TPU (bound in Phase 2)")
        import sys
        from pathlib import Path

        tpu_inf = (Path(__file__).resolve().parents[4] / "third_party"
                   / "tpu-inference")
        if str(tpu_inf) not in sys.path:
            sys.path.insert(0, str(tpu_inf))
        from tpu_inference.kernels.ragged_paged_attention.v3.kernel import (  # noqa: F401
            ragged_paged_attention,
        )
        raise BaselineUnavailable(
            "v3 kernel layout adapter is a Phase-2 (TPU judge) binding step")

    def adversarial_cases(self):
        tiny = self.case_by_name("tiny")

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
            np.testing.assert_allclose(np.asarray(ref), want, rtol=1e-4,
                                       atol=1e-4)

        return [
            AdversarialCase("single-token-decode", single_token,
                            expect_single_token),
            AdversarialCase("page-boundary-lengths", page_boundary,
                            expect_finite),
            AdversarialCase("max-length", max_length, expect_finite),
            AdversarialCase("softmax-saturating", saturating, expect_finite),
        ]

    def bytes_moved(self, case):
        dm = case.dims
        # dominant term: reading the live KV pages once
        avg_len = dm["max_len"] // 2
        return (dm["batch"] * avg_len * dm["kv_heads"] * dm["head_dim"] * 2 * 2
                + dm["batch"] * dm["q_heads"] * dm["head_dim"] * (2 + 4))


PROBLEM = RaggedPagedAttentionProblem()
