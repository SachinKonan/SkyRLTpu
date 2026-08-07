"""Proposed prompt/harness changes for the three dead tasks, plus the fills
that prove a reasonable model answer can satisfy them.

Everything here is evidence-driven: each `API_ADDENDUM` block names the number
of probe candidates (job 3651278) it would have saved, and each fill is
deliberately written the way a competent model's FIRST attempt looks -- short,
obvious, un-tuned -- not the way the harness author's hand-tuned control fill
looks. If a fill this plain cannot pass, the prompt is still too hard.
"""

from __future__ import annotations

# ==========================================================================
# 1. Additions to prompt_seam.API_BLOCK. Four facts, 37 of the 96 failing
#    splash/GMM/RPA candidates between them, and none of them is currently
#    stated anywhere in the prompt.
# ==========================================================================

API_ADDENDUM = r'''
**`dot_general`'s `dimension_numbers` is a PAIR OF PAIRS, never flat.** 13
candidates on these tasks wrote it flat and died at export with
`too many values to unpack (expected 2)` or `dot_general requires ...`:

```python
#                    (( lhs_contract, rhs_contract ), ( lhs_batch, rhs_batch ))
s = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())),      # q[m,k] . k[n,k] -> [m,n]
                        preferred_element_type=jnp.float32)
o = jax.lax.dot_general(p, v, (((1,), (0,)), ((), ())),      # p[m,n] . v[n,d] -> [m,d]
                        preferred_element_type=jnp.float32)
# WRONG, all of these:  ((1,), (1,))   ((1,), (1,), (0, 0))   ((1,), (0,), ((), ()))
```

**A Ref store is not a broadcast.** The right-hand side must ALREADY have the
Ref's shape. 7 candidates wrote a scalar and got
``Invalid shape for `swap`. Ref shape: (8, 4, 128). ... Value shape: ()``:

```python
m_ref[...] = -jnp.inf                          # WRONG: value shape ()
m_ref[...] = jnp.full_like(m_ref, -jnp.inf)    # right
l_ref[...] = jnp.zeros_like(l_ref)             # right
o_ref[0] = jnp.zeros((kv_heads, group, head_dim), jnp.float32)   # also right
```

**`pltpu.VMEM(shape, dtype)` is a SPECIFICATION, not an allocation.** It is
only legal inside `scratch_shapes=` of a `pallas_call` -- which the harness
already wrote. You cannot allocate scratch inside a kernel body; calling it
there returns a `MemoryRef` object and indexing it fails with
`'MemoryRef' object does not support item assignment` (7 candidates):

```python
acc = pltpu.VMEM((bq, dp), jnp.float32); acc[...] = 0.0   # WRONG
acc = jnp.zeros((bq, dp), jnp.float32)                    # right -- accumulate
acc = acc + block                                         # in ordinary values
```

**`@pl.when(cond)` CONSUMES the function and returns `None`.** Decorate and
stop; never call the decorated name, never use it as a context manager
(4 candidates: `'NoneType' object is not callable`, `'function' object does not
support the context manager protocol`):

```python
@pl.when(p == 0)                 # right
def _init():
    l_ref[...] = jnp.zeros_like(l_ref)
# NOT: _init()      NOT: with pl.when(p == 0): ...
```

**Everything you read out of a Ref is a TRACER.** `if`, `while`, `min`, `max`,
`int()` and `bool()` on one are an immediate `TracerBoolConversionError`
(10 candidates). Use `jnp.where` for values and `@pl.when` for whole blocks.
Python control flow is fine on the *static* kwargs (`bq`, `bk`, `page_size`,
`kv_heads`) -- those are real python ints and `for j in range(kv_len // bk)`
unrolls at trace time exactly as you would expect.
'''

# `rg_lru` is the best task in the arena and still loses 7 of its 19 failures
# to two facts about a signature that is currently stated in PROSE only.
RGLRU_ADDENDUM = r'''
**`lax.associative_scan`'s combine takes TWO PYTREES, not four arrays.** Four
rg_lru candidates died on this (`combine() takes 1 positional argument but 2
were given`, `_assoc_op() missing 2 required positional arguments`):

```python
def combine(left, right):        # left and right are each the tuple (a, g)
    a_l, g_l = left
    a_r, g_r = right
    return a_l * a_r, g_l * a_r + g_r
A, G = jax.lax.associative_scan(combine, (a_chunk, g_chunk), axis=1)
```

**Return `(h_chunk, h_last)` in the harness's layout**, `[b, CHUNK, d]` and
`[b, d]`. Three candidates returned the time axis first and the harness's
carry reshape blew up (`cannot reshape array of shape (256, 4, 2560) into
shape (4, 2560)`). If you scanned over time you must `jnp.moveaxis(hs, 0, 1)`
before returning, and `h_last` is then `h_chunk[:, -1, :]`.
'''

# ==========================================================================
# 2. Plausible first-attempt fills under the improved prompt.
# ==========================================================================

# splash: the seam never says the simple strategy is legal, so every candidate
# tried to write FlashAttention. `k_ref` holds EVERY key of the head (the
# harness's BlockSpec is `(1, sk, dp)` with a constant index map), so a single
# materialized softmax is correct and is ~12 lines. Both splash candidates that
# reached silicon failed on numerics, one by 2% -- so the precision knob has to
# be a directive, not an anecdote.
SPLASH_SIMPLE = '''
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

_NEG = -1e30


def choose_blocks(seq, head_dim):
    # one query block of 256 rows; the whole key axis is resident anyway
    return 256, 512


def attn_block(q_ref, k_ref, v_ref, sq_ref, sk_ref, o_ref, *, bq, bk, kv_len):
    i = pl.program_id(1)
    q = q_ref[0].astype(jnp.float32)          # [bq, dp]
    k = k_ref[0].astype(jnp.float32)          # [kv_len, dp]  -- ALL keys
    v = v_ref[0].astype(jnp.float32)

    s = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())),
                            precision=jax.lax.Precision.HIGHEST,
                            preferred_element_type=jnp.float32)

    pos_q = i * bq + jax.lax.broadcasted_iota(jnp.int32, (bq, 1), 0)
    pos_k = jax.lax.broadcasted_iota(jnp.int32, (1, kv_len), 1)
    seg_q = sq_ref[:, :1]
    seg_k = sk_ref[:1, :]
    vis = (pos_k <= pos_q) & (seg_q == seg_k) & (seg_q != 0)

    s = jnp.where(vis, s, _NEG)
    m = jnp.max(s, axis=1, keepdims=True)
    p = jnp.where(vis, jnp.exp(s - m), 0.0)
    l = jnp.sum(p, axis=1, keepdims=True)
    o = jax.lax.dot_general(p, v, (((1,), (0,)), ((), ())),
                            precision=jax.lax.Precision.HIGHEST,
                            preferred_element_type=jnp.float32)
    o_ref[0] = jnp.where(l > 0, o / jnp.maximum(l, 1e-30), 0.0)
'''

# megablox: gemma's own idx-0 answer, with ONLY `dimension_numbers` un-flattened.
# Nothing else about it was wrong.
GMM_SIMPLE = '''
import jax
import jax.numpy as jnp


def choose_tiles(m, k, n, num_groups):
    return 256, 128, 512


def gmm_tile(l_ref, r_ref, o_ref, *, bm, bk, bn, k):
    lhs = l_ref[...]
    rhs = r_ref[0]
    o_ref[...] = jax.lax.dot_general(
        lhs, rhs, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
    )
'''

# RPA under the CURRENT seam, with only the two facts above applied.
RPA_SIMPLE = '''
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

_NEG = -1e30


def decode_block(q_ref, k_ref, v_ref, o_ref, m_ref, l_ref, sl_ref,
                 *, kv_heads, group, page_size, head_dim, num_pages):
    b = pl.program_id(0)
    p = pl.program_id(1)

    @pl.when(p == 0)
    def _init():
        m_ref[...] = jnp.full_like(m_ref, _NEG)
        l_ref[...] = jnp.zeros_like(l_ref)
        o_ref[...] = jnp.zeros_like(o_ref)

    pos = p * page_size + jax.lax.broadcasted_iota(jnp.int32, (1, page_size), 1)
    live = pos < sl_ref[b]

    for h in range(kv_heads):
        q = q_ref[0, h].astype(jnp.float32)
        k = k_ref[0, :, h, :].astype(jnp.float32)
        v = v_ref[0, :, h, :].astype(jnp.float32)
        s = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())),
                                preferred_element_type=jnp.float32)
        s = jnp.where(live, s, _NEG)
        m_old = m_ref[h, :, :1]
        l_old = l_ref[h, :, :1]
        m_new = jnp.maximum(m_old, jnp.max(s, axis=1, keepdims=True))
        pr = jnp.where(live, jnp.exp(s - m_new), 0.0)
        corr = jnp.exp(m_old - m_new)
        m_ref[h] = jnp.broadcast_to(m_new, (group, 128))
        l_ref[h] = jnp.broadcast_to(corr * l_old + jnp.sum(pr, axis=1, keepdims=True),
                                    (group, 128))
        o_ref[0, h] = corr * o_ref[0, h] + jax.lax.dot_general(
            pr, v, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)

    @pl.when(p == num_pages - 1)
    def _fin():
        for h in range(kv_heads):
            ll = l_ref[h, :, :1]
            o_ref[0, h] = jnp.where(ll > 0, o_ref[0, h] / jnp.maximum(ll, 1e-30), 0.0)
'''

# ==========================================================================
# 3. The NARROWED RPA seam: the harness takes the Refs, the `pl.when` init and
#    the finalize; the model writes ONE PURE FUNCTION over plain arrays. Every
#    failure class in RPA's histogram -- scalar-into-Ref (7), pl.when misuse
#    (3), MemoryRef (0 here but 7 arena-wide) -- becomes unreachable by
#    construction, and the fill drops from ~40 lines to ~15.
# ==========================================================================

RPA_NARROW_SCAFFOLD = r'''
# ===================== HARNESS SCAFFOLD -- appended verbatim =================
# Owns: the pallas_call, the grid, every BlockSpec, the scalar-prefetched page
# lookup, the two persistent scratch Refs AND ALL READS AND WRITES OF THEM, the
# first-page initialisation, the ragged `live` mask and the final normalisation.
# You write one PURE function over ordinary arrays: no Refs, no pl.when, no
# scratch, no pallas_call.
import jax as _jax
import jax.numpy as _jnp
from jax.experimental import pallas as _pl
from jax.experimental.pallas import tpu as _pltpu

_NEG = -1e30


def kernel(q, k_pages, v_pages, page_tables, seq_lens):
    b, qh, d = q.shape
    npg, ps, kvh, _d = k_pages.shape
    mp = page_tables.shape[1]
    grp = qh // kvh
    qg = q.reshape(b, kvh, grp, d)

    def _body(pt_ref, sl_ref, q_ref, k_ref, v_ref, o_ref, m_ref, l_ref):
        bb = _pl.program_id(0)
        p = _pl.program_id(1)

        @_pl.when(p == 0)
        def _init():
            m_ref[...] = _jnp.full_like(m_ref, _NEG)
            l_ref[...] = _jnp.zeros_like(l_ref)
            o_ref[...] = _jnp.zeros_like(o_ref)

        pos = p * ps + _jax.lax.broadcasted_iota(_jnp.int32, (1, ps), 1)
        live = pos < sl_ref[bb]                      # [1, ps] bool

        o_new, m_new, l_new = page_step(
            q_ref[0], k_ref[0], v_ref[0], o_ref[0], m_ref[...], l_ref[...], live,
            kv_heads=kvh, group=grp, page_size=ps, head_dim=d)
        o_ref[0] = o_new.astype(_jnp.float32)
        m_ref[...] = m_new.astype(_jnp.float32)
        l_ref[...] = l_new.astype(_jnp.float32)

        @_pl.when(p == mp - 1)
        def _fin():
            ll = l_ref[:, :, :1]
            o_ref[0] = _jnp.where(ll > 0, o_ref[0] / _jnp.maximum(ll, 1e-30), 0.0)

    out = _pl.pallas_call(
        _body,
        grid_spec=_pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(b, mp),
            in_specs=[
                _pl.BlockSpec((1, kvh, grp, d), lambda bb, p, pt, sl: (bb, 0, 0, 0)),
                _pl.BlockSpec((1, ps, kvh, d), lambda bb, p, pt, sl: (pt[bb, p], 0, 0, 0)),
                _pl.BlockSpec((1, ps, kvh, d), lambda bb, p, pt, sl: (pt[bb, p], 0, 0, 0)),
            ],
            out_specs=_pl.BlockSpec((1, kvh, grp, d), lambda bb, p, pt, sl: (bb, 0, 0, 0)),
            scratch_shapes=[
                _pltpu.VMEM((kvh, grp, 128), _jnp.float32),
                _pltpu.VMEM((kvh, grp, 128), _jnp.float32),
            ],
        ),
        out_shape=_jax.ShapeDtypeStruct((b, kvh, grp, d), _jnp.float32),
        compiler_params=_pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary"),
        ),
    )(page_tables.astype(_jnp.int32), seq_lens.astype(_jnp.int32), qg, k_pages, v_pages)
    return out.reshape(b, qh, d)
'''

# The fill the narrowed seam asks for: pure arrays in, pure arrays out.
RPA_NARROW_FILL = '''
import jax
import jax.numpy as jnp

_NEG = -1e30


def page_step(q, k, v, o, m, l, live, *, kv_heads, group, page_size, head_dim):
    """q [kv_heads, group, d] | k, v [page_size, kv_heads, d]
       o [kv_heads, group, d] | m, l [kv_heads, group, 128] | live [1, page_size]
    returns (o, m, l) with the same shapes."""
    o_out, m_out, l_out = [], [], []
    for h in range(kv_heads):
        qh = q[h].astype(jnp.float32)                 # [group, d]
        kh = k[:, h, :].astype(jnp.float32)           # [page_size, d]
        vh = v[:, h, :].astype(jnp.float32)
        s = jax.lax.dot_general(qh, kh, (((1,), (1,)), ((), ())),
                                preferred_element_type=jnp.float32)
        s = jnp.where(live, s, _NEG)
        m_old = m[h, :, :1]
        l_old = l[h, :, :1]
        m_new = jnp.maximum(m_old, jnp.max(s, axis=1, keepdims=True))
        p = jnp.where(live, jnp.exp(s - m_new), 0.0)
        corr = jnp.exp(m_old - m_new)
        o_h = corr * o[h] + jax.lax.dot_general(
            p, vh, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)
        o_out.append(o_h)
        m_out.append(jnp.broadcast_to(m_new, (group, 128)))
        l_out.append(jnp.broadcast_to(corr * l_old + jnp.sum(p, axis=1, keepdims=True),
                                      (group, 128)))
    return jnp.stack(o_out, 0), jnp.stack(m_out, 0), jnp.stack(l_out, 0)
'''


PROPOSED_FILLS = {
    "splash_attention": [("splash-materialized-softmax", SPLASH_SIMPLE)],
    "megablox_gmm": [("gmm-one-dot", GMM_SIMPLE)],
    "ragged_paged_attention": [("rpa-current-seam-two-facts", RPA_SIMPLE)],
}

NARROWED = {
    "ragged_paged_attention": ("page_step", RPA_NARROW_SCAFFOLD, RPA_NARROW_FILL),
}
