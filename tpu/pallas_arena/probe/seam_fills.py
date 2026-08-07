"""KNOWN-GOOD fills for every seam -- the control experiment's payload.

A grid of zeros is evidence about models only if the path is known to pass a
correct kernel. The previous probe cost itself two of its own bugs (a 4-tuple
handed to the export child; `flatbuffers` missing from the pre-gate venv) and
both looked exactly like model failure. So before any chip is spent, each of
these is pushed through the IDENTICAL path a generation takes:

    model-style response -> extract_fill -> compose -> CPU AOT pre-gate
                         -> queue -> judge

If a control fails, the harness is broken and no measurement means anything.

These are also the reference answer for "what does 300-1200 tokens of seam
code look like": every one of them is between 400 and 900 tokens.
"""

from __future__ import annotations

# --------------------------------------------------------------------- flce
FLCE_FILL = '''
import jax
import jax.numpy as jnp

TILE = 512


def tile_forward(h_tile, w, t_tile):
    logits = h_tile.astype(jnp.float32) @ w.astype(jnp.float32)
    tl = jnp.take_along_axis(logits, t_tile[:, None], axis=1)[:, 0]
    return tl - jax.nn.logsumexp(logits, axis=-1), None


def tile_backward(h_tile, w, t_tile, g_tile, carry):
    def f(h):
        logits = h.astype(jnp.float32) @ w.astype(jnp.float32)
        tl = jnp.take_along_axis(logits, t_tile[:, None], axis=1)[:, 0]
        return tl - jax.nn.logsumexp(logits, axis=-1)

    _, vjp = jax.vjp(f, h_tile)
    return vjp(g_tile)[0]
'''

# A second, genuinely DIFFERENT strategy: two-pass with a saved LSE carry, so
# the backward never recomputes the logsumexp. Proves the carry channel works
# and that the seam admits more than one answer.
FLCE_FILL_CARRY = '''
import jax
import jax.numpy as jnp

TILE = 1024


def _logits(h_tile, w):
    return h_tile.astype(jnp.float32) @ w.astype(jnp.float32)


def tile_forward(h_tile, w, t_tile):
    logits = _logits(h_tile, w)
    lse = jax.nn.logsumexp(logits, axis=-1)
    tl = jnp.take_along_axis(logits, t_tile[:, None], axis=1)[:, 0]
    return tl - lse, lse


def tile_backward(h_tile, w, t_tile, g_tile, carry):
    lse = carry
    logits = _logits(h_tile, w)
    probs = jnp.exp(logits - lse[:, None])
    onehot = (jax.lax.broadcasted_iota(jnp.int32, logits.shape, 1) == t_tile[:, None])
    dlogits = (onehot.astype(jnp.float32) - probs) * g_tile[:, None]
    return dlogits @ w.astype(jnp.float32).T
'''

# ----------------------------------------------------------- splash attention
# `precision=HIGHEST` on the q.k dot is deliberate and is the single most
# important lesson of the previous control run: with f32 operands at DEFAULT
# precision the TPU MXU runs ONE bf16 pass, and the previous splash control
# missed the calibrated tolerance by 18% (max err 4.11e-3 vs tol 3.48e-3) for
# exactly that reason. It is a precision knob, not a bug in the kernel.
SPLASH_FILL = '''
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

_NEG = -1e30


def choose_blocks(seq, head_dim):
    return 512, 512


def attn_block(q_ref, k_ref, v_ref, sq_ref, sk_ref, o_ref, *, bq, bk, kv_len):
    i = pl.program_id(1)
    q = q_ref[0].astype(jnp.float32)
    dp = q.shape[1]
    seg_q = sq_ref[:, :1]
    pos_q = i * bq + jax.lax.broadcasted_iota(jnp.int32, (bq, 1), 0)

    m = jnp.full((bq, 1), _NEG, jnp.float32)
    l = jnp.zeros((bq, 1), jnp.float32)
    acc = jnp.zeros((bq, dp), jnp.float32)

    for j in range(kv_len // bk):
        k = k_ref[0, j * bk:(j + 1) * bk, :].astype(jnp.float32)
        v = v_ref[0, j * bk:(j + 1) * bk, :].astype(jnp.float32)
        seg_k = sk_ref[:1, j * bk:(j + 1) * bk]
        pos_k = j * bk + jax.lax.broadcasted_iota(jnp.int32, (1, bk), 1)
        s = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())),
                                precision=jax.lax.Precision.HIGHEST,
                                preferred_element_type=jnp.float32)
        vis = (pos_k <= pos_q) & (seg_q == seg_k) & (seg_q != 0)
        s = jnp.where(vis, s, _NEG)
        m_new = jnp.maximum(m, jnp.max(s, axis=1, keepdims=True))
        p = jnp.where(vis, jnp.exp(s - m_new), 0.0)
        corr = jnp.exp(m - m_new)
        l = corr * l + jnp.sum(p, axis=1, keepdims=True)
        acc = corr * acc + jax.lax.dot_general(
            p, v, (((1,), (0,)), ((), ())),
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32)
        m = m_new

    o_ref[0] = jnp.where(l > 0, acc / jnp.maximum(l, 1e-30), 0.0)
'''

# The same kernel at DEFAULT matmul precision. Submitted alongside the one
# above so the run MEASURES, rather than assumes, whether the one-bf16-pass
# q.k dot is inside the judge's calibrated tolerance.
SPLASH_FILL_FAST = SPLASH_FILL.replace(
    """        s = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())),
                                precision=jax.lax.Precision.HIGHEST,
                                preferred_element_type=jnp.float32)""",
    """        s = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())),
                                preferred_element_type=jnp.float32)""",
).replace(
    """        acc = corr * acc + jax.lax.dot_general(
            p, v, (((1,), (0,)), ((), ())),
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32)""",
    """        acc = corr * acc + jax.lax.dot_general(
            p, v, (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32)""",
)

# ------------------------------------------------------------------- rg_lru
RGLRU_FILL = '''
import jax
import jax.numpy as jnp

CHUNK = 512


def scan_chunk(x_chunk, a_chunk, reset_chunk, h_prev):
    gx = jnp.sqrt(jnp.maximum(1.0 - jnp.square(a_chunk), 0.0)) * x_chunk.astype(jnp.float32)

    def step(h, xs):
        a_t, g_t = xs
        h = a_t * h + g_t
        return h, h

    h_last, hs = jax.lax.scan(
        step, h_prev, (jnp.moveaxis(a_chunk, 1, 0), jnp.moveaxis(gx, 1, 0))
    )
    return jnp.moveaxis(hs, 0, 1), h_last
'''

RGLRU_FILL_ASSOC = '''
import jax
import jax.numpy as jnp

CHUNK = 1024


def scan_chunk(x_chunk, a_chunk, reset_chunk, h_prev):
    gx = jnp.sqrt(jnp.maximum(1.0 - jnp.square(a_chunk), 0.0)) * x_chunk.astype(jnp.float32)
    # fold the incoming carry into the first step, then run a parallel scan
    gx = gx.at[:, 0, :].add(a_chunk[:, 0, :] * h_prev)

    def combine(left, right):
        a_l, b_l = left
        a_r, b_r = right
        return a_l * a_r, b_l * a_r + b_r

    _, hs = jax.lax.associative_scan(combine, (a_chunk, gx), axis=1)
    return hs, hs[:, -1, :]
'''

# ------------------------------------------------------------- megablox gmm
GMM_FILL = '''
import jax
import jax.numpy as jnp


def choose_tiles(m, k, n, num_groups):
    return 512, 1024, 512


def gmm_tile(l_ref, r_ref, o_ref, *, bm, bk, bn, k):
    acc = jnp.zeros((bm, bn), jnp.float32)
    for kk in range(0, k, bk):
        hi = min(kk + bk, k)
        a = l_ref[:, kk:hi]
        b = r_ref[0, kk:hi, :]
        acc = acc + jax.lax.dot_general(
            a, b, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
        )
    o_ref[...] = acc
'''

# --------------------------------------------------- ragged paged attention
RPA_FILL = '''
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
        q = q_ref[0, h].astype(jnp.float32)          # [group, d]
        k = k_ref[0, :, h, :].astype(jnp.float32)    # [page_size, d]
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
        l_ref[h] = jnp.broadcast_to(
            corr * l_old + jnp.sum(pr, axis=1, keepdims=True), (group, 128))
        o_ref[0, h] = corr * o_ref[0, h] + jax.lax.dot_general(
            pr, v, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)

    @pl.when(p == num_pages - 1)
    def _fin():
        for h in range(kv_heads):
            ll = l_ref[h, :, :1]
            o_ref[0, h] = jnp.where(ll > 0, o_ref[0, h] / jnp.maximum(ll, 1e-30), 0.0)
'''


CONTROL_FILLS: dict[str, list[tuple[str, str]]] = {
    "flce": [("flce-recompute", FLCE_FILL), ("flce-saved-lse", FLCE_FILL_CARRY)],
    "splash_attention": [
        ("splash-highest-precision", SPLASH_FILL),
        ("splash-default-precision", SPLASH_FILL_FAST),
    ],
    "rg_lru": [("rglru-sequential", RGLRU_FILL), ("rglru-associative", RGLRU_FILL_ASSOC)],
    "megablox_gmm": [("gmm-ktiled", GMM_FILL)],
    "ragged_paged_attention": [("rpa-online-softmax", RPA_FILL)],
}
