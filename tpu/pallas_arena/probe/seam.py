"""The SEAM: the harness owns the interface, the model owns the strategy.

The previous probe measured two things that together define this file:

  * 47% of candidates died at the static gate and 14% emitted no code at all
    -- models flail on SCAFFOLDING, not on algorithms. Every gemma splash cell
    was 0/16 at export because the model wrote a genuine `pallas_call` and
    then invented its signature (`out_spec`, `out_dtype`, `jax.Shape`,
    `pl.Ref`). Prose gotchas did not fix it.
  * the `tailored` fill-in-the-blank variant passed 16/16 and its within-group
    reward spread was 0.0042 -- BELOW the judge's own measured noise floors
    (0.0069 / 0.0158). Every candidate converged on the same kernel, so there
    was no gradient at all. A prompt can be too good.

So a seam is cut between the two: the harness supplies every structural
decision that is *not* the task (the `pallas_call` and its `BlockSpec`s, the
padding, the `custom_vjp` plumbing, the residual contract, the scan driver),
and the model supplies every decision that *is* (tile and block shapes, one-
pass vs two-pass, online vs materialized softmax, accumulation dtype,
recompute vs save, which blocks to skip). The model writes 300-1200 tokens of
real numerics instead of 2000 tokens of boilerplate it gets wrong.

Mechanically:

    model text --extract_fill--> fill-in source
    fill-in source + SCAFFOLD  --compose--> a complete program defining `kernel`

so the JUDGE CONTRACT IS UNCHANGED: the submitted program is still a single
self-contained module with a module-level `kernel`, and the judge cannot tell
it was assembled. The scaffold is appended AFTER the model's code, which means
the scaffold's definition of `kernel` always wins even if the model pastes a
whole program back.

The scaffold string is the SINGLE SOURCE OF TRUTH: `prompt_seam.py` embeds
exactly this text in the prompt, so what the model is shown is byte-identical
to what it will be wrapped in. A test asserts that.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# The scaffolds. Private names are `_`-prefixed so a model that also imports
# `jax`/`jnp`, or defines a helper called `kernel`, cannot break the harness.


FLCE_SCAFFOLD = r'''
# ===================== HARNESS SCAFFOLD -- appended verbatim =================
# Owns: the token-axis tiling driver, the pad/un-pad arithmetic, the
# jax.custom_vjp registration and the residual contract (residual = the hidden
# tiles plus whatever `carry` you return -- NEVER the logits).
import jax as _jax
import jax.numpy as _jnp


def kernel(hidden, w, targets):
    n, hdim = hidden.shape
    tile = max(1, min(int(TILE), n))
    num_tiles = -(-n // tile)                 # ceil
    pad = num_tiles * tile - n
    flat = _jnp.pad(hidden, ((0, pad), (0, 0))) if pad else hidden
    tgt = _jnp.pad(targets, ((0, pad),)) if pad else targets
    hf = flat.reshape(num_tiles, tile, hdim)  # [T, tile, h]
    tf = tgt.reshape(num_tiles, tile)         # [T, tile]

    def _fwd_body(c, xs):
        lp, carry = tile_forward(xs[0], w, xs[1])
        return c, (_jnp.asarray(lp).astype(_jnp.float32), carry)

    @_jax.custom_vjp
    def _tiled(hf):
        _, (lp, _c) = _jax.lax.scan(_fwd_body, None, (hf, tf))
        return lp

    def _tiled_fwd(hf):
        _, (lp, carry) = _jax.lax.scan(_fwd_body, None, (hf, tf))
        return lp, (hf, carry)                # residual: hidden tiles + carry

    def _tiled_bwd(res, cot):
        hf_res, carry = res

        def _bwd_body(c, xs):
            h_tile, t_tile, g_tile, cr = xs
            gh = tile_backward(h_tile, w, t_tile, g_tile, cr)
            # the cotangent of a bf16 primal must be bf16: the harness casts so
            # a correct-but-f32 gradient is not a plumbing failure
            return c, _jnp.asarray(gh).astype(h_tile.dtype).reshape(h_tile.shape)

        _, grad_hf = _jax.lax.scan(_bwd_body, None, (hf_res, tf, cot, carry))
        return (grad_hf,)

    _tiled.defvjp(_tiled_fwd, _tiled_bwd)
    return _tiled(hf).reshape(num_tiles * tile)[:n]
'''


SPLASH_SCAFFOLD = r'''
# ===================== HARNESS SCAFFOLD -- appended verbatim =================
# Owns: the ENTIRE pallas_call -- grid, every BlockSpec and index_map, the
# out_shape, the explicit padding of both the query axis and the key axis, and
# the final un-pad slice. You never write a pallas_call yourself.
import jax as _jax
import jax.numpy as _jnp
from jax.experimental import pallas as _pl
from jax.experimental.pallas import tpu as _pltpu

_LANES = 128
_SUBLANES = 8


def _ru(x, m):
    return ((x + m - 1) // m) * m


def _blocks(seq, head_dim):
    bq, bk = choose_blocks(int(seq), int(head_dim))
    # sanitized to the TPU native tiling so an unlucky choice is a slow kernel,
    # never a compile error: bq to 8 sublanes, bk to 128 lanes, both clamped.
    bq = min(max(_ru(int(bq), _SUBLANES), _SUBLANES), 2048)
    bk = min(max(_ru(int(bk), _LANES), _LANES), 4096)
    return bq, bk


def kernel(q, k, v, segment_ids):
    h, s, d = q.shape
    bq, bk = _blocks(s, d)
    dp = _ru(d, _LANES)
    sq = _ru(s, bq)          # padded QUERY-axis length (a multiple of bq)
    sk = _ru(s, bk)          # padded KEY-axis length   (a multiple of bk)

    def _pad(x, ts):
        return _jnp.pad(x, ((0, 0), (0, ts - x.shape[1]), (0, dp - d)))

    qp = _pad(q, sq)
    kp = _pad(k, sk)
    vp = _pad(v, sk)
    # padded positions get segment id 0 == PADDING, so the mask kills them free
    sgq = _jnp.pad(segment_ids, (0, sq - s))
    sgk = _jnp.pad(segment_ids, (0, sk - s))
    seg_q = _jnp.broadcast_to(sgq[:, None], (sq, _LANES))
    seg_k = _jnp.broadcast_to(sgk[None, :], (_SUBLANES, sk))

    def _body(q_ref, k_ref, v_ref, sq_ref, sk_ref, o_ref):
        attn_block(q_ref, k_ref, v_ref, sq_ref, sk_ref, o_ref,
                   bq=bq, bk=bk, kv_len=sk)

    out = _pl.pallas_call(
        _body,
        grid=(h, sq // bq),
        in_specs=[
            _pl.BlockSpec((1, bq, dp), lambda hh, i: (hh, i, 0)),   # q  block
            _pl.BlockSpec((1, sk, dp), lambda hh, i: (hh, 0, 0)),   # ALL keys
            _pl.BlockSpec((1, sk, dp), lambda hh, i: (hh, 0, 0)),   # ALL values
            _pl.BlockSpec((bq, _LANES), lambda hh, i: (i, 0)),      # seg_q
            _pl.BlockSpec((_SUBLANES, sk), lambda hh, i: (0, 0)),   # seg_k
        ],
        out_specs=_pl.BlockSpec((1, bq, dp), lambda hh, i: (hh, i, 0)),
        out_shape=_jax.ShapeDtypeStruct((h, sq, dp), _jnp.float32),
        compiler_params=_pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary"),
        ),
    )(qp, kp, vp, seg_q, seg_k)
    return out[:, :s, :d]
'''


RGLRU_SCAFFOLD = r'''
# ===================== HARNESS SCAFFOLD -- appended verbatim =================
# Owns: the outer entrypoint, the time-axis chunking and its padding, the
# carry-state contract between chunks, and the reset contract (a_t is forced to
# 0 wherever reset_t is True, BEFORE you see it, so no state crosses a segment
# boundary).
import jax as _jax
import jax.numpy as _jnp


def kernel(x, a, reset):
    b, t, d = x.shape
    chunk = max(1, min(int(CHUNK), t))
    nch = -(-t // chunk)
    pad = nch * chunk - t
    if pad:
        x = _jnp.pad(x, ((0, 0), (0, pad), (0, 0)))
        a = _jnp.pad(a, ((0, 0), (0, pad), (0, 0)))
        reset = _jnp.pad(reset, ((0, 0), (0, pad)))
    a_eff = a * (1.0 - reset[..., None].astype(a.dtype))   # reset contract

    xs = _jnp.moveaxis(x.reshape(b, nch, chunk, d), 1, 0)      # [C, b, chunk, d]
    as_ = _jnp.moveaxis(a_eff.reshape(b, nch, chunk, d), 1, 0)  # [C, b, chunk, d]
    rs = _jnp.moveaxis(reset.reshape(b, nch, chunk), 1, 0)      # [C, b, chunk]

    def _body(h_prev, ins):
        hc, h_last = scan_chunk(ins[0], ins[1], ins[2], h_prev)
        return (_jnp.asarray(h_last).astype(_jnp.float32).reshape(b, d),
                _jnp.asarray(hc).astype(_jnp.float32).reshape(b, chunk, d))

    _, hs = _jax.lax.scan(_body, _jnp.zeros((b, d), _jnp.float32), (xs, as_, rs))
    return _jnp.moveaxis(hs, 0, 1).reshape(b, nch * chunk, d)[:, :t, :]
'''


GMM_SCAFFOLD = r'''
# ===================== HARNESS SCAFFOLD -- appended verbatim =================
# Owns: the exclusive scan of group_sizes to row offsets, the group-ALIGNED row
# permutation (so no row tile ever straddles two experts -- the zero-size-group
# and boundary-tile correctness traps are the harness's, not yours), the
# pallas_call with scalar-prefetched per-tile expert ids, every BlockSpec, and
# the un-permute of the result.
import jax as _jax
import jax.numpy as _jnp
from jax.experimental import pallas as _pl
from jax.experimental.pallas import tpu as _pltpu


def _ru(x, m):
    return ((x + m - 1) // m) * m


def _tiles(m, k, n, g):
    bm, bk, bn = choose_tiles(int(m), int(k), int(n), int(g))
    bm = min(max(_ru(int(bm), 8), 8), 1024)
    bk = min(max(_ru(int(bk), 128), 128), int(_ru(k, 128)))
    bn = min(max(_ru(int(bn), 128), 128), 2048)
    return bm, bk, bn


def kernel(lhs, rhs, group_sizes):
    m, k = lhs.shape
    g, _k, n = rhs.shape
    bm, bk, bn = _tiles(m, k, n, g)
    np_ = _ru(n, bn)
    # static upper bound on the group-aligned row count: every expert wastes at
    # most bm-1 rows, and a zero-size expert wastes none.
    mp = _ru(m, bm) + g * bm
    n_row_tiles = mp // bm

    sizes = group_sizes.astype(_jnp.int32)
    starts = _jnp.concatenate([_jnp.zeros((1,), _jnp.int32), _jnp.cumsum(sizes)[:-1]])
    psizes = ((sizes + bm - 1) // bm) * bm                      # aligned sizes
    pstarts = _jnp.concatenate([_jnp.zeros((1,), _jnp.int32), _jnp.cumsum(psizes)[:-1]])

    # per PADDED row: which expert owns it, and which source row it is
    prow = _jnp.arange(mp, dtype=_jnp.int32)
    grp = _jnp.sum(prow[:, None] >= pstarts[None, :], axis=1) - 1   # [mp]
    grp = _jnp.clip(grp, 0, g - 1)
    off = prow - pstarts[grp]
    live = (off < sizes[grp]) & (prow < _jnp.sum(psizes))
    src = _jnp.where(live, _jnp.clip(starts[grp] + off, 0, m - 1), 0)

    lhs_p = _jnp.where(live[:, None], _jnp.take(lhs, src, axis=0), 0).astype(lhs.dtype)
    rhs_p = _jnp.pad(rhs, ((0, 0), (0, 0), (0, np_ - n))) if np_ != n else rhs
    # expert id of each ROW TILE: constant within a tile, by construction
    tile_grp = grp[::bm].astype(_jnp.int32)                     # [n_row_tiles]

    def _body(tg_ref, l_ref, r_ref, o_ref):
        gmm_tile(l_ref, r_ref, o_ref, bm=bm, bk=bk, bn=bn, k=k)

    out_p = _pl.pallas_call(
        _body,
        grid_spec=_pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(n_row_tiles, np_ // bn),
            in_specs=[
                _pl.BlockSpec((bm, k), lambda i, j, tg: (i, 0)),          # lhs tile
                _pl.BlockSpec((1, k, bn), lambda i, j, tg: (tg[i], 0, j)),  # expert
            ],
            out_specs=_pl.BlockSpec((bm, bn), lambda i, j, tg: (i, j)),
        ),
        out_shape=_jax.ShapeDtypeStruct((mp, np_), _jnp.float32),
        compiler_params=_pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary"),
        ),
    )(tile_grp, lhs_p, rhs_p)

    # un-permute: original row r lives at padded row pstarts[grp_of_r] + (r - starts)
    row = _jnp.arange(m, dtype=_jnp.int32)
    rg = _jnp.sum(row[:, None] >= starts[None, :], axis=1) - 1
    rg = _jnp.clip(rg, 0, g - 1)
    dst = _jnp.clip(pstarts[rg] + (row - starts[rg]), 0, mp - 1)
    return _jnp.take(out_p[:, :n], dst, axis=0)
'''


RPA_SCAFFOLD = r'''
# ===================== HARNESS SCAFFOLD -- appended verbatim =================
# Owns: the pallas_call, the grid (sequence, page), every BlockSpec, and the
# page-id -> page-slice helper -- the scalar-prefetched `page_tables` in the k/v
# index maps IS that helper, so a grid step is handed the physical page already
# resident in VMEM and you never gather a page yourself.
#
# The kv-head axis is deliberately NOT in the grid. Mosaic requires a block's
# second-minor dimension to be a multiple of 8 or equal to the array's, and
# k_pages is [num_pages, page_size, kv_heads, head_dim] -- so a per-head block
# of [1, ps, 1, d] is rejected at compile time (measured: `block shape are
# divisible by 8 and 128 respectively, or be equal to the respective dimensions
# of the overall array`). One grid step therefore holds ALL kv heads of one
# page, which is also 8x fewer grid steps.
import jax as _jax
import jax.numpy as _jnp
from jax.experimental import pallas as _pl
from jax.experimental.pallas import tpu as _pltpu


def kernel(q, k_pages, v_pages, page_tables, seq_lens):
    b, qh, d = q.shape
    npg, ps, kvh, _d = k_pages.shape
    mp = page_tables.shape[1]
    grp = qh // kvh
    qg = q.reshape(b, kvh, grp, d)

    def _body(pt_ref, sl_ref, q_ref, k_ref, v_ref, o_ref, m_ref, l_ref):
        decode_block(q_ref, k_ref, v_ref, o_ref, m_ref, l_ref, sl_ref,
                     kv_heads=kvh, group=grp, page_size=ps, head_dim=d,
                     num_pages=mp)

    out = _pl.pallas_call(
        _body,
        grid_spec=_pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(b, mp),
            in_specs=[
                # every query head of this sequence: [1, kv_heads, group, d]
                _pl.BlockSpec((1, kvh, grp, d), lambda bb, p, pt, sl: (bb, 0, 0, 0)),
                # ONE physical page of K, chosen by the page table, all heads
                _pl.BlockSpec((1, ps, kvh, d), lambda bb, p, pt, sl: (pt[bb, p], 0, 0, 0)),
                _pl.BlockSpec((1, ps, kvh, d), lambda bb, p, pt, sl: (pt[bb, p], 0, 0, 0)),
            ],
            out_specs=_pl.BlockSpec((1, kvh, grp, d), lambda bb, p, pt, sl: (bb, 0, 0, 0)),
            scratch_shapes=[
                _pltpu.VMEM((kvh, grp, 128), _jnp.float32),  # running row max
                _pltpu.VMEM((kvh, grp, 128), _jnp.float32),  # running row sum
            ],
        ),
        out_shape=_jax.ShapeDtypeStruct((b, kvh, grp, d), _jnp.float32),
        compiler_params=_pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary"),
        ),
    )(page_tables.astype(_jnp.int32), seq_lens.astype(_jnp.int32), qg, k_pages, v_pages)
    return out.reshape(b, qh, d)
'''


# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Seam:
    task: str
    scaffold: str
    required: tuple[str, ...]  # names the model MUST define
    constants: tuple[str, ...] = ()  # of `required`, which are plain constants
    notes: str = ""
    aliases: dict[str, str] = field(default_factory=dict)


SEAMS: dict[str, Seam] = {
    "flce": Seam(
        task="flce",
        scaffold=FLCE_SCAFFOLD,
        required=("TILE", "tile_forward", "tile_backward"),
        constants=("TILE",),
    ),
    "splash_attention": Seam(
        task="splash_attention",
        scaffold=SPLASH_SCAFFOLD,
        required=("choose_blocks", "attn_block"),
    ),
    "rg_lru": Seam(
        task="rg_lru",
        scaffold=RGLRU_SCAFFOLD,
        required=("CHUNK", "scan_chunk"),
        constants=("CHUNK",),
    ),
    "megablox_gmm": Seam(
        task="megablox_gmm",
        scaffold=GMM_SCAFFOLD,
        required=("choose_tiles", "gmm_tile"),
    ),
    "ragged_paged_attention": Seam(
        task="ragged_paged_attention",
        scaffold=RPA_SCAFFOLD,
        required=("decode_block",),
    ),
}

TASKS = tuple(SEAMS)

_CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def _defines(src: str) -> set[str]:
    """Top-level names bound by this source, or () if it does not parse."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set()
    out: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.add(node.target.id)
    return out


def extract_fill(text: str, required) -> tuple[str, str, list[str]]:
    """(fill_source, how, missing_names).

    A seam answer is SHORT and may legitimately arrive as several fenced
    blocks (one per function). `extract_program` takes only the LAST block, so
    it would silently drop `tile_forward` when the model writes it in its own
    block. But naively concatenating every block is wrong in the other
    direction: models routinely show a REJECTED sketch first, and the control
    caught exactly that -- a decoy `def tile_forward: raise NotImplementedError`
    was being carried along with the real answer.

    So: prefer the LAST single block that binds EVERY required name (the model
    answered in one block, and any earlier draft is discarded); only if no
    single block is complete, concatenate the blocks that contribute, in order,
    so a later redefinition still wins.
    """
    required = list(required)
    blocks = [b.strip() for b in _CODE_BLOCK.findall(text)]
    if not blocks:
        m = re.search(r"```(?:python|py)?\s*\n(.*)$", text, re.S)
        if m and "def " in m.group(1):
            blocks, how = [m.group(1).strip()], "unterminated-fence"
        elif re.search(r"^\s*(import |from |def |[A-Z_]+ *=)", text, re.M):
            blocks, how = [text.strip()], "bare"
        else:
            return "", "none", list(required)
    else:
        how = "fenced"

    want = set(required)
    complete = [b for b in blocks if want <= _defines(b)]
    if complete:
        src, how = complete[-1], how + "-single"
    else:
        keep = [b for b in blocks if _defines(b) & want]
        if keep:
            src = "\n\n".join(keep)
            how = f"{how}-multi" if len(keep) > 1 else how
            if not _defines(src):  # concatenation broke parsing; take the best
                best = max(keep, key=lambda b: len(_defines(b) & want))
                src, how = best, how + "-fallback"
        else:
            src, how = blocks[-1], how + "-nomatch"
    return src, how, [r for r in required if r not in _defines(src)]


def compose(task: str, fill_source: str) -> str:
    """fill + scaffold -> one self-contained program defining `kernel`.

    The scaffold goes LAST on purpose: a model that pastes an entire program
    back (including its own `kernel`) is harmless, because the harness's
    definition is the one that survives.
    """
    seam = SEAMS[task]
    return fill_source.rstrip() + "\n\n" + seam.scaffold.strip() + "\n"


def scaffold_of(task: str) -> str:
    return SEAMS[task].scaffold.strip()
