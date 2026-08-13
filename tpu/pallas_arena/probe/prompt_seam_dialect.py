"""SEAM + DIALECT (`sd1`) and SEAM + DIALECT + TYPED SKELETON (`sd2`).

The experiment the prompt ladder pointed at, for the three kernels that have
never produced a single working candidate in this arena -- splash_attention,
ragged_paged_attention, megablox_gmm (0 PASS out of 96 each, at all three
ladder rungs, every one of them dying at `aot_export`).

The three measured facts these prompts are built from, and nothing else:

  * **The seam is the only variant that has ever exported splash** (0/16
    reference -> 2/16 seam, job 3651278). Those three kernels fail at the
    `pallas_call` PLUMBING -- grid, BlockSpecs, index maps -- which no
    whole-program rung supplies by design. So the plumbing is handed over and
    the model writes one strategy function.

  * **The P1 DIALECT list is the only addition that ever cleanly removed a
    failure class.** Across 480 ladder candidates it took `'MemoryRef' object
    does not support item assignment` from 7 to 0, `` Invalid shape for `swap` ``
    from 7 to 0, `custom_vjp.def_fwd` from 1 to 0, and held `no_code` at 0/480.
    Two of those are *precisely* the seam's own modal failures (SEAM-REPORT §5:
    7x MemoryRef, 6x `Invalid shape for swap. Ref shape: (8, 4, 128)`), so it
    is carried here bullet-for-bullet.

  * **The P3 worked example BACKFIRED, and the reason is RANK.** It removed
    splash's invented `pallas_call` kwargs (10 -> 0) and replaced them with 60
    candidates writing a 2-D `BlockSpec` against a 3-D operand, because the
    example was an RMSNorm over `[rows, cols]` and models copied its rank along
    with its dialect. So `sd2` ships no foreign worked example. Its skeleton is
    of the task's OWN fill function and every line of it is a shape: the rank
    and extent of each Ref read, and the exact shape the output store must
    have. A skeleton written against the real rank cannot transfer a wrong one.

What `sd2` still does NOT decide, because that is the `tailored` trap (16/16
PASS with a reward spread of 0.0042 against noise floors of 0.0069/0.0158, i.e.
no gradient at all): block and tile sizes, key/k chunking, which blocks or
pages to skip, and online vs two-pass softmax. Every `...` below is one of
those.

Accumulation dtype used to be on that list, framed as a trade. It is not one.
Measured across all five tasks, the production float32-accumulator path sits at
0.4-0.7x of the calibrated tolerance and the default-dtype (bfloat16
accumulator) path at 1.4-4.1x, i.e. the "choice" was between passing and
failing correctness. It is now stated as a rule at the end of the dialect list,
which is where the wall moved to once the API failure class went to 0 of 192.

--------------------------------------------------------------------------
TOKEN BUDGET -- why the seam's API block is replaced rather than appended to.

gemma is served at 16384 and the driver clamps the completion to whatever is
left. MEASURED over the ladder run's 480 generations: the median generation is
~8100 tokens and 12 of 480 hit `length` against a 12000 budget, so a prompt
that leaves under ~11000 tokens starts destroying candidates for a reason that
has nothing to do with kernels. The seam prompt alone is already ~4750 tokens;
DIALECT is another ~1300 and a skeleton ~450.

So the seam's `API_BLOCK` (~1400 tokens) is REPLACED by the two things in it
that DIALECT does not already say -- the introspected list of the calls you
make inside a kernel body, sliced verbatim out of API_BLOCK itself -- and
DIALECT's two bullets that cannot apply to a fill (the model never writes a
`pallas_call`, and none of these three tasks has a backward) are condensed to
one line each. Everything else is byte-identical to the measured original, and
the slices are asserted at import so they cannot drift.
"""

from __future__ import annotations

from pallas_arena.probe.prompt_ladder import DIALECT
from pallas_arena.probe.prompt_seam import _OUTPUT, API_BLOCK, SEAM_PROMPTS

# ==========================================================================
# 1. The dialect, scoped to a fill answer. Bullets 2-9 are sliced VERBATIM.
# ==========================================================================

_B2 = DIALECT.index("2. **Inside the kernel body")
_B10 = DIALECT.index("10. **`jax.custom_vjp`")
_CLOSER = DIALECT.index("One more, and unlike the rest")

_BULLETS_2_TO_9 = DIALECT[_B2:_B10]
_PRECISION_TRADE = DIALECT[_CLOSER:]

_DIALECT_HEAD = r'''
## Pallas dialect: the mistakes that actually killed candidates here

These are measured. Each line is the verbatim error and how many of the
previous attempts at these five kernels died on it.

1. **Names that DO NOT EXIST at this pin**, every one of them invented by a
   model on these exact tasks: `pl.load`, `pl.store`, `pl.swap`, `pl.Ref`,
   `pltpu.ANY`, `pltpu.TPUCompilerParams`, `jax.Shape`. A Ref is read and
   written by INDEXING, not by a load/store function. (The `pallas_call`
   keyword arguments that killed 4 more are the harness's problem here, not
   yours -- you do not write one.)
'''

_BULLET_10 = r'''10. **You return VALUES, not Refs, and exactly the arity the scaffold
    unpacks.** Killed 5: `too many values to unpack (expected 2)`, and 3 more
    on `not enough values to unpack`. Read the scaffold above for what it does
    with your return.

'''

DIALECT_SEAM = _DIALECT_HEAD + "\n" + _BULLETS_2_TO_9 + _BULLET_10 + _PRECISION_TRADE

# ==========================================================================
# 2. The calls you make inside a kernel body -- sliced verbatim out of the
#    introspected API_BLOCK (seam_dryrun.py re-derives every one of these
#    signatures with inspect.signature at the pin and fails if one drifted).
# ==========================================================================

_A0 = API_BLOCK.index("```python\n# --- YOU CALL these")
_A1 = API_BLOCK.index("```", _A0 + 10) + 4

_CALLS = (
    "\n## The calls you make inside the function you write"
    "\n\n(introspected from the pinned jax 0.10.2, not remembered)\n\n"
    + API_BLOCK[_A0:_A1]
)

# ==========================================================================
# 3. The typed skeletons (sd2 only). Of the FILL functions, never of the
#    entrypoint -- the scaffold already owns that. Ranks are the task's own.
# ==========================================================================

_SKEL_INTRO = '''
## A typed skeleton of exactly the functions you must write

Every shape below is the real one at this seam. Every `...` is a decision that
is yours and that moves the number the judge measures; none is filled in.

'''

_SKELETONS: dict[str, str] = {
    "splash_attention": _SKEL_INTRO + r'''```python
import jax, jax.numpy as jnp
from jax.experimental import pallas as pl   # you must import what you use

def choose_blocks(seq, head_dim):
    return ..., ...                  # (bq, bk). YOUR choice.

def attn_block(q_ref, k_ref, v_ref, sq_ref, sk_ref, o_ref, *, bq, bk, kv_len):
    i  = pl.program_id(1)                    # this query block
    dp = q_ref.shape[2]                      # head_dim padded to 128 (python int)
    q     = q_ref[0]                         # [bq, dp]  bf16 -- cast if you want f32
    seg_q = sq_ref[:, :1]                    # [bq, 1]   int32
    pos_q = i * bq + jax.lax.broadcasted_iota(jnp.int32, (bq, 1), 0)   # [bq, 1]
    ...   # accumulators are ORDINARY VALUES in python variables, e.g. a flash
          # softmax wants m [bq,1], l [bq,1], acc [bq,dp] -- never a VMEM scratch
    # for j in range(kv_len // bk):          # python ints -> unrolls at trace time
    #     k_j   = k_ref[0, j*bk:(j+1)*bk, :]        # [bk, dp] bf16
    #     v_j   = v_ref[0, j*bk:(j+1)*bk, :]        # [bk, dp] bf16
    #     seg_k = sk_ref[:1, j*bk:(j+1)*bk]         # [1, bk]  int32
    #     pos_k = j*bk + jax.lax.broadcasted_iota(jnp.int32, (1, bk), 1)  # [1, bk]
    #     ...                                        # YOUR numerics
    # i and j are both known at trace time, so a fully-masked block need not be
    # emitted at all. Whether to skip is yours.
    o_ref[0] = ...       # EXACTLY [bq, dp] float32 (o_ref itself is [1, bq, dp])
```
''',
    "ragged_paged_attention": _SKEL_INTRO + r'''```python
import jax, jax.numpy as jnp
from jax.experimental import pallas as pl   # you must import what you use

def decode_block(q_ref, k_ref, v_ref, o_ref, m_ref, l_ref, sl_ref,
                 *, kv_heads, group, page_size, head_dim, num_pages):
    b, p = pl.program_id(0), pl.program_id(1)      # sequence, page
    # o_ref/m_ref/l_ref PERSIST across p, so page 0 initialises them, and a
    # store must already have the Ref's own shape:
    # @pl.when(p == 0)
    # def _init():
    #     m_ref[...] = jnp.full(m_ref.shape, -1e30, m_ref.dtype)  # [kv_heads,group,128]
    #     l_ref[...] = jnp.zeros(l_ref.shape, l_ref.dtype)
    #     o_ref[...] = jnp.zeros(o_ref.shape, o_ref.dtype)        # [1,kv_heads,group,d]
    pos  = p * page_size + jax.lax.broadcasted_iota(jnp.int32, (1, page_size), 1)
    live = pos < sl_ref[b]                          # [1, page_size] bool
    ...   # mask the tail, or skip whole pages with @pl.when -- YOUR choice
    # kv_heads and group are PYTHON ints, so `for h in range(kv_heads)` unrolls:
    #     q_h = q_ref[0, h]         # [group, d]      bf16
    #     k_h = k_ref[0, :, h, :]   # [page_size, d]  bf16
    #     v_h = v_ref[0, :, h, :]   # [page_size, d]  bf16
    #     m_h = m_ref[h, :, :1]     # [group, 1] f32  (the 128 lanes hold copies)
    #     o_h = o_ref[0, h]         # [group, d] f32
    # and the stores back are whole slots:
    #     m_ref[h]    = jnp.broadcast_to(m_new, (group, 128))
    #     o_ref[0, h] = ...                              # EXACTLY [group, d] f32
    # Doing all 32 query heads at once against the [page_size, kv_heads, d]
    # block is equally legal and is not the same speed.
    # @pl.when(p == num_pages - 1)  -> where a running softmax gets normalised
```
''',
    "megablox_gmm": _SKEL_INTRO + r'''```python
import jax, jax.numpy as jnp                # you must import what you use

def choose_tiles(m, k, n, num_groups):
    return ..., ..., ...      # (bm, bk, bn). bm/bn set the grid and the VMEM;
                              # bk is internal to gmm_tile only.

def gmm_tile(l_ref, r_ref, o_ref, *, bm, bk, bn, k):
    # l_ref [bm, k] bf16   r_ref [1, k, bn] bf16   o_ref [bm, bn] f32
    # bm, bk, bn, k are PYTHON ints, so any loop over them unrolls.
    ...
    # One [bm, k] x [k, bn] contraction, or bk-wide chunks accumulated in f32:
    # acc = jnp.zeros((bm, bn), jnp.float32)
    # for kk in range(0, k, bk):
    #     a = l_ref[:, kk:min(kk + bk, k)]        # mind the tail
    #     w = r_ref[0, kk:min(kk + bk, k), :]     # [<=bk, bn]
    #     ...                                     # YOUR numerics and precision
    o_ref[...] = ...          # EXACTLY [bm, bn] float32
```
''',
    "rg_lru": _SKEL_INTRO + r'''```python
CHUNK = ...             # timesteps per chunk. A PYTHON INT at module level.

def scan_chunk(x_chunk, a_chunk, reset_chunk, h_prev):
    # x_chunk [b,CHUNK,d] bf16 | a_chunk [b,CHUNK,d] f32, reset already applied
    # reset_chunk [b,CHUNK] bool | h_prev [b,d] f32
    gx = jnp.sqrt(jnp.maximum(1.0 - jnp.square(a_chunk), 0.0)) * x_chunk.astype(jnp.float32)
    ...                 # sequential lax.scan, or associative_scan with h_prev
                        # folded into the first step. YOUR choice.
    return h_chunk, h_last      # [b, CHUNK, d] float32, and [b, d] float32
```
''',
    "flce": _SKEL_INTRO + r'''```python
TILE = ...              # tokens per tile. A PYTHON INT at module level.

def tile_forward(h_tile, w, t_tile):
    # h_tile [tile, hdim] bf16 | w [hdim, v] bf16 | t_tile [tile] int32
    ...
    return logprobs, carry      # [tile] float32, plus anything the backward
                                # needs that is NOT a [tile, v] array

def tile_backward(h_tile, w, t_tile, g_tile, carry):
    # g_tile [tile] float32 is the cotangent of this tile's logprobs
    ...
    return grad_h               # EXACTLY h_tile's shape
```
''',
}


# ==========================================================================
# 4. Assembly. The seam's output contract stays LAST -- a model reads the final
#    instruction best -- and both splices are asserted, not assumed.
# ==========================================================================


def _base(task: str) -> str:
    """The seam prompt with its API block swapped for the in-body call list."""
    base = SEAM_PROMPTS[task]
    if base.count(API_BLOCK) != 1:
        raise AssertionError(f"{task}: seam prompt does not carry API_BLOCK exactly once")
    if not base.endswith(_OUTPUT):
        raise AssertionError(f"{task}: seam prompt no longer ends with its output contract")
    # the seam body points at "the API section" for the precision trade; that
    # section is gone and the trade now lives at the end of the dialect list.
    return base.replace(API_BLOCK, _CALLS).replace(
        "See the precision note in the\n  API section",
        "See the precision note at the end\n  of the dialect list",
    )


TASKS = tuple(SEAM_PROMPTS)
RUNGS = ("sd1", "sd2")


def build(task: str, rung: str) -> str:
    if rung not in RUNGS:
        raise KeyError(f"unknown seam-dialect rung {rung!r}")
    base = _base(task)
    extra = DIALECT_SEAM if rung == "sd1" else DIALECT_SEAM + _SKELETONS[task]
    return base[: -len(_OUTPUT)] + extra + _OUTPUT

SEAM_DIALECT_PROMPTS: dict[str, dict[str, str]] = {
    task: {rung: build(task, rung) for rung in RUNGS} for task in TASKS
}
