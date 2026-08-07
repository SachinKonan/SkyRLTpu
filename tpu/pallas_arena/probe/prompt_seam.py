"""SEAM prompts: the harness owns the interface, the model owns the strategy.

Pure DATA module, with one structural rule that makes it trustworthy: the
scaffold shown in the prompt is `seam.SCAFFOLDS[task]` ITSELF, interpolated,
never a retyped copy. What the model reads is byte-identical to what its code
gets appended to. `seam_dryrun.py` asserts that.

Why these exist at all, from the previous probe's measurements:

  * gemma finished 14-16 of 16 splash generations and exported 0 of 16, dying
    on INVENTED API (`out_spec`, `out_dtype`, `out_shapes`, `block_shapes`,
    `jax.Shape`, `pl.Ref`). The `reference` variant's ten-item prose gotcha
    list did not move that number by a single candidate. Prose about VMEM
    budgets cannot teach a function signature the model does not have, so
    every seam prompt carries a REAL API-signature block (§ API_BLOCK),
    introspected from the pinned jax, not described.
  * the `tailored` variant passed 16/16 with a within-group reward spread of
    0.0042 -- BELOW the judge's own noise floors (0.0069 / 0.0158). Handing
    over the whole scaffold removed the variance RL needs. So the seam gives
    away the plumbing and keeps every decision that changes the NUMBER:
    block and tile shapes, one-pass vs two-pass, online vs materialized
    softmax, accumulation dtype, recompute-vs-save, which blocks to skip.
"""

from __future__ import annotations

from pallas_arena.probe.seam import SEAMS, scaffold_of

# --------------------------------------------------------------------------
# THE API BLOCK. Introspected from the pinned jax (see seam_dryrun.py, which
# re-derives every line with inspect.signature and fails if one has drifted),
# because the measured failure is signature RECALL and prose demonstrably did
# not fix it.

API_BLOCK = r'''
## Real JAX 0.10.2 API -- these signatures are introspected, not remembered

You do not call most of these (the harness does), but you will read them in
the scaffold, and the ones marked YOU CALL are the ones you need.

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# --- the pallas_call the HARNESS writes for you. Shown so you can READ the
# --- scaffold; there is no reason for you to write one.
pl.pallas_call(kernel, out_shape, *, grid_spec=None, grid=(),
               in_specs=NoBlockSpec, out_specs=NoBlockSpec, scratch_shapes=(),
               input_output_aliases={}, debug=False, interpret=False,
               name=None, compiler_params=None, cost_estimate=None,
               metadata=None)

pl.BlockSpec(block_shape=None, index_map=None, pipeline_mode=None, *,
             memory_space=None)
    # POSITIONAL order is (block_shape, index_map). index_map returns BLOCK
    # indices, not element offsets.

pl.GridSpec(grid=(), in_specs=NoBlockSpec, out_specs=NoBlockSpec,
            scratch_shapes=())

pltpu.PrefetchScalarGridSpec(num_scalar_prefetch, grid=(),
                             in_specs=NoBlockSpec, out_specs=NoBlockSpec,
                             scratch_shapes=())
    # scalar-prefetch operands come FIRST in the call, are visible to every
    # index_map as its TRAILING arguments, and arrive in the kernel body as
    # SMEM refs BEFORE the ordinary inputs.

pltpu.CompilerParams(dimension_semantics=None, allow_input_fusion=None,
                     vmem_limit_bytes=None, collective_id=None,
                     has_side_effects=False, flags=None,
                     internal_scratch_in_bytes=None, serialization_format=1,
                     disable_bounds_checks=False, ...)

jax.ShapeDtypeStruct(shape, dtype, *, sharding=None, weak_type=False,
                     manual_axis_type=None, is_ref=False)

pltpu.VMEM(shape, dtype)    # a scratch buffer;  pltpu.SMEM(shape, dtype) too
```

```python
# --- YOU CALL these, inside the function you are asked to write.
pl.program_id(axis: int) -> jax.Array          # index of THIS grid step
pl.num_programs(axis: int) -> int | jax.Array
pl.when(condition)                             # a DECORATOR, not a context mgr:
#     @pl.when(j == 0)
#     def _():
#         acc_ref[...] = jnp.zeros_like(acc_ref)
pl.ds(start, size=None, stride=None)           # == pl.dslice; dynamic slice
pl.cdiv(a, b)          pl.multiple_of(x, values)      pl.next_power_of_2(x)
jax.lax.broadcasted_iota(dtype, shape, dimension)   # position indices in-kernel
jax.lax.dot_general(lhs, rhs, dimension_numbers, precision=None,
                    preferred_element_type=None)
jax.lax.fori_loop / jax.lax.scan                    # both work inside a kernel
```

**Names that DO NOT EXIST at this pin.** Every one of these was invented by a
model on this exact family of tasks, and every one is an immediate failure:

```
pl.load        pl.store        pl.swap        pl.atomic_add      pl.Ref
pltpu.ANY      pltpu.TPUCompilerParams        jax.Shape
pallas_call(..., out_spec=/out_dtype=/out_dtypes=/out_shapes=/in_shapes=
                 /block_shapes=/index_map=)   # none of these are kwargs
```

There is no `pl.load`/`pl.store` in jax 0.10.2 at all: a Ref is read and
written by INDEXING. `ANY` lives at `pl.ANY`, not `pltpu.ANY`. The compiler
params class is `pltpu.CompilerParams`; `TPUCompilerParams` was removed.

A Ref (`q_ref`, `o_ref`, a `pltpu.VMEM` scratch) is NOT an array. Read it with
`ref[...]`, `ref[0]`, `ref[:, :1]`, `ref[0, a:b, :]`; write it with
`ref[...] = value` or `ref[0] = value`. It has `.shape` and `.dtype`, and it
has NO `.astype` -- read first, then cast the result.

**Matmul precision is a real knob and it bit the control run.** On TPU,
`jax.lax.dot_general` on float32 operands runs ONE bfloat16 pass at the default
precision. The previous control's attention kernel missed the judge's
calibrated tolerance by 18% (max error 4.11e-3 against a 3.48e-3 budget) for
exactly that reason and nothing else. `precision=jax.lax.Precision.HIGHEST` is
accurate and roughly 6x the passes; `preferred_element_type=jnp.float32` gets
you an f32 accumulator over bf16 inputs, which is usually what you want. This
is a genuine speed/accuracy trade and it is YOURS to make.
'''

_BUDGET = r'''
## How much to write

Write **300-1200 tokens of code**. If you are past ~2000 you have started
rewriting the scaffold instead of choosing a strategy -- stop and delete it.
Do not restate the scaffold; the harness already has it.
'''

_OUTPUT = r'''
## Output format

Output ONE ```python code block containing ONLY the definitions listed above,
plus whatever imports and private helpers they need. No prose outside it. Do
NOT define `kernel`, do NOT write a `pallas_call`, do NOT rewrite any part of
the scaffold: your code is pasted directly ABOVE it, in the same module, and
anything you redefine that the scaffold also defines will simply be replaced.
'''


def _header(task_line: str, owns: str) -> str:
    return f'''You are an expert JAX/TPU kernel engineer. {task_line}

## What you write, and what you do NOT write

This is a fill-in task with a real seam in it. **The harness owns the
interface; you own the strategy.** The harness has already written {owns} That
code is fixed, it is shown to you verbatim below, and your job is only to
supply the named functions it calls.

Your code is pasted ABOVE the scaffold into one module and submitted as a
single program, so the entrypoint, the shapes and the judge contract are
already satisfied before you write a line. What is NOT decided is every choice
that changes the number the judge measures.
'''


def _scaffold_section(task: str) -> str:
    return f'''
## The scaffold your code is appended to (verbatim, exactly as submitted)

```python
{scaffold_of(task)}
```
'''


# ==========================================================================
# 1. FLCE
# ==========================================================================

FLCE_SEAM = (
    _header(
        "Write the per-tile numerics of a memory-efficient fused linear "
        "cross-entropy (FLCE) kernel.",
        "the token-axis tiling driver, the pad/un-pad arithmetic, the "
        "`jax.custom_vjp` registration and the residual contract.",
    )
    + _scaffold_section("flce")
    + r'''
## What you must define

```python
TILE = <int>            # tokens per tile. A PYTHON INT at module level.

def tile_forward(h_tile, w, t_tile):
    """h_tile : [TILE, h]  bfloat16   one tile of hidden states
       w      : [h, v]     bfloat16   the frozen output head (v = 151936)
       t_tile : [TILE]     int32      target ids for this tile
    returns (logprobs_tile, carry)
       logprobs_tile : [TILE]  logprobs[i] = logits[i, t[i]] - logsumexp(logits[i])
                       (the harness casts it to float32 for you)
       carry         : ANY pytree you want handed back to tile_backward for
                       this same tile -- or `None` to carry nothing. This is
                       the recompute-vs-save knob and it is yours."""

def tile_backward(h_tile, w, t_tile, g_tile, carry):
    """g_tile : [TILE] float32, the cotangent of this tile's logprobs.
       carry  : exactly what YOUR tile_forward returned for this tile.
    returns d(loss)/d(h_tile), shaped like h_tile. The harness casts the dtype,
    so returning float32 is fine."""
```

## Where the real decisions are

* **`TILE`.** Bigger tiles mean fewer scan steps and a bigger transient. One
  tile of logits at v=151936 is `TILE * 151936` elements: 148 MiB at bf16 and
  297 MiB at f32 for TILE=512. The forbidden full [n, v] f32 materialization
  is 2.49 GB at n=4096. 512 divides 4096 and 2048 exactly; the holdout n=3000
  pads (the harness does the padding).
* **Chunk the vocab, or don't.** Nothing forces you to form the whole
  [TILE, v] logits at once. A running (online) max and sum over vocab chunks
  never materializes more than [TILE, chunk].
* **One pass or two.** One pass computes the target logit and the LSE together
  online. Two passes are simpler and read `w` twice.
* **Accumulation dtype.** The matmul is bf16 x bf16; the accumulator, the max
  and the sum do not have to be.
* **Recompute vs save.** `carry=None` and recompute the logits in the backward
  (cheap memory, more flops), or carry the LSE (or the max, or the softmax
  denominator) and skip recomputing it. Whatever you carry is kept alive for
  the WHOLE forward pass, once per tile -- carrying [TILE] floats is free,
  carrying [TILE, v] defeats the entire task.
* **The backward's own formulation.** `jax.vjp` over your forward is correct
  and easy. The closed form `dlogits = (onehot - softmax) * g`, then
  `dlogits @ w.T`, is usually faster. Both are legal.

## Task semantics

`logprobs[i] = logits[i, targets[i]] - logsumexp(logits[i])` where
`logits = hidden @ w` at float32 accumulation. This is the fp32 oracle the
judge grades you against -- it is NOT a valid answer, because it materializes
the full [n, v] logits, which is the one thing this task exists to avoid:

```python
logits = hidden.astype(jnp.float32) @ w.astype(jnp.float32)
tl = jnp.take_along_axis(logits, targets[:, None], axis=1)[:, 0]
return tl - jax.nn.logsumexp(logits, axis=-1)
```

`logsumexp` MUST be max-shifted or computed online: at v=151936 an unshifted
`exp` overflows f32 and the whole row goes non-finite. Log-probabilities must
be <= 0.

## Declared shapes

The composed program must trace and run at ALL of these:

    n=4096, h=2880, v=151936   (scored)
    n=2048, h=2880, v=151936   (scored)
    n=3000, h=2880, v=151936   (HOLDOUT: logged, not scored, but it MUST still
                                trace and be correct -- n=3000 is deliberately
                                not a multiple of any reasonable tile)

## Judge contract

- The composed program is exec'd and `kernel` is serialized with `jax.export`
  once per declared shape plus once for the gradient, with no data and no
  device.
- **Forward AND backward**: the judge differentiates with respect to `hidden`
  only (`w` is frozen) and compares against the fp32 reference at a calibrated
  tolerance. Peak HBM is reported, which exposes anything that materializes
  [n, v].
- Correctness: per-element `|cand - ref| / (|ref| + 1)` on hidden seeds, on
  BOTH the max and the 99th-percentile tail, at 1.5x the reference's own bf16
  error. Non-finite output fails automatically.
- Determinism: 5 runs on the same input must be BITWISE identical.
- Compile budget: every shape plus the gradient within 90 s together.
  `jax.lax.scan` compiles ONE body (the harness already uses it); a Python
  `for` loop over vocab chunks compiles one copy per chunk.
- Score: median wall-clock against the production baseline -- which is our own
  `custom_vjp` tiled kernel at TILE=2048, the thing you are trying to beat.
- Banned imports: `jax.experimental.pallas.ops.*`, `tpu_inference`, `MaxText`,
  `maxtext`, `skyrl`, `recurrentgemma`, and the arena's own package.
- A `pallas_call` is allowed but NOT required here; the baseline is a
  `custom_vjp` over a `lax.scan`, and you may beat it either way.
'''
    + _BUDGET
    + API_BLOCK
    + _OUTPUT
)


# ==========================================================================
# 2. SPLASH ATTENTION
# ==========================================================================

SPLASH_SEAM = (
    _header(
        "Write the inner block of a causal, segment-masked multi-head "
        "attention Pallas kernel for TPU.",
        "the ENTIRE `pallas_call` -- the grid, every `BlockSpec` and "
        "`index_map`, the `out_shape`, the explicit padding of both the query "
        "and the key axis, and the final un-pad slice.",
    )
    + _scaffold_section("splash_attention")
    + r'''
## What you must define

```python
def choose_blocks(seq, head_dim):
    """seq, head_dim: PYTHON INTS (the real, unpadded ones).
    returns (bq, bk): the query-block rows per grid step, and the key-block
    width you want to work in. The harness rounds bq up to a multiple of 8 and
    bk up to a multiple of 128 (the TPU sublane/lane tiles) and clamps them, so
    an unlucky choice costs you speed, never a compile error. It then pads the
    query axis to a multiple of bq and the key axis to a multiple of bk."""

def attn_block(q_ref, k_ref, v_ref, sq_ref, sk_ref, o_ref, *, bq, bk, kv_len):
    """ONE grid step: one head, one query block, ALL of the keys.
       q_ref  [1, bq, dp]      bfloat16  queries, global rows i*bq + 0..bq-1
       k_ref  [1, kv_len, dp]  bfloat16  EVERY key   (already in VMEM)
       v_ref  [1, kv_len, dp]  bfloat16  EVERY value (already in VMEM)
       sq_ref [bq, 128]        int32     segment id of each QUERY row,
                                         broadcast over the 128 lanes;
                                         sq_ref[:, :1] is the [bq, 1] column
       sk_ref [8, kv_len]      int32     segment id of each KEY column,
                                         broadcast over the 8 sublanes;
                                         sk_ref[:1, :] is the [1, kv_len] row
       o_ref  [1, bq, dp]      float32   THE OUTPUT. You must write it.
       bq, bk, kv_len          Python ints. dp is head_dim padded to 128.
    `pl.program_id(0)` is the head; `pl.program_id(1)` is the query-block index
    i, so this block's global query rows are i*bq + 0..bq-1. Nothing persists
    between grid steps: everything you need is in this call."""
```

## Where the real decisions are

* **`bq`, and your own key chunking.** The whole key axis is resident, so how
  you consume it is entirely yours: `for j in range(kv_len // bk)` over
  bk-wide slices `k_ref[0, j*bk:(j+1)*bk, :]` (Python ints, so it unrolls), or
  one giant `[bq, kv_len]` score matrix if you think it fits. It probably does
  not: the scoped VMEM ceiling is ~32 MB per `pallas_call` and a
  `[512, 4096]` f32 score tile is 8 MB before you have made a single
  temporary of it. Going over is a compile-time
  `CompileTimeScopedVmemOom: Scoped allocation with size 36.25M and limit 32.00M`.
* **Skipping work.** A key block strictly above the diagonal contributes
  nothing to any query in this block -- `j*bk > i*bq + bq - 1` -- and you know
  both indices. Roughly half the blocks are skippable and `bq` is a Python int,
  so you can skip them at TRACE time and not emit the code at all.
* **The softmax formulation.** Online (running max, running sum, rescale the
  accumulator each block) or two-pass (find the row max, then accumulate).
  Online costs an extra `exp` and a multiply per block; two-pass reads the keys
  twice.
* **Accumulation dtype and matmul precision.** See the precision note in the
  API section -- it is the difference between passing and failing correctness.

## Task semantics

Causal (query i attends to key j only if j <= i) AND restricted to equal
segment ids. A query whose segment id is 0 is PADDING and must produce
EXACTLY 0.0, never NaN -- and so must any query with no visible key at all.
There is NO 1/sqrt(head_dim) scale: the logits are the raw dot products
(`make_inputs` pre-scales q). Softmax at float32. Forward only -- no backward.

The fp32 oracle, verbatim. NOT a valid answer -- it materializes
[heads, seq, seq] (2.1 GB at heads=8, seq=4096) and contains no `pallas_call`:

```python
NEG_INF = -0.7 * float(np.finfo(np.float32).max)

def causal_segment_attention(q, k, v, segment_ids):
    q32, k32, v32 = q.astype(jnp.float32), k.astype(jnp.float32), v.astype(jnp.float32)
    seq = q.shape[1]
    logits = jnp.einsum("hqd,hkd->hqk", q32, k32)
    idx = jnp.arange(seq)
    causal = idx[:, None] >= idx[None, :]
    same_seg = segment_ids[:, None] == segment_ids[None, :]
    live = segment_ids != 0
    mask = causal & same_seg & live[:, None] & live[None, :]
    logits = jnp.where(mask[None, :, :], logits, NEG_INF)
    row_live = mask.any(axis=-1)
    m = jnp.max(logits, axis=-1, keepdims=True)
    p = jnp.exp(logits - m)
    p = jnp.where(mask[None, :, :], p, 0.0)
    denom = jnp.sum(p, axis=-1, keepdims=True)
    p = jnp.where(row_live[None, :, None], p / jnp.maximum(denom, 1e-30), 0.0)
    return jnp.einsum("hqk,hkd->hqd", p, v32)
```

Do not initialize a running max to `-jnp.inf`: `-inf - -inf` is NaN and a
fully-masked row will poison the output. Use a large finite negative constant.

## Declared shapes

    heads=8, seq=4096, head_dim=128   (scored)
    heads=4, seq=2048, head_dim=128   (scored)
    heads=4, seq=2049, head_dim=128   (HOLDOUT: logged, not scored, but it MUST
                                       still trace and be correct -- seq=2049 is
                                       deliberately not a multiple of any
                                       reasonable block. The harness pads; your
                                       masking has to be right at the seam.)

## Judge contract

- The composed program is exec'd and `kernel` is serialized with `jax.export`
  once per declared shape, with no data and no device.
- The program MUST contain a real `pallas_call` -- the scaffold's counts.
- Correctness on hidden seeds against the fp32 oracle: per-element
  `|cand - ref| / (|ref| + 1)`, on BOTH the max and the 99th-percentile tail,
  at 1.5x the reference's own bf16 error. Non-finite fails automatically.
- Determinism: 5 runs, BITWISE identical.
- Compile budget: 90 s for all shapes together. A `for` loop over key blocks
  unrolls -- 8 blocks is fine, 512 is not.
- Score: median wall-clock against the production baseline.
- Banned: `jax.experimental.pallas.ops.*` (it holds the reference splash
  kernel), `tpu_inference`, `MaxText`, `maxtext`, `skyrl`, `recurrentgemma`,
  the arena's package, and calling `jax.nn.dot_product_attention` or
  `jax.nn.scaled_dot_product_attention`.
'''
    + _BUDGET
    + API_BLOCK
    + _OUTPUT
)


# ==========================================================================
# 3. RAGGED PAGED ATTENTION
# ==========================================================================

RPA_SEAM = (
    _header(
        "Write the inner block of a ragged paged-attention DECODE kernel "
        "(GQA, one query token per sequence, KV read from a paged cache).",
        "the `pallas_call`, the grid over (sequence, page), every `BlockSpec`, "
        "and the page-id -> page-slice helper: the scalar-prefetched "
        "`page_tables` in the k/v index maps means a grid step is handed its "
        "physical page already resident in VMEM.",
    )
    + _scaffold_section("ragged_paged_attention")
    + r'''
## What you must define

```python
def decode_block(q_ref, k_ref, v_ref, o_ref, m_ref, l_ref, sl_ref,
                 *, kv_heads, group, page_size, head_dim, num_pages):
    """ONE grid step: one sequence b, ONE page p of its KV cache, ALL heads.
       q_ref  [1, kv_heads, group, d]      bfloat16  every query head of this
                                                     sequence, grouped by the
                                                     kv head it attends
       k_ref  [1, page_size, kv_heads, d]  bfloat16  physical page p of K,
                                                     already gathered for you
       v_ref  [1, page_size, kv_heads, d]  bfloat16  the same page of V
       o_ref  [1, kv_heads, group, d]      float32   THE OUTPUT. Its index_map
                                                     ignores p, so it PERSISTS
                                                     across the page axis and
                                                     you accumulate into it.
       m_ref  [kv_heads, group, 128]       float32   scratch, persists too
       l_ref  [kv_heads, group, 128]       float32   scratch, persists too
       sl_ref [batch]                      int32     SMEM: sl_ref[b] is
                                                     sequence b's length
       kv_heads, group, page_size, head_dim, num_pages : Python ints.
    `pl.program_id(0)` = b, `pl.program_id(1)` = p. Page p covers cached
    positions p*page_size + 0 .. p*page_size + page_size-1. `k_ref[0, :, h, :]`
    is head h's [page_size, d] slice (h a Python int)."""
```

`m_ref` and `l_ref` are 128 lanes wide only because 128 is the minimum TPU lane
tile; every lane can hold the same per-row scalar, so read `m_ref[h, :, :1]`
and store a full-width broadcast back. You do not have to use them at all.

The kv-head axis is not in the grid on purpose: Mosaic requires a block's
second-minor dimension to be a multiple of 8 or equal to the array's, and
`k_pages` is `[num_pages, page_size, kv_heads, head_dim]`, so a per-head block
of `[1, page_size, 1, d]` is rejected at compile time. One step holds all
8 heads, which is also 8x fewer grid steps.

## Where the real decisions are

* **The ragged tail: mask, or bound the loop.** Positions >= `sl_ref[b]` must
  not contribute. You can mask them inside every page (`pos < sl_ref[b]`,
  cheap, uniform) or skip whole pages with `@pl.when(p * page_size < sl_ref[b])`
  and do no work at all for them. Sequence lengths are random per sequence, so
  the second one wins on average and costs a branch.
* **Online vs two-pass.** With `m_ref`/`l_ref` you can run a running-max online
  softmax across pages. You could instead accumulate unnormalized sums with a
  fixed shift and normalize at the end -- fewer `exp`s, more overflow risk.
* **GQA grouping, and how you sweep the 8 kv heads.** All `group` queries of a
  head share that head's K and V, so head h is one `[group, page_size]` score
  matrix, not `group` separate ones. `group` is 4, below the 8-sublane tile, so
  a per-head `[4, 64]` tile wastes half the sublanes -- you can pay that 8
  times in an unrolled `for h in range(kv_heads)`, or reshape and do all 32
  query heads at once against a `[page_size, kv_heads, d]` block. Both are
  legal and they are not the same speed.
* **Accumulation dtype and matmul precision.** As always.
* **The first and last page.** `@pl.when(p == 0)` to initialize,
  `@pl.when(p == num_pages - 1)` to normalize. Both are yours to place.

## Task semantics

For each sequence b and query head h (attending kv head `h // group`), attend
over cached positions `0 .. seq_lens[b]-1`, softmax at fp32, output the
value-weighted sum. Position t lives in physical page `page_tables[b, t //
page_size]` at slot `t % page_size`. The fp32 oracle:

```python
k = k_pages[page_tables].reshape(b, max_len, kvh, d).astype(jnp.float32)
v = v_pages[page_tables].reshape(b, max_len, kvh, d).astype(jnp.float32)
q32 = q.astype(jnp.float32).reshape(b, kvh, group, d)
logits = jnp.einsum("bhgd,bthd->bhgt", q32, k)
live = jnp.arange(max_len)[None, :] < seq_lens[:, None]
logits = jnp.where(live[:, None, None, :], logits, NEG_INF)
m = jnp.max(logits, axis=-1, keepdims=True)
p = jnp.where(live[:, None, None, :], jnp.exp(logits - m), 0.0)
p = p / jnp.maximum(jnp.sum(p, axis=-1, keepdims=True), 1e-30)
o = jnp.einsum("bhgt,bthd->bhgd", p, v).reshape(b, qh, d)
```

There is no causal mask and no scale -- one query token, `make_inputs`
pre-scales q. `seq_lens[b] >= 1` always, so no row is ever fully masked.

## Declared shapes

page_size=64, kv_heads=8, q_heads=32 (so group=4), head_dim=128 throughout.

    batch=16, max_len=1024, num_pages=264   (scored)
    batch=8,  max_len=512,  num_pages=72    (scored)
    batch=17, max_len=512,  num_pages=144   (HOLDOUT: logged, not scored, but
                                             it MUST still trace -- 17 is prime,
                                             so nothing about the batch tiles)

`max_pages_per_seq = max_len // page_size`, i.e. 16, 8 and 8.

## Judge contract

- Exec'd, then `jax.export`ed once per declared shape, no data, no device.
- MUST contain a real `pallas_call` -- the scaffold's counts.
- Correctness on hidden seeds against the fp32 oracle (max and q99 tail, 1.5x
  the reference's own bf16 error). Non-finite fails automatically.
- Determinism: 5 runs, BITWISE identical. Compile budget 90 s for all shapes.
- This task is **memory-bandwidth bound**; the judge also reports what fraction
  of the chip's peak HBM bandwidth you reached.
- Score: median wall-clock against the baseline. NOTE: this judge's baseline is
  a competent batch-blocked XLA paged decode, not vLLM's production Pallas
  kernel, and it is labelled as such in the run's boot report.
- Banned: `jax.experimental.pallas.ops.*`, `tpu_inference`, `MaxText`,
  `maxtext`, `skyrl`, `recurrentgemma`, the arena's package, and
  `jax.nn.dot_product_attention`.
'''
    + _BUDGET
    + API_BLOCK
    + _OUTPUT
)


# ==========================================================================
# 4. MEGABLOX GMM
# ==========================================================================

GMM_SEAM = (
    _header(
        "Write the inner tile of a grouped matmul (MoE `gmm`) Pallas kernel "
        "for TPU.",
        "the exclusive scan of `group_sizes` to row offsets, the group-ALIGNED "
        "row permutation (so no row tile ever straddles two experts -- the "
        "zero-size-group and boundary-tile correctness traps are the "
        "harness's), the `pallas_call` with scalar-prefetched per-tile expert "
        "ids, every `BlockSpec`, and the un-permute of the result.",
    )
    + _scaffold_section("megablox_gmm")
    + r'''
## What you must define

```python
def choose_tiles(m, k, n, num_groups):
    """All PYTHON INTS. returns (bm, bk, bn):
       bm : rows of `lhs` per grid step   -> rounded up to a multiple of 8
       bk : YOUR contraction chunk        -> rounded to a multiple of 128
       bn : output columns per grid step  -> rounded to a multiple of 128
    bm and bn shape the grid and the VMEM footprint; bk is entirely internal to
    gmm_tile and does not touch the BlockSpecs."""

def gmm_tile(l_ref, r_ref, o_ref, *, bm, bk, bn, k):
    """ONE grid step: one row tile x one column tile, for ONE expert.
       l_ref [bm, k]     bfloat16  this tile's rows of lhs, the FULL k axis
       r_ref [1, k, bn]  bfloat16  this expert's weights, the FULL k axis,
                                   columns j*bn .. j*bn+bn-1
       o_ref [bm, bn]    float32   THE OUTPUT. You must write it.
       bm, bk, bn, k : Python ints.
    Every row in this tile belongs to the SAME expert -- the harness guarantees
    it -- and rows past the expert's true size are already zeroed, so you never
    check a boundary. `pl.program_id(0)` is the row tile, `pl.program_id(1)` the
    column tile."""
```

## Where the real decisions are

* **`bm` and `bn` are the VMEM budget.** `l_ref` is `[bm, 4096]` bf16 and
  `r_ref` is `[1, 4096, bn]` bf16, and Pallas double-buffers both: at
  bm=512, bn=1024 that is 8 MB + 16 MB before the accumulator, against a ~32 MB
  scoped ceiling. Too big is `CompileTimeScopedVmemOom` at compile time; too
  small wastes the MXU. This is the main thing you are choosing.
* **`bm` also sets the padding waste.** The harness aligns every expert to a
  multiple of `bm`, so a large `bm` with many small experts pads a lot of dead
  rows -- and those dead rows are multiplied anyway.
* **The k loop.** `k=4096` is fully resident. One `[bm, 4096] x [4096, bn]`
  `dot_general`, or a loop over `bk`-wide chunks accumulating into an f32
  register. The MXU wants big contractions; the register file does not.
* **Accumulation dtype and matmul precision.** `preferred_element_type=
  jnp.float32` over bf16 operands is the standard grouped-matmul choice.
* **Loop order.** You control only what is inside one tile, but within it the
  order of the k chunks and where you keep the accumulator are yours.

## Task semantics

`out[rows of group g] = lhs[rows of group g] @ rhs[g]`, at fp32 accumulation.
`group_sizes` sums to `m` and **zero-size groups are legal and do occur** (the
harness handles them; you never see one). The fp32 oracle is
`jax.lax.ragged_dot(lhs.astype(f32), rhs.astype(f32), group_sizes)`.

## Declared shapes

Expert widths are the real ones: k=4096, n=14336, num_groups=4.

    m=4096, num_groups=4, k=4096, n=14336, uniform group sizes  (scored)
    m=2048, num_groups=4, k=4096, n=14336, Zipf-skewed          (scored)
    m=3000, num_groups=4, k=4096, n=14336, Zipf-skewed          (HOLDOUT:
        logged, not scored, but it MUST still trace -- 3000 is deliberately not
        a multiple of any reasonable row tile)

Group sizes are freshly sampled from BOTH uniform and Zipf-skewed
distributions on every grading, so a kernel tuned for one balance cannot fake
a win on the other.

## Judge contract

- Exec'd, then `jax.export`ed once per declared shape, no data, no device.
- MUST contain a real `pallas_call` -- the scaffold's counts.
- Correctness on hidden seeds against the fp32 oracle (max and q99 tail, 1.5x
  the reference's own bf16 error). Non-finite fails automatically.
- Determinism: 5 runs, BITWISE identical. Compile budget 90 s for all shapes.
- Score: median wall-clock against the tuned Pallas megablox `gmm` that MaxText
  vendors. Note that the harness's group-aligned permutation costs a gather and
  a scatter that the baseline does not pay, so this denominator is harsh.
- Banned: `jax.experimental.pallas.ops.*` (it holds the megablox kernel),
  `tpu_inference`, `MaxText`, `maxtext`, `skyrl`, `recurrentgemma`, the arena's
  package.
'''
    + _BUDGET
    + API_BLOCK
    + _OUTPUT
)


# ==========================================================================
# 5. RG-LRU
# ==========================================================================

RGLRU_SEAM = (
    _header(
        "Write the per-chunk recurrence of an RG-LRU gated diagonal linear "
        "scan (the RecurrentGemma SSM).",
        "the outer entrypoint, the time-axis chunking and its padding, the "
        "carry-state contract between chunks, and the reset contract.",
    )
    + _scaffold_section("rg_lru")
    + r'''
## What you must define

```python
CHUNK = <int>           # timesteps per chunk. A PYTHON INT at module level.

def scan_chunk(x_chunk, a_chunk, reset_chunk, h_prev):
    """x_chunk     : [b, CHUNK, d]  bfloat16
       a_chunk     : [b, CHUNK, d]  float32, in [0, 1) -- ALREADY forced to 0
                                    wherever reset is True (the harness did it)
       reset_chunk : [b, CHUNK]     bool, for information; you do not have to
                                    use it, the reset is already in a_chunk
       h_prev      : [b, d]         float32, the state entering this chunk
    returns (h_chunk, h_last)
       h_chunk : [b, CHUNK, d] the state at EVERY timestep of this chunk
       h_last  : [b, d]        the state leaving this chunk
    (the harness casts both to float32 for you)"""
```

## Where the real decisions are

* **`CHUNK`.** This is the whole sequential/parallel trade. `CHUNK = t` is one
  chunk and no outer scan; `CHUNK = 1` is a pure sequential scan through the
  harness. Everything in between splits the work between an inner
  parallel scan (log depth, more flops, more memory) and an outer sequential
  one (no extra flops, `t / CHUNK` serial steps). The declared T values are
  2048, 1024 and 1500.
* **Sequential or associative.** `jax.lax.scan` over the time axis inside the
  chunk is the obvious formulation. `jax.lax.associative_scan` on the pair
  `(a, gx)` with `combine((a_l,b_l),(a_r,b_r)) = (a_l*a_r, b_l*a_r + b_r)` is
  explicitly legal and is what the baseline does -- if you use it you must
  still fold `h_prev` in, e.g. by adding `a[:, 0] * h_prev` into `gx[:, 0]`.
* **fp32 state handling.** `a` is fp32 by contract and the gates sit near 1
  (long memory), so `sqrt(1 - a^2)` is small and catastrophic cancellation is
  real. `x` is bf16 and the state is fp32; where you upcast matters.
* **Layout.** `d = 2560` is the lane axis and the time axis is the sequential
  one. Whether you move the time axis to the front once per chunk or index it
  in place is a real cost at these sizes.

## Task semantics

    h_t = a_t * h_{t-1} + sqrt(max(1 - a_t^2, 0)) * x_t

with `h_{-1} = 0` at the start of the sequence, and `a_t` forced to 0 wherever
`reset_t` is True (so no state crosses a segment boundary, and at a reset
`h_t = x_t` exactly). The fp32 oracle:

```python
x32 = x.astype(jnp.float32)
a32 = a.astype(jnp.float32) * (1.0 - reset[..., None].astype(jnp.float32))
gx = jnp.sqrt(jnp.maximum(1.0 - jnp.square(a32), 0.0)) * x32
def step(h, xs):
    a_t, gx_t = xs
    h = a_t * h + gx_t
    return h, h
h0 = jnp.zeros((x32.shape[0], x32.shape[2]), jnp.float32)
_, hs = jax.lax.scan(step, h0, (jnp.moveaxis(a32, 1, 0), jnp.moveaxis(gx, 1, 0)))
h = jnp.moveaxis(hs, 0, 1)
```

## Declared shapes

Width is the real RecurrentGemma-2B one: d=2560.

    b=4, t=2048, d=2560   (scored)
    b=2, t=1024, d=2560   (scored)
    b=2, t=1500, d=2560   (HOLDOUT: logged, not scored, but it MUST still trace
                           -- 1500 is deliberately not a multiple of any
                           reasonable chunk. The harness does the padding.)

## Judge contract

- Exec'd, then `jax.export`ed once per declared shape, no data, no device.
- A `pallas_call` is ALLOWED but NOT required here -- `lax.associative_scan` is
  an explicitly legal strategy.
- Correctness on hidden seeds against the fp32 oracle (max and q99 tail, 1.5x
  the reference's own drift). This task is graded at LONG T where fp32 drift
  is the whole difficulty; non-finite fails automatically.
- Determinism: 5 runs, BITWISE identical. Compile budget 90 s for all shapes.
- This task is **memory-bandwidth bound**; the judge also reports what fraction
  of the chip's peak HBM bandwidth you reached.
- Score: median wall-clock against the baseline. NOTE: this judge's baseline is
  `lax.associative_scan`, not DeepMind's recurrentgemma Pallas scan, and it is
  labelled as such in the run's boot report.
- Banned: `recurrentgemma`, `jax.experimental.pallas.ops.*`, `tpu_inference`,
  `MaxText`, `maxtext`, `skyrl`, the arena's package.
'''
    + _BUDGET
    + API_BLOCK
    + _OUTPUT
)


SEAM_PROMPTS: dict[str, str] = {
    "flce": FLCE_SEAM,
    "splash_attention": SPLASH_SEAM,
    "ragged_paged_attention": RPA_SEAM,
    "megablox_gmm": GMM_SEAM,
    "rg_lru": RGLRU_SEAM,
}

assert set(SEAM_PROMPTS) == set(SEAMS), "a seam without a prompt, or the reverse"
