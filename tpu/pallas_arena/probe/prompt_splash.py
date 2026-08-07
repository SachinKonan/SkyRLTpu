"""Cold-start probe prompts for arena task `splash_attention`.

Pure DATA module: three user-turn prompt variants for the SAME task, differing
only in how much scaffolding the model is handed.

  minimal    task spec + signature + shapes + judge contract, nothing else
  reference  + the fp32 correctness oracle (an INVALID answer -- it
             materializes [heads, seq, seq]) + a Pallas/Mosaic gotcha list
  tailored   + a working fill-in-the-blank scaffold in which everything
             structural is given and exactly one function is a TODO

The variants are strictly nested (`reference` = `minimal` + more, `tailored` =
`reference` + more) so a length/content diff between arms is attributable to
the added block and nothing else. No chat special tokens: the harness renders
those.

Precedent for the register/structure: third_party/discover/examples/gpu_mode/
prompt.py (TRIMUL_PROMPT), transposed from Triton-on-H200 to JAX-on-TPU.
Task contract: pallas_arena/judge/problems/splash_attention.py.
"""

# --------------------------------------------------------------------------
# VARIANT 1 -- minimal
# --------------------------------------------------------------------------

SPLASH_MINIMAL = r'''You are an expert JAX/Pallas TPU kernel engineer. Write a JAX Pallas TPU kernel implementing causal, segment-masked multi-head attention.

## Task

Entrypoint (exact): a module-level function named `kernel`:

    def kernel(q, k, v, segment_ids):  ->  o

  q, k, v      : [heads, seq, head_dim], bfloat16
  segment_ids  : [seq], int32; 0 means PADDING
  o            : [heads, seq, head_dim], float32

Attention is causal (query i attends to key j only if j <= i) AND restricted to equal segment ids; a query in padding (segment 0) must produce EXACTLY 0.0, never NaN. The logits are the raw dot products q.k -- no 1/sqrt(head_dim) scaling is applied. Softmax must be computed at float32 precision. Forward pass only -- no backward is required for this task.

## Declared shapes

One implementation must trace and run at ALL of these:

    heads=8, seq=4096, head_dim=128   (scored)
    heads=4, seq=2048, head_dim=128   (scored)
    heads=4, seq=2049, head_dim=128   (HOLDOUT: logged, not scored, but the kernel MUST still trace and be correct here -- note seq is deliberately NOT a multiple of any reasonable block size)

## Judge contract

- Your program is exec'd, then `kernel` is serialized with `jax.export` once per declared shape, with NO concrete data and NO device. It must trace at every declared shape.
- Banned: importing `jax.experimental.pallas.ops.*` (that tree contains the reference splash kernel), `tpu_inference`, `MaxText`, `maxtext`, `skyrl`, `recurrentgemma`, and the arena's own package. Calling `jax.nn.dot_product_attention` or `jax.nn.scaled_dot_product_attention` is banned. Enforced by a static AST screen AND runtime module poisoning.
- Your program MUST contain a real `pallas_call`.
- Correctness: per-element error `|cand - ref| / (|ref| + 1)` against an fp32 closed-form reference on hidden random seeds, checked on BOTH the max and the 99th-percentile tail, against a tolerance calibrated at 1.5x the reference's own bfloat16 error. Non-finite output is an automatic failure.
- Determinism: 5 repeated runs on the same input must be BITWISE identical.
- Compile budget: all shapes together must trace+compile within 90 seconds.
- Score: median wall-clock speed versus the production baseline, measured interleaved with fresh inputs per iteration; correctness is re-verified on an output from a TIMED invocation.

## Environment

jax 0.10.2, TPU (v6e). `from jax.experimental import pallas as pl`, `from jax.experimental.pallas import tpu as pltpu`.

## Output format

Output ONLY a single self-contained Python program in one ```python code block. No prose outside it. It must define `kernel` at module level and import everything it uses.
'''


# --------------------------------------------------------------------------
# VARIANT 2 -- reference oracle + gotchas
# --------------------------------------------------------------------------

SPLASH_REFERENCE = SPLASH_MINIMAL + r'''

## The correctness oracle

This is the exact fp32 closed form the judge grades you against. It is NOT a
valid answer: it materializes the full [heads, seq, seq] logits array (2.1 GB
at heads=8, seq=4096) and it contains no `pallas_call`. Read it for the exact
masking and normalization semantics you must reproduce.

```python
import jax.numpy as jnp
import numpy as np

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
```

## Pallas / Mosaic gotchas

1. Scoped VMEM ceiling is ~32MB per `pallas_call`. A block that asks for more
   fails at compile time with
   `CompileTimeScopedVmemOom: Scoped allocation with size 36.25M and limit 32.00M`.
   Budget bytes-per-element for EVERY live array inside the kernel body, not
   just the inputs -- an f32 upcast of a bf16 block doubles it. This arena
   measured 23-24 bytes/element for a kernel that upcasts one bf16 input to f32
   and keeps two f32 temporaries.
2. Block sizes must tile the array. A `BlockSpec` block shape that does not
   divide the operand shape is a compile error, not a partial block.
3. Non-block-divisible shapes therefore need EXPLICIT padding: `jnp.pad` up to a
   multiple of the block, run the grid over the padded shape, slice the result
   back down. seq=2049 is in the declared set precisely to force this.
4. The last dimension of a TPU array is tiled to a multiple of 128 and the
   second-to-last to a multiple of 8. Block shapes that ignore this either fail
   or waste VMEM.
5. A raw `pallas_call` is NOT differentiable. (For THIS task the contract is
   forward-only, so you do not need `jax.custom_vjp` here -- but do not assume
   `jax.grad` would work through it.)
6. ONE implementation must trace at EVERY declared shape: block sizes derived
   from the shape are fine, but they must be computed from Python ints at trace
   time, not from traced values.
7. Compile budget is 90 seconds for all shapes together; a deeply unrolled grid
   or a huge `jnp.stack` of per-step results will blow it.
8. jax is pinned at 0.10.2. Do not use APIs added later.
   `pl.BlockSpec(block_shape, index_map)` is the current argument order.
9. `pl.pallas_call` needs `out_shape=jax.ShapeDtypeStruct(...)` and `grid=(...)`;
   `index_map` returns BLOCK indices, not element offsets.
10. Reductions across the key axis must be done with a running (online-softmax)
    max and sum if you tile the key axis -- recomputing a global max needs the
    whole row, which does not fit.

Restated: output ONLY a single self-contained Python program, defining `kernel` at module level, with a real `pallas_call`, in one ```python code block.
'''


# --------------------------------------------------------------------------
# VARIANT 3 -- fill-in-the-blank scaffold
# --------------------------------------------------------------------------

SPLASH_TAILORED = SPLASH_REFERENCE + r'''

## Scaffold

Everything structural is already written below: the head_dim/seq padding, the
block-size choice and its VMEM budget, the grid, every `BlockSpec` and
`index_map`, the flash-attention accumulators (running max `m`, running sum `l`,
running output `o`) and the final un-pad slice. Exactly ONE function is missing:
`_kern`, the inner Pallas kernel body for one grid step.

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# TPU native tiling: the last dim of an array tiles to 128 lanes, the
# second-to-last to 8 sublanes.
_LANES = 128
_SUBLANES = 8

# Block sizes. Scoped-VMEM arithmetic for the biggest scored shape
# (heads=8, seq=4096, head_dim=128, bf16 in / f32 accumulators), BQ=BK=512;
# Pallas double-buffers every operand block, hence the x2 on the I/O terms:
#   q  block  bf16 [512, 128]     131072 B  x2  =    262144
#   k  block  bf16 [512, 128]     131072 B  x2  =    262144
#   v  block  bf16 [512, 128]     131072 B  x2  =    262144
#   o  block  f32  [512, 128]     262144 B  x2  =    524288
#   seg_q     i32  [512, 128]     262144 B  x2  =    524288
#   seg_k     i32  [8, 512]        16384 B  x2  =     32768
#   m, l   f32 [512, 128] scratch 262144 B  x2  =    524288   (not buffered)
#   body temporaries on the [512, 512] f32 score tile:
#                                 512*512*24    =   6291456
#   ------------------------------------------------------------------
#   TOTAL                                       =   8683520 B = 8.28 MB
# Comfortably inside the 24 MB self-imposed budget and the 32 MB ceiling.
_BQ = 512
_BK = 512
_VMEM_BUDGET = 24 * 1024 * 1024
# Bytes of live scoped VMEM per element of the [BQ, BK] score tile: the f32
# scores, the causal and segment mask temporaries, the exponentials and the
# rescaled probabilities are all live at once -- about six f32 tiles.
_BYTES_PER_ELT = 24


def _scoped_bytes(bq: int, bk: int, d: int) -> int:
    """Scoped VMEM for ONE grid step, in bytes."""
    io = 2 * (
        bq * d * 2  # q block, bf16
        + 2 * bk * d * 2  # k and v blocks, bf16
        + bq * d * 4  # o block, f32
        + bq * _LANES * 4  # q segment ids, i32, broadcast over 128 lanes
        + _SUBLANES * bk * 4  # kv segment ids, i32, broadcast over 8 sublanes
    )
    scratch = 2 * bq * _LANES * 4  # running max m and running sum l, f32
    return io + scratch + bq * bk * _BYTES_PER_ELT


def _pick_blocks(d: int) -> tuple[int, int]:
    """Shrink the blocks until one grid step fits the budget. Pure Python ints
    at trace time, so the same source traces at every declared shape."""
    bq, bk = _BQ, _BK
    while bk > 128 and _scoped_bytes(bq, bk, d) > _VMEM_BUDGET:
        bk //= 2
    while bq > 128 and _scoped_bytes(bq, bk, d) > _VMEM_BUDGET:
        bq //= 2
    return bq, bk


def _make_kernel(bq: int, bk: int, n_k: int):
    def _kern(q_ref, k_ref, v_ref, sq_ref, sk_ref, o_ref, m_ref, l_ref):
        # TODO: THIS IS THE ONLY FUNCTION YOU MUST WRITE.
        #
        # One grid step. The grid is (heads, n_q, n_k) with the KEY-block axis
        # innermost, so for a fixed (head, q-block) this body runs for
        # j = 0, 1, ..., n_k - 1 in order, and o_ref / m_ref / l_ref PERSIST
        # across those calls: the output block's index_map does not depend on
        # j (so Pallas keeps the block in VMEM and writes it back once), and
        # the scratch buffers live for the whole grid.
        #
        #   i = pl.program_id(1)   # q-block index,   0 <= i < n_q
        #   j = pl.program_id(2)   # key-block index, 0 <= j < n_k
        #
        # bq and bk are the Python ints closed over by this factory (512 and
        # 512 for every declared shape); dp is head_dim padded up to 128.
        # Refs handed to you, and what they hold:
        #   q_ref   [1, bq, dp]  bfloat16  queries, global rows i*bq + 0..bq-1
        #   k_ref   [1, bk, dp]  bfloat16  keys,    global cols j*bk + 0..bk-1
        #   v_ref   [1, bk, dp]  bfloat16  values,  same rows as k_ref
        #   sq_ref  [bq, _LANES] int32     segment id of each QUERY row,
        #                                  broadcast across the 128 lanes;
        #                                  sq_ref[:, :1] is the [bq, 1] column
        #   sk_ref  [_SUBLANES, bk] int32  segment id of each KEY column,
        #                                  broadcast across the 8 sublanes;
        #                                  sk_ref[:1, :] is the [1, bk] row
        #   o_ref   [1, bq, dp]  float32   OUTPUT block   -- you write it
        #   m_ref   [bq, _LANES] float32   running row max -- you write it
        #   l_ref   [bq, _LANES] float32   running row sum -- you write it
        # m_ref and l_ref are 128 lanes wide only because 128 is the minimum
        # TPU lane tile; every lane holds the same per-row scalar, so read them
        # as m_ref[:, :1] / l_ref[:, :1] and store a full-width broadcast back.
        #
        # WHAT MUST BE TRUE when the last key block for this q-block returns:
        #   o_ref[0] == (softmax over every VISIBLE key) @ V, in float32, where
        #   key column c of block j is visible to query row r of block i iff
        #       (j*bk + c) <= (i*bq + r)                       # causal
        #   and sq_ref[r] == sk_ref[c] and sq_ref[r] != 0      # same live seg
        #   and a query row with NO visible key (a padding row, or a row past
        #   the real seq length) is EXACTLY 0.0, not NaN.
        #   There is NO 1/sqrt(head_dim) scale: the logits are the raw dot
        #   products, exactly as in the oracle above.
        #
        # The standard online-softmax step you need to write:
        #   * on j == 0, initialize m_ref to a large finite negative constant
        #     (NOT -jnp.inf: -inf minus -inf is NaN), l_ref to 0, o_ref to 0;
        #   * s = q_ref[0].astype(f32) @ k_ref[0].astype(f32).T  -> [bq, bk];
        #   * set s to that large negative constant wherever the key is not
        #     visible to the query;
        #   * m_new = maximum(m_old, rowmax(s));   p = exp(s - m_new);
        #     corr = exp(m_old - m_new);
        #     l_new = corr * l_old + rowsum(p);
        #     o_new = corr * o_old + p @ v_ref[0].astype(f32);
        #   * store m_new / l_new / o_new back into their refs;
        #   * on the LAST key block, divide o_ref by l (guarding l == 0 so a
        #     fully masked row yields exactly 0.0 rather than NaN) and store
        #     the final float32 result.
        # Use `pl.when(cond)` for the first-block and last-block special cases;
        # `n_k` is a Python int closed over by this factory. Optional speedup:
        # a key block strictly above the diagonal contributes nothing, so you
        # may skip it -- but then the finalize step must run on the last key
        # block you actually VISIT, not on j == n_k - 1.
        raise NotImplementedError

    return _kern


@jax.jit
def _fwd(q, k, v, segment_ids):
    h, s, d = q.shape
    dp = ((d + _LANES - 1) // _LANES) * _LANES  # pad head_dim to 128 lanes
    bq, bk = _pick_blocks(dp)
    blk = max(bq, bk)  # both are powers of two, so a multiple of max(bq, bk)
    sp = ((s + blk - 1) // blk) * blk  # is a multiple of both
    pad_s, pad_d = sp - s, dp - d

    def _pad(x):
        if pad_s or pad_d:
            return jnp.pad(x, ((0, 0), (0, pad_s), (0, pad_d)))
        return x

    # Zeros in the padded head_dim contribute 0 to every dot product, and the
    # padded seq tail gets segment id 0 == PADDING, so the mask kills it free.
    qp, kp, vp = _pad(q), _pad(k), _pad(v)
    segp = jnp.pad(segment_ids, (0, pad_s)) if pad_s else segment_ids
    seg_q = jnp.broadcast_to(segp[:, None], (sp, _LANES))
    seg_k = jnp.broadcast_to(segp[None, :], (_SUBLANES, sp))

    n_q, n_k = sp // bq, sp // bk
    out = pl.pallas_call(
        _make_kernel(bq, bk, n_k),
        grid=(h, n_q, n_k),
        in_specs=[
            pl.BlockSpec((1, bq, dp), lambda hh, i, j: (hh, i, 0)),  # q
            pl.BlockSpec((1, bk, dp), lambda hh, i, j: (hh, j, 0)),  # k
            pl.BlockSpec((1, bk, dp), lambda hh, i, j: (hh, j, 0)),  # v
            pl.BlockSpec((bq, _LANES), lambda hh, i, j: (i, 0)),  # seg_q
            pl.BlockSpec((_SUBLANES, bk), lambda hh, i, j: (0, j)),  # seg_k
        ],
        out_specs=pl.BlockSpec((1, bq, dp), lambda hh, i, j: (hh, i, 0)),
        out_shape=jax.ShapeDtypeStruct((h, sp, dp), jnp.float32),
        scratch_shapes=[
            pltpu.VMEM((bq, _LANES), jnp.float32),  # m: running row max
            pltpu.VMEM((bq, _LANES), jnp.float32),  # l: running row sum
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
        ),
    )(qp, kp, vp, seg_q, seg_k)
    return out[:, :s, :d]


def kernel(q, k, v, segment_ids):
    return _fwd(q, k, v, segment_ids)
```

Return the COMPLETE program with `_kern` filled in. Do not change anything
outside `_kern`. Output it as one ```python code block and nothing else.
'''


PROMPTS = {
    "minimal": SPLASH_MINIMAL,
    "reference": SPLASH_REFERENCE,
    "tailored": SPLASH_TAILORED,
}
