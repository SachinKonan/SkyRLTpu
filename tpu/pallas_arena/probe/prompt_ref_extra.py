"""`reference` prompts for the three arena tasks that never had one.

The previous probe only ever built prompts for `splash_attention` and `flce`.
The seam-vs-reference comparison is the whole point of this run, so the other
three tasks need a control arm at the same strength: task spec + signature +
declared shapes + judge contract, then the fp32 correctness oracle inline
(labelled as an INVALID answer) and the Pallas/Mosaic gotcha list drawn from
what this repo has actually hit.

Deliberately the same shape as `prompt_flce.FLCE_REFERENCE` and
`prompt_splash.SPLASH_REFERENCE` so "reference" means the same thing in all
five cells and a task-to-task difference is about the task, not the prompt.
"""

from __future__ import annotations

# The gotcha list, verbatim-equivalent to the two existing `reference` prompts,
# with the task-specific item slotted in by the caller.
_GOTCHAS = r'''
## JAX / Pallas / Mosaic gotchas

1. Scoped VMEM ceiling is ~32MB per `pallas_call`. A block that asks for more fails at compile time with `CompileTimeScopedVmemOom: Scoped allocation with size 36.25M and limit 32.00M`. Budget bytes-per-element for EVERY live array inside the kernel body, not just the inputs -- an f32 upcast of a bf16 block doubles it. This arena measured 23-24 bytes/element for a kernel that upcasts one bf16 input to f32 and keeps two f32 temporaries.
2. Block sizes must tile the array. A `BlockSpec` block shape that does not divide the operand shape is a compile error, not a partial block.
3. Non-block-divisible shapes therefore need EXPLICIT padding: `jnp.pad` up to a multiple of the block, run the grid over the padded shape, slice the result back down. The holdout shape is in the declared set precisely to force this.
4. The last dimension of a TPU array is tiled to a multiple of 128 (lanes) and the second-to-last to a multiple of 8 (sublanes). Block shapes that ignore this either fail or waste VMEM.
5. A raw `pallas_call` is NOT differentiable. (This task's contract is forward-only, so you do not need `jax.custom_vjp` -- but do not assume `jax.grad` would work through it.)
6. ONE implementation must trace at EVERY declared shape: block sizes derived from the shape are fine, but they must be computed from Python ints at trace time, not from traced values.
7. Compile budget is 90 seconds for all shapes together. `jax.lax.scan` compiles ONE body; a Python `for` loop over N tiles compiles N copies and will blow it.
8. jax is pinned at 0.10.2. `pl.BlockSpec(block_shape, index_map)` is the current positional order; `pl.pallas_call` needs `out_shape=jax.ShapeDtypeStruct(...)` and `grid=(...)`; `index_map` returns BLOCK indices, not element offsets. There is no `pl.load`, no `pl.store` and no `pltpu.ANY` at this pin.
9. On TPU, `jax.lax.dot_general` on float32 operands runs ONE bfloat16 pass at the default precision. `preferred_element_type=jnp.float32` gives an f32 accumulator over bf16 inputs; `precision=jax.lax.Precision.HIGHEST` is accurate and about 6x the passes. This is a real accuracy knob -- a control kernel in this arena missed the calibrated tolerance by 18% because of it.
10. Reductions across a tiled axis must use a running (online) max and sum: recomputing a global max needs the whole row, which does not fit.
'''

_OUTRO = r'''
Restated: output ONLY a single self-contained Python program, defining `kernel` at module level, in one ```python code block. No prose outside it.
'''


# ==========================================================================
RPA_MINIMAL = r'''You are an expert JAX/Pallas TPU kernel engineer. Write a JAX Pallas TPU kernel implementing ragged paged attention for DECODE (one query token per sequence, KV read from a paged cache, GQA).

## Task

Entrypoint (exact): a module-level function named `kernel`:

    def kernel(q, k_pages, v_pages, page_tables, seq_lens):  ->  o

  q            : [batch, q_heads, head_dim]                    bfloat16 (1 token/seq)
  k_pages      : [num_pages, page_size, kv_heads, head_dim]    bfloat16
  v_pages      : [num_pages, page_size, kv_heads, head_dim]    bfloat16
  page_tables  : [batch, max_pages_per_seq]                    int32 (physical page ids)
  seq_lens     : [batch]                                       int32, always >= 1
  o            : [batch, q_heads, head_dim]                    float32

GQA: `q_heads = kv_heads * group`; query head h attends KV head `h // group`. Cached position t of sequence b lives in physical page `page_tables[b, t // page_size]` at slot `t % page_size`. Positions >= `seq_lens[b]` are masked. There is no causal mask (there is one query token) and NO 1/sqrt(head_dim) scaling -- the logits are the raw dot products. Softmax at float32. Forward pass only.

## Declared shapes

page_size=64, kv_heads=8, q_heads=32 (group=4), head_dim=128 throughout. One implementation must trace and run at ALL of these:

    batch=16, max_len=1024, num_pages=264   (scored)
    batch=8,  max_len=512,  num_pages=72    (scored)
    batch=17, max_len=512,  num_pages=144   (HOLDOUT: logged, not scored, but the kernel MUST still trace and be correct here -- batch=17 is prime, so nothing about it tiles)

`max_pages_per_seq = max_len // page_size`, i.e. 16, 8 and 8.

## Judge contract

- Your program is exec'd, then `kernel` is serialized with `jax.export` once per declared shape, with NO concrete data and NO device. It must trace at every declared shape.
- Banned: importing `jax.experimental.pallas.ops.*`, `tpu_inference`, `MaxText`, `maxtext`, `skyrl`, `recurrentgemma`, and the arena's own package. Calling `jax.nn.dot_product_attention` is banned. Enforced by a static AST screen AND runtime module poisoning.
- Your program MUST contain a real `pallas_call`.
- Correctness: per-element error `|cand - ref| / (|ref| + 1)` against an fp32 closed-form reference on hidden random seeds, checked on BOTH the max and the 99th-percentile tail, at a tolerance calibrated to 1.5x the reference's own bfloat16 error. Non-finite output is an automatic failure.
- Determinism: 5 repeated runs on the same input must be BITWISE identical.
- Compile budget: all shapes together must trace+compile within 90 seconds.
- This task is memory-bandwidth bound; the judge also reports what fraction of the chip's peak HBM bandwidth you reached.
- Score: median wall-clock speed versus the judge's baseline, measured interleaved with fresh inputs per iteration; correctness is re-verified on an output from a TIMED invocation. NOTE: this judge's baseline is a competent batch-blocked XLA paged decode, not vLLM's production Pallas kernel.

## Environment

jax 0.10.2, TPU (v6e). `from jax.experimental import pallas as pl`, `from jax.experimental.pallas import tpu as pltpu`.

## Output format

Output ONLY a single self-contained Python program in one ```python code block. No prose outside it. It must define `kernel` at module level and import everything it uses.
'''

RPA_REFERENCE = (
    RPA_MINIMAL
    + r'''

## The correctness oracle

This is the exact fp32 closed form the judge grades you against. It is NOT a
valid answer: it gathers the whole paged cache into `[batch, max_len, kv_heads,
head_dim]` fp32 and contains no `pallas_call`. Read it for the exact masking
and normalization semantics you must reproduce.

```python
def paged_decode_attention_reference(q, k_pages, v_pages, page_tables, seq_lens):
    b, qh, d = q.shape
    _, page_size, kvh, _ = k_pages.shape
    group = qh // kvh
    max_len = page_tables.shape[1] * page_size

    k = k_pages[page_tables].reshape(b, max_len, kvh, d).astype(jnp.float32)
    v = v_pages[page_tables].reshape(b, max_len, kvh, d).astype(jnp.float32)
    q32 = q.astype(jnp.float32).reshape(b, kvh, group, d)

    logits = jnp.einsum("bhgd,bthd->bhgt", q32, k)          # [b, kvh, group, t]
    pos = jnp.arange(max_len)
    live = pos[None, :] < seq_lens[:, None]                  # [b, t]
    logits = jnp.where(live[:, None, None, :], logits, -0.7 * float(np.finfo(np.float32).max))
    m = jnp.max(logits, axis=-1, keepdims=True)
    p = jnp.exp(logits - m)
    p = jnp.where(live[:, None, None, :], p, 0.0)
    p = p / jnp.maximum(jnp.sum(p, axis=-1, keepdims=True), 1e-30)
    o = jnp.einsum("bhgt,bthd->bhgd", p, v)
    return o.reshape(b, qh, d)
```

A page id may be gathered with a scalar-prefetched index map:
`pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=n, grid=..., in_specs=[...])`
makes the first `n` operands visible to every `index_map` as its trailing
arguments, so a `BlockSpec` can select the physical page for this grid step.
'''
    + _GOTCHAS
    + _OUTRO
)


# ==========================================================================
GMM_MINIMAL = r'''You are an expert JAX/Pallas TPU kernel engineer. Write a JAX Pallas TPU grouped matmul (MoE `gmm`) kernel.

## Task

Entrypoint (exact): a module-level function named `kernel`:

    def kernel(lhs, rhs, group_sizes):  ->  out

  lhs          : [m, k]              bfloat16  (tokens, grouped CONTIGUOUSLY by expert)
  rhs          : [num_groups, k, n]  bfloat16  (per-expert weights)
  group_sizes  : [num_groups]        int32, sums to m; ZEROS ARE LEGAL
  out          : [m, n]              float32

`out[rows of group g] = lhs[rows of group g] @ rhs[g]`, at float32 accumulation. Rows are assigned by an exclusive scan of `group_sizes`: group g owns rows `sum(group_sizes[:g]) .. sum(group_sizes[:g+1]) - 1`. Forward pass only.

THE CORRECTNESS TRAP: a row tile of any fixed size will in general STRADDLE two experts, and a group of size 0 must contribute no rows at all. Group sizes are freshly sampled per grading from BOTH uniform and Zipf-skewed distributions, so a kernel tuned for one balance cannot fake a win on the other.

## Declared shapes

One implementation must trace and run at ALL of these:

    m=4096, num_groups=4, k=4096, n=14336, uniform group sizes  (scored)
    m=2048, num_groups=4, k=4096, n=14336, Zipf-skewed          (scored)
    m=3000, num_groups=4, k=4096, n=14336, Zipf-skewed          (HOLDOUT: logged, not scored, but the kernel MUST still trace and be correct here -- m=3000 is deliberately not a multiple of any reasonable row tile)

## Judge contract

- Your program is exec'd, then `kernel` is serialized with `jax.export` once per declared shape, with NO concrete data and NO device. It must trace at every declared shape.
- Banned: importing `jax.experimental.pallas.ops.*` (that tree contains the megablox reference kernel), `tpu_inference`, `MaxText`, `maxtext`, `skyrl`, `recurrentgemma`, and the arena's own package. Enforced by a static AST screen AND runtime module poisoning.
- Your program MUST contain a real `pallas_call`.
- Correctness: per-element error `|cand - ref| / (|ref| + 1)` against an fp32 closed-form reference on hidden random seeds, checked on BOTH the max and the 99th-percentile tail, at a tolerance calibrated to 1.5x the reference's own bfloat16 error. Non-finite output is an automatic failure.
- Determinism: 5 repeated runs on the same input must be BITWISE identical.
- Compile budget: all shapes together must trace+compile within 90 seconds.
- Score: median wall-clock speed versus the tuned Pallas megablox `gmm` that MaxText vendors, measured interleaved with fresh inputs per iteration; correctness is re-verified on an output from a TIMED invocation.

## Environment

jax 0.10.2, TPU (v6e). `from jax.experimental import pallas as pl`, `from jax.experimental.pallas import tpu as pltpu`.

## Output format

Output ONLY a single self-contained Python program in one ```python code block. No prose outside it. It must define `kernel` at module level and import everything it uses.
'''

GMM_REFERENCE = (
    GMM_MINIMAL
    + r'''

## The correctness oracle

This is the exact fp32 closed form the judge grades you against. It is NOT a
valid answer: it is a single XLA op with no `pallas_call`, and it upcasts both
operands to fp32. Read it for the exact grouping semantics you must reproduce.

```python
def gmm_reference(lhs, rhs, group_sizes):
    return jax.lax.ragged_dot(lhs.astype(jnp.float32), rhs.astype(jnp.float32), group_sizes)
```

Equivalently, and this is the definition that matters when you think about
boundaries and empty experts:

```python
out = np.zeros((m, n), np.float32)
row = 0
for g, size in enumerate(group_sizes):
    if size:
        out[row:row + size] = lhs[row:row + size] @ rhs[g]
    row += size
```

Selecting a per-tile expert needs a scalar-prefetched index map:
`pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=n, grid=..., in_specs=[...])`
makes the first `n` operands visible to every `index_map` as its trailing
arguments, so `rhs`'s `BlockSpec` can pick the expert for this grid step.
'''
    + _GOTCHAS
    + _OUTRO
)


# ==========================================================================
RGLRU_MINIMAL = r'''You are an expert JAX/TPU kernel engineer. Write a fast RG-LRU gated diagonal linear scan (the RecurrentGemma SSM recurrence) in JAX for TPU.

## Task

Entrypoint (exact): a module-level function named `kernel`:

    def kernel(x, a, reset):  ->  h

  x      : [b, t, d]  bfloat16
  a      : [b, t, d]  float32, in [0, 1)   (gates, precomputed)
  reset  : [b, t]     bool
  h      : [b, t, d]  float32

The recurrence, along the time axis, per batch element and per channel:

    h_t = a_t * h_{t-1} + sqrt(max(1 - a_t^2, 0)) * x_t

with `h_{-1} = 0`, and `a_t` forced to 0 wherever `reset_t` is True, so no state crosses a segment boundary (at a reset, `h_t = x_t` exactly). The gates are concentrated near 1 (long memory), so `sqrt(1 - a^2)` is small and float32 drift over long T is the difficulty this task exists for. Forward pass only.

## Declared shapes

Width is the real RecurrentGemma-2B one: d=2560. One implementation must trace and run at ALL of these:

    b=4, t=2048, d=2560   (scored)
    b=2, t=1024, d=2560   (scored)
    b=2, t=1500, d=2560   (HOLDOUT: logged, not scored, but the kernel MUST still trace and be correct here -- t=1500 is deliberately not a multiple of any reasonable chunk)

## Judge contract

- Your program is exec'd, then `kernel` is serialized with `jax.export` once per declared shape, with NO concrete data and NO device. It must trace at every declared shape.
- Banned: importing `recurrentgemma`, `jax.experimental.pallas.ops.*`, `tpu_inference`, `MaxText`, `maxtext`, `skyrl`, and the arena's own package. Enforced by a static AST screen AND runtime module poisoning.
- A `pallas_call` is ALLOWED but NOT required for this task -- `jax.lax.associative_scan` is an explicitly legal strategy.
- Correctness: per-element error `|cand - ref| / (|ref| + 1)` against an fp32 closed-form reference on hidden random seeds, checked on BOTH the max and the 99th-percentile tail, at a tolerance calibrated to 1.5x the reference's own drift. Non-finite output is an automatic failure.
- Determinism: 5 repeated runs on the same input must be BITWISE identical.
- Compile budget: all shapes together must trace+compile within 90 seconds.
- This task is memory-bandwidth bound; the judge also reports what fraction of the chip's peak HBM bandwidth you reached.
- Score: median wall-clock speed versus the judge's baseline, measured interleaved with fresh inputs per iteration; correctness is re-verified on an output from a TIMED invocation. NOTE: this judge's baseline is `jax.lax.associative_scan`, not DeepMind's recurrentgemma Pallas scan.

## Environment

jax 0.10.2, TPU (v6e). If you use Pallas: `from jax.experimental import pallas as pl`, `from jax.experimental.pallas import tpu as pltpu`.

## Output format

Output ONLY a single self-contained Python program in one ```python code block. No prose outside it. It must define `kernel` at module level and import everything it uses.
'''

RGLRU_REFERENCE = (
    RGLRU_MINIMAL
    + r'''

## The correctness oracle

This is the exact fp32 closed form the judge grades you against. It is a VALID
but SLOW answer: a fully sequential `lax.scan` of T steps, which is exactly the
serialization you are being asked to beat.

```python
def rg_lru_scan_reference(x, a, reset):
    x32 = x.astype(jnp.float32)
    a32 = a.astype(jnp.float32) * (1.0 - reset[..., None].astype(jnp.float32))
    gx = jnp.sqrt(jnp.maximum(1.0 - jnp.square(a32), 0.0)) * x32

    def step(h, xs):
        a_t, gx_t = xs
        h = a_t * h + gx_t
        return h, h

    h0 = jnp.zeros((x32.shape[0], x32.shape[2]), jnp.float32)
    _, hs = jax.lax.scan(step, h0, (jnp.moveaxis(a32, 1, 0), jnp.moveaxis(gx, 1, 0)))
    return jnp.moveaxis(hs, 0, 1)
```

The parallel-scan formulation, which is the legal floor and what the baseline
uses: the recurrence is a first-order affine map, so `(a, gx)` composes under
`combine((a_l, b_l), (a_r, b_r)) = (a_l * a_r, b_l * a_r + b_r)` and
`jax.lax.associative_scan(combine, (a32, gx), axis=1)` returns the same `h` in
log depth at the cost of more flops and more live memory.
'''
    + _GOTCHAS.replace(
        "5. A raw `pallas_call` is NOT differentiable. (This task's contract is forward-only, so you do not need `jax.custom_vjp` -- but do not assume `jax.grad` would work through it.)",
        "5. A raw `pallas_call` is NOT differentiable, and it is not required here: `lax.associative_scan` is a legal answer. The contract is forward-only either way.",
    )
    + _OUTRO
)


REFERENCE_PROMPTS: dict[str, str] = {
    "ragged_paged_attention": RPA_REFERENCE,
    "megablox_gmm": GMM_REFERENCE,
    "rg_lru": RGLRU_REFERENCE,
}
