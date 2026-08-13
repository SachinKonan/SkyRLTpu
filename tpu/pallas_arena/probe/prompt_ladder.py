"""The PROMPT LADDER: rungs P1 / P3 / P4, five tasks, one variable per rung.

Pure DATA. Built to answer one question per kernel -- *what is the least you
have to add before gemma-4-31B-it reliably produces WORKING code that still
has room to improve?* -- and built from the measured failure signatures of
job 3651278, never from intuition.

The rungs, and what each one adds (each is a strict superset of the one below):

  BASE  task spec, exact signature, declared shapes, judge contract, and the
        fp32 correctness oracle. Held CONSTANT across every rung so that a
        rung-to-rung difference is about the addition, not about the task
        statement. (This is the previous run's `minimal` + its oracle; the
        `minimal` arm alone was measured at 0/32 export on splash and 2/20
        PASS on flce, so it is the floor, not a candidate.)
  P1    + DIALECT: a compact bullet list, every bullet carrying the verbatim
        error string it eliminates and the number of job-3651278 candidates
        that died on it.
  P3    + a typed SKELETON of the entrypoint (the ladder's P2) and ONE
        complete WORKED EXAMPLE Pallas kernel in the right dialect, which also
        pins the API version. P2 and P3 are merged into a single rung because
        the wall budget affords 15 cells, not 25, and P2 alone (a skeleton
        with no worked example) does not address the modal failure -- an
        invented `pallas_call` signature -- at all.
  P4    + PRIMITIVES: tested helpers, prepended to the model's program by
        `ladder.compose_ladder`, documented here by signature. FLCE's ladder
        differs, as its failure is mathematical rather than dialectal: its P4
        states the backward CONTRACT instead.

What is deliberately NOT here: a fill-in-the-blank scaffold. The `tailored`
variant already measured that trap -- 16/16 PASS with a within-group reward
spread of 0.0042 against noise floors of 0.0069/0.0158, i.e. no gradient at
all. Every rung below leaves block shapes, tile shapes, chunk lengths,
one-pass vs two-pass, which blocks to skip and the matmul precision trade
entirely to the model.
"""

from __future__ import annotations

from pallas_arena.probe.prompt_flce import FLCE_MINIMAL
from pallas_arena.probe.prompt_ref_extra import GMM_MINIMAL, RGLRU_MINIMAL, RPA_MINIMAL
from pallas_arena.probe.prompt_splash import SPLASH_MINIMAL

# ==========================================================================
# The fp32 correctness oracles. Part of BASE, identical at every rung.
# ==========================================================================

_ORACLES: dict[str, str] = {
    "splash_attention": r'''
## The correctness oracle

The exact fp32 closed form you are graded against. NOT a valid answer: it
materializes [heads, seq, seq] logits and contains no `pallas_call`.

```python
NEG_INF = -0.7 * float(np.finfo(np.float32).max)
logits = jnp.einsum("hqd,hkd->hqk", q32, k32)
causal = idx[:, None] >= idx[None, :]                 # idx = jnp.arange(seq)
mask = causal & (seg[:, None] == seg[None, :]) & (seg != 0)[:, None] & (seg != 0)[None, :]
logits = jnp.where(mask[None], logits, NEG_INF)
row_live = mask.any(axis=-1)                          # rows with >=1 visible key
p = jnp.where(mask[None], jnp.exp(logits - jnp.max(logits, -1, keepdims=True)), 0.0)
p = jnp.where(row_live[None, :, None], p / jnp.maximum(p.sum(-1, keepdims=True), 1e-30), 0.0)
out = jnp.einsum("hqk,hkd->hqd", p, v32)              # float32
```
''',
    "ragged_paged_attention": r'''
## The correctness oracle

The exact fp32 closed form you are graded against. NOT a valid answer: it
gathers the whole paged cache and contains no `pallas_call`.

```python
k = k_pages[page_tables].reshape(b, max_len, kvh, d).astype(jnp.float32)
v = v_pages[page_tables].reshape(b, max_len, kvh, d).astype(jnp.float32)
q32 = q.astype(jnp.float32).reshape(b, kvh, group, d)
logits = jnp.einsum("bhgd,bthd->bhgt", q32, k)            # no 1/sqrt(d) scaling
live = jnp.arange(max_len)[None, :] < seq_lens[:, None]   # [b, t]
logits = jnp.where(live[:, None, None, :], logits, -0.7 * np.finfo(np.float32).max)
p = jnp.where(live[:, None, None, :], jnp.exp(logits - logits.max(-1, keepdims=True)), 0.0)
p = p / jnp.maximum(p.sum(-1, keepdims=True), 1e-30)
o = jnp.einsum("bhgt,bthd->bhgd", p, v).reshape(b, qh, d) # float32
```
''',
    "megablox_gmm": r'''
## The correctness oracle

The exact fp32 closed form you are graded against. NOT a valid answer: it is
one `lax.ragged_dot` call and contains no `pallas_call`.

```python
out = jax.lax.ragged_dot(lhs.astype(jnp.float32), rhs.astype(jnp.float32), group_sizes)
# equivalently, with row = exclusive-cumsum(group_sizes):
#   out[row[g] : row[g] + group_sizes[g]] = lhs[row[g] : row[g] + group_sizes[g]] @ rhs[g]
# group_sizes may contain ZEROS, and sum(group_sizes) == m exactly.
```
''',
    "rg_lru": r'''
## The correctness oracle

The exact fp32 closed form you are graded against. It is a sequential
`lax.scan`; a chunked or associative reformulation of the SAME recurrence is
an entirely legal (and much faster) answer.

```python
a32 = a.astype(jnp.float32) * (1.0 - reset[..., None].astype(jnp.float32))  # reset first
gx  = jnp.sqrt(jnp.maximum(1.0 - jnp.square(a32), 0.0)) * x.astype(jnp.float32)

def step(h, xs):          # h: [b, d] float32, starts at zeros
    a_t, gx_t = xs
    h = a_t * h + gx_t
    return h, h
# scanned over the TIME axis; output h is [b, t, d] float32
```
''',
    "flce": r'''
## The correctness oracle

The exact fp32 closed form you are graded against. NOT a valid answer: it
materializes the full [n, v] logits, which is the one thing this task exists
to avoid.

```python
logits = hidden.astype(jnp.float32) @ w.astype(jnp.float32)      # [n, v]
tl = jnp.take_along_axis(logits, targets[:, None], axis=1)[:, 0]
logprobs = tl - jax.nn.logsumexp(logits, axis=-1)                # [n] float32
```
''',
}


# ==========================================================================
# P1 -- the DIALECT list. Every bullet is a VERBATIM error string from job
# 3651278 with the count of candidates it killed. Nothing here is general
# advice; a bullet with no measured victims does not earn its tokens.
# ==========================================================================

DIALECT = r'''
## Pallas dialect: the ten mistakes that actually killed candidates here

These are measured. Each line is the verbatim error and how many of the
previous 160 attempts at these five kernels died on it.

1. **`pallas_call` kwargs are plural.** `out_specs=`, `in_specs=`,
   `out_shape=jax.ShapeDtypeStruct(shape, dtype)`, `grid=`, `scratch_shapes=`,
   `compiler_params=pltpu.CompilerParams(...)`.
   Killed 4: `pallas_call() got an unexpected keyword argument 'out_spec'`.
   These names DO NOT EXIST: `out_spec`, `out_dtype`, `out_shapes`,
   `in_shapes`, `block_shapes`, `index_map=`, `pl.load`, `pl.store`, `pl.Ref`,
   `pltpu.ANY`, `pltpu.TPUCompilerParams`, `jax.Shape`.
2. **Inside the kernel body you are handed Refs; a Ref is not an array.**
   Read `q_ref[...]`, `q_ref[0]`, `q_ref[:, :1]`; write `o_ref[...] = value`
   or `o_ref[0] = value`. A Ref has `.shape` and `.dtype` and NO `.astype` --
   read first, then cast.
3. **`pltpu.VMEM(shape, dtype)` is a SPECIFICATION, legal only inside
   `scratch_shapes=`.** Calling it in a kernel body returns a `MemoryRef`.
   Killed 9: `'MemoryRef' object does not support item assignment`. To
   accumulate inside a body use an ordinary value: `acc = jnp.zeros(...)`,
   `acc = acc + block`.
4. **A store into a Ref is not a broadcast** -- the right-hand side must
   already have the Ref's shape. Killed 6:
   `` Invalid shape for `swap`. Ref shape: (8, 4, 128) ... Value shape: () ``.
   Write `m_ref[...] = jnp.full(m_ref.shape, -1e30, m_ref.dtype)`, not
   `m_ref[...] = -1e30`.
5. **`dot_general`'s `dimension_numbers` is a PAIR OF PAIRS**, never flat.
   Killed 5 (`too many values to unpack (expected 2)`) plus several
   `dot_general requires ...`:
   `jax.lax.dot_general(x, y, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)`
   -- that contracts axis 1 of x with axis 1 of y. WRONG: `((1,), (1,))`,
   `((1,), (1,), (0, 0))`.
6. **`@pl.when(cond)` is a DECORATOR that consumes the function and returns
   `None`.** Decorate and stop. Killed 2 (`'function' object is not iterable`,
   `'NoneType' object is not callable`). Never `with pl.when(...):`, never
   call the decorated name.
7. **Everything read out of a Ref is a TRACER.** `if`, `while`, `bool()`,
   `int()`, python `min`/`max` on one raise `TracerBoolConversionError`
   (killed 3). Use `jnp.where` for values and `@pl.when` for whole blocks.
   Python control flow on the STATIC ints (block sizes, page_size, head
   counts) is fine and unrolls at trace time.
8. **Mosaic tiling.** A block's last dim must be a multiple of 128 (or the
   whole dim) and its second-minor a multiple of 8 (or the whole dim). Killed
   1: `block shape are divisible by 8 and 128 respectively, or be equal to the
   respective dimensions of the overall array`. A `BlockSpec` block must also
   tile the operand: pad with `jnp.pad`, run the grid over the padded shape,
   slice the result back down. The HOLDOUT shape is in the declared set to
   force exactly this.
9. **`kernel` returns ONE array of the declared output shape and dtype**, not
   a tuple, not a Ref. Killed 2 (`function ... returned wrong type from jit`)
   and 2 more on layout (`cannot reshape array of shape (256, 4, 2560) into
   shape (4, 2560)`). If you scanned over time, `jnp.moveaxis` back before
   returning.
10. **`jax.custom_vjp` registers with `.defvjp(fwd, bwd)`**; `fwd` returns
    `(out, residuals)` and `bwd` returns a TUPLE of cotangents, one per
    primal argument. Killed 1: `'custom_vjp' object has no attribute
    'def_fwd'`. And 3 candidates that got the API right still failed the
    gradient gate.

One more, and unlike the rest it is not an API error -- it is the single
biggest cause of wrong ANSWERS here, so treat it as a rule.

**Keep every accumulator in float32. Pass `preferred_element_type=jnp.float32`
on every matmul.**

Your inputs are bfloat16 by contract. The MXU takes bfloat16 operands and sums
them in a wider float32 register -- but `jnp.dot`/`jnp.einsum` on two bfloat16
arrays RETURNS bfloat16, so that float32 accumulator is rounded away at the end
of every matmul unless you ask for it. bfloat16 carries about 3 decimal digits;
these kernels reduce over thousands of keys, pages, vocabulary entries or
timesteps, and the rounding compounds.

What a production TPU kernel does -- splash, vLLM-TPU, megablox, tokamax, all
of them -- is use bfloat16 ONLY as the matmul input format, because that is
what makes the MXU fast, and hold everything between matmuls at float32:
float32 logits, float32 softmax/LSE, float32 output accumulator. Narrow back to
bfloat16 only where a value is about to re-enter the MXU as an operand.

MEASURED against this judge's calibrated tolerance, across all five tasks: the
production path above lands at 0.4-0.7x of the budget, comfortably inside. The
same kernel written with the default dtypes -- bfloat16 logits, bfloat16
softmax, bfloat16 accumulator -- lands at 1.4x to 4.1x, i.e. it FAILS on
correctness while looking completely reasonable.

Do NOT reach for `precision=jax.lax.Precision.HIGHEST` instead. It is about 6x
the passes, no production kernel uses it for bfloat16 inputs, and buying
correctness with it just loses you the speed you are being scored on.
'''


# ==========================================================================
# P3 -- one complete worked Pallas kernel in the right dialect. It is a
# DIFFERENT kernel (RMSNorm) on purpose: it pins the API, the padding
# discipline and the custom_vjp registration without giving away any decision
# that belongs to the task. This code is the arena's own phase-2/4 golden --
# it passes the judge on silicon at three block sizes.
# ==========================================================================

WORKED_EXAMPLE = r'''
## A complete, working Pallas TPU kernel at this exact pin

Not this task -- a fused RMSNorm, included so you can copy the DIALECT rather
than recall it. This program passes this judge on silicon.

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

_BR = 256                       # rows per block: a real choice, budgeted below
_VMEM_BUDGET = 24 * 1024 * 1024  # stay clear of the ~32MB scoped-VMEM limit
_BYTES_PER_ELT = 24              # measured: bf16 in/out + the f32 working set

def _make_kernel(d):
    def _kern(x_ref, g_ref, o_ref):          # Refs, not arrays
        x = x_ref[...].astype(jnp.float32)   # read, THEN cast
        s = jnp.sum(x * x, axis=-1, keepdims=True)
        inv = jax.lax.rsqrt(s / d + 1e-6)
        o_ref[...] = (x * inv * g_ref[...].astype(jnp.float32)).astype(jnp.bfloat16)
    return _kern

@jax.jit
def _fwd(x, g):
    rows, d = x.shape                        # python ints at trace time
    dp = ((d + 127) // 128) * 128            # pad the lane axis to 128
    br = _BR
    while br > 16 and br * dp * _BYTES_PER_ELT > _VMEM_BUDGET:
        br //= 2
    rp = ((rows + br - 1) // br) * br        # pad the row axis to the block
    xp = jnp.pad(x, ((0, rp - rows), (0, dp - d)))
    gp = jnp.pad(g, ((0, dp - d),))
    out = pl.pallas_call(
        _make_kernel(d),
        grid=(rp // br,),
        in_specs=[pl.BlockSpec((br, dp), lambda i: (i, 0)),   # BLOCK indices
                  pl.BlockSpec((1, dp), lambda i: (0, 0))],
        out_specs=pl.BlockSpec((br, dp), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((rp, dp), jnp.bfloat16),
    )(xp, gp.reshape(1, dp))
    return out[:rows, :d]                    # slice the padding back off

@jax.custom_vjp
def _rms(x, g):
    return _fwd(x, g)

def _rms_f(x, g):
    return _fwd(x, g), (x, g)                # (output, residuals)

def _rms_b(res, ct):
    x, g = res
    ...
    return gx.astype(x.dtype), gg.astype(g.dtype)   # one cotangent per primal

_rms.defvjp(_rms_f, _rms_b)

def kernel(x, g):
    return _rms(x, g)
```

Note what it does NOT do: it never allocates scratch inside the body, it never
puts a traced value in a python `if`, and every block size is a python int
computed from the static shape.
'''


# ==========================================================================
# P3 -- the typed skeleton (the ladder's P2), per task.
# ==========================================================================

_SKELETONS: dict[str, str] = {
    "splash_attention": r'''
## Skeleton of the entrypoint

Fill this in however you like; the annotations are the contract, not a design.

```python
def kernel(q, k, v, segment_ids):
    # q, k, v: bf16 [heads, seq, head_dim]   segment_ids: int32 [seq]
    h, s, d = q.shape                    # python ints
    bq, bk = ...                         # YOUR block sizes (multiples of 8 / 128)
    # pad seq up to a multiple of bq (queries) and of bk (keys), and head_dim
    # up to 128; padded query rows get segment id 0 so the mask kills them
    # out = pl.pallas_call(body, grid=(h, sq // bq), in_specs=[...],
    #                      out_specs=..., out_shape=jax.ShapeDtypeStruct(...))(...)
    return out[:, :s, :d]                # float32 [heads, seq, head_dim]
```
''',
    "ragged_paged_attention": r'''
## Skeleton of the entrypoint

Fill this in however you like; the annotations are the contract, not a design.

```python
def kernel(q, k_pages, v_pages, page_tables, seq_lens):
    b, qh, d = q.shape                       # python ints
    npages, ps, kvh, _ = k_pages.shape
    mp = page_tables.shape[1]                # pages per sequence
    grp = qh // kvh                          # GQA group
    # A page id is a DATA-dependent index, so it belongs in an index_map fed by
    # scalar prefetch:
    #   pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=2, grid=(b, mp),
    #       in_specs=[..., pl.BlockSpec((1, ps, kvh, d),
    #                                   lambda bb, p, pt, sl: (pt[bb, p], 0, 0, 0)), ...],
    #       out_specs=..., scratch_shapes=[pltpu.VMEM(...), ...])
    # NOTE a per-head block of [1, ps, 1, d] is REJECTED by Mosaic (second-minor
    # dim must be a multiple of 8 or the whole dim), so one step holds all heads.
    return out.reshape(b, qh, d)             # float32
```
''',
    "megablox_gmm": r'''
## Skeleton of the entrypoint

Fill this in however you like; the annotations are the contract, not a design.

```python
def kernel(lhs, rhs, group_sizes):
    m, k = lhs.shape                         # python ints
    g, _k, n = rhs.shape
    bm, bk, bn = ...                         # YOUR tile sizes
    # group_sizes is TRACED data: no python `if`, no `.item()`. The usual shape
    # is an exclusive cumsum to row offsets, then a per-ROW-TILE expert id fed
    # to the rhs index_map by scalar prefetch, so one row tile never straddles
    # two experts. Zero-size groups are legal and must not break the mapping.
    return out                               # float32 [m, n]
```
''',
    "rg_lru": r'''
## Skeleton of the entrypoint

Fill this in however you like; the annotations are the contract, not a design.

```python
def kernel(x, a, reset):
    b, t, d = x.shape                        # python ints
    a_eff = a * (1.0 - reset[..., None].astype(a.dtype))   # reset BEFORE the scan
    gx = jnp.sqrt(jnp.maximum(1.0 - jnp.square(a_eff), 0.0)) * x.astype(jnp.float32)
    # CHUNK = ...   your chunk length over the TIME axis; pad t up to a multiple
    # of it, carry [b, d] state across chunks, slice back to t at the end.
    return h                                 # float32 [b, t, d], time on axis 1
```
''',
    "flce": r'''
## Skeleton of the entrypoint

Fill this in however you like; the annotations are the contract, not a design.

```python
def kernel(hidden, w, targets):
    n, hdim = hidden.shape                   # python ints
    tile = ...                               # YOUR token tile
    # pad n up to a multiple of `tile`, reshape to [num_tiles, tile, hdim]
    # @jax.custom_vjp over the tiled function; fwd returns (logprobs, residuals)
    # and the residuals must NOT contain any [tile, v] logits.
    # _tiled.defvjp(_tiled_fwd, _tiled_bwd)
    return logprobs                          # float32 [n]
```
''',
}


# ==========================================================================
# P4 -- the primitives, documented by signature. The source of each is in
# `ladder.PRELUDES[task]` and is prepended to the submitted program.
# ==========================================================================

_PRIM_DOT = '''def dot_f32(x, y, cx, cy, hi=False)
    Contract axis cx of x with axis cy of y, float32 accumulation. Its
    dimension_numbers cannot be written wrong. hi=True selects
    Precision.HIGHEST: ~6x the passes, and the only way to hit a tight
    tolerance on f32 operands. THAT TRADE IS YOURS TO MAKE.
'''
_PRIM_IOTA = '''def iota2(shape, dim)
    int32 position indices along `dim`, broadcast over `shape`. In-kernel
    positions, e.g. iota2((bq, 1), 0) and iota2((1, bk), 1).
'''
_PRIM_FILL = '''def fill_ref(ref, value)
    Store a scalar into every element of a Ref at the Ref's own shape --
    the shape-correct form of `m_ref[...] = -1e30`.
'''
_PRIM_SOFTMAX = '''NEG = -1e30

def online_softmax(m, l, o, s, v, live, hi=False) -> (m, l, o)
    ONE streaming (flash) softmax update over a block of keys.
      m [R,1] running max, init to NEG      l [R,1] running sum, init to 0
      o [R,D] running UNNORMALISED output, init to 0
      s [R,K] this block's logits           v [K,D] this block's values
      live    bool [R,K] or [1,K]: True where the key is visible
    Handles the rescale and the masking. Normalise ONCE at the end:
      out = jnp.where(l > 0, o / jnp.maximum(l, 1e-30), 0.0)
    which gives exactly 0 for a fully-masked row instead of NaN.
'''
_PRIM_CHUNK = '''def chunk_scan(a, g)
    h[:, t] = a[:, t] * h[:, t-1] + g[:, t] along axis 1, starting from 0.
    a, g: [B, T, D] float32 -> h: [B, T, D], time still on axis 1 (no
    moveaxis afterwards). Uses lax.associative_scan internally.
'''

_PRIM_HEADERS = {
    "splash_attention": (_PRIM_DOT, _PRIM_IOTA, _PRIM_SOFTMAX),
    "ragged_paged_attention": (_PRIM_DOT, _PRIM_IOTA, _PRIM_FILL, _PRIM_SOFTMAX),
    "megablox_gmm": (_PRIM_DOT, _PRIM_IOTA),
    "rg_lru": (_PRIM_CHUNK,),
}

_PRIM_INTRO = r'''
## Primitives you already have

These are ALREADY DEFINED, immediately above your code, in the same module.
Call them; do not redefine them and do not paste them back. They are tested at
every declared shape. They are also optional -- everything below can be
written by hand, and none of them decides a block shape, a tile shape, a chunk
length, which blocks to skip, or when to stop.

```python
'''

_PRIM_OUTRO = r'''```

None of these contains a `pallas_call`, so the "must contain a real
`pallas_call`" requirement is still entirely yours.
'''


def _primitives(task: str) -> str:
    parts = _PRIM_HEADERS.get(task)
    if not parts:
        return ""
    return _PRIM_INTRO + "\n".join(parts) + _PRIM_OUTRO


# FLCE's ladder differs: it gets the API right and fails at the MATH of the
# recompute backward -- 6 of its 16 candidates cleared every other gate and
# died at `gradient` with q99 error 0.00357 against a 4.55e-05 tolerance.
# Prose about Refs cannot fix that; the gradient formula can.
FLCE_BACKWARD_CONTRACT = r'''
## The backward contract, stated exactly

Six of the previous sixteen FLCE candidates compiled, were numerically correct
FORWARD, and then failed the gradient gate with a q99 error of 0.00357 against
a tolerance of 4.55e-05. The API was fine. The mathematics was not. So:

**What you may save as residuals.** The hidden tiles (and anything of size
[tile] or [tile, 1]). You may NOT save any [tile, v] array -- not the logits,
not the softmax probabilities. If you do, you have kept the thing the task
exists to avoid, and the memory report will show it.

**What the backward must recompute.** One tile's logits at a time, exactly as
the forward computed them, at the same precision. A backward that reuses a
DIFFERENT tiling, a different accumulation dtype, or an approximation of the
softmax is what produces a 0.0036 gradient error: forward and backward must be
the same function.

**The gradient itself.** With `logits = h @ w` at f32 accumulation, and

    lp[i] = logits[i, t_i] - logsumexp(logits[i])

the derivative with respect to `logits` is, for cotangent `c[i]`,

    d(lp)/d(logits)[i, j] = c[i] * (onehot(t_i)[j] - softmax(logits[i])[j])

and therefore

    grad_h[i] = (c[i] * (onehot(t_i) - softmax(logits[i]))) @ w.T      # [h]

The two forms that are actually equivalent and both fine: write that product
by hand, or call `jax.vjp` on the *same* per-tile forward function and let JAX
derive it --

```python
def _bwd_body(_c, xs):
    h_tile, t_tile, c_tile = xs
    _, vjp_fn = jax.vjp(lambda h: _tile_lp(h, t_tile), h_tile)   # SAME _tile_lp
    (grad_h_tile,) = vjp_fn(c_tile)
    return None, grad_h_tile
```

-- which is the production baseline's own backward. `jax.vjp` on the identical
forward is the cheapest way to be exactly right; the hand-written product is
the way to be FASTER, and beating the baseline is the point. Either is legal.

**Shapes.** `grad_h` has `hidden`'s shape, and the cotangent of a bf16 primal
should be cast back to bf16. One candidate died broadcasting `(128, 151936)`
against `(128, 2880)` -- that is `w` un-transposed.

**A forward-only fallback is NOT available**: the judge differentiates
`kernel` and grades the gradient, so a program without a correct backward
scores zero however fast it is.
'''


# ==========================================================================
# Assembly
# ==========================================================================

_MINIMALS = {
    "splash_attention": SPLASH_MINIMAL,
    "ragged_paged_attention": RPA_MINIMAL,
    "megablox_gmm": GMM_MINIMAL,
    "rg_lru": RGLRU_MINIMAL,
    "flce": FLCE_MINIMAL,
}

_CLOSER = r'''
Restated, because it is the one thing that cannot be recovered from: output
ONE ```python code block containing a single self-contained program that
defines `kernel` at module level and imports everything it uses. No prose
outside the block.
'''

TASKS = tuple(_MINIMALS)
RUNGS = ("p1", "p3", "p4")


def _base(task: str) -> str:
    return _MINIMALS[task] + _ORACLES[task]


def build(task: str, rung: str) -> str:
    p1 = _base(task) + DIALECT
    if rung == "p1":
        return p1 + _CLOSER
    p3 = p1 + _SKELETONS[task] + WORKED_EXAMPLE
    if rung == "p3":
        return p3 + _CLOSER
    if rung == "p4":
        extra = FLCE_BACKWARD_CONTRACT if task == "flce" else _primitives(task)
        return p3 + extra + _CLOSER
    raise KeyError(f"unknown rung {rung!r}")


LADDER_PROMPTS: dict[str, dict[str, str]] = {
    task: {rung: build(task, rung) for rung in RUNGS} for task in TASKS
}
