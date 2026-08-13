"""Cold-start probe prompts for arena task `flce` (fused linear cross-entropy).

Pure DATA module: three user-turn prompt variants for the SAME task, differing
only in how much scaffolding the model is handed.

  minimal    task spec + signature + shapes + judge contract, nothing else
  reference  + the fp32 correctness oracle (an INVALID answer) + a JAX/Pallas/
             Mosaic gotcha list
  tailored   + a working fill-in-the-blank scaffold in which everything
             structural is given and exactly one function is a TODO

The variants are strictly nested (`reference` = `minimal` + more, `tailored` =
`reference` + more) so a length/content diff between arms is attributable to
the added block and nothing else. No chat special tokens: the harness renders
those.

Precedent for the register/structure: third_party/discover/examples/gpu_mode/
prompt.py (TRIMUL_PROMPT), transposed from Triton-on-H200 to JAX-on-TPU.
Task contract: pallas_arena/judge/problems/flce.py.
"""

# --------------------------------------------------------------------------
# VARIANT 1 -- minimal
# --------------------------------------------------------------------------

FLCE_MINIMAL = r'''You are an expert JAX/TPU kernel engineer. Write a memory-efficient fused linear cross-entropy (FLCE) kernel in JAX for TPU.

## Task

Entrypoint (exact): a module-level function named `kernel`:

    def kernel(hidden, w, targets):  ->  logprobs

  hidden   : [n, h]  bfloat16   (flattened B*T token axis)
  w        : [h, v]  bfloat16   (frozen output head)
  targets  : [n]     int32
  logprobs : [n]     float32,  logprobs[i] = logits[i, targets[i]] - logsumexp(logits[i])
  where    logits = hidden @ w, with float32 accumulation.

THE POINT OF THE TASK: the full [n, v] logits array must NEVER be materialized -- at the production size that is tens of gigabytes. The judge reports peak HBM, which exposes violators. Tile over the token axis and discard each tile's logits.

CONTRACT IS FORWARD **AND** BACKWARD: the judge differentiates your `kernel` with respect to `hidden` only (w is frozen -- LoRA-only training) and compares the gradient against the fp32 reference at a calibrated tolerance. A naive tiled forward whose residuals keep the per-tile logits defeats the entire purpose, so the backward must RECOMPUTE one tile's logits at a time. In JAX that means `jax.custom_vjp`.

EXPLOIT THE FROZEN HEAD -- this is the main thing that makes this kernel different from a general fused-cross-entropy kernel. Because `w` never receives a gradient, you never form `dw`, which at the production size would be [h, v] = 2880 x 151936, about 437 million values that a general kernel must either hold or recompute per tile. You need only `d/d(hidden)`, which is [tile, h] per tile -- three orders of magnitude smaller. So a tile can be fully retired the moment its `d/d(hidden)` rows are written: nothing about it has to survive into the next tile. Tune for that. A kernel that carries per-tile state as though it still owed a weight gradient is leaving the whole advantage of this setup on the table.

## Declared shapes

One implementation must trace and run at ALL of these:

    n=4096, h=2880, v=151936   (scored)
    n=2048, h=2880, v=151936   (scored)
    n=3000, h=2880, v=151936   (HOLDOUT: logged, not scored, but the kernel MUST
                                still trace and be correct here -- note n=3000 is
                                deliberately NOT a multiple of a reasonable tile size)

## Judge contract

- Your program is exec'd, then `kernel` is serialized with `jax.export` once per declared shape, plus once for the gradient functional, with NO concrete data and NO device. It must trace at every declared shape.
- Banned: importing `jax.experimental.pallas.ops.*`, `tpu_inference`, `MaxText`, `maxtext`, `skyrl`, `recurrentgemma`, and the arena's own package. Enforced by a static AST screen AND runtime module poisoning.
- A `pallas_call` is ALLOWED but NOT required for this task -- the production baseline is a `jax.custom_vjp` over a `jax.lax.scan`, not a Pallas kernel. You may beat it either way.
- Correctness: per-element error `|cand - ref| / (|ref| + 1)` against an fp32 closed-form reference on hidden random seeds, checked on BOTH the max and the 99th-percentile tail, at a tolerance calibrated to 1.5x the reference's own bfloat16 error. Non-finite output is an automatic failure. Log-probabilities must be <= 0.
- Gradient: `d/d(hidden)` of a fixed scalar functional of your output, checked against the fp32 reference at its own calibrated tolerance.
- Determinism: 5 repeated runs on the same input must be BITWISE identical.
- Compile budget: all shapes plus the gradient must trace+compile within 90 seconds.
- Score: median wall-clock speed versus the production baseline, measured interleaved with fresh inputs per iteration; correctness is re-verified on an output from a TIMED invocation.

## Pins

jax 0.10.2, TPU (v6e). If you use Pallas: `from jax.experimental import pallas as pl`, `from jax.experimental.pallas import tpu as pltpu`.

## Output

Output ONLY a single self-contained Python program in one ```python code block. No prose outside it. It must define `kernel` at module level and import everything it uses.
'''


# --------------------------------------------------------------------------
# VARIANT 2 -- minimal + fp32 oracle + gotcha list
# --------------------------------------------------------------------------

FLCE_REFERENCE = FLCE_MINIMAL + r'''
## The correctness oracle

This is the fp32 closed form your output is compared against, verbatim. It is
NOT a valid answer -- it materializes the full [n, v] logits (2.5 GB at n=4096,
44.8 GB at the production token count), which is precisely the thing this task
exists to avoid. Use it only to know exactly what number you must produce.

```python
logits = hidden.astype(jnp.float32) @ w.astype(jnp.float32)
tl = jnp.take_along_axis(logits, targets[:, None], axis=1)[:, 0]
return tl - jax.nn.logsumexp(logits, axis=-1)
```

## JAX / Pallas / Mosaic gotchas

1. Scoped VMEM ceiling is ~32MB per `pallas_call`. A block that asks for more fails at compile time with `CompileTimeScopedVmemOom: Scoped allocation with size 36.25M and limit 32.00M`. Budget bytes-per-element for EVERY live array inside the kernel body, not just the inputs -- an f32 upcast of a bf16 block doubles it.
2. Block sizes must tile the array; a `BlockSpec` block shape that does not divide the operand shape is a compile error, not a partial block.
3. Non-block-divisible shapes therefore need EXPLICIT padding: `jnp.pad` up to a multiple of the tile, run over the padded shape, slice back down. n=3000 is in the declared set precisely to force this.
4. The last dimension of a TPU array is tiled to a multiple of 128 and the second-to-last to a multiple of 8.
5. A raw `pallas_call` is NOT differentiable, and neither is a forward-only tiled scan if you want the memory win: wrap it in `jax.custom_vjp` and recompute each tile's logits inside the backward, keeping only the hidden tiles as residuals.
6. ONE implementation must trace at EVERY declared shape; tile sizes must be Python ints at trace time, not traced values.
7. Compile budget is 90 seconds for all shapes plus the gradient together. `jax.lax.scan` compiles ONE body; a Python `for` loop over 36 tiles compiles 36 copies and will blow it.
8. jax is pinned at 0.10.2. Do not use APIs added later.
9. `logsumexp` must be max-shifted (or computed online) -- at v=151936 an unshifted `exp` overflows f32 and the whole row goes non-finite.
10. `jnp.take_along_axis(logits, targets[:, None], axis=1)[:, 0]` is the target-logit gather; do not use fancy-indexing forms that force a full materialization.

Remember: output ONLY the single self-contained Python program, in one ```python code block, defining `kernel` at module level. No prose outside it.
'''


# --------------------------------------------------------------------------
# VARIANT 3 -- reference + fill-in-the-blank scaffold
# --------------------------------------------------------------------------

FLCE_TAILORED = FLCE_REFERENCE + r'''
## Scaffold (fill in the blank)

Below is a complete, working program EXCEPT for one function. All the structure
is already correct: the tiling and padding arithmetic, the `jax.custom_vjp`
plumbing, the forward scan, the backward scan that recomputes one tile at a
time via `jax.vjp`, the un-pad slice, and the module-level `kernel` entrypoint.

Exactly one function is blank: `_tile_lp`, the per-tile computation. Write it.

```python
import jax
import jax.numpy as jnp

# ----------------------------------------------------------------- tile size
# _TILE = 512 tokens. Transient bytes for ONE tile's logits at v = 151936
# (512 * 151936 = 77,791,232 elements):
#     bf16 [512, 151936] = 77,791,232 * 2 B = 155,582,464 B ~= 148 MiB
#     f32  [512, 151936] = 77,791,232 * 4 B = 311,164,928 B ~= 297 MiB
# so under 0.5 GiB is live at any instant, and it dies with the scan iteration.
# The forbidden full f32 [n, v] materialization, for comparison:
#     n=2048 -> 1.24 GB,  n=4096 -> 2.49 GB,  production n=73728 -> 44.8 GB.
# The residual kept across fwd/bwd is only hf, [T, 512, h] bf16 =
#     n * 2880 * 2 B = 23.6 MB at n=4096.
# 512 divides 4096 (8 tiles) and 2048 (4 tiles) exactly; n=3000 needs 6 tiles
# and pads to 6 * 512 = 3072, i.e. pad = 72 rows. One compiled scan body covers
# all three, so the 90 s compile budget is never in danger.
_TILE = 512


def _tile_lp(h_tile, t_tile, w):
    # TODO: THIS IS THE ONLY FUNCTION YOU MUST WRITE.
    # h_tile: [tile, h] bfloat16; t_tile: [tile] int32; w: [h, v] bfloat16
    # return: [tile] float32 = target logit minus logsumexp, at f32 accumulation,
    #         without keeping the [tile, v] logits alive beyond this call
    raise NotImplementedError


def kernel(hidden, w, targets):
    n, hdim = hidden.shape
    flat = hidden.reshape(n, hdim)
    tgt = targets.reshape(n)

    # tile / pad math -- all Python ints at trace time
    tile = min(_TILE, n)
    num_tiles = -(-n // tile)               # ceil
    pad = num_tiles * tile - n              # 0 when tile divides n; 72 at n=3000
    if pad:
        flat = jnp.pad(flat, ((0, pad), (0, 0)))
        tgt = jnp.pad(tgt, ((0, pad),))     # pad label 0 is a valid vocab id;
                                            # those rows are sliced off at the end
    hf = flat.reshape(num_tiles, tile, hdim)    # [T, tile, h]
    tf = tgt.reshape(num_tiles, tile)           # [T, tile]

    def _fwd_body(carry, xs):
        h_tile, t_tile = xs
        return carry, _tile_lp(h_tile, t_tile, w)

    @jax.custom_vjp
    def _tiled(hf):
        _, lp = jax.lax.scan(_fwd_body, None, (hf, tf))
        return lp                               # [T, tile] float32

    def _tiled_fwd(hf):
        _, lp = jax.lax.scan(_fwd_body, None, (hf, tf))
        return lp, hf   # residual = the hidden tiles ONLY, never the logits

    def _tiled_bwd(hf_res, cot):                # cot: [T, tile]
        def _bwd_body(carry, xs):
            h_tile, t_tile, c_tile = xs
            # recompute THIS tile's logits inside the vjp, then let them die
            _, vjp_fn = jax.vjp(lambda h: _tile_lp(h, t_tile, w), h_tile)
            (grad_h_tile,) = vjp_fn(c_tile)
            return carry, grad_h_tile

        _, grad_hf = jax.lax.scan(_bwd_body, None, (hf_res, tf, cot))
        return (grad_hf,)

    _tiled.defvjp(_tiled_fwd, _tiled_bwd)

    return _tiled(hf).reshape(num_tiles * tile)[:n]
```

Return the COMPLETE program with `_tile_lp` filled in. Do not change anything
outside `_tile_lp`. Output it as one ```python code block and nothing else.
'''


PROMPTS = {
    "minimal": FLCE_MINIMAL,
    "reference": FLCE_REFERENCE,
    "tailored": FLCE_TAILORED,
}
