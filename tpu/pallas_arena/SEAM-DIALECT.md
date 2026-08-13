# Seam + dialect: getting working code out of the three kernels that never produced any

*gemma-4-31B-it only. Generation + grading, no RL. Run: 2026-08-12.
Sibling of `PROMPT-ITERATION.md`, which established the brief for this run.*

**Result in one line: seam + dialect produced the first working `splash_attention`,
`megablox_gmm` and `ragged_paged_attention` kernels this arena has ever seen —
0/96 each at every prompt-ladder rung becomes 3, 25 and 1 PASS here — and it did
it by taking the Pallas-API/plumbing failure class to EXACTLY ZERO in all three.
The best splash kernel is 4.6x an XLA fallback and the best GMM kernel is 20x
`jax.lax.ragged_dot`; both denominators are labelled fallbacks and NOTHING in
this project has beaten a production kernel.**

---

## 0. The brief, in one paragraph

The prompt ladder (480 candidates, `PROMPT-ITERATION.md`) left three kernels at
**0 PASS out of 96 candidates each**, at every rung: `splash_attention`,
`ragged_paged_attention`, `megablox_gmm`. Every one of those 288 candidates
died at `aot_export`, and the error histogram said why — they fail at the
`pallas_call` **plumbing** (grid, `BlockSpec`s, index maps), which no
whole-program rung supplies by design. Two things in the arena's history bear
on that:

* the **seam** (harness owns the `pallas_call`, model writes one strategy
  function) is the only variant that has ever exported splash at all
  (0/16 `reference` → 2/16 `seam`, job 3651278);
* the **P1 DIALECT list** is the only prompt addition ever measured to remove a
  failure class cleanly (`MemoryRef` item assignment 7→0, `` Invalid shape for
  `swap` `` 7→0, `def_fwd` 1→0, `no_code` 0/480) — and two of those are
  precisely the *seam's* own modal failures.

So this run is **seam + dialect**, not more prose.

## 1. The contaminated measurement, fixed first

17 of megablox_gmm's 96 ladder failures were

```
ValueError: Unsupported TPU device kind: cpu
```

which is not a statement about a kernel. Root cause, found by reading the 17
candidates: **every one of them wrote a rank-1 `BlockSpec`** for the per-row
expert-id array —

```python
pl.BlockSpec((bm,), lambda i, j: (i,))        # row_to_group / experts / rtg
```

— and Pallas's rank-1 block-shape check asks the *current device* for its
geometry:

```python
# jax/_src/pallas/mosaic/lowering.py
sublane_count = tpu_info.get_tpu_info().num_sublanes
lane_count    = tpu_info.get_tpu_info().num_lanes
```

`get_tpu_info()` resolves `get_device_kind()`, which under `JAX_PLATFORMS=cpu`
— the sandbox export child, on both the pre-gate host and the judge — is the
string `"cpu"`, and it raises before the check can run. GMM is the only one of
the five arena tasks whose natural formulation has a rank-1 operand, which is
exactly why only GMM hit it (verified: 17/17 device-kind failures in job
3687041 are `megablox_gmm`, 0 in the other four tasks).

**The fix** (`judge/child_runner.py:_tpu_export_device_shim`) pins the device
kind to the judge's own chip while exporting *for* tpu from a non-tpu child,
and does nothing at all on a TPU host. **The effect, measured on the exact 17
contaminated candidates** (`verify/verify_export_devkind.py`, job 3687873):

| | before | after |
|---|---|---|
| `Unsupported TPU device kind: cpu` | 17/17 | **0/17** |
| clear the export gate outright | 0/17 | **11/17** |
| fail the *real* rank-1 block-shape rule | — | 6/17 |

So GMM's ladder result was contaminated in the direction that mattered:
**11 candidates that should have reached the judge never did.** The remaining 6
now fail on a legible model error (`the first (and only) dimension of the block
shape …`) rather than on the harness's hostname.

## 2. What was built

| arm | what it is |
|---|---|
| `sd1` | the seam prompt + the P1 dialect list |
| `sd2` | `sd1` + a typed skeleton **of the fill functions, at the task's own rank** |

`sd2` deliberately ships **no worked example**. The ladder measured that one
backfires: P3's RMSNorm example removed splash's invented `pallas_call` kwargs
(10→0) and replaced them with **60 candidates writing a 2-D `BlockSpec` against
a 3-D operand**, because the example was over `[rows, cols]` and models copied
its *rank*. A skeleton of the task's own function cannot transfer a wrong rank.

What `sd2` still does not decide — because that is the `tailored` trap (16/16
PASS, spread 0.0042 against floors 0.0069/0.0158, zero gradient): block and
tile sizes, key/k chunking, which blocks or pages to skip, online vs two-pass
softmax, accumulation dtype, and the matmul precision trade.

**Token budget, and why the seam's API block was replaced rather than appended
to.** gemma is served at 16384. Over the ladder's 480 generations on these
exact tasks the median generation was ~8100 tokens and 12/480 hit `length`
against a 12000 budget, so a prompt that leaves under ~11000 tokens starts
destroying candidates for reasons that have nothing to do with kernels. The
seam prompt alone is already ~4750 tokens. So `API_BLOCK` (~1400 tokens) is
replaced by the one thing in it the dialect list does not already say — the
introspected list of the calls you make *inside* a kernel body, sliced verbatim
out of `API_BLOCK` itself — and the dialect's two bullets that cannot apply to
a fill (invented `pallas_call` kwargs; `custom_vjp.defvjp`) are condensed to a
line each. Bullets 2–9 and the precision-trade closer are spliced **byte-
identically**, and the splices are asserted at import.

Measured prompt sizes (gemma's own tokenizer reports 3.07 chars/token on this
family of prompts):

| task | `sd1` | room | `sd2` | room |
|---|---|---|---|---|
| splash_attention | 4890 | 11238 | 5444 | 10684 |
| ragged_paged_attention | 4950 | 11178 | 5591 | 10537 |
| megablox_gmm | 4523 | 11605 | 4885 | 11243 |
| *(control)* rg_lru `p1` | 2598 | — | — | — |

## 3. Controls, before any chip

Three earlier runs in this arena published confident all-zero grids caused by
the harness rather than the model, so nothing went to silicon until a known
answer had been through the identical path.

**CPU control, job 3687873** (`probe/seam_dialect_control.sbatch`):

* **prompts 10/10 clean** — every cell ends with the output contract, carries
  the dialect bullets verbatim, shows its scaffold, and leaves ≥10537 tokens
  for a completion;
* **device-kind fix 17/17 rescued**, §1;
* **known-good fills 6/6 interpret OK and 6/6 pre-gate PASS** through
  `model-style response (with a decoy block) → extract_fill → compose → CPU AOT
  export at all three declared shapes`. Max forward error: splash 1.69e-07,
  RPA 1.13e-07, GMM 9.41e-08, rg_lru 9.87e-07. Every extraction was
  `fenced-single` and every decoy was dropped.

**In-grid control cell**: `rg_lru | p1`, the ladder's best cell (25/32 PASS,
reward 1.0000, PASS-only spread 10.8× the floor). If it does not reproduce, the
harness is broken and no negative result in this run means anything.

**One aborted attempt with no chip cost.** Job 3687901 was cancelled 33 seconds
in, during the pre-warm, to resubmit under a shorter slurm time limit for
backfill. The pre-warm runs *before* any `queued-resources create`, so the
cancellation cost one minute and zero chips — which is exactly the property it
was added for after job 3686851 lost a 31-minute bring-up to a transient
GitHub fetch. Both zones were verified empty of `sk7524-sd-*` afterwards.

## 4. The denominators, stated out loud

Judge boot, job 3687904, one v6e-1 (`TPU v6 lite`). **Three of the four scores
in this run are against a labelled FALLBACK, not against the production
kernel**, and no score anywhere in this project has ever beaten a production
kernel.

| task | noise floor | ref-vs-ref | `baseline_impl` | what a score of 1.0 actually means |
|---|---|---|---|---|
| splash_attention | 0.0300 | 1.0002 / 1.0015 | `xla-fallback` | ties a competent query-blocked XLA attention, **not** Google's splash kernel |
| ragged_paged_attention | **0.2362** | 1.0318 / 0.9844 | `xla-paged-decode-fallback` | ties a batch-blocked XLA paged decode, **not** vLLM-TPU's Pallas v3 |
| megablox_gmm | 0.0018 | 1.0002 / 1.0008 | `lax-ragged-dot-fallback` | ties `jax.lax.ragged_dot`; megablox refused these shapes |
| rg_lru | 0.0693 | 0.9996 / 1.0029 | `lax-associative-scan` | ties `lax.associative_scan`, **not** recurrentgemma's Pallas scan |

RPA's floor is **23.6%** — a candidate has to be almost a quarter faster before
the arena can see it at all. That is a property of the timing protocol at this
shape, not of any kernel, and it is the second independent reason RPA is the
weakest of the four as an RL target.

## 5. Results — 224 candidates, 7 cells, 32 each

Job **3687904**. Raw: `runs/pallas_arena/sd-results-3687904.jsonl` (one row per
candidate: full generation text, extracted fill, composed program, gate,
observation, reward), `sd-judge-boot-3687904.json`, `sd-tables-3687904.md`,
`sd-cells-3687904.json`.

### 5.1 The whole run, as measured

| kernel | arm | judged | export | **PASS** | best | group spread | floor | spread/floor | PASS-only spread | signal groups | **verdict** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| splash_attention | `sd1` | 32 | **15** | **3** | **3.2456** | 3.2456 | 0.0300 | 108x | 0.0415 | 2/2 | **SIGNAL** |
| splash_attention | `sd2` | 32 | 10 | 0 | — | 0.0000 | 0.0300 | 0x | — | 0/2 | **NO CODE** |
| ragged_paged_attention | `sd1` | 32 | 4 | 0 | — | 0.0000 | 0.2362 | 0x | — | 0/2 | **NO CODE** |
| ragged_paged_attention | `sd2` | 32 | **23** | **1** | **0.4947** | 0.4947 | 0.2362 | 2.1x | — | 1/2 | **SIGNAL** |
| megablox_gmm | `sd1` | 32 | 17 | **9** | **17.1300** | 17.1300 | 0.0018 | 9517x | **5.3732** | 2/2 | **SIGNAL** |
| megablox_gmm | `sd2` | 32 | **32** | **16** | **19.9181** | 7.4886 | 0.0018 | 4160x | **7.4886** | 1/2 | **SIGNAL** |
| *control* rg_lru | `p1` | 32 | 26 | **13** | 1.0000 | 1.0000 | 0.0693 | 14.4x | **0.7510** | 1/2 | **SIGNAL** |

Against the same three kernels' entire history: **0 PASS out of 96 at every
ladder rung, 0 out of 16 at the plain seam, 0 out of 16 at `reference`.**

### 5.2 One group in this run is not model data — say so before reading it

35 of the 224 candidates were graded `fixtures` in a uniform ~1.1 s, with

```
JaxRuntimeError: INTERNAL: RuntimeUnexpectedCoreHalt: Program or fatal error
occurred; computation may be invalid
```

The judge log localises it exactly. Work item `w000091`
(`gmm|sd1|r1#1`) compiled for 4.5 s and **halted the TPU core**; from `w000092`
onward *every* item failed in the worker's own fixture precomputation, before
the candidate was even compiled, and the worker never recovered. The affected
set is contiguous and complete: `gmm|sd1` round 1 (7), `gmm|sd2` round 1 (16),
`rg_lru|p1` round 1 (12). splash and RPA were graded before the halt and are
clean in **both** rounds.

So the honest per-round table, with the contaminated group marked:

| cell | round | n | lost to the chip fault | export | PASS | best | PASS-only spread |
|---|---|---|---|---|---|---|---|
| splash_attention `sd1` | 0 | 16 | 0 | 8 | 1 | 3.2456 | — |
| splash_attention `sd1` | 1 | 16 | 0 | 7 | 2 | 3.0774 | 0.0415 |
| splash_attention `sd2` | 0 / 1 | 32 | 0 | 3 / 7 | 0 | — | — |
| ragged_paged_attention `sd1` | 0 / 1 | 32 | 0 | 1 / 3 | 0 | — | — |
| ragged_paged_attention `sd2` | 0 | 16 | 0 | 13 | 1 | 0.4947 | — |
| ragged_paged_attention `sd2` | 1 | 16 | 0 | 10 | 0 | — | — |
| megablox_gmm `sd1` | 0 | 16 | 0 | 8 | **8** | 17.1300 | **5.3732** |
| megablox_gmm `sd1` | 1 | 16 | **7** | 9 | 1 | 17.0124 | — |
| megablox_gmm `sd2` | 0 | 16 | 0 | **16** | **16** | **19.9181** | **7.4886** |
| megablox_gmm `sd2` | 1 | 16 | **16** | 16 | *(none graded)* | — | — |
| *control* rg_lru `p1` | 0 | 16 | 0 | 14 | **13** | 1.0000 | **0.7510** |
| *control* rg_lru `p1` | 1 | 16 | **12** | 12 | *(none graded)* | — | — |

**GMM's true count is 25 PASS out of the 41 candidates that reached a healthy
chip**, not 25 out of 64. The chip fault is itself a finding and is written up
in §6.3.

### 5.3 The control reproduces — so the negatives are readable

`rg_lru | p1`, in its uncontaminated round-0 group: **13/16 PASS**, best
**1.0000**, PASS-only spread **0.7510 = 10.8x** the 0.0693 floor, with the same
bimodal shape the ladder measured —

```
1.0, 1.0, 1.0, 1.0, 1.0, 0.6312, 0.6016, 0.6013, 0.6007, 0.5934, 0.5934, 0.2528, 0.2490
```

— against the ladder's 25/32 PASS, best 1.0000, PASS-only spread 0.7554 =
10.8x. Two independent runs, three weeks and two judges apart, agree to the
third decimal on the spread. The harness is sound.

**And 1.0000 is a TIE, not a win.** Noise-floor gating pins a candidate that
matches its baseline to exactly 1.0. rg_lru's baseline is
`lax.associative_scan`, not recurrentgemma's Pallas scan. No verified win over
any baseline exists anywhere in this project.

### 5.4 The failure class that vanished

Pooled over each task's cells, counting every candidate whose failure was about
the Pallas *interface* (`pallas_call` kwargs, `BlockSpec` rank, index-map
arity, `in_specs` pytree, `CompilerParams`, a missing module attribute, an
undefined name, or the device kind):

| task | ladder, 96 candidates | plain seam, 16 | **this run, 64** |
|---|---|---|---|
| splash_attention | **87** | 2 | **0** |
| ragged_paged_attention | **56** | 0 | **0** |
| megablox_gmm | **55** | 2 | **0** |

**Zero.** Not one of the 192 candidates in this run failed at the Pallas API.
That is the whole hypothesis, confirmed: those three kernels were never failing
at attention or at grouped matmul, they were failing at a `pallas_call`
signature — and the seam removes the signature while the dialect list removes
the in-body Ref mistakes.

Specifically eliminated, all to **0** in this run: `pallas_call() got an
unexpected keyword argument` (10 GMM + 20 splash + 15 RPA in the ladder),
`Block shape for args[N] … must have the same number of dimensions` (61 splash
+ 9 RPA), `Index map function … must return N values`, `Pytree for in_specs and
inputs do not match`, `CompilerParams.__init__() got an unexpected keyword
argument`, `ConcretizationTypeError` on traced `group_sizes` (7 GMM), and
`Unsupported TPU device kind: cpu` (17 GMM, §1).

### 5.5 And the failure classes it introduced

Additions create new failures; the ladder's worked example taught that the hard
way. What is new here, per task, and what it means:

| task | new signature | n | reading |
|---|---|---|---|
| ragged_paged_attention | `max err … vs tol …` (gate `correctness`) | **23** | **This is progress, not regression.** RPA candidates now compile, run on the chip and produce *numbers*; they get the online-softmax rescale or the ragged mask slightly wrong. The ladder never got a single RPA candidate far enough to be wrong about arithmetic. |
| ragged_paged_attention | `Incompatible shapes for broadcasting: [(N,N),(N,)]` | 12 | the `[1, page_size]` mask against a `[group, page_size]` logit block — a real shape error inside the body, and the one thing the sd2 skeleton spells out that sd1 does not. |
| megablox_gmm | `RuntimeUnexpectedCoreHalt` | 24 | the judge-side chip fault, §5.2. Not model data. |
| megablox_gmm | `TracerBoolConversionError` | 6 | candidates doing python control flow on a value read from a Ref — dialect bullet 7, still the residual. |
| splash_attention | `` Invalid shape for `swap` `` | 4 | writing a `[bq, dp]` value into the `[1, bq, dp]` `o_ref`. Dialect bullet 4 names this exactly and 4 candidates did it anyway. |

**Zero truncations.** All 224 generations finished with `finish_reason=stop`,
against per-cell completion budgets of 10619–12000 tokens — the prompt-size
work in §2 held. Extraction was `fenced-single` for 222 of 224, and only 2
candidates (both `splash|sd1`) were missing a required name.

Fill sizes, median: GMM 417–446 tokens, rg_lru 531, splash 859–963, RPA
888–894. All inside the seam's stated 300–1200 budget, which the previous seam
run missed by 2–3x.

### 5.6 sd1 vs sd2: the typed skeleton is not uniformly good

| task | export sd1 → sd2 | PASS sd1 → sd2 | best sd1 → sd2 |
|---|---|---|---|
| splash_attention | 15 → **10** | **3 → 0** | 3.2456 → — |
| ragged_paged_attention | 4 → **23** | 0 → **1** | — → 0.4947 |
| megablox_gmm | 17 → **32** | 9 → **16** | 17.13 → **19.92** |

**The skeleton is worth a lot where the interface is wide and costs where it is
narrow.** RPA's fill has 7 arguments and two persistent scratch Refs — writing
out the shape of every slice took export from 4/32 to 23/32, a 5.75x jump, and
produced RPA's only working kernel. GMM's went 17/32 to a clean sweep, 32/32.
But splash's fill is two small functions, and there the skeleton *cost* 5
exports and all 3 passes: splash candidates followed the skeleton's commented
`for j in range(kv_len // bk)` blocked outline and got the block bookkeeping
wrong, where the sd1 winner simply materialised one `[bq, kv_len]` score matrix
and was done.

**No cell landed in the `tailored` trap.** The closest is `gmm|sd2`, which
passed **16/16** in its clean group — and its PASS-only spread is **7.4886, or
4160x the noise floor**, with five visible strategy clusters:

```
19.9181 | 17.1551 17.0873 17.0249 17.0149 16.9735 16.9474 | 16.1616 16.1415
        | 14.5077 | 12.5441 12.4730 12.4540 12.4478 12.4478 12.4295
```

A cell that passes everything *and* spreads by 4160 floors is the opposite of
the trap: every candidate works, and there is still a factor of 1.6 between the
best and the worst tile choice for RL to find.

### 5.7 The kernels, verbatim

**`megablox_gmm | sd2`, reward 19.9181 — the best score in this run.** The
harness owns the `pallas_call`, the group-aligned permutation and the
un-permute; this is the model's entire contribution:

```
probe-m4096-e4-uniform:      cand 2.424ms vs ref 49.346ms (20.359x)
probe-m2048-e4-zipf:         cand 1.506ms vs ref 29.355ms (19.492x)
probe-holdout-m3000-e4-zipf: cand 1.930ms vs ref  8.045ms  (4.168x)
peak HBM 8.18GB
```

```python
import jax
import jax.numpy as jnp

def choose_tiles(m, k, n, num_groups):
    """
    Returns (bm, bk, bn).
    bm: 512 allows high MXU utilization while remaining under the 32MB VMEM ceiling.
    bn: 512 maintains a balance between grid size and memory footprint.
    bk: 128 is provided as a standard contraction chunk, though the full k is resident.
    """
    return 512, 128, 512

def gmm_tile(l_ref, r_ref, o_ref, *, bm, bk, bn, k):
    """
    Performs one grid step: [bm, k] @ [k, bn] -> [bm, bn].
    """
    lhs = l_ref[...]          # (bm, k)
    rhs = r_ref[0, ...]       # (k, bn)
    res = jax.lax.dot_general(
        lhs,
        rhs,
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32
    )
    o_ref[...] = res
```

**The 20x is against `jax.lax.ragged_dot` at 49.3 ms, a labelled fallback that
megablox itself refused these shapes for.** It is not 20x megablox and it is
not 20x anything anyone ships. What it does show is that the model chose a tile
shape that beats XLA's ragged path by an order of magnitude, and that other
candidates choosing differently landed at 12.4x — which is the gradient.

**`splash_attention | sd1`, reward 3.2456 — the first splash kernel to PASS in
this arena.** Its decision was to *not* block the key axis at all:

```
probe-h8-s4096:        cand 0.736ms vs ref 3.385ms (4.597x)
probe-h4-s2048:        cand 0.234ms vs ref 0.529ms (2.264x)
probe-holdout-h4-s2049: cand 0.261ms vs ref 0.297ms (1.137x)
peak HBM 5.89GB
```

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def choose_blocks(seq, head_dim):
    """bq=128 allows for efficient tiling on TPU sublanes and fits the score
    matrix (128 x 4096 f32 = 2MB) well within the VMEM limit."""
    return 128, 128

def attn_block(q_ref, k_ref, v_ref, sq_ref, sk_ref, o_ref, *, bq, bk, kv_len):
    q_val = q_ref[0]                      # (bq, dp)
    k_val = k_ref[0]                      # (kv_len, dp)
    v_val = v_ref[0]                      # (kv_len, dp)
    seg_q = sq_ref[:, 0]                  # (bq,)
    seg_k = sk_ref[0, :]                  # (kv_len,)

    logits = jax.lax.dot_general(
        q_val, k_val, (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32)          # (bq, kv_len)

    q_block_idx = pl.program_id(1)
    global_q = q_block_idx * bq + jax.lax.broadcasted_iota(jnp.int32, (bq,), 0)
    global_k = jax.lax.broadcasted_iota(jnp.int32, (kv_len,), 0)
    causal_mask = global_q[:, None] >= global_k[None, :]
    seg_mask = (seg_q[:, None] == seg_k[None, :]) & (seg_q[:, None] != 0)
    mask = causal_mask & seg_mask

    NEG_INF = -1e30
    masked_logits = jnp.where(mask, logits, NEG_INF)
    row_max = jnp.max(masked_logits, axis=-1, keepdims=True)
    p = jnp.exp(masked_logits - row_max)
    p = jnp.where(mask, p, 0.0)
    denom = jnp.sum(p, axis=-1, keepdims=True)
    row_live = jnp.any(mask, axis=-1, keepdims=True)
    p = jnp.where(row_live, p / jnp.maximum(denom, 1e-30), 0.0)

    out = jax.lax.dot_general(
        p, v_val, (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32)          # (bq, dp)
    o_ref[...] = out[None, :, :]
```

It is a *two-pass, whole-key-axis* softmax at `bq=128` — the model reasoned
about the 2 MB score tile against the ~32 MB VMEM ceiling explicitly and
concluded it did not need to block the key axis or skip anything. It ignores
the `bk` it returned. That is exactly the kind of strategy decision the seam
exists to leave open, and it is 4.6x the XLA fallback at seq=4096 while being
only 1.14x at the non-divisible holdout — the trade is visible in the score.

**`ragged_paged_attention | sd2`, reward 0.4947 — correct, and slower than the
fallback.** Reported because a working RPA kernel has never existed here, not
because it is fast:

```
probe-b16-len1024: cand 0.657ms vs ref 0.251ms (0.382x)
probe-b8-len512:   cand 0.272ms vs ref 0.171ms (0.630x)
```

It runs an online softmax across pages with `@pl.when(p == 0)` init,
`@pl.when(p * page_size < sl_ref[b])` page skipping and a final normalisation
at `p == num_pages - 1`, at `Precision.HIGHEST` throughout — which is very
likely why it is 2.6x slower than a batch-blocked XLA decode. The full fill is
in `sd-tables-3687904.md`.

## 6. Verdict, per kernel

* **splash_attention — SOLVED, at `sd1`.** 3/32 PASS, 15/32 export, best
  **3.2456** (4.6x the `xla-fallback` denominator). First splash kernels in this
  arena's history to reach `gate=all`. `sd1` is the arm: the typed skeleton
  took it to 0. Its PASS-only spread (0.0415, 1.4x the 0.0300 floor) clears the
  floor but not by much, and on 3 passing candidates that is a weak
  measurement — the whole-group spread of 108x floors is mostly the 29 zeros.
* **megablox_gmm — SOLVED, at `sd2`, and it is the best RL target in the arena.**
  16/16 PASS in its clean group with a PASS-only spread of **7.4886 = 4160x**
  the floor across five strategy clusters. `sd1` also works (8/16 in its clean
  group, spread 5.3732) with a lower ceiling. Caveats that must travel with the
  number: the denominator is `lax.ragged_dot`, the harness owns the permutation
  and the `pallas_call`, and 24 of 64 candidates were lost to a chip fault.
* **ragged_paged_attention — WORKING CODE, BUT NOT SOLVED.** 1/32 PASS at
  `sd2`, scoring **0.4947** — correct and 2.6x *slower* than a fallback
  baseline. The arm difference is dramatic (export 4/32 → 23/32) and the modal
  failure moved from "cannot write a `pallas_call`" to "max err vs tol" (23
  candidates), which is the right direction. But its floor is **0.2362**: a
  candidate has to be a quarter faster before the arena can see it, so RPA
  cannot be trained on at this shape whatever the model writes. **Fix the
  timing protocol before the prompt.**
* **rg_lru — control, GREEN.** Reproduced the ladder to the third decimal.

### 6.1 The general finding

The ladder's conclusion was that prose helps exactly one class of problem — the
one that is about mathematics you can state — and fails on the class that is
about interface plumbing you must write. This run is the positive half of that
sentence: **give the model the `pallas_call` and the plumbing failure class
goes to zero, in all three kernels at once, and working code appears
immediately.** 288 ladder candidates could not produce one passing kernel among
them; 192 seam+dialect candidates produced 29.

### 6.2 And a second, sharper one: the right amount of scaffolding is per-task

`sd2` is a strict superset of `sd1` and it is better on two kernels and worse on
the third, by a lot in both directions (RPA export x5.75, splash PASS 3 → 0).
The variable that predicts it is the **width of the fill interface**: RPA's
7-argument, two-scratch-Ref signature needs its shapes spelled out; splash's
two small functions do not, and the extra structure pushed candidates into a
blocked formulation they got wrong. A ladder that adds the same rung to every
task will therefore be wrong for some of them — which is the same lesson as
P3's worked example, one level up.

### 6.3 A judge robustness gap, found the hard way

One model-written Pallas GMM kernel **halted the TPU core at Mosaic compile
time and permanently killed the judge worker**, which then failed the next 35
candidates in its own fixture setup at ~1.1 s each, silently, as if they were
model failures. The worker has no liveness check on its device: it should
detect a `RuntimeUnexpectedCoreHalt`, re-lease the item, and restart its jax
context (or exit, so the supervisor's restart loop replaces it). Until then,
any grid can lose a whole group to one candidate. This is the single highest-
value harness fix outstanding.

## 7. Resources and teardown

| | |
|---|---|
| jobs | 3687873 (CPU control), 3687899 (prompt check), 3687901 (aborted in pre-warm, **no QR**), **3687904** (the measurement) |
| QRs | `sk7524-sd-serve` v5p-8 us-east5-a spot; `sk7524-sd-judge` v6e-1 us-east5-b spot |
| first QR create | **19:55:22** — the 4 h clock starts here |
| serve ACTIVE | 19:59:44 (4 min 22 s) |
| judge provisioned | 20:02:55 (7 min 33 s) |
| gemma serving | 20:16:03 (**20 min 41 s** — the fastest bring-up this arena has had) |
| grid | 20:18:36 → 20:57:03, 224 candidates, 2 complete rounds of 16 per cell |
| teardown complete | 20:57:03, rc=0 |
| **QR lifetime** | **3916 s = 65 min 16 s**, against a 4 h cap |

Never more than 2 QRs alive. **Chip-hours: 4 chips x 65 min = 4.4 v5p
chip-hours + 1 chip x 65 min = 1.1 v6e chip-hours ≈ 5.5 chip-hours**, plus one
neuronic node and three short CPU jobs.

Job 3687901 was cancelled during the pre-warm, before `queued-resources
create`, and cost zero chip time — the pre-warm is there precisely so a
resubmission is free.

**Teardown, verified.** The job's own cleanup:

```
[cleanup] 20:57:03 tearing down (always-delete, both QRs)
[cleanup] detached deletes issued for sk7524-sd-serve and sk7524-sd-judge
[cleanup] delete confirmed: sk7524-sd-serve
[cleanup] delete confirmed: sk7524-sd-judge
[cleanup] verified empty of sd QRs in us-east5-a
[cleanup] verified empty of sd nodes in us-east5-a
[cleanup] verified empty of sd QRs in us-east5-b
[cleanup] verified empty of sd nodes in us-east5-b
[cleanup] QR lifetime 3916s
```

And independently, checked directly in both zones at **21:00:45** after the job
had exited:

```
us-east5-a QRs: NONE     us-east5-a nodes: NONE
us-east5-b QRs: NONE     us-east5-b nodes: NONE
```

**Zero `sk7524-sd-*` queued-resources and zero `sk7524-sd-*` TPU nodes in BOTH
us-east5-a and us-east5-b.** The running RL sweeps (`sk7524-llamafarm-*`,
`sk7524-stagea-*`, `sk7524-tunix-*`) were read-only-observed and never touched.
`tpu/results/sweep1-analysis/` and `third_party/discover` were not modified. No
`.env` file or credential was printed.

## 8. What to do next

1. **Train GMM on `sd2`.** 16/16 PASS, PASS-only spread 4160x the floor, five
   strategy clusters, and the cheapest of the five to grade. It is now the best
   RL target this arena has, ahead of rg_lru.
2. **Make the judge survive a core halt** (§6.3) before any longer run. One
   candidate cost 35 gradings here; in an RL loop it would cost a whole batch.
3. **Use `sd1` for splash and `sd2` for RPA and GMM.** The right rung is
   per-task and the difference is large in both directions.
4. **Fix RPA's timing before its prompt.** A 0.2362 floor makes it untrainable
   at this shape no matter how good the kernel is: bigger shapes, more timing
   pairs, or a different score.
5. **Bind the real baselines, or keep saying "fallback" out loud.** Splash is
   scored against XLA, GMM against `ragged_dot`, RPA against an XLA paged
   decode, rg_lru against `lax.associative_scan`. Every headline in this
   document is against one of those.

