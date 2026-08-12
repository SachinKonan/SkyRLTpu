# Iterating the arena prompts, per kernel, until the code works *and still has room to improve*

*gemma-4-31B-it only. Generation + grading, no RL. Run: 2026-08-12.*

**STATUS: IN PROGRESS — this file is written as the run proceeds. The results
sections are filled in from measured artifacts only; anything not yet measured
says so.**

---

## 0. The question, and the metric that answers it

Not "what fraction passes". The arena has already been burned by that: the
`tailored` fill-in-the-blank variant passed **16/16** with a within-group
reward spread of **0.0042** against judge noise floors of **0.0069 / 0.0158**.
Every candidate converged on the same kernel, so the advantage vector
ttt-discover would compute is exactly zero and there is nothing to learn. A
prompt can be too good.

So each (kernel x rung) cell gets a verdict, not a rate:

| verdict | meaning |
|---|---|
| `NO CODE` | nothing PASSed the judge. The rung did not produce a working kernel. |
| `NO GRADIENT` | candidates PASS, but no complete 16-candidate group's reward spread clears the judge's own measured noise floor for that task. Working code, frozen answer. **This is a failure for our purposes and is reported as one.** |
| `SIGNAL` | at least one complete group PASSes *and* spreads further than the noise floor. The only outcome that is a win. |

Two spreads are reported because they answer different questions: the
whole-group spread (which a 6/16-pass group gets for free — some zeros, some
ones) and the **PASS-only** spread (once it works, is there still anything to
learn?). A cell with 32/32 PASS and a PASS-only spread under the floor is the
`tailored` trap wearing a new prompt, and the report labels it `SATURATED`.

## 1. What was already measured, and therefore not re-measured

From job **3651278** (160 candidates, `runs/pallas_arena/seam-results-3651278.jsonl`)
and job **3650240**, the failure signatures this run had to move:

| kernel | modal failures (job 3651278) |
|---|---|
| `splash_attention` | `'MemoryRef' object does not support item assignment` (x5); `pallas_call() got an unexpected keyword argument 'out_spec'` (x3); no `pallas_call` found (x3); `'PjitFunction' object is not subscriptable` (x2) |
| `ragged_paged_attention` | `` Invalid shape for `swap`. Ref shape (8,4,128) `` (x6); no `pallas_call` (x4); syntax errors (x2) |
| `megablox_gmm` | `too many values to unpack (expected 2)` (x3); `'function' object is not iterable` (x2); no `pallas_call` (x2); `MemoryRef` item assignment (x2) |
| `rg_lru` | returned wrong type from jit (x2); `cannot reshape (256,4,2560) into (4,...)` (x2); `out_spec` (x1); block shape divisibility (x1) |
| `flce` | **gate `gradient`, q99 err 0.00357 vs tol 4.55e-05 (x6)** — the backward is mathematically wrong; `TracerBoolConversionError` (x2); `custom_vjp has no attribute 'def_fwd'` (x1); broadcasting `(128,151936)` vs `(128,2880)` (x1) |

Two different diseases. splash / RPA / GMM die at the **Pallas API and layout
contract**. FLCE gets the API right and dies at the **mathematics of the
recompute backward**. They get different treatment below.

Also already measured, and the reason the ladder does not start at P0:

* `gemma | minimal | splash_attention`: **0/32** export, two separate runs.
* `gemma | reference | splash_attention` (P0 + oracle + a ten-item prose
  gotcha list): **0/16** export. Prose did not move it by a single candidate.
* `gemma | minimal | flce`: **2/20** PASS — so P0 already works on FLCE, and
  P0 is a measured data point rather than a gap.
* `gemma | reference | rg_lru`: 2/8 PASS, spread 0.5986, 15.6x its floor.

## 2. The ladder actually run

Fifteen cells: 5 kernels x 3 rungs x 32 samples (2 groups of 16). Each rung is
a strict superset of the one below, so exactly one thing changes per step.

| rung | what it adds | module |
|---|---|---|
| **BASE** *(held constant, not a cell)* | task spec, exact signature, declared shapes incl. the non-divisible holdout, judge contract, fp32 correctness oracle | `probe/prompt_ladder.py:_base` |
| **P1** | the **DIALECT** list: ten bullets, each carrying the verbatim error string it eliminates and the number of job-3651278 candidates that died on it | `prompt_ladder.DIALECT` |
| **P3** | a typed **SKELETON** of the entrypoint (the ladder's P2) **and** one complete **WORKED EXAMPLE** Pallas kernel in the right dialect — the arena's own phase-2/4 RMSNorm golden, which passes this judge on silicon. Pins `pallas_call`/`BlockSpec`/`ShapeDtypeStruct`/`defvjp` by demonstration rather than description | `prompt_ladder.WORKED_EXAMPLE`, `_SKELETONS` |
| **P4** | **PRIMITIVES**: tested helpers prepended to the submitted program (`dot_f32`, `iota2`, `fill_ref`, `online_softmax`, `chunk_scan`), documented by signature. FLCE instead gets the **backward CONTRACT** stated exactly, because its failure is mathematical | `probe/ladder.py`, `prompt_ladder.FLCE_BACKWARD_CONTRACT` |

**P2 and P3 are merged into one rung.** The wall budget affords 15 cells, not
25, and a typed skeleton on its own does not address the modal failure — an
invented `pallas_call` signature — at all.

**What no rung gives away.** Block and tile shapes, chunk length, one-pass vs
two-pass, materialized vs online softmax, which blocks to skip, accumulation
dtype, and the `precision=HIGHEST` accuracy/speed trade. Every primitive is
optional and none of them decides any of those. That is deliberate: it is the
difference between this ladder and `tailored`.

**The prelude is additive by construction.** It is *prepended*, so a model that
ignores it produces exactly the program it would have produced at P3, and a
model that redefines a helper still wins (its definition comes later). This is
what makes one control program valid for both rungs.

## 3. Control before any chip — placeholder

*(filled in from `runs/pallas_arena/ladder-control-*.json`)*

## 4. Results — placeholder

*(filled in from `runs/pallas_arena/ladder-results-*.jsonl`)*

## 5. Resources and teardown — placeholder
