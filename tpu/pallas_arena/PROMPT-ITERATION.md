# Iterating the arena prompts, per kernel, until the code works *and still has room to improve*

*gemma-4-31B-it only. Generation + grading, no RL. Run: 2026-08-12.*

**Result in one line: rg_lru is solved at the LOWEST rung tried (25/32 PASS,
reward 1.0000, PASS-only spread 10.8x the noise floor); FLCE is solved by
stating the backward contract (8/32 PASS, an arena record); and splash,
ragged-paged-attention and megablox-GMM produce ZERO working kernels at every
rung — 0 out of 96 candidates each — for a reason the error histogram names.**

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

Re-audited across all 145 non-passing judged candidates of job 3651278 with a
regex over each candidate's observation and violations — this is the list the
P1 DIALECT bullets were written from, one bullet per line:

| n | signature |
|---|---|
| 8 | `TracerBoolConversionError` |
| 7 | `'MemoryRef' object does not support item assignment` |
| 7 | `` Invalid shape for `swap` `` |
| 5 | `pallas_call() got an unexpected keyword argument 'out_spec'` |
| 5 | `dot_general requires ...` |
| 5 | `pl.when` misuse (`'function' object is not iterable`, `'NoneType' object is not callable`, context-manager) |
| 5 | `too many values to unpack (expected 2)` |
| 3 | `cannot reshape array ...` |
| 2 | `block shape ... divisible by 8 and 128` |
| 1 | `'custom_vjp' object has no attribute 'def_fwd'` |

*(The per-bullet counts written into the prompt itself were taken from the
per-task tallies and are within ±2 of this stricter whole-corpus audit. The
prompt was frozen before the audit and was not edited mid-run.)*

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

## 3. Control before any chip — GREEN

Three earlier runs in this arena produced confident all-zero grids from
infrastructure faults (a token budget that 400'd every request; a
signature-tuple bug that cost 14 of 32 candidates; unreachable engines). So
nothing went to silicon until a known-good answer had been through the
*identical* path. Job **3686829**, CPU only, `runs/pallas_arena/ladder-control-3686829.json`.

**Prompt budget — all 15 fit, with room.** Pessimistic chars/3 against
gemma's served 16384:

| rung | tokens (worst kernel) | room left for the completion |
|---|---|---|
| P1 | 2829 (RPA) | 13299 |
| P3 | 3920 (RPA) | 12208 |
| P4 | 4497 (RPA) | **11631** |

Every cell keeps ≥11631 tokens for generation against a 12000 request, and the
driver additionally asks the server's own `/tokenize` per cell and clamps, so
the failure mode that cost a whole cell last time (a 4.4k prompt + a 12000
request against a 16384 window returning `HTTPError 400` sixteen times, read
as sixteen model failures) cannot recur.

**Whole-program control — 10/10 PASS.** Each task's verified known-good
program, wrapped in a model-style response *with a decoy `raise
NotImplementedError` block first*, through the real `extract_program`, the
real `compose_ladder`, and the real sandbox AOT-export child at every declared
probe shape, at rung p1 and rung p4:

```
splash_attention p1 PASS / p4 PASS      ragged_paged_attention p1 PASS / p4 PASS
megablox_gmm     p1 PASS / p4 PASS      rg_lru                 p1 PASS / p4 PASS
flce             p1 PASS / p4 PASS
```

The decoy was dropped in 10/10 (the extractor takes the last complete block),
and the prelude was prepended in exactly the four p4 cells that have one and
in none of the p1 cells.

**Primitives — checked against their own semantics, not asserted:**

| helper | check | result |
|---|---|---|
| `dot_f32` | vs `einsum`, default and `hi=True` | max err **0.0** both |
| `online_softmax` | streamed in 3 blocks vs one dense masked softmax | max err **2.4e-07** |
| `online_softmax` | a fully-masked row | **exactly 0.0**, all finite |
| `chunk_scan` | vs a python loop over the recurrence | max err **2.4e-07**, layout `[B,T,D]` |
| `fill_ref` | inside a real `pallas_call` body | every element set |

**And one end-to-end P4 answer**: the plain rg_lru kernel a model would write
under the primitives prompt (a chunk length, one `chunk_scan` call, a carry)
— exports at every declared shape and lands **1.67e-07** from the fp32
reference. If an answer that obvious could not pass, the rung would be wrong.

## 4. Attempt 1 (job 3686851): a complete bring-up lost to one HTTP fetch

Recorded because it is the fourth time this arena has nearly published an
all-zero grid caused by infrastructure, and because the fix generalises.

Everything worked: the v5p-8 was ACTIVE 11 minutes after create, the judge
provisioned at 16:11 and booted all five problems with noise floors, the gemma
engine was serving at 16:32:14 (26 min from create), and the chat rendering
verified token-for-token against the server's own template
(`template_agrees=True`, `special_tokens_single=True`, `sample_nonempty=True`).

Then the driver's `uv run --isolated` died in **13 seconds**:

```
error: Failed to generate package metadata for `causal-conv1d==1.6.1 @
  direct+https://github.com/erictang000/causal-conv1d/releases/download/...`
  Caused by: http2 error: stream error received: refused stream
=== ladder probe done rc=2 16:37:22 ===
```

`causal-conv1d` is pulled from a GitHub *release URL* by the `dev` extra and is
used by nothing in this run. Zero candidates were generated; the job tore
itself down and both zones verified empty. **This is a harness fault, not a
finding, and none of it is reported as data.**

Three changes (commit `8ee94229`), in the order they fire:

1. **Pre-warm before any QR exists.** The exact grid environment is built and
   the prompt/ladder modules imported *before* `queued-resources create`. A
   resolution flake, a syntax error in a prompt module or a missing config name
   now costs a minute and no chips.
2. **`uv_retry`** (4 attempts, linear backoff) around the driver and the render
   check, with `UV_HTTP_TIMEOUT` raised.
3. **Offline fallback** (`UV_OFFLINE=1`) after the retries, resolving from the
   local uv cache — the only path that does not depend on github.com.

It paid for itself immediately. On attempt 2 the pre-warm hit the *same* host,
now returning a sustained `503 Service Unavailable`, retried, and passed on the
third attempt with `[prewarm] OK 15 configurations; 15 prompts` — **before a
single chip was provisioned.**

## 5. Results — 480 candidates, 15 cells, 32 samples each

Job **3687041**. Raw: `runs/pallas_arena/ladder-results-3687041.jsonl` (one row
per candidate: full generation text, extracted code, composed program, gate,
observation, reward), `ladder-judge-boot-3687041.json` (noise floors).
Every cell is 32 candidates = 2 complete groups of 16. Nothing is extrapolated.

| kernel | rung | judged | export | **PASS** | best | group spread | floor | spread/floor | **PASS-only spread** | signal groups | **verdict** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| splash_attention | `p1` | 32 | 0 | **0** | 0.0000 | 0.0000 | 0.0312 | 0x | - | 0/2 | **NO CODE** |
| splash_attention | `p3` | 32 | 0 | **0** | 0.0000 | 0.0000 | 0.0312 | 0x | - | 0/2 | **NO CODE** |
| splash_attention | `p4` | 32 | 0 | **0** | 0.0000 | 0.0000 | 0.0312 | 0x | - | 0/2 | **NO CODE** |
| ragged_paged_attention | `p1` | 32 | 0 | **0** | 0.0000 | 0.0000 | 0.2525 | 0x | - | 0/2 | **NO CODE** |
| ragged_paged_attention | `p3` | 32 | 0 | **0** | 0.0000 | 0.0000 | 0.2525 | 0x | - | 0/2 | **NO CODE** |
| ragged_paged_attention | `p4` | 32 | 0 | **0** | 0.0000 | 0.0000 | 0.2525 | 0x | - | 0/2 | **NO CODE** |
| megablox_gmm | `p1` | 32 | 0 | **0** | 0.0000 | 0.0000 | 0.0013 | 0x | - | 0/2 | **NO CODE** |
| megablox_gmm | `p3` | 32 | 0 | **0** | 0.0000 | 0.0000 | 0.0013 | 0x | - | 0/2 | **NO CODE** |
| megablox_gmm | `p4` | 32 | 0 | **0** | 0.0000 | 0.0000 | 0.0013 | 0x | - | 0/2 | **NO CODE** |
| flce | `p1` | 32 | 18 | **5** | 0.4990 | 0.4990 | 0.0106 | 47x | 0.2327 | 2/2 | **SIGNAL** |
| flce | `p3` | 32 | 18 | **2** | 0.2666 | 0.2666 | 0.0106 | 25x | 0.0001 | 2/2 | **SIGNAL** |
| flce | `p4` | 32 | 24 | **8** | 0.2666 | 0.2666 | 0.0106 | 25x | 0.2000 | 2/2 | **SIGNAL** |
| rg_lru | `p1` | 32 | 26 | **25** | 1.0000 | 1.0000 | 0.0699 | 14x | 0.7554 | 2/2 | **SIGNAL** |
| rg_lru | `p3` | 32 | 24 | **20** | 1.0000 | 1.0000 | 0.0699 | 14x | 0.7526 | 2/2 | **SIGNAL** |
| rg_lru | `p4` | 32 | 25 | **15** | 1.0000 | 1.0000 | 0.0699 | 14x | 0.4221 | 2/2 | **SIGNAL** |

**Headline: 12 signal groups out of 30 — and 12 is the maximum obtainable.**
Only the flce and rg_lru cells produce any passing code at all, which is 12
complete groups; *every one of them* clears its noise floor. Not one cell
landed in the `tailored` trap (working code, spread below the floor).

### 5.1 First rung that produced working code, per kernel

| kernel | first rung with working code | first rung with signal | best reward | verdict |
|---|---|---|---|---|
| **rg_lru** | **P1** (the lowest rung tried) | **P1** | **1.0000** | 25/32 PASS. Solved by the dialect list alone. |
| **flce** | **P1** | **P1** | 0.4990 | but **P4 is the right rung**: 8 PASS vs 5, see §5.2 |
| splash_attention | **none** | none | — | 0/96 across all three rungs |
| ragged_paged_attention | **none** | none | — | 0/96 across all three rungs |
| megablox_gmm | **none** | none | — | 0/96 across all three rungs |

### 5.2 FLCE: the backward contract is the rung that matters

FLCE exports fine at every rung; its gate histogram shows where it actually dies.

| rung | `aot_export` | `gradient` | `correctness` | **`all`** |
|---|---|---|---|---|
| P1 (dialect) | 14 | 10 | 3 | **5** |
| P3 (+worked example) | 13 | 15 | 1 | **2** |
| **P4 (+backward CONTRACT)** | **7** | 16 | 0 | **8** |

P4 halves the export failures (14 → 7) and produces **8 passing kernels, the
most FLCE has ever produced in this arena** (previous best: 2, job 3651278).
Conditional on exporting, the pass rate goes 5/18 → 8/24. The `gradient` gate
still claims 16 of 32 — stating the formula does not make everyone get it right
— but it is now a minority failure among exporters rather than the wall.

`p3`'s PASS-only spread is **0.0001** on 2 passing candidates: those two kernels
are numerically identical. Its whole-group spread (0.2666, 25x the floor) comes
entirely from the 30 zeros beside them. That is exactly the distinction the
PASS-only column exists to expose, and it is why P3 is not FLCE's rung despite
being nominally "SIGNAL".

### 5.3 rg_lru: solved at the bottom of the ladder, with a bimodal reward

`p1 | rg_lru` is the best cell in the run: **25/32 PASS**, best reward
**1.0000**, PASS-only spread **0.7554 = 10.8x the 0.0699 noise floor**. The
passing rewards are visibly bimodal —

```
1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6187, 0.5966, 0.5959, 0.5950, 0.5928, 0.5915, ...
```

— two strategy clusters (a full-sequence `lax.associative_scan` that ties the
baseline, and chunked/sequential formulations at ~0.6) separated by ~0.41. That
is a real, learnable gradient between two real kernel designs, not timing
jitter. **Adding rungs made it worse**: 25 PASS at P1, 20 at P3, 15 at P4.

### 5.4 The three kernels that resist every rung — and why

splash, RPA and GMM are **0 PASS out of 96 candidates each**. Every one dies at
`aot_export`. The modal errors say what happened, and they are not the same
errors at every rung:

| kernel | P1 modal failure | P3/P4 modal failure |
|---|---|---|
| splash_attention | invented `pallas_call` kwargs (`in_shapes`, `out_shapes`) | **`Block shape for args[0] (= (Blocked(128), Blocked(128))) must have the same number of dimensions as the array shape (8, 4096, 128)` — 60 candidates** |
| ragged_paged_attention | `pallas_call() got an unexpected keyword argument 'out_shapes'` (13) | same, plus rank mismatches (6) and `divisible by 8 and 128` (3) |
| megablox_gmm | `ConcretizationTypeError` on traced `group_sizes` (7) | `Unsupported TPU device kind: cpu` (17), pytree mismatch on `in_specs` (5) |

**The worked example backfired, measurably.** At P1 splash fails by *inventing*
`pallas_call` kwargs. At P3/P4 those invented kwargs are gone — the example
taught the real signature — and are replaced by a **2-D `BlockSpec` applied to a
3-D operand**, in 60 candidates. The worked example is a fused RMSNorm over a
`[rows, cols]` array, so it demonstrates `pl.BlockSpec((br, dp), lambda i: (i, 0))`;
models copied that *rank* onto splash's `[heads, seq, head_dim]`. A worked
example transfers its shape assumptions along with its dialect. That is the
single most useful negative result here.

**One measurement to distrust.** 17 of megablox_gmm's 96 failures are
`Unsupported TPU device kind: cpu`, raised while exporting on a CPU host. That
is a property of the export environment (`JAX_PLATFORMS=cpu` in the sandbox
child), not necessarily of the kernel — a candidate that queries the device kind
to pick a tile size would fail here and might compile on the judge. GMM's true
denominator is therefore somewhere between 0/96 and 0/79, and the honest
statement is **0 passing, with 17 failures I cannot cleanly attribute to the
model.** Fixing that (export with a TPU device kind stubbed) is the first thing
to change before GMM is called impossible.

### 5.5 Which measured failure modes each addition eliminated

| addition | measured effect |
|---|---|
| **P1 DIALECT list** | `no_code` **0/480**. `'MemoryRef' object does not support item assignment` (7 in job 3651278): **0 occurrences**. `Invalid shape for swap` (7): **0**. `pl.when` misuse (5): 3. `custom_vjp def_fwd` (1): **0**. It cleanly removed the in-kernel Ref-semantics class. |
| **P3 typed skeleton + worked example** | Removed invented `pallas_call` kwargs on splash (10 at P1 → 0 at P3/P4) — and **introduced** the 2-D-BlockSpec-on-3-D-array failure (0 → 60). Net effect on PASS: negative everywhere (flce 5→2, rg_lru 25→20). |
| **P4 primitives** (`dot_f32`, `iota2`, `fill_ref`, `online_softmax`, `chunk_scan`) | No effect on the three dead kernels — they never reach the body where the helpers would be used. Slight negative on rg_lru (20→15). |
| **P4 FLCE backward CONTRACT** | **The one unambiguous win.** `aot_export` 14→7, PASS 5→**8**, the most FLCE has produced in this arena. |

## 6. Verdict, per kernel

* **rg_lru — SOLVED at rung P1.** 25/32 PASS, reward up to 1.0000, PASS-only
  spread 10.8x the noise floor, two distinct strategy clusters. Use `p1`. Do not
  add rungs; they cost passes.
* **flce — SOLVED at rung P4.** 8/32 PASS (arena record), export 24/32, and the
  backward contract is what did it. Use `p4`.
* **splash_attention — NOT SOLVED.** 0/96. Hypothesis, supported by the error
  histogram: its difficulty is the *entrypoint plumbing* (a five-operand
  `pallas_call`, double padding of query and key axes, segment-id broadcasting),
  which no rung here supplies. The only variant that has *ever* exported splash
  is the `seam`, which hands over the `pallas_call` itself (2/16, job 3651278).
  For splash the plumbing must be given, not described — and the next experiment
  is seam-plus-DIALECT, not another prose rung.
* **ragged_paged_attention — NOT SOLVED.** 0/96, same diagnosis. Its 0.2525
  noise floor is a second, independent problem: a candidate would have to be
  >25% faster before the arena could see it.
* **megablox_gmm — NOT SOLVED, with a caveat.** 0/96, but 17 failures are a
  CPU-export artifact rather than a clean model failure. Re-measure before
  concluding.

**The general finding.** Adding context to a whole-program prompt helps exactly
one class of problem: the one that is about *mathematics you can state*
(FLCE's backward). For the class that is about *interface plumbing you must
write*, prose, worked examples and primitives all fail — and a worked example
can actively hurt by transferring its own shape assumptions. The lever that has
ever worked on those kernels is structural: give the model the `pallas_call`.


## 7. Gate histograms, every cell

| kernel | rung | gate histogram (32 judged) |
|---|---|---|
| splash_attention | `p1` | `aot_export` 29, `ast` 2, `exec` 1 |
| splash_attention | `p3` | `aot_export` 32 |
| splash_attention | `p4` | `aot_export` 32 |
| ragged_paged_attention | `p1` | `aot_export` 29, `ast` 2, `exec` 1 |
| ragged_paged_attention | `p3` | `aot_export` 28, `ast` 4 |
| ragged_paged_attention | `p4` | `aot_export` 26, `ast` 6 |
| megablox_gmm | `p1` | `aot_export` 28, `ast` 4 |
| megablox_gmm | `p3` | `aot_export` 31, `ast` 1 |
| megablox_gmm | `p4` | `aot_export` 32 |
| flce | `p1` | `aot_export` 14, `gradient` 10, `all` 5, `correctness` 3 |
| flce | `p3` | `gradient` 15, `aot_export` 13, `all` 2, `exec` 1, `correctness` 1 |
| flce | `p4` | `gradient` 16, `all` 8, `aot_export` 7, `ast` 1 |
| rg_lru | `p1` | `all` 25, `aot_export` 6, `correctness` 1 |
| rg_lru | `p3` | `all` 20, `aot_export` 8, `compile_budget` 2, `correctness` 2 |
| rg_lru | `p4` | `all` 15, `aot_export` 7, `fixtures` 6, `worker` 2, `correctness` 2 |


## 8. Resources and teardown

Job **3687041** (the measurement) and **3686851** (attempt 1, no candidates).

| | attempt 1 (3686851) | attempt 2 (3687041) |
|---|---|---|
| QRs | `sk7524-ladder-{serve,judge}` | `sk7524-ladder2-{serve,judge}` |
| serve | v5p-8, us-east5-a, spot | v5p-8, us-east5-a, spot |
| judge | v6e-1, us-east5-b, spot | v6e-1, us-east5-b, spot |
| created | 16:06:12 | 16:49:34 |
| serve ACTIVE | 16:17:29 (11 min) | 17:19 (30 min; zone contended) |
| gemma serving | 16:32:14 (26 min) | 17:38:16 (49 min) |
| grid | **never ran** (uv fetch, §4) | 480 candidates, 2 full rounds |
| finished | 16:37:22 | 19:27:34, rc=0 |
| QR lifetime | 32 min | 158 min |

**Total QR-alive time ≈ 3 h 10 m** across both attempts (32 min + 158 min), with
a 6-minute gap at 16:38–16:44 where nothing was provisioned — inside the 4-hour
budget measured from the first create (16:06:12). Never more than 2 QRs alive.
**Chip-hours: 4 chips x 190 min = 12.7 v5p chip-hours + 1 chip x 190 min = 3.2
v6e chip-hours ≈ 15.9 chip-hours**, plus one neuronic node and two short CPU
jobs.

Both attempts tore themselves down. Attempt 1's cleanup printed `verified empty
of ladder QRs` and `verified empty of ladder nodes` for both us-east5-a and
us-east5-b, and I independently confirmed zero QRs and zero nodes in both zones
at 16:39:16. Attempt 2's teardown is recorded in §9.

The running RL sweep (`sk7524-tunix-qwen35-v5p32-dbtest-{d,e}`, `sk7524-league-*`,
`sk7524-llamafarm-*`, `forever_sweep`) was read-only-observed and never touched.
No `.env` file or credential was printed.

## 9. Teardown, verified

Attempt 2's own cleanup, from `runs/pallas_arena/ladder-probe-3687041.log`:

```
[cleanup] 19:27:34 tearing down (always-delete, both QRs)
[cleanup] detached deletes issued for sk7524-ladder2-serve and sk7524-ladder2-judge
[cleanup] delete confirmed: sk7524-ladder2-judge
[cleanup] delete confirmed: sk7524-ladder2-serve
[cleanup] verified empty of ladder QRs in us-east5-a
[cleanup] verified empty of ladder nodes in us-east5-a
[cleanup] verified empty of ladder QRs in us-east5-b
[cleanup] verified empty of ladder nodes in us-east5-b
[cleanup] QR lifetime 9638s
```

And independently, checked directly in both zones at 19:31:16 after the job had
exited:

```
QRs   in us-east5-a: NONE      nodes in us-east5-a: NONE
QRs   in us-east5-b: NONE      nodes in us-east5-b: NONE
```

**Zero ladder queued-resources and zero ladder TPU nodes in BOTH us-east5-a and
us-east5-b.** The `setsid nohup` detached deletes fired before the verification
pass, as designed after job 3650988 lost both QRs to a `scancel` mid-delete.

## 10. What to do next

1. **Train rg_lru on `p1` and FLCE on `p4`.** Both are `SIGNAL`, both have
   PASS-only spread well above the floor, and rg_lru has two visibly distinct
   strategy clusters 0.41 apart. Do not use `p3` for anything.
2. **Stop adding prose to splash / RPA / GMM.** Three rungs and 288 candidates
   say it does not work. Run the `seam` (harness owns the `pallas_call`)
   *combined with* the P1 DIALECT list — the seam is the only variant that has
   ever exported splash, and DIALECT is the only addition here that cleanly
   removed a failure class.
3. **Fix the worked example before reusing it.** It must demonstrate a
   3-D-operand `BlockSpec`, or it will keep teaching rank-2 to rank-3 tasks
   (60 candidates lost to exactly that).
4. **Re-measure GMM with a TPU device kind available at export.** 17 of its 96
   failures are `Unsupported TPU device kind: cpu` and cannot be attributed to
   the model.
5. **Give RPA a bigger shape or more timing pairs.** Its 0.2525 noise floor
   means a candidate must be >25% faster before the arena can see it at all.
