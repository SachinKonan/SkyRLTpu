# The seam — a prompt that gives away the plumbing and keeps the decisions

*Run: 2026-08-07. TPU jobs 3650988 (preempted) and **3651278** (the
measurement). CPU jobs 3650754/76/96, 3650817, 3650852, 3650953. Raw data:
`runs/pallas_arena/seam-results-3651278.{json,jsonl}` (160 candidates, one row
each: full generation text, extracted fill, composed program, gate,
observation, reward), `seam-judge-boot-3651278.json`, `seam-tables-3651278.md`,
`seam-control-3650852.json`, `seam-results-3650988.jsonl` (the preempted arm).*

The question: does a prompt that hands over the **scaffolding** and keeps the
**strategy** raise export rate without collapsing the within-group reward
spread the way `tailored` did?

**Answer: on the API-hallucination failure it works decisively — invented
`pallas_call` kwargs fell from 8/80 to 1/80 and the failures moved to real
in-kernel semantics. On the headline it is a draw at 5/20 signal groups, and
every one of those 5 groups clears its noise floor by 18–31x, so nothing
collapsed into noise anywhere. But the seam did not raise pass rate: it beat
reference on splash (first non-tailored exports ever) and lost to it on FLCE.**

Also, for the first time in this arena, **a candidate beat its baseline**:
`gemma4-31b | seam | rg_lru` at **1.1941**.

---

## 1. Headline

```json
{"overall_signal_groups": "5/20", "overall_signal_group_frac": 0.25,
 "overall_nonuniform_groups": "5/20",
 "best_config_by_max_score": "gemma4-31b|seam|rg_lru",
 "best_score": 1.1940660961115808, "any_candidate_passed": true}
```

`signal groups` = complete groups whose within-group reward spread exceeds the
**judge's own measured noise floor** for that task. **Signal == non-uniform,
5 and 5** — every group that varied at all varied by far more than the timing
jitter. That is the direct contrast with the previous run, where
`tailored|flce` passed 16/16 with a spread of 0.0042 against floors of
0.0069/0.0158 and was counted as trainable while being pure noise. **No cell
in this run collapsed into the noise floor.**

## 2. Seam vs reference, per task

Groups of 8 (see §7), one round, 160 candidates, all with a terminal verdict.

| task | noise floor | variant | judged | export | PASS | max reward | spread | spread/floor | **signal groups** |
|---|---|---|---|---|---|---|---|---|---|
| flce | 0.0143 | reference | 16 | **50.0%** | **12.5%** | **0.2630** | 0.2630 | **18.4x** | **1/2** |
| flce | 0.0143 | seam | 16 | 31.2% | 0.0% | — | — | — | 0/2 |
| rg_lru | 0.0383 | reference | 16 | **50.0%** | 43.8% | 1.0000 | 1.0000 | 26.1x | **2/2** |
| rg_lru | 0.0383 | seam | 16 | 37.5% | 37.5% | **1.1941** | 1.1941 | **31.1x** | **2/2** |
| splash_attention | 0.0312 | reference | 16 | 0.0% | 0.0% | — | — | — | 0/2 |
| splash_attention | 0.0312 | seam | 16 | **12.5%** | 0.0% | — | — | — | 0/2 |
| megablox_gmm | 0.0017 | reference | 16 | 0.0% | 0.0% | — | — | — | 0/2 |
| megablox_gmm | 0.0017 | seam | 16 | 0.0% | 0.0% | — | — | — | 0/2 |
| ragged_paged_attention | 0.2239 | reference | 16 | 0.0% | 0.0% | — | — | — | 0/2 |
| ragged_paged_attention | 0.2239 | seam | 16 | 0.0% | 0.0% | — | — | — | 0/2 |

Per configuration (8 candidates each; `sig` = signal groups / complete groups):

| model | variant | task | export | PASS | max reward | spread | sig |
|---|---|---|---|---|---|---|---|
| gemma4-31b | reference | flce | 75% | 2 | 0.2630 | 0.2630 | **1/1** |
| gemma4-31b | seam | flce | 50% | 0 | — | — | 0/1 |
| gemma4-31b | reference | rg_lru | 38% | 2 | 0.5986 | 0.5986 | **1/1** |
| gemma4-31b | **seam** | **rg_lru** | 50% | **4** | **1.1941** | 1.1941 | **1/1** |
| qwen35-27b | reference | rg_lru | 62% | 5 | 1.0000 | 1.0000 | **1/1** |
| qwen35-27b | seam | rg_lru | 25% | 2 | 1.0989 | 1.0989 | **1/1** |
| qwen35-27b | reference | flce | 25% | 0 | — | — | 0/1 |
| qwen35-27b | seam | flce | 12% | 0 | — | — | 0/1 |
| gemma4-31b | seam | splash_attention | 12% | 0 | — | — | 0/1 |
| qwen35-27b | seam | splash_attention | 12% | 0 | — | — | 0/1 |
| *(the other 10 cells)* | | | 0% | 0 | — | — | 0/1 |

**rg_lru is the best RL target of the five, by a distance**: 4/4 cells carry
signal, 13 of 32 candidates pass, the spread clears the floor by 26–31x, and
it is the only task where anything beats its baseline.

## 3. The one thing the seam unambiguously fixed

The previous run's diagnosis was that gemma writes real Pallas structure and
then invents the signature, and that prose gotchas do not fix it. Both halves
replicated, and the seam fixed it:

| | reference | seam |
|---|---|---|
| candidates emitting an invented `pallas_call` kwarg | **8/80** | **1/80** |

And the failure *modes* moved wholesale. Top `aot_export` messages:

| reference (n=42) | seam (n=59) |
|---|---|
| 5 × `pallas_call() got an unexpected keyword argument 'out_spec'` | 7 × `'MemoryRef' object does not support item assignment` |
| 4 × `'PjitFunction' object is not subscriptable` | 7 × `TracerBoolConversionError` |
| 2 × `module 'jax.experimental.pallas.tpu' has no attribute 'pallas_call'` | 6 × `Invalid shape for swap. Ref shape: (8, 4, 128)` |
| 2 × `block shape are divisible by 8 and 128 …` | 5 × `too many values to unpack (expected 2)` |
| 1 × `module 'jax.experimental.pallas' has no attribute 'Ref'` | 3 × `not enough values to unpack (expected 2, got 1)` |

Reference fails at *"I do not know this library's API"*. Seam fails at *"I
wrote the wrong thing into a Ref"* and *"I got the return contract wrong"* —
dense, specific, learnable errors about the actual kernel. That is exactly the
shift the design was aiming at, and it is visible in the observations the RL
side would be fed.

`no_code` also went to **zero**: 0/160, against 28/155 (18%) in the previous
run — far under the 30% harness-alarm threshold. The extractor found every
required name in **74/80** seam candidates (92.5%), overwhelmingly from a
single fenced block.

## 4. Qwen is measurable now

The previous run could not measure Qwen at all: 3 of 80 generations reached
`stop`, so every Qwen cell was zero and the honest reading was *unmeasured,
not incapable*. With the `<think>\n\n</think>\n\n` disable-thinking renderer
(verified token-for-token against the server's own template before samples
were spent):

| model | `stop` | `length` | best cell |
|---|---|---|---|
| gemma4-31b | **78/80** | 2 | seam \| rg_lru, 4/8 PASS, **1.1941** |
| qwen35-27b | **63/80** (was 3/80) | 17 | reference \| rg_lru, 5/8 PASS |

Qwen produced its first passing kernels in this arena. It is now a real second
model, not a hole in the grid.

## 5. Where the seam did NOT help, and why that is informative

* **FLCE: the seam lost.** reference 50% export / 2 PASS / max 0.2630; seam
  31% export / **0 PASS**. Both gemma cells concentrate on gate `gradient`
  (4 each) — kernels that compile and are numerically correct forward and then
  miss `d/d(hidden)`. For FLCE the harness already owned the `custom_vjp`, so
  the seam gave away less than it did elsewhere while adding a two-function
  return contract (`(logprobs, carry)`) that models got wrong 8 times
  (`too many values to unpack`). **On a task whose plumbing is a `lax.scan`,
  the seam's interface is a net cost.** Nothing beat FLCE's baseline; the best
  anywhere is still **0.7008** from the previous run, and this run's best FLCE
  score was 0.2630.
* **splash_attention: the seam is the only thing that has ever exported.**
  0/16 reference (replicating last run's 0/16), **2/16 seam**, and *both* seam
  candidates reached the chip and got a `correctness` verdict rather than
  dying at export. First non-`tailored` splash candidates ever to run on
  silicon. Still 0 PASS.
* **megablox_gmm and ragged_paged_attention: 0/16 in every cell.** Not a seam
  failure specifically — reference is 0/16 too. These are the two seams with
  the most intricate interface (scalar-prefetched index maps, an 8-argument
  `decode_block` with two persistent scratch Refs), and the modal seam failure
  is `Invalid shape for swap. Ref shape: (8, 4, 128)` — models writing a
  wrong-shaped value into the `m_ref`/`l_ref` scratch. **The seam's own
  interface became the difficulty.** If these two are worth pursuing, the next
  move is to shrink their fill signature, not to add more prose.

**RPA carries a second, independent warning: its noise floor is 0.2239** —
22%. The op is short enough that the timing protocol cannot resolve better
than that, so a candidate would have to be >22% faster before the arena could
tell. As configured, RPA is close to untrainable by timing regardless of what
the model writes. GMM is the opposite extreme at 0.0017.

## 6. The judge now grades all five tasks — three for the first time

The persistent worker calls `jax.jit(problem.baseline)` at boot and treats any
exception, including `BaselineUnavailable`, as *"problem not served"*. It does
not fall back. `ragged_paged_attention` raised on **both** branches
unconditionally; `rg_lru` raised on TPU whether or not `recurrentgemma`
imported. **Neither could ever have booted.** Fixed with labelled fallbacks —
a judge that refuses to boot has graded nothing at all, which is strictly
worse than a slower-but-real denominator.

Boot report, all five on one v6e-1:

| task | noise floor | ref-vs-ref | `baseline_impl` | denominator, stated honestly |
|---|---|---|---|---|
| splash_attention | 0.0312 | 0.998 / 1.019 | `xla-fallback` | a competent query-blocked XLA attention, **not** Google's splash kernel |
| flce | 0.0143 | 0.999 / 0.997 | *(production)* | our own `custom_vjp` tiled kernel |
| ragged_paged_attention | 0.2239 | 1.032 / 1.004 | `xla-paged-decode-fallback` | **not** vLLM-TPU's Pallas v3 kernel |
| megablox_gmm | 0.0017 | 1.000 / 1.000 | `lax-ragged-dot-fallback` | **megablox refused these shapes**; this is `ragged_dot`, not the tuned kernel |
| rg_lru | 0.0383 | 0.996 / 1.015 | `lax-associative-scan` | **not** recurrentgemma's Pallas scan |

Every ref-vs-ref is inside the ±2% band. **Three of five scores are against a
labelled fallback, and the 1.1941 headline is "1.19x `lax.associative_scan`",
not "1.19x recurrentgemma".** The megablox fallback is a live finding: the
guard I added is the only reason GMM booted at all.

Plus one-chip probe shape sets for the three new tasks, each keeping the axis
the task exists for at production width (k=4096/n=14336, d=2560, page_size 64 /
8 kv heads / 32 q heads / 128 head_dim) and each with a non-block-divisible
holdout (batch=**17**, m=**3000**, t=**1500**).

## 7. Method, and what to distrust

* **Groups of 8, not 16.** The first attempt's serving slice was preempted, and
  the second had ~80 minutes. ttt-discover normalizes over 16, so 8 is a proxy.
  It errs *conservative*: a smaller group has a smaller expected max−min
  spread, so this under-reports signal rather than over-reporting it. One
  round, so exactly one complete group per cell and 20 in total — enough for a
  per-task verdict, not enough to separate two cells whose spreads are close.
* **Fills are 2–3x over budget.** Target was 300–1200 tokens; measured medians
  were ~1000 (gmm) to ~2500 (splash) tokens, max 26 050 chars. Models restate
  the scaffold despite being told not to. Worth tightening.
* **`aot_export` still dominates** at 101/160 overall. The seam changed *which*
  errors these are, not how many candidates clear the gate.
* One arm is discarded: job **3650988** was preempted five minutes into
  generation and 640 of its 672 rows are `error:URLError(ConnectionRefused)`
  with `gen_chars=0`. Those rows are **not** model failures and are excluded
  from every number here. Its 32 real generations are consistent with the
  above (gemma reference splash 16/16 `stop`, 0/16 export, modal error
  `unexpected keyword argument 'out_spec'`).

### Three guards added because of that preemption

Two consecutive probe runs have now burned budget sampling against engines
that were not serving (first a two-host mesh hang, then this), and both
recorded a full grid of confident zeros. Now, in the driver:

1. **Pre-flight (hard).** One *real* completion per engine that must return
   non-empty text, before a single grid candidate. It fired correctly on the
   measurement run: `[preflight] gemma4-31b: ALIVE`, `qwen35-27b: ALIVE`.
   `/v1/models` returning 200 is not evidence an engine can generate.
2. **Mid-run liveness.** A preemption happens in the middle — the engines
   answered at 14:12 and were gone by 14:25 — so a pre-flight alone is not
   enough. A whole group of transport errors now re-checks the engine and
   aborts the grid if it is dead, instead of writing zeros. Candidates already
   on the judge are still drained.
3. **`no_code` alarm** at 30% per config (the last good run's baseline was
   14%), stopping the grid after three tripped configs. It did not fire: the
   measured rate was 0%.

And teardown was hardened: `scancel` sends TERM then KILL after ~30 s while one
queued-resource delete takes ~90 s, so on job 3650988 the trap was killed
mid-delete and **both QRs survived a cancellation meant to remove them**. I
deleted them by hand and verified both zones. Cleanup now issues `setsid nohup`
deletes *first* so they outlive the shell, then verifies.

## 8. Control first — it caught three real bugs before any chip

Every seam's known-good fill through the identical path, wrapped in a
model-style response with a **decoy block first**:
`fill -> extract_fill -> compose -> CPU AOT export -> [judge]`, plus CPU
numerics against each problem's own fp32 reference.

**interpret 8/8 OK, pre-gate 8/8 PASS** (job 3650852). Max relative error:
FLCE fwd and grad **0.0**, splash **1.69e-07**, rg_lru **9.87e-07**, gmm
**9.41e-08**, RPA **1.13e-07**. Fills 110–423 tokens. Two genuinely different
strategies per seam where possible (recompute vs saved-LSE; `HIGHEST` vs
default matmul precision; sequential vs associative scan).

Caught before chips: (1) the extractor carried a decoy block along with the
real answer — fixed to prefer the last *complete* block; (2) an RPA `BlockSpec`
whose second-minor block dim (1 of 8 kv heads) violates Mosaic's tiling rule —
**exactly the class of failure the seam exists to absorb**, and had it been in
the prompt every RPA candidate would have died at export and read as "models
cannot write paged attention"; (3) a `SyntaxError` in the control's own
`interpret=True` injection.

Arena regression battery: **175 passed, 0 failed** (job 3650953) — unchanged
after three rewritten baselines, fifteen new shape cases and three
`adversarial_cases()` fixes.

The prompt's API block is **introspected, not remembered**:
`seam_dryrun.py` re-derives every signature with `inspect.signature` at the pin
(jax 0.10.2) and asserts that the names it says do not exist really do not —
`pl.load`, `pl.store`, `pltpu.ANY`, `pltpu.TPUCompilerParams`, `jax.Shape`, and
every invented `pallas_call` kwarg. 33 assertions, all green. Two of those
facts would have been wrong from memory.

## 9. Example fill-ins

**Best — `gemma4-31b | seam | rg_lru`, reward 1.1941, the first candidate in
this arena to beat its baseline.** Chunked associative scan at `CHUNK = 256`;
the whole decision is the chunk length and the fold-in of the carry:

```
GATE all | rg_lru | PASS reward=1.1941
probe-4x2048x2560:          cand 1.268ms vs ref 2.207ms (1.740x)
probe-2x1024x2560:          cand 0.358ms vs ref 0.295ms (0.825x)
probe-holdout-2x1500x2560:  cand 0.558ms vs ref 0.808ms (1.449x)
peak HBM 28.63GB
```

```python
# Trade-off between parallel associative scan (log-depth, high memory/flops)
# and sequential scan (linear-depth, low memory). 256 is a typical TPU
# sweet spot for SSMs of this dimension (d=2560) to maintain VMEM efficiency.
CHUNK = 256

def scan_chunk(x_chunk, a_chunk, reset_chunk, h_prev):
    x_fp32 = x_chunk.astype(jnp.float32)
    g_chunk = jnp.sqrt(jnp.maximum(1.0 - jnp.square(a_chunk), 0.0)) * x_fp32

    def combine(left, right):
        a_l, g_l = left
        a_r, g_r = right
        return (a_l * a_r, g_l * a_r + g_r)
    ...
```

It wins where it should — 1.74x on the long sequence, 1.45x on the holdout —
and loses (0.825x) on the short one, which is precisely the chunk-length trade
the seam left open. That is a real strategy decision producing a real score
difference, which is the whole hypothesis.

**Representative failure — the modal seam error, RPA:**

```
GATE aot_export | ragged_paged_attention | ValueError
Invalid shape for `swap`. Ref shape: (8, 4, 128). Expected shape: ...
need: the SAME entrypoint must trace at EVERY declared shape
```

The model understood the algorithm and wrote a wrong-shaped store into the
scratch Ref. Compare the modal *reference* failure — `pallas_call() got an
unexpected keyword argument 'out_spec'` — which carries no information about
attention at all.

## 10. Resources and teardown

| | attempt 1 (3650988) | attempt 2 (3651278) |
|---|---|---|
| QRs | `sk7524-seam-serve` v5p-16 us-east5-a, `sk7524-seam-judge` v6e-1 us-east5-b | `sk7524-seam2-serve`, `sk7524-seam2-judge` (same shapes/zones) |
| created | 13:38:28 | 14:42:59 |
| serve ACTIVE | ~13:58 (20 min) | ~14:47 (5 min) |
| both engines serving | 14:12:50 (34 min) | 15:01:21 (18 min, warm caches) |
| outcome | **spot-preempted** ~14:25, serve QR `SUSPENDED` | completed, 160 candidates |
| QR lifetime | 61.8 min | **59.98 min** |

Never more than 2 QRs alive at any moment. Hard cap 3 h from the first create
(16:38) — never approached; the second attempt was additionally capped at
16:28 and finished at 15:42.

**Chip-hours: 16.2 v5p chip-hours + 2.0 v6e chip-hours ≈ 18.3 chip-hours**
(8 chips × 61.8 min + 8 chips × 60.0 min; 1 chip × 61.8 min + 1 chip ×
60.0 min). Plus two neuronic compute nodes and six short CPU jobs (~17 min).

**Teardown, verified.** Attempt 2 tore itself down: `verified empty of seam QRs
in us-east5-a`, `… in us-east5-b`, and the same for nodes. Attempt 1's trap was
killed mid-delete by `scancel`; I deleted both QRs by hand at 14:38–14:40 and
independently confirmed. Final state, checked directly in both zones after both
jobs exited:

```
us-east5-a QRs: (no seam QRs)     us-east5-a nodes: (no seam nodes)
us-east5-b QRs: (no seam QRs)     us-east5-b nodes: (no seam nodes)
```

**Zero probe QRs and zero probe TPU nodes in BOTH us-east5-a and us-east5-b.**
The running RL sweep (`sk7524-tunix-qwen35-v5p32-dbtest-{d,e}`,
`sk7524-league-*`, `forever_sweep`) was read-only-observed and never touched.

## 11. Verdict

**The seam does not collapse gradient signal — that was the risk, and it did
not happen.** All 5 signal groups clear their noise floor by 18–31x; nothing
landed in the `tailored` trap. But at 5/20 it does not *beat* reference on the
headline either: it wins on splash (0% → 12.5% export, first silicon), loses
on FLCE (2 PASS → 0), and ties on rg_lru while producing the better kernel.

**Its unambiguous win is the failure mode.** Invented `pallas_call` kwargs
8/80 → 1/80, `no_code` 18% → 0%, and the residual errors are about Refs, shapes
and return contracts instead of about the library's API. That is the difference
between an environment that teaches JAX trivia and one that teaches kernels.

**What I would change before the RL run:**

1. **Train on `rg_lru` first.** 4/4 cells carry signal, 13/32 pass, 26–31x the
   floor, and it is the only task where a candidate beats its baseline. It is
   also the cheapest to grade.
2. **Use `reference` for FLCE and `seam` for splash.** The seam's value is
   task-dependent and this run says which way for each.
3. **Simplify the RPA and GMM fill signatures** — 8 arguments and two
   persistent scratch Refs made the *interface* the difficulty. And RPA's
   0.2239 noise floor needs a bigger shape or more timing pairs before it can
   be trained on at all.
4. **Rerun at groups of 16 and ≥3 rounds.** 20 groups is a per-task verdict,
   not a cell-vs-cell one.
5. **Bind the real baselines, or keep saying "fallback" out loud.** Three of
   five denominators are fallbacks; megablox refused our shapes outright.
