# Pallas Kernel Arena — Prompt A/B Status

_Last updated: 2026-09-05. Owner: sk7524. Branch: `agent/ttd-discover-erdos`._

## The experiment in one paragraph

Two models (**qwen3.5-27B**, **gemma-4-31B**) are each handed a **working** TPU
kernel and asked, in a single turn, to make it **faster** while staying correct.
Two kernels are tested — **splash attention** and **rg_lru** (the Griffin gated
linear recurrence) — giving a **2×2** grid of (model × kernel), 32 attempts per
cell. Reward = `baseline_time / candidate_time`, geomean over graded shape cases,
forward and backward. The **seed** program each model starts from already scores
a fixed value; **beating the seed is the bar**. We are comparing two prompt
versions, **v1** (baseline) and **v2** (new), to see whether the v2 additions make
the models write faster kernels.

## Prompt v1 vs v2

Both prompts are seed-improvement prompts with the same seven sections: framing,
graded shapes, backward-pass contract, output format, **the full working seed
kernel pasted inline**, the seed's measured judge feedback, and a strategy note.
They are archived byte-exact and hash-verified against what the models received:

| prompt | splash sha256[:12] | rg_lru sha256[:12] |
|---|---|---|
| v1 (served) | `d81e11e8dc68` | `901162646b5c` |

Archived at `runs/pallas_arena/prompts/served-*.txt`.

**v2 adds exactly two sections and rewrites Output/Strategy to match; nothing
else changes.** Verified by diff: two headings added, seven lines removed.

1. **`## Rules your rewrite must satisfy`** — states, with a worked example each,
   the constraints the seed code silently obeys but v1 never named:
   the (8,128) tiling rule, no Python control flow on traced values, ref-write
   shape matching, and "kernel must return." These target ~43% of v1 failures.
2. **`## Reusing what you are not changing`** — offers the seed's top-level names
   as `from lib import ...`, so a model can keep the parts it is not changing
   instead of retyping the whole program (which is where qwen breaks on its own
   syntax). The `lib_splice` module resolves the imports back into a
   self-contained program before grading.

## Baselines and shapes: tokamax

The "production kernel" each candidate must beat, the canonical benchmark
**shapes**, and the judge's **device timer** (xprof) all come from
`openxla/tokamax` (vendored at `third_party/tokamax`, installed on every judge
via `phase2/provision_judge.sh`). Two tokamax-derived tasks were **retracted** as
unfair baselines (megablox untuned — XLA `ragged_dot` beats it 38×; FLCE unfair).

## Reward denominator: production vs XLA

The judge runs `general_mode`: it times every entry of `baseline_candidates()`
per shape and grades against the **fastest**. For the two arena tasks that set is
`{production, xla-*}` (splash) / `{production, lax-associative-scan}` (rg_lru), so
"production" (the tuned tokamax kernel) wins the election almost everywhere — the
effective bar is "beat the expert kernel," which nothing does.

**`ARENA_BASELINE=xla` (added 2026-09-02) drops production from the denominator**,
leaving the naive XLA baseline the seed already beats. Rationale: against
production, every candidate scores ≤1.0 and piles below the seed — a **validity**
gradient but no **speed** gradient. Against XLA, rewards spread above 1.0 and the
slope keeps going. It is a grade-time change over banked gens (no regeneration).
The flag is carried in the ray `cfg` dict (shipped by value to each task) and set
in the worker before the reward election reads it — shell/ray env inheritance
proved unreliable (three failed attempts; see `rl_judges.sbatch` / `ray_pool.py`
history).

## Results

Two graded scales exist and both are reported. The **XLA denominator** (below)
is the current, correct one: it moves rewards off the 1.0 noise-floor gate so
near-seed candidates keep their true spread. Production-graded numbers are kept
for provenance but superseded.

Seed bars, XLA denominator, v6e: **splash 0.40**, **rg_lru 1.99**. (The rg_lru
seed BEATS naive XLA scan ~2×; the splash seed is ~2.5× SLOWER than XLA
attention — the handed splash kernel is mediocre.) Production seed bars were
splash 0.232 / rg_lru 1.000.

**Important caveat on interpretation:** whether a candidate beats the SEED is
~denominator-invariant (`cand/seed = seed_time/cand_time`, baseline cancels).
The XLA denominator's real value is (a) un-gating near-seed candidates from the
1.0 collapse so RL sees their real spread, and (b) making "beat XLA" (1.0) a
meaningful milestone. It does not change the beat-the-seed count.

### V1 — COMPLETE on the XLA denominator (v6e, all 4 cells 32/32)

| cell | passed | best reward | seed bar | beat seed? |
|---|---|---|---|---|
| qwen · splash | 10/32 | **0.39** | 0.40 | no |
| gemma · splash | 7/32 | **0.27** | 0.40 | no |
| qwen · rg_lru | 4/32 | **2.02** | 1.99 | **1 of 4** |
| gemma · rg_lru | 3/31 | **2.03** | 1.99 | **1 of 3** |

**The v1 finding, now on trustworthy hardware and a beatable baseline:** on
**splash**, neither model reaches the seed (best 0.39 vs 0.40) — they write
working kernels slower than the one handed to them. On **rg_lru**, each model
squeaks exactly one candidate marginally past the seed (~2.02 vs 1.99). The XLA
scale reveals the rg_lru "ties" (all 1.000 under production) were actually a
real spread — e.g. qwen·rg_lru is [1.18, 1.27, 1.54, 2.02], not four ties.

(Production-denominator v1, superseded: qwen·splash best 0.199, gemma·splash
0.192, both rg_lru best exactly 1.000.)

**V1 failure modes** (128 candidates; 24 valid, 2 judge-fault, 102 real fails):

| root cause | count | targeted by v2? |
|---|---|---|
| TPU tiling/layout (tile-(8,128) 18, aot_export 5, reshape 3, VMEM 3) | 29 | ✅ constraints |
| Python control flow on traced values (`if` 14, `max()` 2) | 16 | ✅ constraints |
| long-program bookkeeping (syntax 5, wrong self-sig 5, IndexError 3, NameError 1) | 14 | ✅ lib-imports |
| Pallas ref semantics (write-shape 7, returned None 2) | 9 | ✅ constraints |
| Pallas internal verify error | 5 | — |
| numerically wrong | 3 | — |
| unclassified early CPU rejections (pregate) | 21 | not yet cracked open |

Per-cell signals are **task-specific, not model-specific**: gemma·rg_lru is
16/32 the same tile error; gemma·splash is 9/32 `if`-on-traced; qwen fails
diffusely (long-program collapse). splash → control-flow, rg_lru → tiling.

### V2 — generation COMPLETE (all 4 cells); XLA grading IN PROGRESS

The four v2 cells are being re-graded on v6e against the XLA denominator
(`regrade_v2only.sh` on the live judge). Not yet complete: v6e-8 spot slices
keep getting **preempted mid-grade** (judge13 at 23h, judge14 at 12h), so v2
cells came back partial (14/32, 2/32) and are being re-run. **No v2 conclusion
until each cell is a full 32/32** — the partials were preemption damage, not a
v2 property. The generation-side effects below ARE complete and hardware-
independent.

### V2 — generation-side (complete) and the superseded v5p grade

**Generation-side (hardware-independent, all 32/cell):**

| cell | lib-import used | max lines v1→v2 | static parse v1→v2 |
|---|---|---|---|
| qwen · splash | **7/32** | 578 → 468 | 29 → 30 |
| qwen · rg_lru | **6/32** | 486 → 347 | 28 → 30 |
| gemma · splash | 0/32 | 336 → 346 | 32 → 32 |
| gemma · rg_lru | 0/32 | 302 → 165 | 31 → 31 |

- lib-imports is **qwen-only**: gemma ignores the offer entirely (0/64).
- v2 shrinks the program-length **tail** (the pathological 300–500-line programs
  where qwen breaks), not the median.
- qwen parse rate ticks up slightly.

**Graded (v5p, denominator=production — read with caveats):**

| cell | graded | result |
|---|---|---|
| qwen · rg_lru | 32/32 | 0 passed — **15 VMEM compile-OOMs (v5p artifact)** + 14 pregate |
| qwen · splash | 6/32 | 2 passed at **0.253, 0.255** — *above* the 0.232 seed |
| gemma · splash | 4/32 | 2 passed at 0.141, 0.179 |

The **only** glimmer of a genuine improvement in the whole campaign is
qwen·splash v2 at 0.255 vs the 0.232 seed — **unconfirmed** (6/32, and v5p).

## What is blocked, and why

**Only the v2 XLA grades remain.** v1 is done; both seeds are done (rg_lru 1.99
CONFIRMED the XLA denominator is live — it was 1.0 vs production). The four v2
cells are re-grading on v6e via `regrade_v2only.sh`.

**Blocker: v6e-8 spot in us-east5-b is both slow to get AND unstable once
gotten.** Slices take hours to land (judge13 ~18h) and get preempted mid-grade
after ~12–24h. A clean 4-cell v2 pass needs several uninterrupted hours; the
slices don't reliably provide them. Mitigations in place: the regrade is
resumable + writes verdicts incrementally (a preemption loses nothing already
graded), and the XLA baseline compile cache is warm on GCS (a fresh judge skips
the one-time ~3h cold compile that judge13/14 each paid).

The XLA denominator required a **four-layer fix** (driver env → ray-start env →
cfg-in-worker → **cache contract key**); each layer only surfaced once the prior
was fixed. The last was subtle: the reward cache served the seed's stale
production verdict because `_contract_tag()` did not include the denominator.
All four are committed.

## Key files

| path | what |
|---|---|
| `probe/prompt_ref_first.py` | prompt builder (v1 base + v2 constraints/lib sections) |
| `probe/lib_splice.py` | resolves `from lib import ...` back to a self-contained program |
| `probe/gen_smoke.py` | generator + async submit-on-completion grading client |
| `probe/coserve_cell.sbatch` | one-slice generate+grade (league pattern), per-chip mesh |
| `probe/regrade_v6e.sh` | grade all 10 datasets on one v6e judge |
| `judge/ray_pool.py` | per-test ray dispatch; `--baseline` (xla/all) via cfg |
| `judge/problems/*.py` | tasks: reference, seed, tokamax baseline, shapes, `elected_candidates` |
| `rl_judges.sbatch` | standalone judge launcher; forwards `ARENA_BASELINE` |
| `runs/pallas_arena/prompts/served-*.txt` | v1 prompts, byte-exact, hash-verified |

## Hard-won gotchas (do not relearn)

- **v6e uses VFIO, not `/dev/accel*`** — count TPUs with `lspci | grep -ci 'processing accelerators'`. Validate any health probe against a known-good node; keep it advisory.
- **RUNTIME must match the chip** — `v2-alpha-tpuv6e` for v6e; the default `-tpuv5` gives `num_chips=0`.
- **v6e-8/-16 permitted only in us-east5-b** (us-east5-a denies with code 7); v5p in us-east5-a.
- **Hold a queued resource, don't churn it** — re-creating on a deadline sends you to the back of the line.
- **XLA cache name encodes ctx/max_num_seqs/chips** — any mismatch forces a full recompile.
- **Env for ray tasks must go through cfg/runtime, not the shell** — the raylet's env does not reliably reach task workers.
