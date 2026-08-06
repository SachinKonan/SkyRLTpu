# Phase 2 report — single-judge v6e-1 RMSNorm shakedown (2026-08-06)

Two attempts, both via `tpu/pallas_arena/phase2/run_phase2.sbatch` (one spot
QR `sk7524-pallas-judge-shakedown`, 2h floundering cap, always-delete trap).
**Both QRs deleted; zero idle spot resources remain** (verified empty QR
list after each run).

## Attempt 1 (sbatch 3645906) — bring-up green, grades killed by RLIMIT

Spot v6e-1 ACTIVE in **5.5 min**; provision (uv venv py3.12 +
`jax[tpu]==0.10.2` + chip sanity) in ~50 s. Every grading child then died
at jax init: `TPU initialization failed: Couldn't mmap` — the grader's
16 GB `RLIMIT_AS` (sized for CPU Mosaic-compile blowups) is far below
libtpu's VA reservations. Fixed via `ARENA_RLIMIT_GB` (512 on the judge;
commits a8f110fc, 752fae26). QR deleted by the trap.

## Attempt 2 (sbatch 3645912) — shakedown RAN END TO END

Log: `runs/pallas_arena/phase2-3645912.log`; full JSON:
`runs/pallas_arena/phase2-shakedown-results-3645912.json`; host log:
`runs/pallas_arena/phase2-shakedown-3645912.log`.

Bring-up: create 10:37:26 → ACTIVE 10:44:43 (**7.3 min**, second sample of
spot v6e-1 landing latency in -b despite the v6e-8 crunch). Shakedown wall:
**36 min** for noise floor + 17 full grades + HTTP server acceptance.

### Judge machinery — GREEN across the board

| Invariant | Result |
|---|---|
| Whole cheater battery ON SILICON (cached-output, aliased ref [ast], obfuscated import [poison_stub], seed-reader, memoizer, split-personality [determinism], wrong-grad [gradient], nondeterministic [determinism], no-kernel [exec], wrong-eps [correctness — the small-magnitude vector, 2.7e-1 vs 6.3e-3 tol]) | **all rejected at the right gate** |
| Layer-2 golden: naive-but-correct Pallas kernel (pallas-br16) | **PASS**, honestly scored 0.364 (naive < XLA, as it should be) |
| honest XLA candidates | PASS, scores 0.799 / 0.806 |
| Same-kernel regrade (independent copies) | ratio **0.991** — inside ±3% |
| Determinism N=5 bitwise | enforced in every passing grade |
| Peak HBM reported | yes; max **23.8 GB** across grades |
| children-on-tpu | all grades on the chip; parent stayed on CPU |
| Server: launch-flag problem lock | 400 on wrong problem |
| Server: boot noise floor | measured (0.0888) and injected into grades |
| Server: full grade over POST /grade | PASS, 181.3 s cold |
| Server: hash→reward cache | **2.1 ms** hit with byte-identical reward |

### Measured numbers (throughput-table inputs)

- **Per-candidate cost (cold fork, full production protocol): mean 112 s,
  min 50 s (AST reject), max 186 s.** Passing grades ≈ 180 s; gate-failed
  cheaters ≈ 86–140 s. This is ~30× the DESIGN's ~4 s/candidate chip
  estimate because the shakedown pays per-candidate: fresh-process jax+libtpu
  init (~15–20 s), correctness at 3 seeds × 3 production shapes, N=5
  determinism, gradient check, and 4 × 23 timed iterations — serially. The
  phase-3 judge must amortize (resident compiled reference per chip is
  already in the design; add a persistent grading worker and/or a lighter
  per-grade protocol) before the 110/min fleet table can hold.
- **Noise floors (per case)**: 8192x4096 → 0.094; 32768x8192 → 0.030;
  73728x2880 → 0.030. Floor measurement wall: 52 s.
- **Ref-vs-ref scores: 1.053 / 1.021 / 1.019 — FAILED the 1.00±2%
  invariant.** Systematic bias, not noise: in every timed pair the
  first-position leg (right after on-device input generation) runs slower.
  Judge fix queued: counterbalance pair order (R,C / C,R alternation) in
  child_runner timing — this is precisely the kind of defect the shakedown
  exists to catch, and it also explains the inflated small-case floor.

### The five FAILs, diagnosed (none is a judge defect)

1. **ref-vs-ref-1.00±2%** — real protocol finding (above); fix:
   counterbalanced pair order before any scoring run.
2. **pallas-br256 / br512** — my oversized golden variants genuinely blow
   the 32 MB scoped-VMEM limit (`CompileTimeScopedVmemOom`, 32.18M/32.00M);
   the judge correctly failed them at runtime. Real candidates must budget
   VMEM; br16 is the working golden.
3. **unjitted-honest** — caught by the near-overflow-bf16 adversarial
   vector: eager op-by-op mean-of-squares overflows f32 to inf where the
   fused/jitted reference survives. The adversarial library doing its job;
   the variant was mislabeled "honest at all extremes".
4. **timer-tamperer** — its own 200-op waste chain OOMs HBM under the
   gradient probe (128 M short). The tamper defense itself was proven in
   the CPU battery; on TPU the variant needs a smaller waste chain.
5. (attempt-1 RLIMIT — fixed, regression-proofed by `ARENA_RLIMIT_GB`.)

### honest-xla scoring 0.80, explained

The CPU-battery honest variant outputs **f32**; the baseline casts back to
**bf16**. RMSNorm is HBM-bound, so writing 2× the output bytes is a real
~20% slowdown — the judge measured a true difference. Contract note added:
candidates should emit bf16 (as the task docstring says) to be
traffic-comparable.

## Verdict

**Phase-2 objective met**: QR lifecycle, provisioning, judge server, FIFO
grading, cheater rejection on silicon, calibrated tolerances, determinism,
peak-HBM reporting, cache, and cost measurement all validated end to end on
a real v6e-1 — with two genuine findings (timing position bias; true
per-candidate cold cost) that phase 3 must incorporate. Both QRs deleted.
