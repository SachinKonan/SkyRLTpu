# Pallas Kernel Arena — implementation plan (updated 2026-08-06, session-final)

RL kernel-generation arena for TPU (the GPU-MODE/trimul idea, on Pallas).
Slate (user-settled): RMSNorm (warm-up) → **splash attention @ [4,18432] causal**
(train flagship) → **ragged paged attention** (inference) → **FLCE** (vs our own
custom_vjp kernel, 73728 tok × ~150k vocab) → **megablox gmm** (MoE) → **RG-LRU
scan** (SSM). Judge: warm fleet of 5× spot **v6e-1 in us-east5-b** (v6e-1..256
confirmed available there; fresh QR quota pool). Arena GCS bucket (created):
`gs://sk7524-pallas-arena-us-east5` (regional US-EAST5 — serves -b at full
locality) — layout: `judge-registry.json`, `reward-cache/`, `kernel-corpus/`,
`goldens/`. Reward: gate-then-score, geomean latency ratio vs the reference,
%-of-speed-of-light logged for memory-bound tasks.

**Grading idiom (settled): NO containers.** Same shape as today's
SandboxRewardEvaluator.execute_code grading everywhere else: fork a plain
subprocess per candidate — compile + trace + run all in the child — with the
eval timeout, RLIMIT_AS (64 parallel Mosaic compiles of pathological unrolls
can OOM a host; timeouts alone don't catch it), and the reference modules
stubbed to raise in the child's sys.modules (runtime backstop for
wrap-the-baseline, stronger than AST). Process isolation alone guarantees a
candidate cannot touch the judge's timer, reference outputs, or reward cache.

## Phase 0 — Prerequisites (no TPU, ~half a day)
- [x] Zone: us-east5-b. Bucket: gs://sk7524-pallas-arena-us-east5 (created).
- [x] Confirm QueuedResourcePerProjectPerZone in us-east5-b ≥ 5 free slots.
      (2026-08-06: limit 20, 13 in use by another user's fleet → 7 free.
      CAPACITY warning: a v6e-8 QR in-zone FAILED with "no more capacity";
      see PHASE0-REPORT.md.)
- [x] Verify each baseline imports at our pins, CPU interpret mode, tiny shapes
      — GREEN, sbatch job 3645849, all six at jax 0.10.2; exact module paths
      + versions in `PHASE0-REPORT.md` (kept there instead of problems/README).
- [ ] Service-account IAM grant (admin): without it the keeper dies at each
      daily gcloud expiry — the single 24/7-availability blocker.

## Phase 1 — Judge core + test battery (~1.5 days; code anywhere, tests via
##           sbatch on neuronic compute nodes or a TPU host's vCPUs — NEVER the
##           login node)
Layout `tpu/pallas_arena/judge/`: `server.py` (FastAPI FIFO, one worker/chip,
problem = launch flag — rejecting other requests is itself an anti-cheat
property; compiled reference resident per chip), `grader.py` (subprocess-per-
candidate per the idiom above), `timing.py`, `cache.py` (GCS hash→reward),
`problems/<task>.py` (reference, shape set incl. one HOLDOUT shape, tolerance
calibration, adversarial vector generator, golden candidates).

Test battery — the three layers PLUS the sufficiency additions:
- [x] References vs closed forms (interpret mode, tiny shapes); reward math on
      synthetic latencies.
- [x] Cheater battery, every one must fail: cached-output (fails fresh seeds);
      aliased reference call (AST); **obfuscated import**
      (`importlib.import_module("jax."+x)` — caught by child module stubs, not
      AST); **timer-tamperer** (defeated by process isolation); seed-reader
      (impossible by construction — inputs generated on-device from a seed the
      child never receives); Python-level memoizer (defeated by fresh inputs
      per timed iteration). (Plus: split-personality, wrong-backward,
      nondeterministic, RLIMIT hog, sleeper.)
- [x] **Adversarial vector library per task** (tolerance-exploitation defense —
      approximation passes allclose on Gaussian seeds forever): softmax-
      saturating logits / fully-masked rows must yield 0 not NaN (attention);
      label-in-tail, LSE stability at 150k vocab (FLCE);
      group_size=0 and max-skew single-expert (megablox); a→1 long memory,
      reset boundaries, non-divisible T (RG-LRU); page-boundary + single-token
      + max-length (paged attention); small-magnitude rows where var≈eps
      (RMSNorm — this is what catches wrong-epsilon). Per-element error TAILS
      (max + q99), not global allclose.
- [x] **Gradient checks** for fwd+bwd tasks (RMSNorm, FLCE; splash bwd is
      phase-2 of its task): jax.grad vs fp32 reference at its own calibrated
      tolerance + finite-difference spot checks. Contract pinned per task via
      Problem.has_bwd.
- [x] Timing protocol: interleaved R,C median-of-20, **fresh inputs per
      iteration**, block_until_ready, **correctness verified on an output from
      a timed invocation** (kills fast-garbage/slow-correct split kernels).
- [x] CPU AOT pre-gate helper (client side) — judge/aot_gate.py.
Acceptance: full battery green under sbatch; every cheater rejected.
**DONE 2026-08-06: 98/98 green, sbatch job 3645902
(runs/pallas_arena/phase1-tests-3645902.log); first run 3645889 caught the
wrong-eps-inside-bf16-tolerance gap → fixed with the small-magnitude-rows
vector.**

## Phase 2 — Single-judge shakedown — **DONE 2026-08-06** (2 QRs, both deleted)
Infrastructure fully green (5.5/7.3-min spot landings, 50s provision, artifact
round-trip, always-delete 3/3). Three findings → the Phase 3 fixes:
(1) cold per-candidate cost 112s mean vs 4s planned → persistent worker;
(2) ref-vs-ref 1.019–1.053, FAILED ±2% → first-position bias, counterbalance
R,C/C,R; (3) block-256/512 + unjitted honest goldens failed the 1.5×-ref-only
tolerance → recalibrate across legitimate implementations. Reports:
PHASE2-REPORT.md + runs/pallas_arena/phase2-shakedown-results-3645912.json.

## Phase 3 — Fixes + pull-queue, SIMULATED locally (zero TPU, sbatch only)
Architecture (user-settled): work queue lives with the training client (v5p
host); judges are pure stateless pollers. Queue leases items; requeue on lease
timeout (~2× max grade time) / missed heartbeat; double-grades harmless
(idempotent + hash cache). No GCS registry, no client-side routing; keeper's
only job = fleet size 5.
- [x] Persistent worker per chip (`judge/worker.py`): jax/TPU init +
      reference compile + counterbalanced noise floor ONCE at boot;
      candidate serialized via jax.export in the throwaway sandbox child
      (mode="aot_export": AST+stubs+timeout+RLIMIT, JAX_PLATFORMS=cpu, no
      device); worker loads + compile-warms the artifact off the clock and
      times steady-state (warm_chip_s). Bonus: no candidate python ever
      runs in the worker — timer-tamper/split-personality/host-RNG cheats
      become structurally impossible. Needs `flatbuffers` (jax.export
      serialize) in the judge venv.
- [x] Counterbalanced R,C/C,R timing (`timing.counterbalanced_pair`) in all
      three timing loops + the worker.
- [x] Tolerance recalibration: `Problem.calibrated_tolerance` = 1.5x the
      worst spread across `honest_variants()` (rmsnorm: bf16-out, chunked
      reduction, overflow-robust scaled mean; non-finite variants never
      widen the margin); goldens re-armed as regression tests (bf16-out
      honest, UNJITTED_ROBUST, VMEM-budgeted row-padded pallas kernels).
- [x] Queue server (`judge/queue.py`): thread-locked FIFO + leases +
      heartbeat + lazy expiry sweep + idempotent duplicate accounting;
      memoryless by design — ArenaQueueClient resubmits through restarts.
- [x] **Local simulation** under sbatch: queue + 5 mock-timed worker
      processes; chaos green — SIGKILL workers mid-lease → requeue +
      completion on survivors; queue kill/restart → client resubmits all;
      exact lease-expiry/heartbeat/duplicate accounting unit-tested.
Acceptance MET 2026-08-06: **121/121** green under sbatch (job 3646312;
legacy 98 still green inside it). Logs runs/pallas_arena/phase1-tests-*.log.

## Phase 4 — Small-scale real revalidation (1× spot v6e-1, ~1 chip-hour)
- [ ] One judge polling the (login-node or compute-node) queue over the wire;
      RMSNorm battery with the three fixes: warm cost ≈4s, ref-vs-ref within
      ±2%, recalibrated goldens all pass, N≥5 bitwise determinism, peak-HBM.
      Always-delete unchanged.
Acceptance: every Phase-2 red is green on silicon.

## Phase 5 — GRADING-SERVER FLEET DEMONSTRATION (user-scoped 08-06: NO RL)
The deliverable is proof the grading server works at fleet scale — not a
training run. No v5p, no vLLM, no policy: a fixed corpus of candidate kernels
is submitted through the queue and graded by 5 judges.
- [ ] 5× spot v6e-1 (us-east5-b) all polling ONE queue (queue runs wherever —
      login-adjacent host or a compute node).
- [ ] Corpus: honest variants (block sizes, unjitted, bf16-out) + the full
      cheater battery + a fast/slow spread, submitted as one batch.
- [ ] Acceptance: every cheater rejected at the correct gate; every honest
      candidate scored reproducibly (regrade within ±3%); ref-vs-ref 1.00±2%;
      rewards byte-identical for repeats via the shared cache.
- [ ] Chaos: kill a judge mid-lease → its work requeues and completes on the
      survivors, no lost or double-counted results.
- [ ] Measured fleet throughput (candidates/min) + zero resources left behind.

## (deferred, NOT now) The real RL run: v5p + fleet
- [ ] `examples/pallas_arena/` env in discover (gpu_mode pattern): prompt =
      spec + reference signature + declared shapes; evaluator = AOT pre-gate →
      hash-cache → enqueue + await result. Runner `_ENVS` entry with a DEFAULT
      problem_type (the circle lesson, f2b522d1).
- [ ] Training side: v5p-8 is ONE host (4 chips) — needs split-host serving
      (train TP/FSDP on 2 chips + vLLM TP=2 on 2 chips) at reduced ctx (kernel
      prompts ≪ erdos; arena scales quadratically down). Agent sizes this; if
      27B genuinely doesn't fit both roles, fall back to v5p-16 (w0 train +
      w1 vLLM) and say so.
- [ ] Fleet keeper (reconcile to 5 judges; provision = the phase-2 script);
      queue on the v5p w0; short RL run (RMSNorm, ~5 steps) end-to-end.
Acceptance: nonzero valid rate, finite distinct rewards, cache hits on
repeats, queue depth ~0 while sampling streams, no idle spot left behind.

## Phase 6 — The science (per-run, standard sweep machinery)
Run order: RMSNorm (harness-calibration run) → splash → ragged paged attention
→ FLCE → megablox → RG-LRU. Each: 15-step variant via a driver spec line +
forever supervisor, unchanged.
- [ ] Per champion: **in-situ validation** — drop the kernel into a real fb
      step (or vLLM decode batch) and confirm end-to-end time improves; report
      peak-HBM delta; holdout-shape check for declared-set overfit; re-time
      independently before claiming (the erdos verification discipline).
- [ ] Writeups in tpu/results/ per task.

## Sequencing vs the running sweep
Phases 0–1 touch nothing the sweep uses (sbatch + new zone). Phase 2–3 spend is
in us-east5-b only. Phase 4 borrows a v5p slice briefly; Phase 5 replaces a
sweep slot or gets its own slice — decision then.

## Risks / open items
- Baseline importability at our jax/libtpu pins (Phase 0 kills early).
- Pin ONE jaxlib for the whole arena run (Mosaic compile determinism).
- Spot v6e capacity in us-east5-b unknown — measured in Phase 2 before
  committing to 5 slices.
- Anti-cheat is never "done": every new hack found in RL rollouts becomes a
  cheater-battery regression test.
- Kernels don't transfer between generations: v6e wins are Trillium claims; a
  champion we want for OUR v5p steps gets re-timed on a v5p chip during in-situ
  validation anyway (that test runs on the training slice).
