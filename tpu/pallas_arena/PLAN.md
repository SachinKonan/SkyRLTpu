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
- [ ] Confirm QueuedResourcePerProjectPerZone in us-east5-b ≥ 5 free slots.
- [ ] Verify each baseline imports at our pins, CPU interpret mode, tiny shapes:
      `jax.experimental.pallas.ops.tpu.splash_attention`; vLLM-TPU
      `ragged_paged_attention`; MaxText megablox gmm; recurrentgemma RG-LRU
      Pallas scan; our in-tree FLCE (198f41fa/2e85086f). Record module paths +
      versions in `problems/README`. This is the early-kill gate.
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
- [ ] References vs closed forms (interpret mode, tiny shapes); reward math on
      synthetic latencies.
- [ ] Cheater battery, every one must fail: cached-output (fails fresh seeds);
      aliased reference call (AST); **obfuscated import**
      (`importlib.import_module("jax."+x)` — caught by child module stubs, not
      AST); **timer-tamperer** (defeated by process isolation); seed-reader
      (impossible by construction — inputs generated on-device from a seed the
      child never receives); Python-level memoizer (defeated by fresh inputs
      per timed iteration).
- [ ] **Adversarial vector library per task** (tolerance-exploitation defense —
      approximation passes allclose on Gaussian seeds forever): softmax-
      saturating logits / fully-masked rows must yield 0 not NaN (attention);
      label-in-tail, ignore-index, LSE stability at 150k vocab (FLCE);
      group_size=0 and max-skew single-expert (megablox); a→1 long memory,
      reset boundaries, non-divisible T (RG-LRU); page-boundary + single-token
      + max-length (paged attention). Check per-element error TAILS (max +
      quantiles), not just global allclose.
- [ ] **Gradient checks** for fwd+bwd tasks (RMSNorm, FLCE; splash bwd is
      phase-2 of its task): jax.grad vs fp32 reference at its own calibrated
      tolerance + finite-difference spot checks. Pin the fwd-only vs fwd+bwd
      contract per task so candidate and baseline do the same work.
- [ ] Timing protocol: interleaved R,C median-of-20, **fresh inputs per
      iteration**, block_until_ready, **correctness verified on an output from
      a timed invocation** (kills fast-garbage/slow-correct split kernels).
- [ ] CPU AOT pre-gate helper (client side).
Acceptance: full battery green under sbatch; every cheater rejected.

## Phase 2 — Single-judge shakedown on one v6e-1 @ us-east5-b (first TPU spend, ~1 day)
- [ ] QR + provision script (baked venv, judge tmux) — becomes the keeper's
      provision step verbatim. Measures real spot landing latency in -b
      (decision input before committing to a 5-slice fleet).
- [ ] RMSNorm end-to-end; on-chip goldens (naive-correct passes, wrapped
      reference rejected, subtly-wrong fails); **determinism N≥5 bitwise**.
- [ ] Timing invariants on silicon: ref-vs-ref = 1.00±2%, regrade ±3%,
      per-chip noise floor measured at boot and logged; **reward > 1.0 only
      when the win exceeds the noise floor; ties score exactly 1.0**.
- [ ] **Peak HBM/VMEM measured and reported per grade** (memory honesty — a
      kernel that wins latency by tripling workspace is useless for our arena
      pressure).
- [ ] Measure true compile + chip costs → recalibrate the throughput table.
Acceptance: `POST /grade` on ~20 hand-written RMSNorm variants returns sane,
stable, noise-floor-consistent rewards.

## Phase 3 — Fleet keeper (~half a day, alongside Phase 4)
- [ ] `keeper.sh` reconcile loop (generalize forever_sweep): list judge-{1..5}
      QRs in us-east5-b → delete SUSPENDED/FAILED → create to target 5 →
      provision newly ACTIVE → health-check by grading a golden kernel
      end-to-end → publish name→IP registry to
      gs://sk7524-pallas-arena-us-east5/judge-registry.json.
- [ ] Client: judge-URL list from the registry, round-robin, failover-on-timeout
      (the vLLM round-robin pattern). Judges stateless; shared GCS reward cache
      keeps repeat rewards byte-identical across fleet churn.
- [ ] Chaos test: kill one judge mid-batch → batch completes on survivors,
      keeper replaces within one cycle + provision (~5 min).

## Phase 4 — RL env integration (~half a day, borrowed v5p slice between sweep variants)
- [ ] `examples/pallas_arena/` in discover (copy the gpu_mode pattern): prompt =
      problem spec + reference signature + declared shapes; evaluator = AOT
      pre-gate → hash-cache lookup → judge POST with failover.
- [ ] Runner `_ENVS` entry, task as `problem_type` **WITH a default** (the
      circle/acineq lesson, f2b522d1).
- [ ] 2-step RL smoke vs RMSNorm at 8×8 on a v5p slice between sweep variants.
Acceptance: nonzero valid rate, finite distinct rewards, cache hits on repeats,
judge queue depth 0 when sampling ends.

## Phase 5 — The science (per-run, standard sweep machinery)
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
