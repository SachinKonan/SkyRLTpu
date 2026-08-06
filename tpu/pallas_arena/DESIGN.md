# Pallas kernel arena — design spec (2026-08-06)

PROVISIONED SO FAR: `gs://sk7524-pallas-arena-us-east5` (regional US-EAST5 —
buckets are regional, never zonal, so it serves the judge fleet at full
intra-region locality; holds judge registry + hash→reward cache + kernel
corpus, lifecycle-separated from the training bucket). Judge fleet zone:
**us-east5-b** (v6e-1 through v6e-256 confirmed available there; fresh QR
quota pool vs our crowded -a). Everything else is spec-only.

A TriMul-style kernel-generation arena for Pallas/TPU: self-contained tasks, fast
TPU-silicon grading, significant-if-beaten baselines. Consolidated from the
design fork (session db8bd534, agent i-was-wondering). Plugs into the existing
ttt-discover harness (PUCT pool, drivers, supervisors) unchanged.

## The slate — single-problem-at-a-time server

Final slate (TriMul dropped 2026-08-06 — it was the one weak-baseline entry):
every headline target is a hand-tuned Google/community production kernel, which
is what makes wins significant rather than "beat naive XLA":

| # | Task | Baseline to beat | Claim |
|---|---|---|---|
| 0 | RMSNorm fwd+bwd (warm-up) | XLA fusion, scored as % of speed-of-light HBM bandwidth | judge shakedown only |
| 1 | **Splash attention @ [4,18432] causal + one non-block-divisible length** | Google's production Pallas kernel | flagship — external claim + directly cuts our own fb arena pressure |
| 2 | Ragged paged attention (decode, GQA 8 KV heads) | vLLM-TPU's Pallas kernel | a win speeds our own sampling fleet |
| 3 | Megablox grouped matmul (MoE), uniform + Zipf-skewed `group_sizes` | MaxText's tuned Pallas gmm | strong external baseline |
| 4 | FLCE (fused linear cross-entropy) @ 73728 tokens × ~150k vocab | **our own** custom_vjp tiled kernel (198f41fa/2e85086f) | hard honest baseline we understand; a win cuts every train step |
| 5 | RG-LRU linear scan | DeepMind's recurrentgemma Pallas kernel | production-tuned; `associative_scan` legal for candidates |

Build order: 0 → 1 → 2 → 4/3 (FLCE/megablox) → 5.

**Resources by phase:** 0–1 (harness + cheater battery + AOT pre-gate) = zero
TPUs, CPU pytest anywhere, ~1.5 days; 2 (end-to-end shakedown, RMSNorm, timing
invariants) = one spot v6e-1 — every task's eval fits <2–8GB HBM (FLCE never
materializes logits by construction); 3 = the judge fleet below; RL training =
the existing v5p-32 stack, one `_ENVS` entry per task, phase-4 smoke on a
borrowed slice between sweep variants. Phase-0 gate: verify all baselines
import at our JAX pins before building anything.
(TriMul @ AF2 shapes, GPU MODE spec parity vs XLA lowering, stays on the bench
as an optional cross-silicon leaderboard demo.)

Notable per-task specs:
- **Task 1 shapes**: our real fb shard shapes [4, 18432, heads, 128] plus one
  deliberately non-block-divisible length (the case splash itself rejects — bit us
  at TP=4).
- **Task 3**: (tokens=32k, experts=8 and 64, k=4096, n=14336); `group_sizes`
  freshly sampled per grading from uniform AND Zipf-skewed distributions so
  imbalance-tuned kernels can't fake wins. `jax.lax.ragged_dot` is the floor.
- **Task 5 scope**: just the gated diagonal linear scan
  `h_t = a_t*h_{t-1} + sqrt(1-a_t^2)*(i_t*x_t)`, gates precomputed as inputs —
  kernel-vs-kernel against DeepMind's scan. Shapes: RecurrentGemma-2B width,
  [8,4096,2560] and [1,32768,2560]; tolerance calibrated vs the reference's own
  drift at long T (not fixed atol). `lax.associative_scan` is a legal candidate
  strategy; only importing recurrentgemma is AST-banned.
- **SSM alternates**: Mamba-1/SSD have NO tuned TPU kernels (weak baselines,
  "first-good-kernel" claims only) — RG-LRU is the proven-baseline SSM task.

## Evaluation — confirmed feasible

Per candidate: Mosaic compile ~10–40s on host CPU (parallelizes across vCPUs),
then ~4s exclusive chip time (correctness at 3 hidden seeds + interleaved
ref/candidate median-of-20 timing over the shape set). The funnel does the heavy
lifting: hash-dedup (30–50% cut, already in our rollout cache) + CPU AOT pre-gate
(kills the 40–70% of early-RL kernels that don't compile, at zero chip cost) →
~500 of 1k rollouts need chip time.

**Throughput (v6e-8):** chip ceiling 120/min (8 lanes × 4s), compile ceiling
~150/min (64 parallel Mosaic ≈ 2.5/s) → effective ~110–120 candidates/min.
Realistic grading wall ≈ 5 min per 1k-rollout step (worst case all-unique ≈ 8–9
min). Steady-state is better still: rollouts stream in over 30–90 min of
sampling (demand ~11–33/min vs ~110/min supply), so the judge grades as they
arrive and the end-of-step wall is ~zero — the 5-min figure is the cold-burst
case. RMSNorm warm-up grades ~3× faster. Grading is never the loop's critical
path; sampling and fb remain ~10× slower.

**Judge placement — SETTLED (user, 08-06): the 5× spot v6e-1 warm fleet below.**
Build order when green-lit: CPU harness tests + RMSNorm shakedown → fleet keeper
→ splash. The v5p carve-out analysis is retained for the record (it remains the
right call if self-payoff on v5p ever outranks fleet resilience):

**(superseded) v5p one-chip carve-out:**
Deciding fact: kernels don't transfer between TPU generations, and the arena's
strongest argument is self-payoff — task 1 is OUR fb shape on OUR chips, task 2
OUR sampling kernel. That's only true graded on v5p. Zero-QR path: on one vLLM
host, TP=2 (chips 0,1) + TP=1 (chip 2; 27B fits one v5p chip) + judge (chip 3)
≈ 8% sampling tax; no new spot resource to hunt. TP=3 is invalid (8 KV heads
must divide by TP). Launcher tweak needed: two vLLM processes/ports on the split
host + extra URL in client round-robin. One-chip throughput ~15 graded/min vs
streaming demand ~6–11/min → keeps up in steady state; cold burst queues
~35 min. Upgrade trigger: nonzero judge queue when sampling finishes → take a
second chip (17% tax) or spin a v5p-8 QR when the pool has room.
**v6e-8 becomes the port-and-claim-both option** (fresh per-zone QR quota,
~110/min): use it if/when a winning v5p kernel should also claim Trillium.
Everything else (funnel, gates, timing protocol, plumbing) is silicon-agnostic.

**PHASE-3 ARCHITECTURE (user-settled 08-06, supersedes registry/push): pull
queue with leases.** The work queue lives on the v5p training host (where
rollouts are born): a thread-locked queue the client appends candidates to as
sampling produces them. Each v6e-1 judge polls GET /work against that one
stable address — judges are pure stateless workers; no GCS registry, no
client-side routing or failover. The queue LEASES items (in-flight until the
result returns); requeue on lease TIMEOUT (~2x max grading time) or missed
heartbeat — a dead server can't release, so expiry is the requeue trigger.
Double-grades after false-death are harmless: grading is idempotent and the
hash→reward cache makes repeat results byte-identical. A preempted judge is a
non-event by construction; the keeper's only job is fleet size = 5.
Post-shakedown fixes folded in: persistent worker per chip (jax/TPU init +
reference compile once at boot; sandbox only the candidate AOT compile);
counterbalanced R,C/C,R timing order (first-position bias measured at 2-5%);
tolerance recalibrated across multiple legitimate implementations (block-256/
512 + unjitted honest goldens failed the 1.5x-ref-only margin). Reference
TIMING stays interleaved per candidate (drift cancellation, ~2s) even though
reference COMPILE is once; can drop to every-Nth if ever needed.

**(superseded) warm v6e-1 fleet with GCS registry:**
5× spot v6e-1 (not one slice): ~75 graded/min aggregate, preemption costs 20%
not 100%; five QRs fit the v6e zone's own fresh quota pool. Keeper = ~80-line
reconciliation loop generalized from forever_sweep: classify judge-{1..5} QRs
every 2–5 min, delete SUSPENDED, create until 5 healthy+pending; newly ACTIVE →
provision (stateless — references re-derive from seeds at boot, ~3–5 min) →
health-check by grading a golden kernel end-to-end → publish live-judge registry
(name→IP) to a GCS object. Client evaluator reads the registry list,
round-robins with failover-on-timeout (our vLLM pattern) — RL never notices a
preemption. hash→reward cache lives in GCS (fresh judges give byte-identical
repeat rewards); per-chip noise floors re-measured at each boot. Hard
dependency: the keeper can only submit QRs while gcloud creds are alive — the
pending service-account IAM grant is the real 24/7-availability bottleneck.

## Reward frame (anti-cheat is the hard 20%)

Gates first — compiles in 60s; genuinely a `pallas_call` (AST ban on wrapping the
reference or `jax.nn` attention); allclose to fp32 reference at 1.5× the
reference's OWN bf16 error, fresh hidden seeds per grading (inputs generated
on-device from a seed the program can't read); edge shapes (non-divisible
lengths, empty expert groups, mask boundary rows, single-token decode);
deterministic across 2 runs; no NaN/inf.

Score = geomean latency ratio vs baseline over the declared shape set, timed
interleaved R,C,R,C (cancels clock/thermal drift). Memory-bound tasks
additionally logged as fraction of speed-of-light bandwidth (QuACK's yardstick —
prevents weak-baseline pseudo-wins; v6e ≈ 1.6 TB/s, v5p ≈ 2.8 TB/s). Persistent
hash→reward cache gives repeat kernels instant, CONSISTENT rewards (de-noises
advantage estimation).

## Test suite — three layers

1. **CPU-only pytest**: references vs closed forms on tiny shapes; reward math
   (geomean, interleave estimator on synthetic latencies); deliberate cheaters
   that must fail — cached-output kernel (fails fresh seeds), aliased reference
   call (AST-rejected), seed-reader (impossible by construction).
2. **On-chip goldens per task** (~seconds): naive-but-correct Pallas kernel must
   pass; wrapped reference must be AST-rejected; subtly-wrong (off-by-one mask
   row / dropped empty group / wrong LN epsilon) must fail numerics; edge shapes.
3. **Timing invariants**: ref-vs-ref grades 1.00±2%; same-kernel regrade ±3%;
   per-chip noise floor measured and logged at suite start.

### Adversarial addendum (08-06) — necessary-but-not-sufficient closed

The three layers alone are gameable; six additions make every known hack class
have a test that must fail for the attack to succeed:

1. **Tolerance-exploitation (biggest gap):** silent approximators (dropped KV
   blocks, truncated softmax tails, skipped small expert groups) pass allclose
   on Gaussian seeds forever. Add per-task adversarial structured inputs
   (softmax-saturating logits, outlier rows, near-overflow bf16, fully-masked
   rows that must yield 0 not NaN, label-in-tail for FLCE, max-skew groups for
   megablox, a→1 for RG-LRU) and check the per-element error DISTRIBUTION
   (max + tail quantiles), not a global allclose.
2. **Timing-cache attacks:** regenerate/permute inputs per timed iteration
   on-device; block_until_ready; verify correctness ON AN OUTPUT PRODUCED BY A
   TIMED INVOCATION (kills fast-garbage/slow-correct split-personality kernels
   and lru_cache memoization).
3. **Gradient contract:** tasks with bwd in the baseline (FLCE, RMSNorm) need
   jax.grad checked vs fp32 reference at its own calibrated tolerance + finite
   -difference spot checks; pin fwd-only vs fwd+bwd per task.
4. **Isolation-as-gate (SETTLED 08-06: containerless, our existing grading
   idiom):** fork a plain subprocess per candidate with timeout — the exact
   SandboxRewardEvaluator.execute_code pattern that has graded thousands of
   erdos/circle/AHC programs. Process isolation alone delivers reward
   integrity: a child cannot patch the judge's timer, references, or hash
   cache in the parent. Compile+trace in the child too; one-line RLIMIT_AS on
   the child (64 parallel Mosaic compiles of pathological unrolls can OOM a
   host); reference modules stubbed in the child's sys.modules (stronger than
   AST parsing, zero infra). Threat model is our own policy reward-hacking,
   not a hostile adversary — no Docker, no network isolation. Cheater battery
   still gains obfuscated-import and timer-tampering cheaters (must fail).
5. **Memory honesty + in-situ validation:** report peak HBM/VMEM alongside
   speedup (workspace-tripling latency wins are useless — arena pressure IS the
   point); any champion's final test is a real fb step / vLLM decode batch with
   end-to-end step time confirmed improved.
6. **Statistical honesty:** reward > 1.0 only when speedup exceeds the logged
   per-chip noise floor (ties = exactly 1.0); one HOLDOUT shape logged-unscored
   to detect declared-set overfitting; determinism = N≥5 bitwise, not 2.

Ops note: the "CPU-only" battery is real compute — per cluster rules it runs
via sbatch on neuronic compute nodes (run_frontiercs_smokes.sh pattern) or on a
TPU host's ~200 idle vCPUs, never the login node. Cost: +~half a day for the
adversarial vector library; phase estimates otherwise hold.

## Plumbing (nearly free)

- Copy `examples/gpu_mode` → `examples/pallas_attention` (env.py + prompt.py +
  evaluator that HTTP-POSTs the judge instead of exec-ing locally).
- One `_ENVS` entry in tpu/run_ttd_smoke_gptoss20b.py with the shape-set as
  `problem_type` — WITH a default (see f2b522d1's lesson).
- Judge = ~150-line FastAPI FIFO on the judge host: one worker per chip, problem
  type as a LAUNCH FLAG (rejecting other requests is itself an anti-cheat
  property), compiled reference resident per chip, launched from bring-up as a
  tmux session (same pattern as the lora-prune daemon).
- PUCT pool semantics unchanged — kernels refine tile sizes/pipelining/DMA
  double-buffering the way erdos refined step-function resolution.
