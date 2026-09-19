# Full 17-case circuit comparison — launched 2026-09-19

Status: running. `summary.json` is updated after completed cases. No complete
17-case result is claimed yet. The launch runs on della-vis2, A40 GPU 0, with
four concurrent CPU case workers. GPU 1 has substantial unrelated allocations.
GPU-heavy pipelines queue on GPU 0; the 85 final method/case evaluations are
not 85 simultaneous GPU jobs. Other users' processes are untouched.

## Frozen selection

The selection was refreshed from durable GRPO and PWC pools before launch.
One program per model is selected using the existing four-case discovery reward;
full-suite feedback does not change the selected source during this evaluation.

| Method | Source | Discovery mean cost (four cases only) |
|---|---|---:|
| Gemma | Job 1060, GRPO, state c9781730-fd95-48d4-ab2a-f5f7a4b6ffcc | 0.9269662052 |
| Qwen | Job 1067, PWC, state 915626b5-5af9-430d-8e24-0aa90be4fab6 | 0.9501483142 |
| Muse | Job 1066, PWC, state bc789833-1bff-4c33-8a5a-d5919bf38d64 | 0.9619377702 |
| AbuPlace | Pinned local abuplace/placer.py, XplacePlacer | New full-suite run |
| ArchGen | Pinned local submissions/final_placer.py, OptimalPlacer | New full-suite run |

`selection.json` contains pool identities and source hashes. `provenance.json`
contains repository commits/status, harness/scorer hashes, runtime versions,
and hashes of all 34 native input files. The three source programs and original
states are retained alongside this document.

## Protocol

- Cases: ibm01–ibm04 and ibm06–ibm18 (17 total; no ibm05).
- Per-case objective: wirelength + 0.5*density + 0.5*congestion.
- Qualifying score: arithmetic mean of all 17 costs, lower is better.
- Every case must pass the independent scorer's bounds, fixed-object,
  finite-output and zero-hard-overlap checks. Soft-cluster overlaps follow the
  official task semantics. An incomplete or invalid suite has no qualifying mean.
- Candidate outputs are saved before independent scoring; no post-hoc repair of
  human/model outputs is performed. Xplace initialization explicitly includes
  the same deterministic legalization used to create the training starts.
- All final outputs use our pinned challenge/TILOS scorer. This is a local
  reproduction, not judge confirmation; scorer hashes are recorded for audit.
- Original four Xplace starts remain byte-identical to training. The additional
  13 starts are generated with the same pinned route-aware Xplace pipeline,
  independently validated, and supplied identically to all three models.
- Models: JAX CPU environment, injected fast_proxy_v1 helper, seed 42, 170 seconds
  supplied to the runner including initialization; 180-second external cap,
  four-CPU affinity. Aggregate descendant RSS is monitored every 250 ms and jobs
  exceeding 8 GiB are terminated. This local watchdog is NOT the TPU hosts'
  kernel-enforced cgroup memory limit. Trusted scoring has a 120-second cap to
  accommodate local full-suite auditing, separate from candidate execution.
- Humans: their original benchmark inputs and full published pipelines;
  A40 GPU, 16-CPU affinity, 3450-second external candidate cap. Existing ArchGen
  wrapper settings target 3150 seconds. Xplace is included in those pipelines.
- Model reports include Xplace precomputation runtime separately and added to
  end-to-end runtime. The model/human compute budgets differ; this comparison
  measures these exact saved solutions, not equal-compute model superiority.
- Hardware is shared. Time-bounded searches can vary with host contention.
  Local JAX is 0.10.1; all runtime versions are recorded. Do not interpret small
  changes from recorded TPU-host CPU scores as proof of invalidity.

## Algorithm analysis from the frozen code

### AbuPlace

The human system uses a multi-stage pipeline: Xplace global placement, refinement,
polishing, escape moves and an operator stack. Its default best-of-N wrapper runs
multiple route-demand settings and selects a legal low-cost result. It combines
C incremental proxy evaluation, C surrogate refinement/legalization and GPU
proposal evaluation. Periodic full rescoring and regression guards limit errors
from incremental drift. These implementation features are visible in the pinned
README, abuplace/placer.py and extension modules; their effectiveness will be
assessed from the new per-case results and logs, including fallback paths.

### ArchGen

The pinned Scope-Selector implementation creates alternative legal layouts from
congestion-aware spreading, an internal analytical/evolutionary lane, and optional
route-aware Xplace scouts. It selects a promising lane using proxy components,
then allocates remaining time to GPU coordinate-descent proposals with legality
and proxy acceptance gates. The upstream local README distinguishes its public
package score (0.961653 across 17) from its reported verified leaderboard score
(0.9507). We are executing the pinned public package, not assuming those scores
will reproduce exactly on this A40/runtime/budget.

### Qwen

Uses a single Xplace start. Each iteration selects one movable block, generates
16 Gaussian proposals, clips them to canvas bounds, filters hard overlaps, and
scores legal moves with the provided helper. It selects the best proposal and
uses simulated-annealing acceptance, retaining the best encountered layout.
Hard and soft blocks have different fixed move scales. Temperature decays per
iteration rather than elapsed time. It periodically rebuilds the evaluator.
It does not import JAX; its algorithm is NumPy plus the C helper.

### Muse

Uses a single Xplace start. It batches proposals across both hard and soft blocks,
with Gaussian moves plus occasional wider jumps. It picks the lowest-cost move
from the batch, applies annealing, and adapts move radii after acceptance/rejection.
The first 70% of the budget explores; the remainder uses smaller moves and nearly
zero temperature for polishing. It periodically rebuilds/rescores, preserves the
best layout, and has a final legality guard that can return the initial layout.
It also uses NumPy plus the C helper, not JAX computations.

### Gemma

Uses a single Xplace start. It builds a weighted connectivity matrix and computes
block centroids with JAX on CPU. Proposals combine the centroid, eight local
radial directions, and a larger random jump. Legal proposals are scored by the
C helper, with time-dependent annealing/radius schedules and best-layout retention.
JAX supplies centroid proposals, not the authoritative objective or gradients.
The inner loop traverses movable blocks without an internal deadline check;
large-case runtime overrun is a specific possibility tested by the unchanged
180-second external cap. We will not patch the frozen solution to hide failures.

## Questions the full suite will answer

1. Does each saved program remain legal on the 13 cases outside training feedback?
2. Which method has the lowest complete 17-case mean?
3. Are gains concentrated on training cases, certain sizes, or congestion/density?
4. How much does each model improve its common Xplace start?
5. Do human pipelines execute their intended paths or fall back after failures?

No per-case mix-and-match of different model programs is reported as one algorithm.
No four-case mean is compared directly with a 17-case leaderboard mean.

## Operations

Supervisor PID and command: `launch.json` / `supervisor.json`.
Results: `<method>-<case>/report.json`, `metrics.json`, `positions.npy`, and logs.
Inputs: `inputs/<case>.npz` and provenance JSON.
Progress: `summary.json`; completion/errors: `completion.json`.

Run entrypoint: `python -m tpu.science.full17_eval --out <this-directory>`.
The supervisor takes an exclusive lock and skips completed reports on restart.
Interrupted partial case directories are intentionally not overwritten: inspect
and preserve them before assigning a new attempt. Active training is unchanged.
