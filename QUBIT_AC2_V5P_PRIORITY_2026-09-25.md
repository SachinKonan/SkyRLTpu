# Qubit and AC2 v5p priority, 2026-09-25

The user requested qubit and AC2 before other experiments, and the legacy v5p-32 Muse shape under the Ray v2 executor.

## Active submissions

| Work | Job | Priority | Intended continuation |
|---|---:|---:|---|
| Qubit Muse on Gemma20 pool | 1751 | 710 | Ten new steps; Muse16 adapter, Gemma20 search pool, fresh optimizer, reset PUCT |
| AC2 Gemma | 1752 | 700 | Continue from its own durable state to step 20; latest previously verified checkpoint 13 |
| AC2 Muse | 1753 | 690 | Continue from its own durable state to step 20; latest previously verified checkpoint 16 |

These use `tpuswarm-v5p32-east5a-erdos`. All runtime bucket/cache/checkpoint paths point to East5. Qubit Qwen 1736 remains on its existing v4-64 worker, with priority raised to 700.

Placement and process readiness must be read from live evidence; these entries are submission identities, not proof of completed updates.

## Qubit Muse correction

1742 had no successful optimizer updates: its Muse adapter was expanded to eight logical KV heads on v4, whereas TP1 MaxText instantiated Muse's native two KV heads. The first backward pass tried to reshape `(32,1024)` into `(32,2,128)`.

All eight K/V LoRA B tensors had four exactly identical replicas per native head. `tpu/science/weights_only_checkpoint.py::collapse_repeated_kv` verifies exact replica equality, retains one replica, verifies exact expansion back to the original tensor, and writes a separate weights-only archive. It rejects divergent replicas. It omits saved mesh placement metadata so the new trainer uses its current template. Other tensors and the adapter configuration are retained; the original archive is unchanged.

The converted archive is staged at:
`gs://sk7524-tinker-tpu-us-east5/ray-training/qubit-gen1-muse16-on-gemma20-pwc05-legacy-v5p-20260925/checkpoints/model_81f4355b/000016.tar.gz`.

The replacement uses a new run ID and the original 711-program donor seed, not the failed attempt's generated batch. PUCT counts/values and timeline restart at zero; parent links and evaluated program feedback are preserved. Three local conversion/optimizer-stripping tests passed. This establishes conversion correctness; actual TPU backward success remains a separate runtime verification.

### Legacy Muse execution settings

- Ray v2 executor, one TP1/FSDP4 trainer host.
- Native two KV heads; LoRA rank 32, alpha 32, carried Muse16 parameters.
- Uniform training row 18,432; token budget 73,728 (four rows per call); full rematerialization, FLCE tile 1024, vocabulary tiling 32, host parameter offload, free base state.
- Three serving hosts, each running two TP2 engines: six engines total.
- 64 sequences per engine, memory utilization 0.90, chunk size 8192, engine model length 22,528.
- JAX wrapper backend, Muse vLLM implementation, batched RPA and ragged convolution enabled, precompile skipped.
- Client context 18,432, phase-1 cap 13,824.
- 16 groups of 32 samples, adaptive PWC rho 0.5, importance-sampling loss, learning rate 4e-5, ten new steps.
- Full routing suite with the existing parallel-v2 bounded grader; eight candidate slots per host.

The detailed before/after profile is `.science/qubit-muse-legacy-v5p-20260925/profile-diff.json`. AC2 keeps its already-tested continuation settings and optimizer state.

## Allocation and storage repair

Four healthy v5p workers had idle TPUs after failed attempts but remained reserved by recovering lower-priority jobs. Seven unsuccessful waiting/recovering jobs were cancelled and re-created as held pending jobs, preserving original task/environment/resume configuration. After provider loss reduced usable capacity to two workers, lower-priority job 1724 was also deferred. Its step-22 checkpoint was verified in East5 before cancellation.

| Old job | Held replacement |
|---:|---:|
| 1719 | 1744 |
| 1720 | 1745 |
| 1721 | 1746 |
| 1722 | 1747 |
| 1723 | 1748 |
| 1725 | 1749 |
| 1726 | 1750 |
| 1724 | 1754 |

Held replacements are intentionally `INACTIVE` in native scheduler state. They require admission after qubit/AC2 priority has been satisfied. Their exact configurations and receipts are in `.science/qubit-muse-legacy-v5p-20260925/deferred/`; do not create duplicate submissions.

Audited all four hosts on workers 1758, 1759, 1760, 1763: no TPU owners or active workload/grading units before cleanup. Removed only unused Muse base-model download caches after verifying durable East5 sources and checking process/file references. No experiment checkpoints or logs were deleted. Head disk availability rose to 64–73 GiB; inference-host availability is 66–77 GiB.

## Monitors and evidence

On della9:
- `qubit-muse-legacy-v5p-dispatch-20260925.service`: executes only this job's exact stalled native request, after identity, request-state, remote active-job, and disk checks.
- `qubit-muse-legacy-v5p-progress-20260925.service`: records host readiness, completions, errors, and local checkpoints every two minutes.
- Existing `ac2-gemma-v5p20-dispatch-20260925`, `ac2-muse-v5p20-dispatch-20260925`, and `ac2-v5p20-progress-20260925` continue tracking AC2.

Live and historical evidence:
- `.science/qubit-muse-legacy-v5p-20260925/three-live.json`
- `.science/qubit-muse-legacy-v5p-20260925/latest-progress.json`
- `.science/qubit-muse-legacy-v5p-20260925/progress-history.jsonl`
- `.science/ac2-v5p20-monitor-20260925/latest.json`
- Conversion, bucket-region, bundle, state-copy, worker-audit, and cleanup proofs in the same Muse directory.

A controller reporting RUNNING is insufficient: check trainer load, ready engine counts, completed generations, and subsequent durable optimizer checkpoints.

## Ray port conflict follow-up

The first reallocated attempts exposed a host-specific SkyPilot Ray worker range of 20000–29999. Ray v2 defaults overlapped that range and bootstrap correctly refused startup. Rebuilt the three immutable bundles with worker ports 30000–30999, fixed service ports 31679–31807, and systemd coordinator 31900. Validated against both observed SkyPilot ranges (11002–19999 and 20000–29999) and Linux ephemeral ports 32768–60999.

Cancelled 1743/1739/1741 and submitted 1751/1752/1753 respectively, keeping the SAME run IDs, region-local checkpoints, optimizer recovery rules, and search state. Updated the existing monitors and queue receipts to the new job IDs. This is an execution-port change, not an experiment restart.

An isolated TPU backward/optimizer probe is prepared at `.science/qubit-muse-legacy-v5p-20260925/isolated_backward_probe.py`. It creates a separate adapter slot from the converted checkpoint and unloads it after testing. Its presence is not proof that it has run; require a successful runtime receipt.

## Recovery follow-up at 08:06 UTC

Provider nodes for workers 1758, 1759, and 1760 were deleted during startup. Cloud audit records show internal TPU deletion events; they do not establish who or what initiated the loss. The surviving workers were 1763 and 1765. AC2 Gemma 1752 is restoring caches on 1763; AC2 Muse 1753 awaits capacity.

Worker 1763 rank 2 had a stale apt metadata update holding the package-list lock for over two hours. Verified it was not a dpkg installation, terminated that exact process and descendants, and verified lock release.

Legacy Qwen 1724 cancellation left trainer/client and vLLM processes alive on worker 1765. Verified their identities against the cancelled job, stopped those process trees, and removed only unused Qwen base-model download caches after verifying East5 source objects. The step-22 experiment checkpoint and logs remain intact. Free disk is now 69.4 GiB on the trainer and 56.3–56.5 GiB on serving hosts. All four TPUs were verified idle before clearing the failed qubit attempt's stale allocation claim; native recovery performs the next allocation.

Receipts: `worker1765/host*-orphan-clean.json`, `worker1765/host*-clean.json`, `worker1765-claim-repair.json`, `worker1763/stale-apt-stopped.json`, `deferred/1724-durable-proof.json` beneath `.science/qubit-muse-legacy-v5p-20260925/`.

At 08:05 UTC, qubit 1751 was reallocated to 1765 and passed the disk preflight. At the 08:06 UTC check, its host task was running setup; no inference or trainer health endpoint was ready yet. Gemma 1752 had finished trainer environment preparation and was finishing serving cache/environment preparation.

`qubit-muse-isolated-probe-20260925.service` on della9 waits for trainer HTTP readiness, launches the disposable backward/optimizer check at most once, and records output to `isolated-probe-result.json`. It uses an explicit gcloud PATH. Monitor startup initially lacked that PATH and was corrected before any remote probe launched.

## Actual TPU validation at 08:18 UTC

Qubit Muse 1751 has a healthy trainer API and all six TP2 inference engines healthy. The full 72-case routing reference evaluation passed on all four hosts with identical reward 0.5193640461472073.

The isolated adapter test PASSED: restored converted Muse16 weights into `model_46e18ffc`, executed four rows using the full uniform 18,432 training shape, completed backward in 33.824 seconds, applied an Adam step at learning rate 4e-5 with finite gradient norm 210.0, and unloaded the disposable model. Total test time including model initialization was 264.18 seconds. Receipt: `.science/qubit-muse-legacy-v5p-20260925/isolated-probe-result.json`. This is a real TPU backward/optimizer check, not a completed experimental rollout batch or durable experimental step. The real run uses a separate adapter slot.

AC2 Gemma 1752 also has healthy trainer/inference endpoints and its resumed adapter was accepted by all three serving engines. Durable Gemma checkpoint remains 13/20, Muse remains 16/20. AC2 Muse 1753 is still waiting for a third worker. Eight deferred jobs remain held.

At 08:20 UTC, both running jobs had 16 group requests in the ingress. Qubit had all 512 requested sequences accounted for across six engines (319 running, 193 waiting), with the first 65 generated tokens observed during warmup. Gemma had 48 sequences running, 461 waiting, and three completed engine requests; generated-token counters summed to 284,093. These counters establish active inference, not completed 512-rollout grading/training updates. `runtime-use.json` contains the observation and client logs proving Gemma resumed optimizer/search step 13 and qubit used fresh optimizer/search step 0.

## Provider loss at 08:22 UTC

GCP reported worker 1763's queued resource SUSPENDING with `stateInitiator=SERVICE`, followed by its TPU VM entering DELETING. AC2 Gemma 1752 entered native recovery without resubmission. Its last observed generation counters totaled 403,697 tokens and 31 completed engine requests; no new durable step was observed. Checkpoint 13 remains the resume point.

Qubit 1751 on worker 1765 remains healthy and generating. All six engines advanced beyond initial warmup, with the previously slower host reporting approximately 1,350 and 1,434 generated tokens/s per engine at 08:23 UTC. AC2 Muse 1753 and Gemma 1752 await replacement capacity. The provider queue showed 32 reservations WAITING_FOR_RESOURCES, three SUSPENDED, one SUSPENDING, and one ACTIVE at the queried instant. Do not interpret provisioning records as usable workers.

Added and verified `qubit-ac2-v5p-performance-20260925.service` on della9. Every two minutes it follows the current three submission receipts, reads live controller state, ingress state and per-engine Prometheus counters, and appends `runtime-use-history.jsonl`. It stops when all three submissions are terminal. This supplements the durable GCS checkpoint monitors and avoids describing a live process as completed training. The first recorded cycle observed two completed qubit engine requests and growing counters on all six engines. Both AC2 recovery dispatch services remained active.

## Capacity block audit

Three consecutive goal turns observed insufficient provider capacity for the complete three-job objective. Latest reconciliation: worker 1765 is the only READY v5p-32 worker; 32 provider reservations WAITING_FOR_RESOURCES, four SUSPENDED, one ACTIVE. Qubit has six healthy generating engines (1,921,041 generated tokens; 49 completed engine requests at this observation), but both AC2 jobs require new capacity. The full goal is not complete. Native recovery controllers and all five progress/performance/AC2-dispatch services remain active. No job, worker, or monitor is stopped by the goal-status change. Audit receipt: `goal-capacity-block-audit.json`.
