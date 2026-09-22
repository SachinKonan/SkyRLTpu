# Circuit bootstrap migration to v4-64, September 22, 2026

User authorized three circuit runs on the three idle v4-64 pool workers. Destination: `tpuswarm-v4-64-central2-qwen35-erdos`, `us-central2-b`. Initial inventory: idle workers 744, 749, 750; no active training/bootstrap processes on their heads. Existing qubit jobs 1493, 1482, 1483 are untouched.

Each model (Gemma4-31B, Muse-Glimmer-30B, Qwen3.5-27B) retains adaptive `piecewise_valid_entropic_centered_adaptive`, rho 0.5, `importance_sampling`, 10 optimizer steps, 16 groups x 32 rollouts, 22,528 context / 16,384 phase-one cap, and seed 1. Learning rates remain 4e-5 for Gemma/Muse and 1.5e-4 for Qwen.

Placement: all 17 IBM cases; 300-second placement budget, 180-second scorer cap; 4 CPU / 4 GiB per case; 32 case slots per host. Inputs retain frozen fallback plus the exact off/rudy/rudy_hv Xplace portfolio. No regrading of already journaled candidates due to hardware migration.

## Resume data

Original east bucket: `gs://sk7524-tinker-tpu-us-east5/ray-training/circuit300-v5p32-{model}-pwc05-three-starts-10step-20260921/client/`.

New destination: `gs://sk7524-tinker-tpu-us-central2/ray-training/circuit300-v464-{model}-pwc05-three-starts-10step-20260922/client/`.

| Model | Saved groups (16 each) | Saved candidate grades | Pending grades among saved generations |
|---|---:|---:|---:|
| Gemma | 64 | 1,022 | 2 |
| Muse | 41 | 656 | 0 |
| Qwen | 58 | 793 | 135 |

All copied objects are pinned to source generations and verified by size and CRC32C. Originals remain unchanged. Contracts retain prompt, tokenizer, generation request, root, and bootstrap implementation hashes. The reviewed config diff changes hardware topology, bucket/cache locations, run paths, and v4 execution settings only. Qwen enables the required v4 ragged convolution implementation. Source contracts and any source completion records are retained under `client/migration/`.

Gemma's source completion record reported 1,024 total / 357 valid / 108 retained, but no promoted pool snapshot was present in the source client prefix and two grade records were absent. Therefore its old completion marker is preserved as provenance, not used to skip recovery. Bootstrap finishes the two missing grades and rebuilds the pool. Muse/Qwen finish missing journal work and continue toward the existing 512-unique-valid target / 1,024-draft cap. No optimizer database or checkpoint is imported.

## Runtime and farm

Ray v2 executor uses eight hosts per v4-64 slice. Bootstrap permits all hosts for inference. Training uses the established four-host v4 topology and remaining hosts for local inference. CPU grading can use all eight hosts. Run-scoped farm borrowing, dynamic discovery, scheduler, and attestation remain enabled. Discovery's existing service already targets this pool.

Bundles include the pending RunBorrower fix: late discovery updates and periodic heartbeat checks retry lease preparation during an active phase, without blocking local generation. That change's existing regression run passed 74 tests. Compile output uses fresh v4 namespaces; v5p compiled executables are not reused. Existing Central model-weight caches are used.

## Evidence and reproducibility

Operational files: `.science/launch/v4-resume/` contains source object inventories, original/migrated contracts, per-model config diffs, source-generation/CRC copy receipts, build outputs, preflight results, tests, and eventual launch receipts. Preparation: `prepare.py`; generation-pinned GCS copy: `migrate.py`; build: `build.sh` (Slurm 14268617); submission: `.science/launch/submit_v4_resume.py`.

Every launch bundle is hash-verified after upload, includes the exact profile, and is checked for the exact bootstrap and RunBorrower source bytes. Completed journal grades require their corresponding saved generation group before migration. Cloud writes use fresh destination run prefixes and create-only generation preconditions.

Validation: bounded-bootstrap and placement-budget tests passed **30/30** (Slurm 14268662). The initial invocation lacked Discover on PYTHONPATH (25 passed, five import failures); rerun with the deployment-equivalent source path passed fully. Previous farm retry regressions: **74 passed**. These are local regression checks, not evidence of a completed v4 training step.

## Submission result

All three jobs were submitted and reconciled against the live controller queue:

| Model | Job | Pool worker | Initial controller state |
|---|---:|---:|---|
| Gemma | 1548 | 744 | STARTING |
| Muse | 1549 | 749 | STARTING |
| Qwen | 1550 | 750 | STARTING |

This establishes submission and allocation to the three previously idle workers. It does not yet establish farm lease acquisition, successful runtime bootstrap restoration, or an optimizer step. Final receipts: `.science/launch/v4-resume/submissions.json`, `queue-after.json`, `submit.log`.

## Muse inference recovery, same day

Job 1549 failed during fresh bootstrap generation with JAX `RESOURCE_EXHAUSTED: E0101: RuntimeProgramAllocationFailure` while loading `jit__select_from_array_fn` on local inference host 1 (10.130.0.170). The bootstrap client received HTTP 500; the local-inference fatal error stopped the controller. Cloud state retained 656 candidate grades and 41 groups; no training update had begun.

At the user's request, cancelled 1549 and prepared `circuit300-v464-muse-pwc05-three-starts-10step-20260922-r2`. Requested changes only: inference memory utilization 0.75 -> 0.7; prefix caching true -> false; max sequences remains 16. Also use fresh compile namespaces and a separate run ID. Keep the previous root to reuse installed environments and the verified model cache, avoiding duplicate disk/tmpfs consumption. The runtime automatically clears the old compile cache when its namespace changes.

All eight hosts of worker 749 passed the read-only clean-host gate after cancellation: no TPU owners, stray workloads/private executor Ray clusters, or active grading units. Free disk: 68.87–71.03 GiB; available host RAM: 323.53–339.38 GiB; existing private tmpfs model cache: about 55.6 GiB of 128 GiB. No disk deletion or unrelated process termination was needed.

The new bootstrap prefix contains the generation-pinned, CRC-verified journal, with original migration provenance retained. Source prompt, tokenizer, root, request, and implementation hashes remain intact. Adaptive PWC rho 0.5, learning rate, seed, and 10-step cap are unchanged. Explicitly verified generated command flags: `--gpu-memory-utilization 0.7 --max-num-seqs 16 --no-enable-prefix-caching`.

Evidence: `.science/launch/v4-muse-r2/` (worker audit, migration manifest and object receipts, build job 14270234, submission receipt). Submission helper: `.science/launch/submit_v4_muse_r2.py`. Gemma 1548 and Qwen 1550 are untouched.

Replacement submission verified: **job 1551**, **STARTING on worker 749**. Previous job 1549 was confirmed CANCELLED before launch. Successful generation under the revised settings remains to be observed.
