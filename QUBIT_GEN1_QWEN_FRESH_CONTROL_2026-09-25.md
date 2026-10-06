# Qubit Gen-1 Qwen fresh-adapter comparison — September 25, 2026

User authorized a parallel Qwen-on-Gemma20 run using fresh weights after observing
copy-heavy generation in the Qwen22-adapter branch. Here, fresh means the same
pretrained Qwen3.5-27B base with a newly initialized LoRA and optimizer. It does
not mean randomly initializing the base model. Existing Qwen and Muse branches
are left running.

## Matched experiment

| Setting | Existing Qwen job 1736 | Fresh arm |
|---|---|---|
| Initial policy | Qwen Gen-0 step-22 adapter | Pretrained Qwen with fresh LoRA |
| Initial optimizer | Fresh at branch start | Fresh at branch start |
| Program pool | Exact Gemma step 20, 711 programs | Identical |
| Initial best pool reward | 0.5553563436932767 | Identical inherited score |
| PUCT counts/values/total | Reset at branch start | Reset at branch start |
| Program ancestry, feedback, scores | Preserved | Identical |
| Additional updates | 10 | 10 |
| Batch | 16 groups × 32 rollouts | Identical |
| Context / prompt-plus-thinking limit | 22,528 / 16,384 | Identical |
| Temperature | 1.0 | Identical |
| Advantage / loss | Adaptive PWC rho 0.5 / importance_sampling | Identical |
| Learning rate | 1.5e-4 | Identical |
| Execution | Ray v2, v4-64, 4 trainer hosts TP8/FSDP2, 4 local TP4 inference engines | Identical |
| Inference capacity | 16 sequences/engine, memory utilization 0.8, farm discovery enabled | Identical settings; actual availability may vary |

Run ID: `qubit-gen1-qwen-fresh-on-gemma20-pwc05-10step-v464-20260925`.
Pool: `tpuswarm-v4-64-central2-qwen35-erdos`, zone `us-central2-b`.
Queue priority: 680. No existing run is cancelled or preempted by this submission.

`TTD_INIT_STATE_PATH_QWEN` is explicitly empty. No donor checkpoint, optimizer,
trainer database, or member checkpoint log is staged. The new output prefix was
checked for absence of those artifacts. Normal checkpoint recovery remains
enabled for this new run's future checkpoints. The Gemma seed snapshot has
`step=0`, `puct_n={}`, `puct_m={}`, and `puct_T=0`.

The immutable job-1736 bundle is reused with only its selected profile replaced;
622 other source files were verified byte-identical. Current uncommitted runtime
changes are not included. Profile validation passed. All GCS references in the
task and profile use the verified US-CENTRAL2 bucket. No inter-region copy is
needed. Worker 801 was unassigned and all eight hosts passed a read-only audit:
no active jobs or TPU device owners; at least 83.27 GiB free disk per host.

## Interpretation and measurement

This tests whether carrying the trained Qwen adapter contributes to copying. It
does not independently test the thinking budget. Fresh initialization is not a
guarantee of more diverse outputs. Compare exact-parent-copy fraction, unique
valid programs, valid fraction, and accepted new programs at matched rollout
counts and optimizer steps. Repeated identical-code reward fluctuations must
not be presented as new discoveries. Confirm any apparent improvement by repeat
grading before making a strong claim.

Preparation, source comparisons, clean-host evidence, submission receipt, and
live status are in `.science/qubit-gen1-qwen-fresh-20260925/`. In particular,
`submitted.json` is the durable job receipt; `status.json` is a timestamped live
snapshot. Submission/startup does not establish a completed optimizer update.

## Launch receipt

Submitted as job **1758**, assigned to worker **801**. Both managed-job and
host-local job records now report RUNNING; the observed stage is runtime/CPU
grader preparation, with archive checksums passing. No optimizer update is yet
verified. The stalled SkyPilot `sky.exec` request was reconciled and executed
once using the existing guarded dispatcher; its request status is SUCCEEDED.
Read-only monitoring is active on della9 as
`qubit-qwen-fresh-progress-20260925.service`, polling every 60 seconds. Jobs 1736
(trained Qwen adapter) and 1755 (Muse) remained RUNNING at admission.

## Startup correction

Job1758 did not complete any training update. Its archive retained an old PAX
`path` header when the selected profile's TarInfo.name was renamed, so the
launch command referenced a nonexistent profile. Corrected packaging removes
the stale PAX path and verifies the extracted launch profile and fresh-adapter
setting. Job1758 was cancelled and replaced by **1762**, with the same seed
pool, recipe, and run namespace. The original seed-import receipt is retained;
the corrected manifest is published separately under `profile-fixes/`.
The fresh-run monitor now follows the replacement and executes only its exact
guarded stalled SkyPilot dispatch request when necessary.
