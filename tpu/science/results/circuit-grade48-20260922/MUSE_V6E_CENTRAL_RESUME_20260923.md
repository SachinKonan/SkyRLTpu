# Muse circuit: v6e-central continuation, September 23, 2026

User authorized moving the paused Muse circuit run from v4-64 to v6e-central. Source job 1594 was verified CANCELLED before copying or submission. Qwen/Gemma continuations are unchanged; Muse stays excluded from the v4 continuation watcher.

Run identity is intentionally retained: `circuit300-v464-muse-pwc05-three-starts-10step-20260922-r4`. The historical `v464` in its name does not describe the new hardware.

Source: `gs://sk7524-tinker-tpu-us-central2/ray-training/circuit300-v464-muse-pwc05-three-starts-10step-20260922-r4/`.
Destination: same prefix under `gs://sk7524-tinker-tpu-us-central1/`.
All 1,337 objects (21.586 GiB), including database, checkpoint archives, bootstrap, sampler pool, and metrics, were copied server-side with source-generation guards and destination size/CRC32C verification. Source is preserved.

Resume requires checkpoint 6 (`model_3d182644/weights/000006`), with target 15 total steps and best saved reward 0.5104587089575159. The loader reconstructs LoRA and optimizer arrays against the destination templates; runtime restoration still needs confirmation from the new job.

Destination pool: `tpuswarm-v6e32-central1b`; zone `us-central1-b`; `tpu-v6e-32`, eight hosts. Worker 264 was idle and all eight hosts passed read-only storage, RAM, TPU-device, stale-process, and grading-unit checks. Available RAM was 688–700 GiB and free disk about 85 GiB per host. No unrelated workloads were stopped or workers released.

Profile: `tpu/swarm/ray_train/profiles/circuit300-muse-v6e-central-resume6-to15-20260923.json`.
Changes versus the prepared v4 continuation: accelerator, zone, bucket, trainer physical process bounds `1,1,4` -> `2,2,1`, independent compile-cache destinations, and resume minimum 10 -> 6. Adaptive PWC rho 0.5, importance-sampling loss, 16×32 batches, model/context, grader budgets, 48 slots/host, memory settings, and inference-farm discovery/borrowing remain unchanged. HF/Orbax base-model caches still reference the existing central2 copies.

Bootstrap reuse has a narrow explicit allowance for this reviewed v4-central2 -> v6e-central1 placement migration. Original contract/pool hashes and all scientific settings remain checked. Targeted tests: 24 passed, including rejecting changed run IDs, bucket, tensor parallelism, and rho.

Operational evidence and submission receipt: `.science/launch/muse-v6e-central/`. Build Slurm job 14309458. Raw prior baseline reports and current grading code are unchanged.

Submission confirmed: **job 1609**, controller state **STARTING**, assigned **worker 254**. Worker 254 became available when its preceding RG-LRU grading job 1605 **SUCCEEDED**; SkyPilot selected it rather than the worker 264 audited earlier. No unrelated job was cancelled. At the initial startup check no driver log existed yet, so checkpoint restoration and new training progress are not yet proven.

## Recovery diagnosis and corrected bundle

Job 1609 failed twice at the placement resource gate: 48 slots require 224 CPUs including service reserve, but v6e hosts expose 180. Corrected the migration profile to 32 slots (128 grading CPUs + 32 service CPUs). Rebuilt in Slurm 14310612; bootstrap tests: 24 passed. This changes concurrency, not grading budgets or scoring.

A subsequent attempt on worker 254 failed TPU ownership checks. Inspection identified actively changing TPU-owning `grader_child` processes under `/home/gcpuser/.cache/rglru-pregate-fix-20260923/runs/rglru-compact-ab-20260923/`, in session-76.scope, outside the SkyPilot assignment. These were not treated as stale leftovers. Job 1609 was cancelled to stop retries. Worker 254 appears unassigned in SkyPilot but is not actually idle; other READY workers are assigned. The active RG-LRU experiment is preserved pending explicit direction or genuinely idle capacity. No worker was released. Corrected bundle was uploaded and SHA256 verified; replacement has not yet been submitted. Step-6 central1 state remains intact.

## Corrected relaunch, job 1612

Subsequent inventory found two SkyPilot-unassigned workers: 254 and newly provisioned 269. Worker 254 still ran independent active RG-LRU graders; worker 269 passed all eight host cleanup and resource gates, including `180 >= 32*4+32` CPUs.

Submitted lightweight managed occupancy job **1611** (`external-rglru-254-occupancy-20260923`) to represent the actual external work on worker 254 and prevent Muse landing there. It checks the expected head hostname and merely waits while the specific existing RG-LRU grader processes are present; it does not stop them, launch grading, or touch TPU devices. It exits once those graders disappear. Controller confirmed assignment to 254.

Submitted corrected Muse job **1612**, same central1 durable state and target 15, with 32 grading slots per host. Bundle manifest and receipt are under `.science/launch/muse-v6e-central/`; corrected receipt is `submission-grade32.json`. Previous job 1609 remains cancelled.

Controller confirmed job 1612 assigned to **worker 269**. At the follow-up check the managed job was still STARTING: API execution request `e903ea76-9d66-4f1f-98dc-552407238470` was PENDING alongside other launch/exec requests, and no remote Muse driver log existed yet. This is controller dispatch delay, not a demonstrated new workload error or successful checkpoint restore. Do not report training active until remote evidence confirms it.
