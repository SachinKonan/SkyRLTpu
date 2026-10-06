# Qwen qubit returns to v4-64 — September 23, 2026

At the user's request, cancelled Muse qubit job1625 on worker747 and Qwen's
pending v6e-central job1610, then submitted Qwen job1636 to the existing
`tpuswarm-v4-64-central2-qwen35-erdos` pool. The scheduler assigned worker747.
Gemma qubit1603 and unrelated workloads were not cancelled or changed.

Qwen resumes `tinker://model_bdc7f571/weights/000016`, continuing to 25 total
updates. Its saved best reward is 0.5505479576624515. Verified the checkpoint
index, step16 metrics and search snapshot, completed checkpoint registration
inside the portable database, and nonempty training/sampler archives. Muse's
step16 state and best reward 0.5261619332546773 remain saved in GCS.

## Regional artifacts and compilation caches

Every configured GCS URI in the launch task and training profile uses
`sk7524-tinker-tpu-us-central2`, verified as a regional US-CENTRAL2 bucket.
This includes code/runtime bundles, Hugging Face/Orbax caches, checkpoints,
search state, portable database, and compilation-cache reads and writes.
Regional storage does not select a single availability zone.

Existing v4 warm-start caches are retained:

- Trainer: `science-q20-v4-qwen-grpo-clean-20260918-trainer_compile-v1`,
  382 objects, approximately 0.232 GiB.
- Inference: `science-q20-v4-qwen-grpo-clean-20260918-inference_compile-v1`,
  349 objects, approximately 0.271 GiB.

New compilation entries are written under
`qubit-v4-continue25-20260923/qubit-v4-qwen-parallel2-pwc-rho05-20260921/`
in that same central2 bucket, separated into trainer/inference prefixes.
No v6e/east5 cache was copied. Hardware-specific executables are reused only
when their compilation keys match; this does not guarantee zero recompilation.

The immutable bundle is
`2672523a637db9f56b717c5310ba92c751877c0c46a611eefdedf48186444bb6`.
Compared with the previously prepared v4 cache-handoff bundle, only the profile
changed: minimum resume step15 to16 and the two compilation write/read prefixes.
The learning recipe, farm assistance, runtime code, and v4 topology are retained.
Task priority is110. Profile validation passed on Slurm CPU job14321870.

## Worker preparation and launch evidence

After cancellation, all eight worker747 hosts had no TPU device owners or
active runtime/grading services. The head had22.28 GiB free, below admission.
Nine obsolete local Muse checkpoint archives were matched by size and MD5 to
their durable GCS objects before removal; this freed13.75 GiB, leaving36.03 GiB.
Latest step16 archives, databases, logs, search state, and all GCS data remain.

Launch receipts, artifact generations, inventories, and cleanup evidence are in
`.science/routing-relaunch-20260921/qwen-v4-return-20260923/`.
Submission/assignment alone is not proof of restored optimizer or a new update.

At 14:00 EDT, controller job1636 and worker-local job7 were RUNNING. All eight
hosts passed clean-host admission; startup was installing CPU dependencies.
Optimizer restoration has not yet been observed for this attempt. The existing
pending execution request was validated against job/run/bundle/worker and
dispatched through SkyPilot's locked request executor; it completed successfully.
The continuation preflight now permits a newer minimum resume step within the
authorized 25-step target; all21 tests passed on CPU job14321965.
