# Qwen qubit continuation moved to v6e-central

On September 23, the user requested moving the queued Qwen qubit continuation
from v4-64 to v6e-central. Queued job1608 was cancelled before replacement
job1610 was submitted to `tpuswarm-v6e32-central1b` in `us-central1-b`.
The scheduler initially assigned worker254. No running Gemma, Muse, AC2, or
circuit job was cancelled as part of this move.

The first attempt passed all eight clean-host gates but failed before TPU
startup because the previous job had left interrupted dpkg transactions on
worker254. Completed package configuration on all eight hosts and reinstalled
the interrupted bubblewrap package on host5; every subsequent dpkg audit was
clean. After verifying no remaining TPU/workload owners and the terminal local
job, cleared job1610's stale recovery reservation under the pool scheduler lock.
Its controller then selected worker264. One interrupted libssl configuration
there was completed before dispatch; all eight package audits were clean.
Recovery retains managed job1610 and the same immutable bundle/checkpoint.
At 09:14 EDT, job1610 was RUNNING on worker264 (worker-local job3), had passed
all eight host ownership gates, and was installing/building dependencies without
reported startup errors. Model/checkpoint restoration and a new optimizer step
had not yet been observed.

## Preserved experiment

- Run ID: `qubit-v4-qwen-parallel2-pwc-rho05-20260921`. The historical `v4`
  substring remains so checkpoint paths and inference-farm discovery retain
  the same identity; the new accelerator is `tpu-v6e-32`.
- Resume at step15 and continue to 25 total optimizer steps, with strict resume.
- Indexed state: `tinker://model_622babca/weights/final`, batch15.
- Existing database, optimizer checkpoint, metrics, and step15 PUCT snapshot
  remain under `gs://sk7524-tinker-tpu-us-central2/ray-training/<run_id>/`.
- Best saved reward before migration: `0.5499024273434641`.
- Adaptive centered PWC, rho0.5, importance-sampling loss, learning rate1.5e-4,
  16 groups ×32 rollouts, 22,528-token context, 16,384-token phase1 cap.
- Four trainer hosts, TP8/FSDP2, and four local TP4 inference engines. Existing
  inference-farm borrowing and attestation settings remain enabled.

## Deployment changes

Only the profile JSON was replaced in the already-prepared immutable bundle.
Every other archived file was compared byte for byte, preserving the cache
ownership fix and the Ray v2 executor.

- Accelerator/zone: v6e-32, us-central1-b.
- Trainer process grid: `2,2,1`, the validated v6e four-host block, replacing
  the v4 grid `1,1,4`.
- New local root ends in `-continue25-v6e-central`; sick-marker path follows it.
- New compilation-cache output prefixes under
  `gs://sk7524-tinker-tpu-us-central1/qubit-continue25-v6e-20260923/<run_id>/`.
  Existing v6e trainer/inference cache seeds were checked for readable objects.
- Task uses `v2-alpha-tpuv6e` and the pool's 150-GiB disk size. Model caches and
  durable run storage retain their previous locations. RAM-cache limits remain
  96GiB trainer, 128GiB inference, and 240GiB host reserve.

Profile:
`tpu/swarm/ray_train/profiles/qubit-v4-qwen-parallel2-pwc-rho05-20260921-continue25-v6e-central.json`.
Bundle SHA256:
`a83d578651ec52ce9b0f538011570ee3ad1856bff4348fbcbd1725bdc213f7a2`.

## Validation and operations

Nine targeted continuation/v6e tests passed on CPU job14309789. The final state
archive and database backup were verified present and nonempty, and the metrics,
checkpoint index, and PUCT snapshot agreed on step15 before cancellation.
Service-account, provider, pool, and storage checks passed.

Worker264 was initially free and audited on all eight hosts. Worker254 became
available during preparation and was selected by the scheduler instead; its
eight hosts were then checked for free TPU devices, stale TPU locks, active
workload units, and disk space. Its prior circuit job had already failed and
moved elsewhere. The task also retains the per-host startup ownership gate.

The continuation watcher receipt and prepared record now reference job1610 and
the v6e bundle. Its exact-request dispatch guard checks the pool in that record,
rather than assuming every continuation is on v4. The watcher was restarted
as PID1585125; Gemma and Muse continuation records were retained unchanged.

Evidence and exact launch receipts are under
`.science/routing-relaunch-20260921/qwen-v6e-central-migration/`.
Submission and allocation alone do not prove successful cross-hardware
checkpoint restoration or an optimizer update; inspect live startup/client logs.
