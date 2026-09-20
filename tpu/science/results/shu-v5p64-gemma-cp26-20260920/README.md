# Gemma circle packing on the dedicated v5p-64

Machine: `vision-mix/us-central1-a/shu-v5p-64-on-demand-machine1`, provider state READY, eight hosts with four chips each. Run: `shu-v5p64-gemma-cp26-grpo-lr4e5-s1-20260920`.

- Rank 0: trainer TP4/FSDP1, CPU client, CPU grading.
- Ranks 1–7: independent TP4 inference engines, 16 sequences each, 0.80 memory utilization, native thinking budget and prefix caching.
- GRPO, LR 4e-5, 16 x 32 rollouts, 15 epochs, 16,384 prompt-plus-thinking allowance, 22,528 total context.
- Existing single-host v5p Gemma trainer recipe: logical KV heads 4, full rematerialization, sequence buckets 18,432/22,528, token budget 22,528.
- Imported 51 graded bootstrap states from Gemma v6e circle packing. Original pool hash matches its completion marker. Fresh model/optimizer state; no failed-attempt checkpoint is treated as successful progress.
- New central1 run/output and compile-cache destinations. HF/Orbax and compile-cache seeds reuse the existing configured assets.

`config.py` adds a narrow v5p-64 hardware mapping and validates one trainer host plus seven single-host TP4 inference engines. Existing v5p-32 profiles still validate; invalid v5p-64 host counts, trainer splits and zones were rejected by local checks. This does not yet establish TPU training success.

All eight hosts passed the device-owner, stray-runtime, grading-unit, RAM and disk audits before launch (`host-audit.json`). No existing workload was killed. The first launch failed because this standalone machine lacked `uv`, normally installed by pool setup. Installed user-local uv 0.8.22 on all eight hosts; `uv-install.json` and `retry-uv.json` record the correction and restart.

This is an existing provider machine, not a newly provisioned SkyPilot pool member. It runs the same Ray v2 executor via per-host systemd launcher unit `skyrl-shu-gemma-cp26-launch`; there is no managed Sky job ID. The runner retains its nested cgroup supervision, PID/start-time parent checks and slice heartbeat coordination. The outer launch has no automatic retry policy. Stop this exact launcher unit on all eight hosts to request shutdown; do not stop unrelated Ray processes or delete the machine.

The live controller confirmed `train_ranks=[0]` and `inference_ranks=[1,2,3,4,5,6,7]`, with all eight hosts entering cache setup. Service readiness and an optimizer update are not yet claimed. The four-minute monitor includes this direct run.

`launch.py` uploads and verifies the immutable bundle and seed pool, writes a receipt before dispatch, and starts the eight units. Before using it on another newly prepared standalone host, install uv 0.8.22 in the user environment. Do not rerun over an existing receipt without reconciling all host units and GCS state. `prepare.py`, `seed-import.json` and the profile record the reproducible setup.

## Stale-lock repair

The first trainer startup after uv installation failed before model loading: libtpu reported the TPU was already in use. All eight hosts had no device owners and all launcher units had stopped. Rank 0 still had `/tmp/libtpu_lockfile`, owned by uid 2014 with mode 0600, inaccessible to gcpuser. No process held the file. The error therefore came from the inaccessible stale lock, not a concurrent trainer.

A targeted root check confirmed no device or lock-file handles, acquired both advisory lock forms without blocking, verified the inode, and removed that one file. The clean-host gate now rejects inaccessible TPU lock files early and remains read-only. The updated gate was inserted into all eight existing launch scripts. Every host passed it before restarting the same systemd units and run namespace. Seeds and runtime settings were preserved. See `lock-failure-evidence.json`, `lock-repair-audit.json`, and `retry-lock-repair.json`.

The retry cleared the lock failure but exposed the outer standalone systemd unit's default 8 MiB `LimitMEMLOCK`. The nested runtime correctly inherited that limit; TPU mmap initialization then failed. Normal SSH sessions have unlimited locked memory. Set `LimitMEMLOCK=infinity` only on the eight dedicated launcher units and added it to `launch.py` for future launches. Before restarting, all eight hosts passed the clean-host audit again.

Each host also had a `/tmp/tpu_logs` directory owned by the previous user. After confirming no open handles, preserved each directory at `/tmp/tpu_logs.uid2014.before-skyrl-20260920` and created a writable gcpuser-owned replacement. No previous logs were deleted. `memlock-repair.json` and `retry-memlock.json` record these changes.

## Live acceptance after repair

At 17:09:26 UTC the controller reported `services_ready` with one trainer and seven inference replicas and started the GRPO client. The client restored all 51 PUCT seed states and entered step 0 with 16 rollout groups. At 2026-09-20T17:11:19.713674+00:00, every inference host had accepted the same `model_2635337d_ss0_seq1` adapter (HTTP 200) and had positive generated-token counters. There were 112 active generations across the seven engines; the latest per-engine log samples totaled approximately 4091 generated tokens/s. These are live generation measurements, not a completed optimizer-step or grading result. See `live-generation-evidence.json`.
