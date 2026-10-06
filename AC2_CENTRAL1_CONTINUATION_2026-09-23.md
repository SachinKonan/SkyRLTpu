# AC2 adaptive-PWC continuation on central1-b

The user authorized moving Qwen and Gemma AC2 from their recovering v4-64 controllers to existing v6e-32 capacity in `tpuswarm-v6e32-central1b`, continuing to **15 total optimizer steps**. Muse AC2 remains on v4-64 job 1597, including its existing continuation to step 15. Unrelated qubit and circuit work is outside this migration.

## Source state and isolated destinations

| Model | Source v4 job | Saved step | Best search reward | Checkpoint model ID |
|---|---:|---:|---:|---|
| Qwen3.5-27B | 1596 | 4 | 0.9339839439571375 | `model_6228d8ec` |
| Gemma4-31B | 1598 | 4 | 0.9456047867148081 | `model_28181b01` |

Source run IDs are `ac2-v464-{qwen,gemma}-adaptive-pwc-rho05-seed1-resume-20260922`, under `gs://sk7524-tinker-tpu-us-central2/ray-training/`.

Destination run IDs are `ac2-v6e-central-{qwen,gemma}-adaptive-pwc-rho05-seed1-resume-20260923`, under `gs://sk7524-tinker-tpu-us-central1/ray-training/`. New prefixes prevent source and destination controllers from writing to the same checkpoint namespace. The original state is retained.

Each copy contains the full step-4 LoRA and optimizer checkpoint, retained sampler exports, the portable API database with embedded request payloads, client checkpoint indices, global step, metrics, and matching PUCT search snapshots. Client log directories are renamed to the destination run ID; checkpoint model IDs are preserved. Dead-client pending requests are retired only in the copied database (Qwen 10, Gemma 16), because the continuation resumes the completed checkpoint rather than replaying interrupted training requests.

Generation IDs, CRC32C checksums, sizes, and source/destination object names are recorded in `.science/ac2-v6e-central-resume-20260923/{model}/manifest.json` and `published.json`. Local generation-pinned copies were validated before source cancellation.

## Configuration and runtime

- Each v6e-32 has eight hosts: four trainer hosts and four local TP4 inference engines. Trainer process bounds change from `1,1,4` to `2,2,1` for the verified v6e topology.
- Qwen retains TP8/FSDP2 and eight logical KV heads; Gemma retains TP4/FSDP4 and sixteen logical KV heads. Existing trained parameter shapes and optimizer settings are preserved.
- Adaptive PWC with rho 0.5 and `importance_sampling`; seed 1; 16 groups × 32 rollouts; learning rates Qwen 1.5e-4, Gemma 4e-5. Context length 22,528 and phase-1 cap 16,384 remain unchanged.
- Strict checkpoint resume requires at least step 4. `NUM_EPOCHS=15` is the total target, not fifteen additional updates. Fresh bootstrap generation is disabled.
- Ray grading retains 128 dispatch threads, two Ray CPUs per task, and a 1,100-second evaluation timeout. This migration does not change grading semantics or establish a new grading throughput benchmark.
- RAM-backed caches retain 96 GiB trainer / 128 GiB inference capacity and a 240 GiB host memory reserve. A target host reported approximately 689 GiB available RAM during preflight. HF and Orbax caches already exist in central1; compile-cache publication uses new run-specific prefixes and preserves the source profile's v6e compile seeds.
- Run-scoped inference borrowing remains enabled, with four requested farm engines, discovery updates, adapter attestation, and a 300-second initial reservation window before local fallback. The existing discovery supervisor already covers the central1 pool. No global farm-management configuration was changed.
- At preflight, Gemma farms were reachable; Qwen farms were recovering. Farm assistance being enabled does not imply a live lease or completed remote generation.
- The deployment bundles preserve all source runtime and AC2 overlay files byte-for-byte (625 Qwen files, 633 Gemma files). Only the new profile is added. The task wrapper uses the existing executor startup and clean-host gate, but omits unrelated routing/Qiskit/Rust CPU setup.

## Validation and cutover

Both exact bundles passed configuration and overlay-hash validation. Both portable databases restored and passed SQLite integrity checks. Each full checkpoint contains LoRA weights and optimizer state, with all three optimizer counters equal to four. Learning-rate values in the optimizer state match the profiles. The previous v4 step-10-to-15 watcher has a tested per-model retirement guard; its seven guard tests passed, including preventing a retired source from submitting another continuation.

`cutover.py` first rechecks the durable source steps and destination checksums, retires only Qwen/Gemma entries in the v4 continuation watcher, cancels only 1596/1598, verifies terminal controller state, and submits each prepared v6e task once. Muse's watcher remains active. Submission intents and receipts prevent blind duplicate launches following an ambiguous CLI response.

Operational evidence and scripts live in `.science/ac2-v6e-central-resume-20260923/`. Read `qwen/submission.json` and `gemma/submission.json` for assigned job IDs, `cutover-inventory.json` for pre-cancellation inventory, and `source-cancelled.json` for source terminal state. A submitted controller is not proof of runtime checkpoint restoration; runtime evidence must be recorded separately.

## Submitted jobs

At 08:19 EDT on September 23:

| Model | New job | Worker | Pool | Resume → total target |
|---|---:|---:|---|---|
| Qwen | 1606 | 257 | `tpuswarm-v6e32-central1b` | 4 → 15 |
| Gemma | 1607 | 261 | `tpuswarm-v6e32-central1b` | 4 → 15 |

Both were assigned and `STARTING` at the initial check. Source jobs 1596 and 1598 were verified `CANCELLED`. Muse 1597 remained `RUNNING` on v4 worker 726; its continuation watcher remained active. No new optimizer update or farm lease was claimed at submission.
