# Additional Qwen and Muse qubit runs on v6e-32

User authorized two additional runs while retaining the existing v4-64 runs.
Submitted 2026-09-17 to `tpuswarm-v6e32-east5b-qwen35`:

| Model | Managed job | Assigned worker | Starting state |
|---|---:|---:|---|
| Qwen | 991 | 4029 | Verified checkpoint 3 plus matching search pool |
| Muse | 992 | 4009 | Six verified seeds from completed draft/repair shards |

Profiles were committed at `490cbed9` before packaging/submission. They use
systemd lifecycle ownership, four trainer hosts TP8/FSDP2, four TP4 inference
engines, 16 sequences per engine, and CPU grading on all eight hosts at 16
slots per host (4 CPUs / 8 GiB per candidate). Qwen KV allocation stays 80%;
Muse stays 75%. Native 16,384 prompt-plus-thinking / 22,528 total context,
16x32 rollouts, optimizer/loss/reward settings are unchanged from v4.

Each run has distinct local roots, GCS run/checkpoint/client-state prefixes,
and trainer/inference compilation-cache write destinations in the east5
bucket. Neither publishes compilation caches to a v4 or Gemma destination.
Pinned HF base weights and base Orbax weights remain shared read-only sources.
See `config-deltas.json` for exact paths and the exhaustive v4-to-v6e diff.

Qwen's checkpoint and archived payload-preserving database were copied using
no-clobber writes and verified by size/CRC32C. Search/client state was cloned
under the new run name and verified byte-for-byte. Muse's canonical seed hash
was checked before staging, with exact GCS readback after upload. The v4 run
state and live databases were not modified. See `clone-manifest.json`.

Required gcloud and refreshed ADC identity were verified. TPU/storage reads
passed. All 16 spare hosts passed the clean-host device/process/storage gate;
minimum available RAM exceeded 688 GiB and minimum free home disk was 84 GiB.
Both uploaded immutable packages were read back and hash-verified before launch.
Submitting or reaching RUNNING does not establish an optimizer step.
