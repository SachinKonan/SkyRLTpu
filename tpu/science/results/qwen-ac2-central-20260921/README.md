# Qwen AC2 recovery in central, 2026-09-21

Qwen AC2 managed job 1334 repeatedly lost east workers and was RECOVERING on
worker 5689. At the live preflight, east had zero READY pool workers and its
provider replacements were still CREATING. Central had nine READY idle workers;
worker 173 passed an SSH clean-host audit with unowned TPU devices and no stray
workload. Existing central training, farm, qubit, and dedicated circuit jobs
were inventoried and left running.

Only task resources.zone changes from us-east5-b to us-central1-b. The structured
YAML was compared with the original after reversing that field. The original
immutable bundle was checked locally and against the generation-pinned GCS
object. Its profile still enables checkpoint resume and NUM_EPOCHS=10. Run ID,
source, model, loss, caches, storage bucket, seeds, and admission settings are
unchanged.

Preflight verified checkpoint 3, matching PUCT step 3, training/sampler archives,
and the database backup. The pinned database passed SQLite quick_check and
contained COMPLETED training and sampler registrations. See prepared.json and
state-verification.json for object generations and paths.

Job 1334 was confirmed CANCELLED, and no other active writer with this run ID
existed, before replacement 1372 was submitted to tpuswarm-v6e32-central1b.
The submission receipt was matched to the live queue. This continues toward ten
total steps, not ten additional steps. Actual optimizer restore and resumed
sampling must still be observed. The AC2 completion watcher selects the newest
job for the canonical run ID and therefore follows 1372 automatically.
