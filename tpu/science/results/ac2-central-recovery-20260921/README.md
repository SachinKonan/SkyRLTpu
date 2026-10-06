# Resume displaced AC2 jobs in central

At approximately 06:22 UTC on 2026-09-21, GCP reported all four original east
v6e workers as `UNHEALTHY_MAINTENANCE`. SkyPilot's SSH probes received connection
refused. Qwen AC2 acquired an east replacement; Gemma and Muse AC2 remained in
recovery on their old unavailable worker records. Central had seven READY idle
workers in the initial inventory and at least two at the immediate preflight.

The two waiting jobs were moved to central:

| Model | Cancelled job | Replacement | Assigned worker | Saved step |
|---|---:|---:|---|---:|
| Gemma AC2 | 1335 | 1363 | central1b-168 | 2 |
| Muse AC2 | 1336 | 1364 | central1b-170 | 4 |

Each original job was confirmed CANCELLED, and the absence of any other active
job with that run ID was checked, before its replacement was submitted. The
replacement receipts were then matched to the live queue. Qwen's existing
recovery, all qubit jobs, the circuit queue, and farm jobs were left in place.

These are checkpoint continuations, not fresh experiments. Only the task YAML
`resources.zone` changes from `us-east5-b` to `us-central1-b`; its structured
contents were compared against the original task after reversing that field.
Both tasks use the exact original immutable code archive, SHA256
`8b885c71e8a2a4bd4e6d0a412eedeea21a937874e6ea4f3d33d15dd8a8b41d3d`.
Local archive hashes and the existing GCS object were checked. Inspection of
the archived source confirms profile zone is used for task construction, not
runtime placement. The archived profiles retain checkpoint resume, their
canonical run IDs, and NUM_EPOCHS=10 (ten total steps, not ten more).

Run IDs, buckets, model and optimizer settings, cache reads/writes, adapters,
seed pools, and farm admission behavior remain identical. Before cancellation,
GCS contained the saved client checkpoint index, matching step pool, database
backup, trainer checkpoint archive, and sampler-weight archive for each run.
`prepared.json` records their object generations and sizes. Recovery is not
counted as complete until the new clients actually restore and continue.

The AC2 follow-up gate selects the newest job ID for each canonical run ID, so
it automatically follows 1363/1364. Its supervised source guard was rearmed
after reviewing this placement-only commit; no handoff or source winner was
changed. The handoff still requires all three originals to finish step 10.
