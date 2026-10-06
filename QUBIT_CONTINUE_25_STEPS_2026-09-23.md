# Qubit continuation to 25 total steps

September 23 morning handoff correction: Qwen1590 completed successfully at
15 steps. Its recovery wrote `final` under a new model ID, while the numbered
step-15 archive belongs to the prior model ID. The continuation watcher had
incorrectly required both archives under the newest model ID. It now validates
the actual `state_path` from the checkpoint index and the database backup,
retaining the strict saved-step, metrics, and PUCT checks. Six regression tests
passed on CPU job14308536. Only the local continuation watcher was restarted;
running TPU jobs and immutable training bundles were unchanged by this fix.

On September 23 the user authorized Qwen, Gemma, and Muse to continue to **25 total optimizer steps**. These are continuations of each model's own checkpoint, optimizer state, and search pool. No cross-model search-state transfer is included.

| Model | Current source job | Source finishes at | Continuation target | Last verified saved step / best reward |
|---|---:|---:|---:|---|
| Qwen | 1590, worker 760 | 15 | 25 | 13 / 0.5496452437350237 |
| Gemma | 1493, worker 726 | 10 | 25 | 9 / 0.5531269293274382 |
| Muse | 1600, worker 758 | 15 | 25 | 11 / 0.5254921145935254 |

The saved-step observations are from preparation around 12:25 a.m. EDT; read the live status receipt for subsequent progress. The active jobs were not interrupted. Gemma's previous 10-to-15 launcher (PID 16799) was stopped after verifying its identity, obtaining its submission lock, and confirming it had not submitted a continuation. Its replacement goes directly from 10 to 25.

## Execution and state preservation

All continuations use the existing `tpuswarm-v4-64-central2-qwen35-erdos` pool. Managed job names remain identical to runtime run IDs so inference-farm discovery continues to recognize them. The same bucket and durable checkpoint/search-state paths are retained. New local roots end in `-continue25` to avoid restoring stale state from a reused worker's old local run directory.

The three `*-continue25.json` profiles set `NUM_EPOCHS=25` and strict minimum resume steps of 15 for Qwen/Muse and 10 for Gemma. The sick-marker path follows the new local root. All retain the working 96-GiB trainer RAM cache, existing Ray v2 executor, adaptive PWC rho 0.5, importance-sampling loss, 16 groups of 32 rollouts, model-specific runtime settings, grading, and inference-farm borrowing configuration.

Each immutable bundle was derived from its existing continuation bundle, replacing only the selected profile JSON. All other archived file contents were compared byte for byte. Gemma's parent bundle was the prepared 15-step continuation, which itself preserves the running source's non-profile files.

| Model | New bundle SHA256 |
|---|---|
| Qwen | `39d004f958f15b18022eacfc2ef62f735872d8da355d5c68ea72288c5ec8b2f3` |
| Gemma | `075b606b18c823fd1bbdff1bf948fdb8017d1ccc6e5cba06277b7d042a3e309b` |
| Muse | `d2dcb86a0eb588a68f167c76719d6756632210e6b9c35d6ae2985f0259d66e29` |

All three were uploaded under `gs://sk7524-tinker-tpu-us-central2/code-bundles/science-training-<sha>.tar.gz`. No fleet sizing or unrelated workload changes were made.

## Automatic handoff

Launcher: `python -m tpu.science.ops.qubit_continue25 --watch`.

Initial watcher PID: **3070412**, detached on **della-vis2.princeton.edu**. This is a local background launcher, not a SkyPilot dependency job. It polls every 60 seconds for up to seven days and exits after all three successors have reached RUNNING, or records any branch that needs manual reconciliation. RUNNING alone does not establish resumed optimizer progress; check worker logs and checkpoints afterward.

Before each submission it requires:

- The designated source job has succeeded and no other active job owns that run ID.
- Matching final-step metrics, checkpoint index, and a nonempty search snapshot in GCS.
- The database backup, indexed checkpoint archive, and numbered final-step archive exist.
- The prepared profile, task, and bundle hashes match.
- The expected service-account identity and healthy allocated v4-64 capacity are present.

Submission locks, intents, and receipts prevent duplicate launches; an ambiguous launch requires reconciliation. The launch tasks retain their per-host cleanliness gates. The watcher holds the superseded Gemma watcher's lock as well as its own.

For the previously observed stuck Sky execution queue, the watcher can dispatch an existing request through SkyPilot's normal locked execution wrapper after it has remained PENDING for two minutes. The child verifies the exact continuation's managed job ID, cluster, run name, bundle hash, and STARTING state before acting. It never creates an additional job for this repair and does not restart the API server.

## Evidence and validation

Private receipts are under `.science/routing-relaunch-20260921/extend25/`:

- `prepared.json` and each source-job directory's `manifest.json`, task, and archive.
- `upload-checks.json`: uploaded bundles and durable progress observed during preparation.
- `watch-process.json`, `watch-status.json`, and `watch.log`: current launcher state.
- Per-source `intent.json`, `submitted.json`, and optional error/dispatch receipts appear as handoffs occur.

Profile validation and archive equivalence passed on Slurm CPU job **14296877**. Six launcher guard checks passed on CPU job **14296987**, covering live/failed sources, conflicting run ownership, ambiguous submissions, receipt idempotency, and changed prepared inputs. The initial live one-pass check correctly left all three sources running and reported `waiting_source` for their step-25 successors.

## Cache ownership repair, September 23 around 1:28 a.m. EDT

Gemma's first continuation attempt, job **1603** on worker **726**, failed before model startup with `not enough available memory for RAM cache plus runtime reserve`. Its failing host had the inference role: admission required 128 GiB plus the 240-GiB runtime reserve. A subsequent measurement showed 330.81 GiB available and 59.37 GiB occupied by the preceding Gemma run's tmpfs cache. The retention audit had preserved that cache because it considered the old run active.

The process check treated a matching `RAY_NAMESPACE` as proof of ownership even when `SKYPILOT_TASK_ID` identified a different managed job. Continuations intentionally retain the run ID/namespace and use a new local root. The repair shares one predicate between the ordinary and privileged process audits: explicit old task identity, old run-directory arguments, or `TTD_RUN_DIR` still protect the old cache; namespace-only processes remain protected when task identity is missing. A known different task with only the same namespace no longer prevents reclaiming an idle retired root. File locks, mount-user checks, unknown-content checks, and fail-closed handling remain in force.

Validation: **57 passed** across cache-admission and checkpoint-retention tests on CPU job **14298520**. Regression cases cover both audit implementations, a same-run continuation reusing the old cache, and preserving old-task/path users and unknown-identity orphans.

All eight hosts on idle worker 726 were inventoried before cleanup. Only the two retired Gemma roots were eligible. Under the admission, root, and run locks, with a fresh privileged process and mount-user audit and matching boot identity, **11 unused tmpfs mounts totaling 550.78 GiB were normally unmounted**. Each unmount was verified. Available RAM afterward ranged from 372.56 to 390.38 GiB. Checkpoint files, optimizer archives, run logs, and GCS state were not deleted.

During this work SkyPilot's existing Gemma retry progressed on worker **758**. It reclaimed that worker's retired Muse caches, passed cache admission on all eight hosts, and began model/cache restoration. The prepared cancellation was deliberately not executed once this progress was observed. **Gemma1603 remains on its original immutable bundle** `075b606b...`; the repaired Gemma bundle below is prepared and uploaded but not substituted into the active job. At this snapshot Gemma had no new optimizer step and its inference endpoint was not ready yet.

The pending Qwen and Muse step-25 handoffs now use the repaired bundles. Their profiles and all other archived files are unchanged from the previously prepared continuations; only `checkpoint_retention.py` and `cache_admission.py` changed.

| Model | Repaired bundle SHA256 | Deployment |
|---|---|---|
| Qwen | `0be275a21ecab575eb56bb81720f2d137d3520b43231e00570aa69e35ada56ce` | Armed after 1590 completes step15 |
| Gemma | `4e54072ce25f240d3e7182601dfec2de9161d6069c1ec696d683d0e129a6e441` | Prepared only; active1603 retained |
| Muse | `e79448782d48d6ebd4739609a81221c72b9ac2a18a9e98321f50f4132c058324` | Armed after 1600 completes step15 |

The continuation watcher was restarted as PID **3492649** on della-vis2. Its latest receipt confirms Qwen/Muse waiting for their source jobs and Gemma1603 running. Private evidence is in `.science/routing-relaunch-20260921/extend25-cache-handoff/`: immutable manifests, activation record, before/after cache audits in `cleanup726/`, and `gemma-live-after-fix.json`. Archive equivalence validation ran on CPU job **14298530**.
