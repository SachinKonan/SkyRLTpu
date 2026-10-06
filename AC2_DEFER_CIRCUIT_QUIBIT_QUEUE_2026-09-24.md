# AC2 deferred for circuit and qubit Gen-1 — September 24, 2026

User requested cancellation of AC2 on the existing v4-64 worker, complete worker cleanup, circuit Gen-1 ahead of qubit Gen-1 Qwen, and AC2 requeued afterward. No TPU was released.

## Actions and evidence

- Cancelled AC2 Gemma managed job **1686** using the native consolidated-controller cancellation signal. Controller consumed the signal and marked the job CANCELLED.
- Verified durable AC2 checkpoint **12**, best reward **0.9539225274166804**, in Central2. Preserved the completed registry backup, latest training archive, client history and PUCT state.
- Inspected all eight hosts of worker **747**. Cancellation initially left head-host executor processes and a TPU owner; after cancellation settled, explicitly stopped the remaining run-owned private Ray GCS process. No SkyPilot control cluster was stopped.
- Compared fourteen old local training/sampler archives against Central2 object sizes and MD5 hashes before deleting duplicate local copies. Freed 23.72 GiB from these archives.
- Audited and unmounted the retired AC2 tmpfs caches on all eight hosts, and removed unused AC2 environments/sources and uv caches after process/FD/maps checks. Final free disk: approximately **41.0, 40.2, 49.2, 40.1, 50.3, 50.1, 50.3, 49.6 GiB**. TPU owners, grading units and workload process checks were empty before replacement startup.
- Requeued AC2 as **1711**, same run and executable bundle, strict checkpoint resume, target 15 steps, priority **90**. Used native consolidation submission state transitions on shared SkyPilot state; existing controllers picked it up. No duplicate controller was started on the login node.
- Circuit Gen-1 Muse **1702** was assigned worker **747**. Its existing pending sky.exec request `07df98d3-e916-4e69-a84a-6849b0fc1097` was dispatched through the native request executor after cleanup; request completed SUCCEEDED. That proves submission, not completed model startup/training.
- Circuit Gen-1 Qwen **1701** on worker **775** was left running.

## Persisted queue priorities

| Job | Work | Priority |
|---|---|---:|
|1701|Circuit Gen-1 Qwen|160|
|1702|Circuit Gen-1 Muse|150|
|1705|Qubit Gen-1 Gemma20 → Qwen|140|
|1706|Qubit Gen-1 Gemma20 → Muse|130|
|1711|Deferred AC2 Gemma, resume step12|90|

Priorities were updated in both managed-job state and persisted DAGs. Existing pool controller retries may already be in flight; priority values alone are not proof of future assignment order. At this handoff the two healthy allocated workers were occupied by the two circuit jobs, with qubit and AC2 awaiting capacity.

## Evidence directory

`.science/ac2-defer-for-qubit-gen1-20260924/` contains cancellation/requeue intents and receipts, original task and resume proof, backup generations/checksums, per-host audits/cleanup receipts, priority changes, and circuit request dispatch evidence. Do not blindly rerun mutation scripts: they have exact job/process guards and submission-intent checks.

The login node could not reach the existing API at its configured localhost address. The shared native cancellation signal and native submission records were processed by the existing remote controllers; the API server was not restarted.
