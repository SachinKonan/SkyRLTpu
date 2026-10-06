# AC2 priority recovery on v4-64 — September 24, 2026

AC2 Gemma 1638 and Qwen 1637 are being resumed on the existing
`tpuswarm-v4-64-central2-qwen35-erdos` pool in `us-central2-b`.
Their run IDs, published code bundles, optimizer checkpoints, search state,
loss, and 15-step targets are preserved.

| Model | Managed job | Worker | Durable step before this recovery | Best AC2 bound/reward |
|---|---:|---:|---:|---:|
| Gemma | 1638 | 747 | 6/15 | 0.9470895673605969 |
| Qwen | 1637 | 761 | 5/15 | 0.9357378134873612 |

These are last durable training results, not new updates from this recovery.
Muse AC2 remains cancelled at its completed 15-step target.

## Why priority alone did not start them

The installed pool recovery path bypasses the ordinary priority scheduler.
Its worker selection also treats a nonterminal job's stale cluster reservation
as occupying that worker, even after the worker-local attempt has failed.
Raising Gemma's stored priority to 120 did not by itself solve placement.
Qwen's stored priority is now 119, with Gemma at 120. These persisted settings
express AC2 priority but do not repair the pool recovery path's scheduling bypass.

Three failed recovery attempts were deferred only after confirming that their
exact local attempts had failed and that their saved checkpoints were durable:

| Original job | Workload | Saved step | Original target | Disposition |
|---:|---|---:|---:|---|
| 1635 | Gemma circuit | 14 | 15 | Requeued as 1663 from exact saved task |
| 1626 | Qwen circuit | 15 | 15 | Already at target; no extra training requested |
| 1636 | Qwen qubit | 20 | 25 | Requeued as 1664 from exact saved task |

The active Muse circuit job 1646 on worker 726 was left running. Gemma qubit
1603 remains eligible for worker 773. No worker was released or pool resized.

AC2 launch requests were also waiting behind unrelated provisioning requests
in the shared API execution queue. Only the existing pending AC2 `sky.exec`
requests were dispatched through SkyPilot's request execution wrapper, after
verifying the request's managed job ID, name, cluster, and pool. This preserves
the request result and controller ownership; no duplicate job was launched.

## Startup repairs

Worker 747 lacked head disk space. Only recreatable code/environment/source
caches from inactive runs were removed, followed by `uv cache prune --ci`.
The final head check after this cleanup reported about 31.5 GiB free. Checkpoints,
client databases, run logs, and remote durable state were preserved.

Both models then hit `not enough available memory to grow the RAM cache` when
reusing 64-GiB inference caches with their configured 128-GiB capacity. The
current growth guard compares available RAM with `new_cap + reserve - old_free`,
which is too conservative for these populated caches. No training configuration
or published bundle was changed in this recovery.

Idle inference caches were resized in place, retaining their contents. Before
each remount, checks verified tmpfs identity, no TPU owners, no users of that
mount, no active local job, no pending dispatch, and enough RAM to leave at least
the configured 240-GiB reserve even if the enlarged cache became full.

- Qwen ranks 2, 4, 5, 6: 64 → 128 GiB; calculated remaining RAM at full cache
  was approximately 253 GiB per host.
- Gemma ranks 4, 5, 6: 64 → 128 GiB; calculated remaining RAM at full cache
  was approximately 262 GiB per host. Rank 2 already had a 128-GiB cache.

The stale AC2 reservations were cleared under the pool lock only after idle-host
and pending-request checks. Their existing controllers then selected workers
and generated new launch requests. The shared API and controller processes were
not restarted; pool scheduling source was not modified.

## Validation boundaries and evidence

`RUNNING` in SkyPilot proves a launched process, not an optimizer update or a
working inference endpoint. This recovery requires separate runtime/startup
checks; new completed steps and farm leases must be reported from live evidence.
Farm discovery and borrowing configuration were retained. An available farm is
not a confirmed lease.

Detailed receipts, exact deferred task YAML, verified profiles from immutable
published bundles, dispatch IDs, cache checks, and replacement-job receipts are
under `.science/ac2-priority-control-20260924/`.

The reservation, API queue, and conservative RAM-cache resize issues remain
code-level follow-ups. The cache remounts are an operational repair for these
workers, not a deployed general source fix.

At approximately 01:00 EDT, both AC2 jobs had worker-local `RUNNING` attempts:
Qwen local job 7 on worker 761 and Gemma local job 12 on worker 747. Topology
validation passed with four training hosts and four local inference hosts each.
Both were restoring caches; Qwen had begun completing host preparation. No new
optimizer update or active farm lease was established by this snapshot.

At 01:08 EDT, both jobs still had running local attempts. All eight Qwen hosts
were prepared, its ingress returned HTTP 503 during engine warmup, and a directly
inspected TPU engine was advancing through compiled input sizes (16 through 512
tokens). Gemma had four prepared trainer hosts and four inference hosts restoring
compilation caches. Neither had reached `services_ready` or `client_started` yet.
Muse circuit 1646 remained running. Replacement jobs 1663 and 1664 had no worker
assigned. The bounded read-only startup observer records subsequent snapshots in
`startup-history.jsonl` and `startup-latest.json` under the evidence directory.
