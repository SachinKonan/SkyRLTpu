# Direct LoRA Ray Serve canary

Isolated experiment on one SkyPilot-managed v4-32 or v4-64 pool worker. All files and
runtime directories are canary-specific. No production launchers are edited.

- Host 0: workload Ray head, Serve ingress, client, CPU grading. No advertised TPU.
- Hosts 1-2: one independent TP4 engine each, Serve replica, bounded CPU grading.
- Remaining hosts: CPU grading; no advertised TPU.
- SkyPilot Ray remains separate. Workload Ray uses port 16379 and private ports/temp.
- Ray Serve supplies replica scheduling, request routing and health-based recovery.
- The existing vLLM TPU upload server handles adapter loading inside each replica.
- The ingress streams a client upload to local disk, drains admitted requests,
  explicitly loads both replicas, and commits the version only after both ACK.
- Upload failure is fail-closed. New requests cannot accidentally mix versions.
- A replacement replica reloads the committed upload from the surviving head.
- The head is still a single point of failure; this is not durable adapter storage.

This is **not** native Ray Serve LLM's cloud-storage LoRA loader. Native Ray Serve
is the routing/recovery layer; direct-upload coordination is canary integration.

The client generates small Erdos density constructions, grades completed results
immediately as Ray CPU tasks, updates the adapter version, then stops one canary
engine. Assertions cover overlapping generation/grading, both inference hosts,
grading on all hosts, peer survival, and replacement registration with LoRA loaded.
It uses structured generated data, not execution of model-generated Python, and
does not run a trainer or optimizer. One real exported adapter is reused under two
immutable names; changing weight values and full RL rollout semantics remain later
tests. Full reservation/head loss recovery is outside this canary.

Submit with the repo's configured SkyPilot CLI:

```bash
sky jobs launch -p tpuswarm-v4-32-central2-smoke \
  tpu/swarm/ray_serve_canary/canary.yaml -y -d
```

The eight-host variant keeps two TP4 engines and grades across all eight hosts:

```bash
sky jobs launch -p tpuswarm-v4-64-central2-qwen35-erdos \
  tpu/swarm/ray_serve_canary/canary_v4_64.yaml -y -d
```

For this variant, the runtime root (including `fixture.tar` and logs) is
`~/ray-serve-canary-v4-64-v3`, and `CANARY_HOST_COUNT=8` is required.

Transfers use one native gcloud batch, with `CANARY_GCLOUD_PROCESSES=4` and
`CANARY_GCLOUD_THREADS=8`. Completed model files are reused; missing blobs are
downloaded into staging, size-checked, and hard-linked into their manifest names
without a second disk copy. The XLA restore batches only completed `-cache`
objects through `gcloud storage cp`, excludes uploaded temporary files, and
validates cached files against GCS size and MD5 metadata. Both commands work
with the TPU image's older CLI, without requiring `storage rsync`.
Transfer logs and 30-second metrics are in `transfer-{hf,xla}.{log,jsonl}`.
Logical file sizes include buffered/incomplete bytes; disk-write rates measure
the destination filesystem's whole block device, not exclusively gcloud traffic.

Slicing defaults off for the initial comparison. Future runs can set
`CANARY_GCLOUD_SLICE_THRESHOLD=150M` and `CANARY_GCLOUD_SLICE_COMPONENTS=8`.
Tune this separately from cross-file concurrency and the destination disk.

The v4-64 configuration uses a private 96 GiB tmpfs on each inference host,
with a 128 GiB available-memory reserve checked before mounting and swap required
to be disabled. `model`, `model-downloads`, and `xla` point into that mount.
Prior disk directories are preserved under `disk-cache-before-ram`; neither
trainer caches nor checkpoints are modified. The RAM files survive engine/job
restarts on the same host, but not a reboot/unmount. Readiness markers are
revalidated after setup. Code, venvs, logs, and uploaded LoRA archives remain on disk.

The code-only archive in `CANARY_BUNDLE` must contain these files at its root.
It uses the frozen v42 worker bundle for the existing TPU fork and upload server,
not the dirty shared worktree. After assignment, deliver an uncompressed PEFT tar
containing `adapter_config.json` and `adapter_model.safetensors` to the assigned
head's `~/ray-serve-canary-v1/fixture.tar` using atomic rename after transfer.
The client waits for this direct transfer; it never fetches the adapter from GCS.

Logs: `~/ray-serve-canary-v1/{serve.log,engine-*.log,events.jsonl}`.
The result events and Serve log are uploaded to `RESULT_GCS` at test completion.
Local validation: `srun -p cpu ... python -m pytest tests/tpu_swarm/test_ray_serve_canary.py`.
