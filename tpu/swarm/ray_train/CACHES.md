# Compilation-cache writeback

This is the new, experimental Ray training path. These changes do not update
already-running pool jobs or their frozen bundles.

Both trainer and inference processes enable the JAX persistent cache and write
to the host's private `ram/compile` directory. Inference also sets
`VLLM_XLA_CACHE_PATH` to that directory. RAM is a replaceable cache, not durable
training state.

## Restore and Publish

- `CacheStore.restore_compile()` accepts an empty GCS prefix as a cold start.
  Existing entries are checksum-verified and reused or restored before startup.
- After the cache barrier, the controller schedules each host's writeback
  independently, including while engines are starting and compiling. The
  interval is `cache.sync_seconds`, default **60 seconds**, checked on the
  controller's `log_seconds` tick. Slow uploads do not overlap another upload
  of the same kind on the same host, or hold up other hosts.
- `CacheStore.publish_compile()` selects new flat `*-cache` entries, excluding
  temporary files, access-time files, and symlinks. It copies a batch into
  private tmpfs staging, checks for concurrent source modification, and verifies
  the complete zlib/zstd frame before uploading. A final-looking filename alone
  is insufficient: JAX may still be writing it.
- Uploads use one native `gcloud storage cp --no-clobber` invocation per batch,
  with process/thread concurrency from the profile (default 4 x 8). Existing GCS
  objects are never overwritten or deleted. Uploaded snapshots are verified
  against GCS size and checksum metadata before reporting success.
- Incomplete entries stay local and are deferred to a later cycle. Upload
  errors are logged and retried on subsequent cycles. Staging is cleaned on
  success/failure and before the next attempt; model caches are not removed.
- Normal controller shutdown stops producers and attempts a final flush on
  every host. A failed host does not skip healthy hosts' uploads. Per-host locks
  serialize final and periodic writeback.

The two destination settings are `cache.trainer_compile` and
`cache.inference_compile`. The optional `cache.inference_compile_seed` is an
additional restore source, not the upload destination. Both v4-64 and v5p-32
profiles use this same implementation with separate accelerator-specific GCS
prefixes. A remote entry does not guarantee a compiler cache hit when the
runtime, topology, or compilation signature differs.

## Observability and Limits

Per-host `writeback/cache-events.jsonl` records `compile_cache_published`,
including destination, verified object count and `deferred_incomplete` count.
Heartbeats expose `last_cache_sync` and `cache_sync_error`. A successful cycle
can have zero uploads or deferred entries; it does not mean every local file
is durable. Controller logs record retries and final-flush errors/timeouts.

This is periodic, best-effort writeback, not synchronous durability. Forced
process termination or abrupt VM loss can prevent the final flush and lose
entries generated since the last successful upload. Such entries must compile
again. LoRA/optimizer checkpoints, sampler state, and the client run directory
use the separate run/checkpoint persistence paths; they are not compilation
caches and must not depend on this mechanism.

## Local Checkpoint Retention

Downloaded base weights and compilation caches use tmpfs. Generated training
checkpoint archives still use disk under `runs/RUN_ID/checkpoints`, with the
trainer's synchronous, verified GCS mirror providing durability.

`checkpoint_retention.py` reclaims old local checkpoint replicas during host
preflight, before the 10 GiB free-disk check. Bootstrap records the exact run,
managed task and checkpoint GCS prefix in `.checkpoint-owner.json`, and holds
`.checkpoint-owner.lock` until services and final writebacks stop. Cleanup:

- Excludes the current run, held leases, live task/namespace processes, and
  processes whose ownership cannot be inspected safely.
- Only considers executor-owned `checkpoints/**/*.tar.gz`. It rejects symlink
  paths and hardlinked files. Unknown/legacy runs without ownership records are
  preserved; it does not infer ownership from a directory name.
- Verifies each candidate's size and checksum against the recorded GCS prefix,
  rechecks remote generations after hashing, and checks local inode/timestamps
  before unlinking. Missing, mismatched, changing or inaccessible copies stay.
- Never deletes GCS objects, client/PUCT state, databases, model/compile caches,
  or current-run checkpoints. Per-file audit events include GCS generation and
  reclaimed bytes. Metadata errors are logged and skipped, not deletion authority.

`checkpoint_cleanup_timeout` defaults to 600 seconds; `0` disables cleanup.
Cancellation/time-budget checks stop further deletion. An outstanding GCS
metadata call can take up to its existing 300-second timeout; preflight allows
that additional time. If disk is still insufficient, the ordinary preflight
refusal remains. This is startup retention, not eviction during active training;
a single long-running job can still need a separate active-checkpoint policy.

This behavior applies to newly built Ray v2 bundles on v4 and v5p. Existing
immutable job bundles are not patched or restarted, and legacy run directories
require an explicit ownership audit before any local checkpoint removal.

## CPU Tests

Use an isolated Python 3.12 environment with `pytest`, `ray[serve]==2.58.0`,
`httpx`, `jinja2`, `google-crc32c` and `zstandard==0.25.0`. No TPU allocation,
Ray cluster, or GCS writes are needed for these unit tests:

```bash
srun -p cpu --cpus-per-task=2 --mem=4G --time=00:10:00 \
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$TEST_PYTHON" -m pytest -q \
  tests/tpu_swarm/test_ray_train_cache.py \
  tests/tpu_swarm/test_ray_train_commands.py \
  tests/tpu_swarm/test_ray_train_writeback.py
```
