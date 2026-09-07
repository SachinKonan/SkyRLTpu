# Ray Training Trials

This is a separate implementation under `tpu/swarm/ray_train`, not a replacement
for the existing jobman cell launcher. All profiles use Qwen3.5-27B and generic
Ray Serve over independent per-host vLLM engines, not native vLLM Ray DP.

## Profiles

| Profile | Trainer | Inference | Pool |
| --- | --- | --- | --- |
| qwen_v5p_32.json | 1 host, TP1/FSDP4 | 3 hosts, TP4 each | tpuswarm-v5p32-east5a-erdos |
| qwen_v4_32.json | 1 host, TP1/FSDP4 | 3 hosts, TP4 each | tpuswarm-v4-32-central2-smoke |
| qwen_v4_64.json | 4 hosts, TP8/FSDP2 | 4 hosts, TP4 each | tpuswarm-v4-64-central2-qwen35-erdos |
| qwen_v4_64_retry.json | 4 hosts, TP8/FSDP2 | 4 hosts, TP4 each, s16/u0.80 | tpuswarm-v4-64-central2-qwen35-erdos |
| qwen_v4_32_inference.json | None | 4 hosts, TP4 each, s16/u0.80 | tpuswarm-v4-32-central2-smoke |

The inference-only profile starts no trainer, client, grading, or automatic
generation. It waits with all engines serving until stopped, with normal
health monitoring and compilation-cache writeback. The head also runs an
engine; `/health`, `/status` and `/v1/completions` are on head port 19800.
This is a bring-up test, not evidence that long generation is stable.

The active v4 training target is v4-64: four trainer hosts with TP8/FSDP2
and full rematerialization, plus four independent TP4 inference hosts. The
v4-32 profile is retained only as a failed reproduction: job 349 exhausted
HBM during initial LoRA template creation, before sampling or training. Do
not submit that profile for the v4 training comparison. The v4-64 topology selector is only
used by the eight-host profile. The four-host profiles isolate the trainer
to rank 0 and inference to ranks 1-3.

Defaults: full rematerialization, LoRA rank 32, sequence length 22528,
16 groups of 32 samples, learning rate 1.5e-4, 15 epochs, grader timeout 1100s.
The v5p-32 trainer overrides this with the proven 18432-token uniform rows
and a 73728-token budget (4 x 18432 for FSDP4). Its client context and training
limit are also 18432; the inference server still allows 22528. The trainer
compile cache uses a separate `s18432` prefix. This corrects job 353's
4 x 22528 warm-up HBM overflow despite base-state release and cache eviction.
The v4-64 trainer remains at 22528 with TP8/FSDP2. These profile edits require
a new bundle and workload restart; they do not change running processes.
Inference uses TP4 and 4096-token chunks. The current v5p-32 trial uses
32 sequences and 0.90 memory utilization; v4-64 uses 16 sequences and 0.87.
Job 348's v5p baseline (16 sequences) measured about 630 output tokens/sec
across three engines over a five-minute steady generation window. Job 350's
v4 inference compilation exceeded 30.75 GiB HBM by 21.82 MiB at utilization
0.90. The new settings are trials, not validated speed or memory-fit claims.
Both profiles retain their original run/checkpoint identity and seed the new
compilation-cache prefix from the prior configuration's cache.
CPU grading and serving share workload Ray on port 19679; SkyPilot's Ray is
separate. Never use global `ray stop` on a pool host.

## Build And Submit

Use the configured service-account gcloud configuration. Build only these new
files; the training/client source is a separate SHA256-pinned base archive.

```bash
python -m tpu.swarm.ray_train.build \
  tpu/swarm/ray_train/profiles/qwen_v5p_32.json \
  --output /shared/path/build --upload
sky jobs launch -p tpuswarm-v5p32-east5a-erdos \
  /shared/path/build/qwen-ray-v5p-32-001.yaml -y -d
```

Use this repo's `third_party/TPUSwarm/.venv/bin/sky`, the existing API at
`http://127.0.0.1:46580`, and the established SkyPilot config. Do not restart
the API or resize pools to submit these jobs. Inspect `sky jobs pool status
POOL -a` and match actual VM addresses to cluster handles, not an assumed
first idle worker: assignments can change between observations.

## Startup And Persistence

Each host acquires an ownership lock and joins a private workload Ray cluster.
Host actors perform preflight, restore complete role-specific model caches
into tmpfs, and install private environments. A cache barrier precedes
concurrent trainer and inference startup; the client starts only after both
services are ready. Existing valid model caches are reused and partial owned
cache restores are replaced. Unrelated disk caches are not deleted.

Logs are under `~/.cache/skyrl-ray/runs/RUN_ID/` on each TPU host. The head's
`controller.jsonl`, `driver.log`, `trainer.log` and `client.log` distinguish
setup, readiness, generation and actual optimizer progress. RUNNING in
SkyPilot alone does not prove training progress.

Run state is mirrored to `BUCKET/ray-training/RUN_ID`. Trainer checkpoints are
staged under the private local run directory; `checkpoint_mirror_gcs` uploads
and verifies each save synchronously before acknowledgment. Do not pass a
GCS URI as `--checkpoints-base` while enabling this local-file mirror: the
frozen backend converts the mirror source with `Path(str(output_path))`.
LoRA updates are streamed directly to the ingress, then committed only after
all engines acknowledge the adapter. Replacement engines fetch that adapter
before serving. Compilation caches are written back every 60 seconds and at
shutdown; concurrent create-only upload failures are accepted only when every
remote object matches the immutable local snapshot.

## Retired Legacy Workloads

The v4-64 retry profile uses a fresh run identity (`qwen-ray-v4-64-002`),
preserves the trainer shape, and lowers inference memory utilization to 0.80.
Job 355 was verified FAILED before its exact task ID was authorized for cleanup.
Two detached serving API roots and an EngineCore child still carried that ID.
Bootstrap now performs authorized retirement before checking service ports,
after acquiring worker ownership. Preflight also recognizes orphaned `VLLM::`
processes. Unknown processes still fail closed; there is no global vLLM kill.
An engine child is covered by an authorized ancestor, but a reparented engine
requires its own matching task ID or audited process identity. This retry is
not proof that job 355's fatal TPU runtime interrupts were transient.

Job 375 exposed another cleanup compatibility issue: Ray's bundled psutil
`wait_procs()` used `pidfd_open`, which returned EINVAL on the TPU kernel.
Retirement now polls process identity/status with bounded TERM/KILL waits,
without requiring pidfd support. The old job 355 processes were verified gone;
job 376 then passed cleanup and eight-host topology validation on worker 56.
Its executor bundle is `647d6fb40d62bc80b20704e5a511f57ef1600db73ae7edf601ae4e08e51ca23a`.
Regression coverage includes unsupported pidfd, unknown orphan refusal, cleanup
before port checks, and parent death cleanup preserving an unrelated process.

The v4-64 profile audits the leftover trainer processes from gradient-analysis
job 326 on pool worker 56. SkyPilot reported SUCCEEDED and the GCS result at
`gradient-conflict/20260906-qwen-v4-64-v1/result.json` was verified with
`complete: true` and `groups_completed: 6`. All five recognized trainer
processes matched release `1788722531793827-gradient-20260906-qwen-v4-64-v1-r3`.
Their boot IDs, PIDs, creation times and command hashes are checked only after
the new managed job owns the worker. No experiment files are removed.

Legacy successful cells intentionally leave services warm. This is not proof
of an active client, nor does an idle pool worker prove its TPU processes are
gone. Our default preflight refuses existing trainer/inference processes.

`retired.py` provides opt-in cleanup inside the newly allocated managed job.
An operator must first verify the old job is terminal and checkpointed. It
accepts exact retired SkyPilot task IDs, or a per-IP audited manifest containing
host boot ID, PID, process creation time and SHA256 of the exact command line.
The latter is necessary for legacy SSH launches without task IDs in their
environment. Unknown workload roots cause refusal before any signal on that
host. Only recognized roots and their descendants are terminated; generic
tmux-session kills, unrelated Ray cleanup and cache deletion are not used.

The v5p profile also audits job 329's completed Qwen services on worker 127.
SkyPilot reported SUCCEEDED, its client reported CONVERGED and final sync,
and both final checkpoint objects under `skyrl-checkpoints/model_86d158e9/`
were verified in the east5 bucket before authorizing retirement. Job 351
correctly refused these processes before this explicit audit was added.

The v5p profile's temporary manifest identifies job 271's completed Muse
processes on the four hosts of worker 148. The head's old cell worker had task
ID `sky-managed-2026-09-05-19-47-24-615848_meta-wt16-carry-g0-muse_271-0`.
Before retirement, both `model_9cbb280e/final.tar.gz` and
`model_9cbb280e/sampler_weights/final.tar.gz` were verified under the east5
bucket's `skyrl-checkpoints` prefix. Process identity checks cannot authorize
a replacement process with the same PID. A new host boot invalidates the
manifest; ordinary preflight checks still apply.

## Diagnosed Failures

- 338: the pinned MaxText source contains two matching return blocks, one in
  each transformer implementation. The FLCE patch incorrectly required one.
  It now handles both and remains idempotent.
- 337: Rich wrapped the trainer readiness message. Exact string matching
  missed a ready trainer; the matcher now accepts whitespace wrapping within
  the current owned process's log segment. After an in-place readiness repair,
  the first adapter upload exposed a second bug: Ray's FastAPI wrapper treated
  postponed `Request` annotations as a required query parameter, returning
  HTTP 422. Concrete annotations fix this; a test exercises the actual frozen
  ASGI application rather than calling handlers directly.
- 339: preflight refused an older Gemma engine on worker 142. That worker was
  subsequently assigned to another agent's job 340 and was not cleaned by us.
- 343: preflight refused completed job 271's Muse engines on worker 148.
- 344: audited cleanup succeeded on all four hosts; trainer and all engines
  became ready. The initial 638 MB adapter was acknowledged by all three engines
  in 9.75 seconds, proving the HTTP fix. The following checkpoint mirror failed
  because the launcher passed a GCS URI where a local staging file was required.
  The command now uses a local staging path plus synchronous GCS mirroring.
- 348 (v5p-32): the initial adapter reached all three engines and the initial
  sampler checkpoint was independently verified in GCS. Generation requests
  started; this observation alone does not establish an optimizer step.
- 349 (v4-32): services started, but initial LoRA template creation failed
  with HBM allocation exhaustion. The user clarified that this comparison
  requires v4-64, not a single-host v4 trainer. Keep the v5p trial independent.

These are setup and integration fixes, not proof of a completed training step.
Verify initial adapter upload, generation/grading, forward/backward, optimizer
step, and durable checkpoint before declaring either profile working.

## Tests

Run the five `tests/tpu_swarm/test_ray_train_*.py` files on `srun -p cpu`, using
the shared `~/.cache/skyrl-ray-tests/venv` environment. Avoid viz-node `/tmp`
paths as Slurm test inputs: `/tmp` is node-local. Tests mock TPU execution;
they do not establish device memory fit or training correctness.
