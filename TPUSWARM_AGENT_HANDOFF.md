# TPUSwarm / SkyPilot operational handoff

Status snapshot: 2026-09-04 11:03 EDT. Live status is volatile; refresh it
before making any decision. This document describes the current dirty
worktree and the long-lived controller state on the Princeton visualization
node.

## Read this first

1. Do not restart, stop, signal, or replace the SkyPilot API server, a pool
   controller, TPUSwarm, or any TPU VM while jobs or queued resources are live.
2. Do not delete a queued resource by name alone. First prove the embedded
   `skypilot-user` label is ours, map it to a SkyPilot pool worker, and confirm
   that it has no live TPU node. Never delete another user's resource.
3. Do not run `git reset`, `git checkout --`, submodule reset/update, or any
   cleanup command in this checkout. The parent and several submodules contain
   intentional, uncommitted production changes.
4. Do not reapply a pool YAML merely because a CLI call timed out. Inspect the
   pool, the GCP queued resource, and the TPU VM first. A duplicate request can
   lose queue position or consume additional quota.
5. A SkyPilot `RUNNING` job may still be restoring caches or compiling vLLM.
   It does not mean the first rollout or trainer step has happened.
6. Observe at five-minute intervals unless actively debugging a concrete
   failure. Avoid polling loops that overload the API server.

Read these documents and script headers before changing runtime behavior:

- `TPUSWARM_V6E32_HANDOFF.md`: original Asia v6e-32 design and identity issue.
- `tpu/swarm/README.md`: TPUSwarm task and pool concepts.
- `tpu/jobman/RUNBOOK.md`: the current cell, venv, model, and cache contract.
- `tpu/start_vllm_tpu.sh`, `tpu/jobman/cell_worker.sh`, and
  `tpu/jobman/cell_monitor.sh`: actual worker setup and supervision.

## Checkout state

The active branch and recorded submodule commits are:

```text
parent branch                         agent/tunix-multihost
parent HEAD                           7d4b4208235113a216e9db1b1e74e591254b7585
third_party/TPUSwarm                  a039683c33ff4333c06d9275224e98ccffbaa5f5
TPUSwarm/third_party/skypilot         8260324adbd8a3d2bf0cb2af332c4e363b4e2e71
third_party/discover                  c95227a36111233acf7c0691bda5dddbc5cc7b1f
third_party/jobman                    a7b770e7d3b9107dd75db1fa15018c84b43293c9
third_party/tpu-inference             afe0cb9e9bf259a072242c6f3279d92b702f9f2a
```

These hashes are not the whole deployed source. The parent worktree, nested
SkyPilot checkout, Discover, and TPU inference checkout are dirty. Run
`git status --short` in each checkout and preserve all existing changes. GCS
worker bundles are the code actually running on TPU VMs; a bundle URL plus its
object generation and `TPUSWARM_BUNDLE_ID` identify a deployment more reliably
than the parent Git SHA alone.

The current v4-64 task bundle is:

```text
gs://sk7524-tinker-tpu-us-central2/code-bundles/tpuswarm-skyrl-v4-mixed-v31.tar.gz
generation: 1788531765925921
sha256: dcf9058e68058e20339cace274e7d0f9cd2fa49515e304680f3bee7f86f55eb8
```

## Controller environment

Use this environment for every `sky` command. The isolated `HOME` contains the
live SkyPilot database, pool state, logs, generated SSH key, and client ID.

```bash
cd /scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-multihost

export CLOUDSDK_CONFIG=/home/sk7524/.config/gcloud-tpuswarm-compute-sa-v6e32
export GOOGLE_APPLICATION_CREDENTIALS=/home/sk7524/.config/gcloud/vision-mix-compute-sa-key.json
export HOME=/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/sky-home-v6e32
export SKYPILOT_CONFIG=/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/skypilot-config.yaml
export SKY_API_SERVER_URL=http://127.0.0.1:46580
export SKY_API_SERVER_ENDPOINT=http://127.0.0.1:46580
export SKYPILOT_API_SERVER_ENDPOINT=http://127.0.0.1:46580
export SKYPILOT_DISABLE_LOCAL_API_SERVER=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SKY="$PWD/third_party/TPUSwarm/.venv/bin/sky"
```

The one authorized identity is
`289186856710-compute@developer.gserviceaccount.com` in project `vision-mix`.
Both gcloud and ADC must resolve to that account. ADC refresh requires the
cloud-platform scope; an unscoped `google.auth.default()` refresh can report
`invalid_scope` even with the correct key.

```bash
gcloud auth list --filter=status:ACTIVE --format='value(account)'
gcloud config get-value project

third_party/TPUSWARM/.venv/bin/python - <<'PY'
import google.auth
from google.auth.transport.requests import Request

creds, project = google.auth.default(
    scopes=['https://www.googleapis.com/auth/cloud-platform'])
creds.refresh(Request())
print(getattr(creds, 'service_account_email', None), project)
PY
```

Do not print the key JSON, bearer token, `HF_TOKEN`, or other secrets. Do not
bypass SkyPilot credential checks. The service account has already been used
successfully for TPU and regional GCS access, but recheck before a fresh server
start.

## The three control planes

Keep these states separate in every status report:

1. **SkyPilot API and pool controller:** the API on port 46580 stores requests
   under `$HOME/.sky/api_server/requests.db`. Each pool is implemented by a
   long-lived `sky.serve.service` process. SkyPilot's own Ray cluster on each
   TPU slice is part of this control plane.
2. **SkyPilot Job Pool worker:** one reusable worker is one whole TPU pod slice,
   not one TPU VM. A worker moves through provisioning, setup, ready, used,
   preempted, and cleanup states. Historical failed rows remain visible and do
   not equal current GCP usage.
3. **GCP:** each provisioning worker owns a queued-resource wrapper. That
   wrapper may exist in `WAITING_FOR_RESOURCES` or `PROVISIONING` before a TPU
   VM node exists. Treat queued-resource and TPU VM state as separate facts.

TPUSwarm is a fourth, thin application scheduler on port 8787. It stores stable
task IDs, admission/recovery-reserve policy, and mappings to SkyPilot jobs in
`/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/tpuswarm.db`. SkyPilot continues
running and recovering already-submitted jobs if TPUSwarm is temporarily down.
Most recent v4-32/v4-64 experiments were submitted directly to SkyPilot pools;
they do not depend on TPUSwarm's HTTP process after submission.

## Running services and restart rules

At the snapshot time:

- SkyPilot API PID 97684 has run since 2026-09-02 13:36 EDT on `127.0.0.1:46580`.
- TPUSwarm PID 3897908 has run since 2026-08-31 21:58 EDT on `127.0.0.1:8787`.
- TPUSwarm health reports leader `controller-946af44a-cbdd-4b65-87b0-c6a79600c981`.
- Pool service processes are long-lived children adopted by PID 1.

Inspect before starting anything:

```bash
ps -eo pid,lstart,args | rg 'sky.server.server|sky.serve.service|tpuswarm'
curl -fsS http://127.0.0.1:8787/healthz
```

If the API process is healthy, use it. Do not restart it to pick up source
changes: request executors use multiprocessing `spawn`, so newly launched or
retried requests import the current source from disk. Existing service
controllers must remain alive to preserve active reservations.

Only if the API is confirmed absent, start it with the complete environment
above and the same durable `HOME`:

```bash
nohup third_party/TPUSwarm/.venv/bin/python -m sky.server.server \
  --host=127.0.0.1 --port=46580 \
  >/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/sky-api.log 2>&1 &
```

Only if TPUSwarm is confirmed absent, restore its token from the durable token
file without displaying it and start against the same database:

```bash
export TPUSWARM_TOKEN="$(</scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/token)"
export TPUSWARM_SERVER=http://127.0.0.1:8787
nohup third_party/TPUSwarm/.venv/bin/tpuswarm --log-level INFO serve \
  --database /scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/tpuswarm.db \
  --registry-module skyrl.tpu_swarm --host 127.0.0.1 --port 8787 \
  >/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/server.log 2>&1 &
```

The SkyPilot web dashboard is not currently usable in this editable checkout:
its built `dashboard/out/index.html` is absent. Use the CLI. From a laptop, an
SSH tunnel can expose the two loopback APIs, but do not bind them publicly:

```bash
ssh -N -L 46580:127.0.0.1:46580 -L 8787:127.0.0.1:8787 USER@VIZ_HOST
```

## What a pool is

A pool is a desired number of reusable, preconfigured TPU pod slices. The pool
controller continuously attempts to reach the target and binds queued jobs to
ready workers. Jobs can be submitted before all workers are ready; they remain
`PENDING` until a compatible worker is available. Do not wait for an entire
pool to become ready before submitting a test job.

Slice sizes used here:

- `tpu-v4-32`: four TPU VMs, 16 chips total.
- `tpu-v4-64`: eight TPU VMs, 32 chips total.
- `tpu-v5p-32`: one complete four-host, 32-chip slice.
- `tpu-v6e-32`: eight TPU VM ranks, 32 chips total.

The first number in output such as `2/46` is ready workers. The denominator can
include historical or extra lifecycle rows and is not the configured target.
Read the `Autoscaling from N to N workers` text for the target, then inspect the
verbose worker table.

## Live pools at the snapshot

| Pool | Zone / slice | Target | Snapshot | Purpose |
| --- | --- | ---: | --- | --- |
| `tpuswarm-v6e32-asia-qwen35` | `asia-northeast1-b`, v6e-32 | 40 | 0 ready; one starting and several provisioning | Primary Asia Qwen pool |
| `tpuswarm-v6e32-east5b-qwen35` | `us-east5-b`, v6e-32 | 32 | 0 ready; one live provisioning row | East v6e saturation |
| `tpuswarm-v6e32-europe-w4a-qwen35` | `europe-west4-a`, v6e-32 | 2 | 0 ready; one provisioning | Europe v6e saturation |
| `tpuswarm-v5p32-east5a-erdos` | `us-east5-a`, v5p-32 | 45 | 2 ready; most new requests quota-blocked | Erdős / quota saturation |
| `tpuswarm-v4-32-central2-smoke` | `us-central2-b`, v4-32 | 10 | 9 ready; seven jobs running, one recovering | Train/inference smokes |
| `tpuswarm-v4-64-central2-qwen35-erdos` | `us-central2-b`, v4-64 | 6 | 2 ready and used by Jobs 113/114; four provisioning | Mixed GRPO |
| `tpuswarm-v4-64-central2-quota-fill` | `us-central2-b`, v4-64 | 1 | no ready worker | Extra quota request |
| `tpuswarm-v4-64-central2-quota-fill-2` | `us-central2-b`, v4-64 | 1 | one provisioning | Extra quota request |

The v5p preemptible per-zone project limit is 1536 chips. Forty-five v5p-32
slices is only a theoretical maximum after accounting for 80 chips observed in
other active slices. The TPU API has returned
`TPUV5PPreemptiblePerProjectPerZoneForTPUAPI exhausted`; queued and suspended
requests from other users also consume shared quota. A displayed quota usage
of zero in a generic gcloud quota view is not authoritative for TPU queued
resources.

## Inspecting pools and GCP

```bash
$SKY jobs pool status -a
$SKY jobs pool status tpuswarm-v4-64-central2-qwen35-erdos -a -v
$SKY jobs queue -a

gcloud alpha compute tpus queued-resources list \
  --project=vision-mix --zone=us-central2-b \
  --format='table(name,state.state,tpu.nodeSpec[0].nodeId)'

gcloud compute tpus tpu-vm list \
  --project=vision-mix --zone=us-central2-b \
  --format='table(name,state,acceleratorType,networkEndpoints[0].ipAddress)'
```

Repeat the GCP commands for all zones in use:
`asia-northeast1-b`, `europe-west4-a`, `us-central2-b`, `us-east1-d`,
`us-east5-a`, and `us-east5-b`.

For a worker, inspect its controller log before touching GCP:

```bash
$SKY jobs pool logs --no-follow --tail 300 POOL_NAME WORKER_ID
```

To change only the desired size of an existing pool without replacing setup or
workers:

```bash
$SKY jobs pool apply -p POOL_NAME --workers N -y
```

Then update the intended YAML or leave a dated note; otherwise a later full
apply can restore an old size. Full YAML applies create a new pool version and
can replace workers when setup/resources differ. Never run one casually on a
live pool.

## Provisioning sequence

1. `sky jobs pool apply` creates or updates a local pool service.
2. The service launches one replica request per missing slice.
3. The GCP provisioner creates a labeled queued-resource wrapper and waits
   indefinitely. Retries adopt the same nonterminal wrapper for the cluster so
   the request keeps its spot queue position.
4. A waiting wrapper may have no TPU VM yet. SkyPilot now reports this as
   initializing instead of treating `all([])` as complete preemption.
5. When GCP allocates the slice, SkyPilot discovers every TPU VM, installs its
   control-plane Ray runtime, and places the pod SSH key on rank 0.
6. Pool `setup` runs on every VM. It resolves the GCS bundle generation,
   installs it under `~/.cache/tpuswarm/bundles/$generation`, atomically points
   `~/SkyRLTpu-tpuswarm` to it, and validates the expected node count.
7. Pool setup retries forever at 60-second intervals by default. A transient
   apt/GCS/archive failure therefore retains the allocated TPU. Inspect the log
   if the worker stays in setup; an infinite retry also preserves permanent
   configuration errors until fixed.
8. Once setup/probing succeeds, the worker becomes `READY`. A pending job is
   bound to it and the job's `run` stanza executes on every TPU VM.

## Submitting jobs

Direct SkyPilot pool submission is the current path for v4 experiments:

```bash
$SKY jobs launch --pool tpuswarm-v4-64-central2-qwen35-erdos \
  tpu/swarm/examples/v4-64-qwen35-grpo-erdos.yaml -d -y

$SKY jobs launch --pool tpuswarm-v4-64-central2-qwen35-erdos \
  tpu/swarm/examples/v4-64-qwen35-grpo-erdos-tp8-fsdp2.yaml -d -y
```

`-d` detaches. If submission output is lost or times out, run
`$SKY jobs queue -a` and search by YAML `name` before retrying. Before launching
a second experiment, give it a unique `name`, `RUN_DIR_NAME`,
`EXPERIMENT_NAME`, `GCS_RUN`, topology-specific `TPUSWARM_BUNDLE_ID`, and cache
prefixes where compilation shapes differ. Reusing a `GCS_RUN` means resume, not
a clean independent run.

The original typed TPUSwarm submission remains available for v6e Qwen:

```bash
export TPUSWARM_SERVER=http://127.0.0.1:8787
export TPUSWARM_TOKEN="$(</scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/token)"
third_party/TPUSWARM/.venv/bin/python tpu/swarm/submit_qwen35_grpo.py \
  --server "$TPUSWARM_SERVER" \
  --task-id qwen35-v6e32-grpo-001 \
  --run-dir qwen35-v6e32-grpo-001
```

The task ID is its idempotency key. Reuse it after a client timeout. TPUSwarm
resource policy can hold one warm worker as a recovery reserve; for a five
worker resource, `target_workers=5,recovery_reserve=1` admits four ordinary
jobs. Do not assume this admission policy applies to jobs launched directly by
`sky jobs launch`.

## v4-64 mixed job sequence

The v4-64 task uses one eight-VM slice:

- Four physically adjacent hosts form the 16-chip trainer.
- Four hosts run four independent TP4 vLLM engines.
- The standard arm is trainer TP4/FSDP4 with a 90,112-token batch.
- The comparison arm is trainer TP8/FSDP2 with full rematerialization and a
  45,056-token batch.
- Both use Qwen3.5-27B, LoRA rank 32, 22,528-token rows, 16 groups x 32
  rollouts, GRPO mean baseline, and the Erdős minimum-overlap environment.

SkyPilot rank order is not physical v4 topology order and changes after a spot
recreation. `run_qwen35_v4_64_grpo.sh` makes rank 0 the head, probes all eight
hosts, selects the physical four-host row containing rank 0 for training,
validates the subset mesh, and gives the other row to inference. Never hardcode
`TRAIN_WORKERS=0,1,2,3` for v4-64.

After topology selection, role setup runs concurrently:

1. `reconcile_v4_64_role_caches.sh` stops only stale opposite-role processes.
2. Trainer hosts remove HF weight payloads but retain/restage tokenizer and
   config metadata, retain or restore the complete Orbax checkpoint, and
   restore the topology-specific trainer JAX cache.
3. Inference hosts remove the MaxText/Orbax model, retain or restore the full
   55.6 GB HF snapshot, and restore the vLLM XLA/JAX cache.
4. `cell_worker.sh` starts vLLM engines and the local SkyRL Tinker-compatible
   trainer asynchronously, then starts the isolated grader Ray cluster.
5. `cell_monitor.sh` waits for every engine, trainer, and grader worker before
   starting the Discover client. It periodically verifies identity/health and
   syncs durable run state and executable caches to GCS.

The local Tinker API is intentionally unauthenticated; upstream client code
still requires a nonempty `TINKER_API_KEY`, so the task uses the explicit dummy
value `tml-local-skyrl-no-auth`. This is not a cloud secret.

Current venv split:

- Serving: `~/.venvs/vllm-tpu`, vLLM TPU 0.23.0 plus the forked TPU inference
  overlay. Qwen uses transformers 5.8.0.
- Trainer: SkyRL Tinker server plus JAX/Tunix/MaxText on the training hosts.
- Client: the bundled `third_party/discover` environment on rank 0.
- Grader: `~/.venvs/grader`, with a private Ray cluster distinct from
  SkyPilot's Ray.
- Topology probe: `~/.venvs/tpu-topology`, pinned to JAX/JAXlib 0.11.1,
  libtpu 0.0.46, and Python 3.12.

## Current v4-64 jobs

At the snapshot, the fixes for old Jobs 70 and 74 are deployed as replacements:

| Job | Name | Worker | State | Current phase |
| ---: | --- | ---: | --- | --- |
| 113 | `qwen35-v4-64-grpo-erdos-tp8-fsdp2-002` | 38 | RUNNING, 0 recoveries | caches reconciled; v31 engine bring-up / precompile |
| 114 | `qwen35-v4-64-grpo-erdos-005` | 41 | RUNNING, 0 recoveries | HF/cache setup completed or completing; engine bring-up |

Old Job 70 failed and old Job 74 exhausted recovery; they are terminal and
must not be confused with the replacements. A log such as `trainer bundle
mismatch: running=<old> expected=<v31 id>` is the supervisor detecting a stale
session and restarting it. It is only benign if followed by new engine/trainer
bring-up and improving health.

Monitor every five minutes:

```bash
$SKY jobs queue -a
$SKY jobs logs 113 --no-follow --tail 500
$SKY jobs logs 114 --no-follow --tail 500
$SKY jobs pool status tpuswarm-v4-64-central2-qwen35-erdos -a -v
```

Look for, in order: topology selected, role caches reconciled, trainer
checkpoint complete, all vLLM engines serving `/v1/models`, trainer UP, grader
Ray ready, client rollouts, and the first trainer forward/backward/update. Long
HF restores and JAX precompiles can take 30-60 minutes on a fresh host.

## Important SkyPilot changes in this checkout

The nested fork at `third_party/TPUSWARM/third_party/skypilot` has uncommitted
changes and tests. Major behavior changes are:

- Pure TPU queued resources skip generic GCP Compute VM IAM/network bootstrap
  unless explicit network/IP configuration requires it. Controllers run in
  local consolidation mode; no CPU controller VM is requested.
- `gcp.tpu_queued_resource_timeout_seconds: infinite` is plumbed through the
  schema/template/provisioner. Capacity waits do not time out and give up queue
  position.
- Retries adopt an existing same-cluster nonterminal queued resource rather
  than creating a replacement.
- A queued resource with no TPU node is initializing, not evidence that all
  nodes were preempted. The prior `all([]) == True` path no longer deletes a
  valid waiting request.
- `FAILED`, `SUSPENDING`, `SUSPENDED`, and `DELETING` wrappers are reconciled as
  terminal for their exact cluster. Actual preempted TPU nodes are cleaned up.
- On a queued-resource create/readiness failure, SkyPilot performs a best-effort
  sweep of `FAILED`/`SUSPENDING`/`SUSPENDED` wrappers only in the same zone and
  only when their embedded TPU node labels contain the current `skypilot-user`.
  There is deliberately no name fallback. A create is retried once only when
  that sweep deleted something.
- Resources created by the current failed attempt are cleaned exactly. Existing
  adopted requests are preserved.
- API interruption and setup-in-progress are distinguished from true replica
  failure, preventing a controller restart race from tearing down a live
  provisioning request.
- Pool readiness is tri-state so setup still in progress does not consume the
  readiness timeout. Replica IDs resume above persisted IDs.
- The TPU pod SSH key is copied to rank 0 so head-driven scripts can reach all
  other TPU VMs.

The live config at
`/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/skypilot-config.yaml` is:

```yaml
allowed_clouds:
  - gcp
jobs:
  controller:
    consolidation_mode: true
gcp:
  force_enable_external_ips: true
  tpu_queued_resource_timeout_seconds: infinite
```

## Important repo changes

- Added regional and accelerator-specific pool YAMLs for Asia/east/europe
  v6e-32, east5a v5p-32, central2 v4-32, and central2 v4-64.
- Pool setup installs immutable GCS bundle generations and retries setup
  indefinitely so apt/GCS/archive errors do not discard allocated TPUs.
- Added v4-64 physical topology discovery and role assignment.
- Added role-aware cache reconciliation, size/completeness checks for Orbax and
  HF snapshots, resumable single-process GCS transfers, incomplete-file purge,
  HF weight pruning on trainer hosts, and HF metadata restaging.
- Added bundle identity checks so a stale trainer/vLLM process from an older
  task is restarted before the client is launched.
- Added separate `grader_ray.sh` and `vllm_ray.sh` control paths. They identify
  processes by private temp root and exact GCS address and must not touch
  SkyPilot Ray.
- Removed fixed grader Ray runtime-env/metrics ports that collided with
  SkyPilot. Grader startup now retries three times and health is checked with
  `ray health-check` plus exact raylet membership.
- Hardened vLLM startup, offline HF cache validation, LoRA adapter loading,
  request timeouts, JAX/XLA cache restore/writeback, and engine identity checks.
- Hardened trainer/checkpoint setup, monitor behavior, and external inference
  adapter coordination. Added focused tests under `tests/tpu_swarm/` and the
  relevant backend/Tinker test directories.

The fixed Ray incident was not caused by SkyPilot using Ray in general. Old
Job 70 started a second grader Ray with a fixed runtime-env port already used
by SkyPilot, producing `OSError: [Errno 98] Address already in use`. Also, a
Ray placement result list was incorrectly treated as logical TPU rank order.
The grader now has isolated ports/health checks, and v4 host roles come from
physical topology probes instead of result-list order.

Never run bare `ray stop` on a pool worker. It can kill SkyPilot's control-plane
Ray, make a healthy TPU look unreachable, and trigger teardown. Use only the
role-specific scripts and exact private cluster address.

## Suspended resources and quota cleanup

We previously removed verified, same-user, no-node terminal wrappers in Asia,
east5b, central2, and east5a. The automatic owner-scoped cleanup behavior is now
in the provisioner for `FAILED`, `SUSPENDING`, and `SUSPENDED`. It activates only
on a provisioning failure and only for embedded owner labels; it does not
continuously sweep the project.

Before any manual deletion, capture JSON for the candidate and answer all of:

1. Is the state terminal (`FAILED`, `SUSPENDING`, or `SUSPENDED`)?
2. Does `tpu.nodeSpec[*].node.labels.skypilot-user` equal our SkyPilot client?
3. Is there no live TPU VM for every referenced node ID?
4. Does no current pool worker/controller map to it as a live reservation?

On 2026-09-04 the user explicitly authorized deleting the unlabeled terminal
wrappers `sk7524-stageb-m-lr-n_1` and
`sk7524-meta-wt16-fresh-gemma_1_1`; both were verified to have no TPU node and
were deleted. Name matching remains disallowed for automatic cleanup.

## Tests and validation

Run substantial CPU tests through Slurm unless the user explicitly confirms
this visualization node may be used. The user has permitted local viz-node
tests during this incident, but `srun -p cpu` remains the default.

Validated during this work:

- Nested SkyPilot focused GCP queued-resource, GCP cloud, serve autoscaler,
  implementation, and replica-manager tests.
- `tests/tpu_swarm/test_grader_ray.py` and
  `tests/tpu_swarm/test_v4_64_grpo_config.py` (16 tests passed together).
- Python compilation, shell syntax, YAPF, and `git diff --check` for touched
  paths.
- Live grader Ray startup and health on retained worker 38 while SkyPilot Ray
  remained healthy.

The broad nested SkyPilot test harness may fail during collection because the
local editable environment lacks optional `boto3`; use its focused GCP tests
with `--confcutdir=tests/unit_tests` or install the intended test extras. Do not
misreport that dependency-collection failure as a product regression.

## Immediate continuation point

1. Keep Jobs 113 and 114 and their workers alive. Check them every five minutes
   until both reach client rollouts and a first successful trainer update.
2. Keep all existing queued/provisioning requests. Do not restart the API to
   propagate patches; fresh spawned executors already load them.
3. Watch v5p-32 target 45. Current 429s are shared TPU quota pressure, not proof
   that the pool controller stopped retrying.
4. Reconcile every status report across SkyPilot pool workers, GCP queued
   resources, and TPU VMs. State explicitly which layer is waiting or failing.
5. Before editing runtime setup, inspect the active bundle generation and the
   TPU-side logs. A local source edit does nothing to an already downloaded
   bundle until a new GCS bundle is published and referenced.
