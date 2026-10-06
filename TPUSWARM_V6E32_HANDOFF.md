# v6e-32 Qwen TPUSwarm handoff

Status captured on 2026-08-31 before moving off the current Neuronic login
node. This is the operational handoff for the mixed trainer/inference
Qwen3.5-27B GRPO pool. [`PLAN.md`](PLAN.md) contains the longer implementation
history and model-runtime investigation.

## Stop point

- Implementation branch: `agent/tunix-multihost`
- Published implementation head before this handoff document:
  `bfce189b2f446f5e526b6589222b1f2781daf5f7`
- The branch and all commits below are pushed to
  `origin/agent/tunix-multihost`.
- The one-time Asia Qwen Hugging Face cache is complete.
- The Asia Orbax/MaxText checkpoint cache and the pinned worker bundle exist.
- The trainer JAX cache, vLLM XLA cache, and training run prefix do not have
  objects yet; they are intentionally populated on the first real pool/run.
- No TPUSwarm pool was created, no TPUSwarm task was submitted, and no
  queued resource owned by this pool exists.
- The local SkyPilot API server was stopped cleanly at handoff.
- A TPUSwarm server/database/token was never started or created.
- Existing `taiming-tpuhosta-*` queued resources and old suspended experiment
  resources in `asia-northeast1-b` are unrelated. Do not delete them as part of
  this work.

The immediate blocker is GCP identity consistency, documented below. Do not
run `sky jobs pool apply` until the SkyPilot API process uses one identity for
both `gcloud` and Application Default Credentials (ADC), and that identity has
the permissions required by the chosen TPU-only SkyPilot path.

## What was implemented

Relevant parent commits, oldest to newest:

- `0a88ea46`: add the TPUSwarm runtime submodule.
- `724f6af9`: add the initial SkyRL task adapters.
- `cbc866f7`: add multi-host Tunix v6e training and exact resume.
- `b0bfc48c`: add the five-worker v6e-32 Qwen GRPO pool, task adapter, worker
  preparation, bundle builder, and submit helper.
- `2e6b616c`: fix the Slurm Qwen cache snapshot shell variable.
- `0154948f`: make pool setup wait for the regional model-cache completion
  marker, allowing capacity requests and cache construction to overlap safely.
- `bfce189b`: remove the memory-heavy Xet fast path from cache staging and
  document the successful 32 GiB CPU allocation.

Pinned submodules at handoff:

| Component | Commit | Notes |
| --- | --- | --- |
| TPUSwarm | `a039683c33ff4333c06d9275224e98ccffbaa5f5` | Published `main` |
| SkyPilot fork | `8260324adbd8a3d2bf0cb2af332c4e363b4e2e71` | Queued-resource terminal-state cleanup and Managed Job recovery |
| Discover | `c95227a36111233acf7c0691bda5dddbc5cc7b1f` | Published `agent/tunix-multihost` |
| Jobman | `a7b770e7d3b9107dd75db1fa15018c84b43293c9` | Published `agent/resumable-autorestart` |

After the initial handoff, the exact pinned Discover and Jobman tips were
published to the branches shown above. A clean `git clone
--recurse-submodules --branch agent/tunix-multihost` was then verified through
all parent and nested submodules.

The main new operational files are:

- [`tpu/swarm/examples/v6e32-qwen35-grpo-pool.yaml`](tpu/swarm/examples/v6e32-qwen35-grpo-pool.yaml)
- [`tpu/swarm/examples/qwen35-v6e32-grpo-task.json`](tpu/swarm/examples/qwen35-v6e32-grpo-task.json)
- [`skyrl/tpu_swarm/tasks.py`](skyrl/tpu_swarm/tasks.py)
- [`tpu/swarm/prepare_qwen35_v6e32.sh`](tpu/swarm/prepare_qwen35_v6e32.sh)
- [`tpu/swarm/run_qwen35_v6e32_grpo.sh`](tpu/swarm/run_qwen35_v6e32_grpo.sh)
- [`tpu/swarm/submit_qwen35_grpo.py`](tpu/swarm/submit_qwen35_grpo.py)
- [`tpu/swarm/build_skyrl_bundle.sh`](tpu/swarm/build_skyrl_bundle.sh)
- [`tpu/swarm/stage_qwen35_hf_cache.sh`](tpu/swarm/stage_qwen35_hf_cache.sh)
- [`tpu/swarm/README.md`](tpu/swarm/README.md)

The SkyPilot fork treats GCP queued-resource `FAILED`, `SUSPENDING`,
`SUSPENDED`, and `DELETING` states as terminal, deletes pool-owned stale
wrappers and TPU nodes, handles partial multi-node failures, and lets Managed
Jobs continue recovery. TPUSwarm pins that exact fork revision both as a
nested submodule and in its dependency metadata.

## Intended topology and training configuration

Region and zone are fixed to `asia-northeast1` / `asia-northeast1-b`.
Historical availability made this the best observed quota island: the snapshot
used for the decision was Asia at 128/1536 chips used (8%), `us-east5-b` at
344/1536 (22%), and `us-central1-b` at 1140/1536 (74%). Asia also produced the
last accepted v6e model runs while east repeatedly returned capacity code 8.

One SkyPilot pool worker is one complete `tpu-v6e-32` slice:

- 32 chips total;
- four physical 8-chip hosts;
- eight 4-chip TPU VMs/process ranks;
- ranks 0-3 are the 16-chip trainer, sharded `TP=8 x FSDP=2`;
- ranks 4-7 are four independent `TP=4` vLLM engines, one per TPU VM;
- the two inference physical hosts therefore each run two engines.

The configured pool has exactly five workers. This means five v6e-32 slices,
20 physical hosts, 40 TPU VMs, and 160 chips when fully allocated. TPUSwarm
admits at most four ordinary tasks and holds one already-warm slice as the
recovery reserve.

The Qwen run is pinned to:

- model: `Qwen/Qwen3.5-27B`;
- Hugging Face snapshot:
  `fc05daec18b0a78c049392ed2e771dde82bdf654`;
- LoRA rank 32;
- learning rate `1.5e-4`;
- context and maximum target length 22,528;
- Tunix token budget 45,056, or two 22,528-token rows per compiled trainer
  call;
- GRPO mean baseline with 16 groups x 32 rollouts;
- four asynchronous, client-round-robin inference engines streaming data to
  the trainer;
- fixed single-zone `FAILOVER` recovery;
- three restarts for ordinary nonzero application exits;
- unlimited infrastructure/preemption recovery and unlimited recovery for
  explicit exit codes 33/34 without consuming those three application
  restarts.

## Durable GCS state

All production objects are in the same Asia bucket to avoid repeated
cross-region transfer:

| Purpose | URI | State at handoff |
| --- | --- | --- |
| HF cache | `gs://sk7524-tinker-tpu-asia-northeast1/hf-cache-qwen35-v1` | Complete, 55,586,173,563 bytes |
| HF marker | `gs://sk7524-tinker-tpu-asia-northeast1/hf-cache-qwen35-v1/HF_CACHE_COMPLETE` | Generation `1788220190768279`; marker records snapshot above and `seeded_at=2026-08-31T23:49:48Z` |
| Orbax/MaxText | `gs://sk7524-tinker-tpu-asia-northeast1/skyrl-maxtext-ckpts/qwen3.5-27b` | Present, 41,892,404,608 bytes |
| Worker bundle | `gs://sk7524-tinker-tpu-asia-northeast1/code-bundles/tpuswarm-skyrl-qwen35-v6e32-v1.tar.gz` | Present, 93,600,556 bytes, generation `1788219843656988` |
| Trainer JAX cache | `gs://sk7524-tinker-tpu-asia-northeast1/jax-compile-cache-v6e-qwen35-tp8-fsdp2-r32-s22528-b45056-v1` | Empty; first trainer run populates it |
| vLLM XLA cache | `gs://sk7524-tinker-tpu-asia-northeast1/vllm-xla-cache-v6e-qwen35-tp4-s22528-v1` | Empty; pool preparation populates it |
| Training run | `gs://sk7524-tinker-tpu-asia-northeast1/skyrl-runs/qwen35-v6e32-grpo-001` | Empty; no task submitted |

The recorded SHA-256 of worker-bundle generation `1788219843656988` is
`d7d087223072922262537381e8f5769f1a556c35d0adde9999e3a7bd9959fcd3`.
It was built from clean parent commit `0154948f`. The later `bfce189b` change is
only to the Neuronic cache-seeding utility and its documentation; TPU workers
do not invoke that utility, so do not pay another cross-region bundle upload
solely for that change.

Pool setup resolves and records the GCS object generation, installs the bundle
under a generation-specific directory, waits for `HF_CACHE_COMPLETE`, restores
the HF and Orbax data, and prewarms the empty executable-cache prefixes.

## Identity issue: what was actually being used

Two different service accounts were accidentally mixed in the last local
SkyPilot API process.

### Cloud SDK identity

The normal Neuronic `gcloud` configuration reports:

```text
289186856710-compute@developer.gserviceaccount.com
project: vision-mix
```

This is the Neuronic VM's attached/default Compute service account. It was
confirmed to be able to list queued TPU resources in `asia-northeast1-b` and
read the Asia cache bucket. This is the identity used successfully by the
direct `gcloud` checks and cache upload.

### Application Default Credentials

`~/.config/gcloud/application_default_credentials.json` exists, has type
`authorized_user`, and is stale. A token refresh returns
`invalid_grant: Bad Request`. Do not rely on it on the next login node.

### Separate TPU bot key

The file below exists locally and was never committed:

```text
~/.config/gcloud/sk7524-tpu-bot-key.json
```

It identifies:

```text
sk7524-tpu-bot@vision-mix.iam.gserviceaccount.com
```

Despite its name, direct read-only checks showed that this account currently
has no usable access to `vision-mix`: `tpu.nodes.list`, `compute.zones.get`, and
`storage.objects.list` were all denied. Do not start SkyPilot with
`GOOGLE_APPLICATION_CREDENTIALS` pointing at this key unless its IAM roles are
fixed first.

The last SkyPilot API process had this mixed state:

```text
gcloud identity: 289186856710-compute@developer.gserviceaccount.com
GOOGLE_APPLICATION_CREDENTIALS: ~/.config/gcloud/sk7524-tpu-bot-key.json
```

SkyPilot obtains the displayed identity with `gcloud auth list`, but its
permission probe uses `google.auth.default()`. It therefore printed the Compute
service-account email while actually testing permissions with the unauthorized
bot key. That explains the confusing preflight message.

An isolated local Cloud SDK profile was also created at
`~/.config/gcloud-skypilot-tpuswarm`; it activates the same unauthorized bot
account and should not be used. A harmless partial profile also exists at
`/n/fs/vision-mix/sk7524/.config/gcloud-skypilot-tpuswarm` from an initial
wrong-path attempt. Neither directory contains code required by the repo.

## Why no TPU request was sent

The pool apply reached SkyPilot's credential preflight and stopped before any
GCP creation call. Generic SkyPilot GCP enablement checks permissions for
ordinary Compute VMs, networks, firewalls, disks, IAM bootstrap, API
enablement, and bucket creation/deletion. `vision-mix` is a TPU Research Cloud
project and the intended path only has TPU capacity. The bot account then made
the check fail completely because it lacks even TPU/storage reads.

The next agent must choose and validate one of these paths:

1. Preferred: obtain/fix a service-account key with the necessary
   `vision-mix` TPU, storage, and SkyPilot control-plane permissions. Make that
   same account both the active `gcloud` identity and ADC identity.
2. If TRC policy cannot grant generic Compute permissions, finish a scoped
   TPU-only SkyPilot path. It must use the working attached service account,
   avoid generic Compute/IAM bootstrap for pure TPU VM resources, and run the
   jobs/pool controller locally in consolidation mode so SkyPilot never tries
   to allocate a CPU controller VM. This needs code review and a preflight
   test; merely forcing GCP into SkyPilot's enabled-cloud database is not a
   sufficient or safe fix.
3. A direct Jobman queued-resource launch remains the already-proven fallback,
   but switching this demonstration away from TPUSwarm should be confirmed
   with the user first.

Do not grant roles, modify project IAM, delete queued resources, or bypass
credential checks silently.

## Local-only state at handoff

- `third_party/TPUSwarm/.venv` is an ignored Python 3.13.7 environment with
  TPUSwarm, SkyPilot 1.0.0-dev0, GCP extras, and server dependencies installed.
  Recreate it on the next machine instead of assuming it is portable.
- `~/.sky/` contains the failed local API-server request database and logs.
  The relevant log was
  `~/.sky/api_server/request_logs/102ef593-370f-4a44-b2dc-bf9c2081fe95.log`.
- The SkyPilot API server on `127.0.0.1:46580` was stopped.
- No `~/.sky/config.yaml` existed.
- `TPUSWARM_SERVER`, `TPUSWARM_TOKEN`, `WANDB_API_KEY`, and
  `SKY_API_SERVER_URL` were unset. `HF_TOKEN` was set in the shell.
- No repo-root `.env` or `third_party/discover/.env` existed.
- No TPUSwarm SQLite database or bearer-token file was created.
- Four running `arena-coserve` Slurm jobs belonged to other work and were not
  touched.

The Qwen submit helper resolves `HF_TOKEN` and `WANDB_API_KEY` from the
TPUSwarm server environment because both secret values are `null` in the task
contract. Supply them securely before submission or deliberately revise the
contract if W&B is not wanted. Never commit either value.

## Fresh-machine bootstrap

Start from a fresh clone/worktree or update this one:

```bash
cd /n/fs/vision-mix/sk7524/SkyRLTpu-multihost
git fetch origin
git switch agent/tunix-multihost
git pull --ff-only origin agent/tunix-multihost
git submodule update --init --recursive
git status --short
```

The final command must be empty before rebuilding a worker bundle. Rebuilding
is not required for the existing generation unless runtime code changes.

Recreate the controller environment:

```bash
uv venv third_party/TPUSwarm/.venv --python 3.13
uv pip install --python third_party/TPUSwarm/.venv/bin/python \
  --editable 'third_party/TPUSwarm[server,skypilot]'
uv pip install --python third_party/TPUSwarm/.venv/bin/python \
  --editable 'third_party/TPUSwarm/third_party/skypilot[gcp]'
```

Always cap numerical-library threads on this login node. Without these limits,
SkyPilot's API workers attempted 64 OpenBLAS threads each and became unhealthy:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export SKYPILOT_API_SERVER_LOCAL_PORT=46580
```

Before starting the server, confirm the intended account consistently in both
credential systems. The following must name the same usable service account
and the refresh must succeed:

```bash
gcloud auth list --filter=status:ACTIVE --format='value(account)'
gcloud config get-value project
third_party/TPUSwarm/.venv/bin/python - <<'PY'
import google.auth
from google.auth.transport.requests import Request

credentials, project = google.auth.default()
credentials.refresh(Request())
print(type(credentials).__name__,
      getattr(credentials, "service_account_email", None), project)
PY
```

Then start the local API server and check GCP:

```bash
third_party/TPUSwarm/.venv/bin/sky api start \
  --host 127.0.0.1 --port 46580
export SKY_API_SERVER_URL=http://127.0.0.1:46580
third_party/TPUSwarm/.venv/bin/sky check gcp
```

Do not continue unless the check reflects the identity you intended and the
TPU-only controller path has been resolved.

## Launch sequence after credentials are fixed

The model cache is already complete, so pool workers will not race Hugging
Face. First verify the marker and current absence/presence of a pool:

```bash
gcloud storage objects describe \
  gs://sk7524-tinker-tpu-asia-northeast1/hf-cache-qwen35-v1/HF_CACHE_COMPLETE \
  --format='value(name,generation)'

third_party/TPUSwarm/.venv/bin/sky jobs pool status -a
```

Request the five queued v6e-32 slices exactly once:

```bash
third_party/TPUSwarm/.venv/bin/sky jobs pool apply \
  --pool tpuswarm-v6e32-asia-qwen35 \
  tpu/swarm/examples/v6e32-qwen35-grpo-pool.yaml -y
```

Immediately verify that the request exists instead of resubmitting blindly:

```bash
third_party/TPUSwarm/.venv/bin/sky jobs pool status \
  tpuswarm-v6e32-asia-qwen35 -a -v

gcloud alpha compute tpus queued-resources list \
  --project=vision-mix --zone=asia-northeast1-b \
  --format='table(name,state.state)'
```

Create durable, private TPUSwarm controller state outside the repo. The token
and database must survive a login-shell disconnect/restart:

```bash
state_root=/n/fs/vision-mix/sk7524/tpuswarm-state
mkdir -p "$state_root"
chmod 700 "$state_root"
umask 077
test -s "$state_root/token" || openssl rand -hex 32 > "$state_root/token"
export TPUSWARM_TOKEN="$(<"$state_root/token")"
export TPUSWARM_SERVER=http://127.0.0.1:8787
export SKY_API_SERVER_URL=http://127.0.0.1:46580
# Export HF_TOKEN and WANDB_API_KEY here from an approved secret source.

nohup third_party/TPUSwarm/.venv/bin/tpuswarm serve \
  --database "$state_root/tpuswarm.db" \
  --registry-module skyrl.tpu_swarm \
  --host 127.0.0.1 --port 8787 \
  >"$state_root/server.log" 2>&1 &
```

Confirm health, then install the admission policy that holds one of five
workers for recovery:

```bash
curl -fsS "$TPUSWARM_SERVER/healthz"
curl -fsS -X PUT \
  "$TPUSWARM_SERVER/v1/resources/gcp-tpu-v6e-32-asia" \
  -H "Authorization: Bearer $TPUSWARM_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"target_workers": 5, "recovery_reserve": 1}'
```

Wait until the SkyPilot pool reports all five workers ready. Then submit the
first task idempotently:

```bash
third_party/TPUSwarm/.venv/bin/python \
  tpu/swarm/submit_qwen35_grpo.py \
  --server "$TPUSWARM_SERVER" \
  --task-id qwen35-v6e32-grpo-001 \
  --run-dir qwen35-v6e32-grpo-001
```

The task ID is also its idempotency key. Reusing it is safe; do not invent a
new ID merely because a client command times out.

## Monitoring

Use five-minute observation intervals, not repeated pool submissions:

```bash
watch -n300 'third_party/TPUSwarm/.venv/bin/sky jobs pool status \
  tpuswarm-v6e32-asia-qwen35 -a'

watch -n300 'third_party/TPUSwarm/.venv/bin/sky jobs queue \
  --skip-finished -v'

curl -fsS "$TPUSWARM_SERVER/v1/tasks/qwen35-v6e32-grpo-001" \
  -H "Authorization: Bearer $TPUSWARM_TOKEN"
```

For GCP-side allocation state:

```bash
gcloud alpha compute tpus queued-resources list \
  --project=vision-mix --zone=asia-northeast1-b \
  --format='table(name,state.state)'
```

If provisioning reaches a patched terminal queued-resource state, inspect the
SkyPilot controller log before acting. The fork should delete only the
cluster-owned wrapper/nodes and recover. Do not manually delete unrelated
queued resources.

## Validation already completed

- The focused SkyRL TPUSwarm CPU tests passed 5/5 under a Slurm CPU allocation.
- The pinned SkyPilot fork passed 66 GCP tests, including 34 new
  queued-resource tests; formatting, mypy, and pylint passed.
- TPUSwarm passed 10 tests and Ruff.
- The pinned SkyPilot parser accepted the v6e-32 pool YAML.
- The worker bundle dry build and manifest validation passed.
- The real Neuronic cache seed succeeded using `srun -p cpu`, eight CPUs, and
  32 GiB memory after the Xet high-performance mode was removed.
- No live TPUSwarm v6e-32 pool allocation or suspend/preempt recovery smoke
  test has happened. That remains the deployment-time validation.

Any new tests on Neuronic must follow the user requirement to use the Slurm CPU
partition and `srun`; do not run substantial tests directly on the login node.
