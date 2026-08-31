# SkyRL + TPUSwarm

TPUSwarm is included as the [`third_party/TPUSwarm`](../../third_party/TPUSwarm)
submodule. The standalone project implements application task/workflow
contracts and delegates execution to SkyPilot Managed Jobs. This directory is
the thin SkyRL port for the current Erdős minimum-overlap training scripts.

## What SkyPilot now does

- A real SkyRL training task is submitted to the Job Pool. There is no
  long-running queue consumer monopolizing each pool worker.
- Managed Jobs recover spot preemptions and node failures, retry configured
  application exits, preserve logs/status, and clean up resources.
- The job uses `EAGER_NEXT_REGION` recovery, SkyPilot priorities, and a native
  preemption lifecycle hook that invokes `cell_sync.sh`.
- The existing startup path probes completion and restores its latest GCS
  LoRA/optimizer/trainer/tree state before continuing.

TPUSwarm retains stable task IDs, idempotent client submission, the deliberate
warm recovery reserve, dynamic child generations, and checkpoint barriers.

## Initialize

After cloning SkyRL:

```bash
git submodule update --init --recursive
uv sync --extra swarm
```

Start the thin controller on a durable CPU host. It uses the configured remote
SkyPilot API server:

```bash
export SKY_API_SERVER_URL=https://your-skypilot-api.example.com
export TPUSWARM_TOKEN="$(openssl rand -hex 32)"

uv run --isolated --extra swarm tpuswarm serve \
  --database /var/lib/tpuswarm/tpuswarm.db \
  --registry-module skyrl.tpu_swarm
```

Restarting this command reloads the logical database and adopts still-running
SkyPilot jobs by job ID or the stable name derived from the task ID. SkyPilot
continues recovering the jobs while this thin process is unavailable.

## Publish the separate worker bundle

TPUSwarm does not share or overwrite the live Jobman
`stagea-league.tar.gz`. Build its worker artifact at the separate URL encoded
in the pool examples:

```bash
bash tpu/swarm/build_skyrl_bundle.sh \
  gs://sk7524-tinker-tpu-us-east5/code-bundles/tpuswarm-skyrl-v1.tar.gz
```

The builder excludes `.env` files and local run state. Pass credentials through
the task's SkyPilot `secrets` mapping. For a new code release, use a new object
name (`v2`, or preferably a release/hash) and update both pool YAMLs; changing
the URL makes SkyPilot replace workers whose setup changed.

Each pool worker downloads that object and extracts it under a directory keyed
by the GCS object generation, then atomically points
`~/SkyRLTpu-tpuswarm` at it. The typed submit helper defaults to the matching
`/home/sk7524_princeton_edu/SkyRLTpu-tpuswarm` path; override
`TPUSWARM_REMOTE_SKYRL_ROOT` if SkyPilot uses another remote username.

## Create warm pools

Apply one pool for each independently useful quota island. The pool owns setup,
workdir distribution, mounts, and the fixed/auto-scaled worker target:

```bash
sky jobs pool apply --pool tpuswarm-v5p32 \
  tpu/swarm/examples/v5p32-pool.yaml -y

sky jobs pool apply --pool tpuswarm-v6e8 \
  tpu/swarm/examples/v6e8-pool.yaml -y
```

The checked-in pool setup now installs the versioned source bundle. Before
production use, complete the remaining environment preparation (system and
training dependencies, GCS access, internal SSH key, and model caches) or build
an equivalent image from the matching Jobman `prepare` contract.

If the v5p pool has three workers and one must remain immediately available for
recovery, configure admission as:

```bash
curl -fsS -X PUT "$TPUSWARM_SERVER/v1/resources/gcp-tpu-v5p-32" \
  -H "Authorization: Bearer $TPUSWARM_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"target_workers": 3, "recovery_reserve": 1}'
```

Only two ordinary logical jobs are sent to SkyPilot at once. A recovery or
blocking dynamic child uses priority 800/900 and may occupy the reserve.

## Submit the existing Erdős job

JSON interface:

```bash
curl -fsS -X POST "$TPUSWARM_SERVER/v1/tasks" \
  -H "Authorization: Bearer $TPUSWARM_TOKEN" \
  -H 'Content-Type: application/json' \
  --data-binary @tpu/swarm/examples/erdos-task.json
```

Typed Python interface:

```bash
uv run --isolated --extra swarm python tpu/swarm/submit_erdos.py \
  --task-id erdos-stagea-ttd-n-001 \
  --cell ttd-n \
  --run-dir stageA2-ttd-n \
  --gcs-run gs://sk7524-tinker-tpu-us-east5/skyrl-runs/stageA2-ttd-n
```

The registered `ErdosMinOverlapTask` compiles this payload into one native
SkyPilot Managed Job on `tpuswarm-v5p32`. The head executes
`run_erdos_min_overlap.sh`; non-head ranks remain no-ops while the existing
head-driven cell script reaches them over the slice's internal IPs.

## Multi-component workloads

Use a native SkyPilot Job Group when every member is known up front and its
placement constraints fit one group. Job Groups already run tasks in parallel
and recover one preempted component without restarting siblings. SkyPilot 0.13
does not allow a Job Group to target a Pool, so use separate TPUSwarm pool jobs
when retaining reusable warm workers matters more than grouping them as one
native unit.

Use TPUSwarm `MultiAutoResumable` for the tree-search case discussed here:
members/generations are created dynamically, a failed terminal component must
be replaced at top priority, or shared algorithmic state advances only after a
checkpoint barrier. [`ensemble-workflow.json`](examples/ensemble-workflow.json)
shows the current fixed workflow API, while
`erdos_ensemble_workflow()` provides the typed builder.

The current `ttt_discover.rl.ensemble` still keeps shared PUCT state in one
process and enforces equal-step resume. Until that controller is extracted to
durable state and member checkpoints publish barrier sequences, the whole
existing ensemble must remain one `AutoResumable` leaf. Splitting it into child
jobs without that refactor would not be correct.

## Mixed v6e-32 Qwen GRPO pool

[`v6e32-qwen35-grpo-pool.yaml`](examples/v6e32-qwen35-grpo-pool.yaml) is the
single-zone production shape for Qwen3.5-27B. One pool worker is one complete
`v6e-32` slice (8 TPU VMs / 32 chips), not one physical host:

- ranks 0-3: 16-chip trainer, TP8 x FSDP2 (two physical hosts);
- ranks 4-7: four independent TP4 vLLM engines (two physical hosts);
- rank-32 LoRA, 22,528-token context, and a 45,056-token trainer budget;
- GRPO mean baseline, 16 groups x 32 rollouts, streamed asynchronously;
- checkpoints, run state, HF weights, and v6e executable caches remain in the
  `sk7524-tinker-tpu-asia-northeast1` bucket.

The pool pins `asia-northeast1-b`, uses GCP queued resources, and keeps exactly
five complete slices warm. The TPUSwarm admission policy below admits at most
four ordinary jobs, preserving one already-warm slice for recovery:

```bash
curl -fsS -X PUT "$TPUSWARM_SERVER/v1/resources/gcp-tpu-v6e-32-asia" \
  -H "Authorization: Bearer $TPUSWARM_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"target_workers": 5, "recovery_reserve": 1}'
```

Before allocating the pool, seed the public Qwen weights once from a Neuronic
CPU allocation. This prevents twenty serving TPU VMs from racing to download
the same checkpoint and does not request a GCP CPU VM (the `vision-mix` TRC
project has TPU-only capacity):

```bash
srun -p cpu --time=02:00:00 --cpus-per-task=8 --mem=16G \
  bash tpu/swarm/stage_qwen35_hf_cache.sh
```

Publish only from a clean parent worktree. The bundle embeds a manifest with
the exact parent and recursive submodule commits; the GCS object generation and
SHA-256 additionally pin the bytes used by pool setup:

```bash
bash tpu/swarm/build_skyrl_bundle.sh \
  gs://sk7524-tinker-tpu-asia-northeast1/code-bundles/tpuswarm-skyrl-qwen35-v6e32-v1.tar.gz

sky jobs pool apply --pool tpuswarm-v6e32-asia-qwen35 \
  tpu/swarm/examples/v6e32-qwen35-grpo-pool.yaml -y
```

Submit the first idempotent training task after the five workers report ready:

```bash
uv run --isolated --extra swarm python tpu/swarm/submit_qwen35_grpo.py \
  --task-id qwen35-v6e32-grpo-001 \
  --run-dir qwen35-v6e32-grpo-001
```

Infrastructure loss, preemption, and capacity failures recover without a fixed
attempt limit. Exit codes 33/34 also recover without consuming the application
retry allowance; other nonzero application exits receive three restarts (four
total attempts). `FAILOVER` intentionally preserves the single-zone placement.
