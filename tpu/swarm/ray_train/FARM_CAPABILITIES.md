# Farms as capabilities: a zero-engine CPU farm (della)

A farm is one runtime everywhere: bootstrap, then Ray v2 (`ray start`), then the controller, then
the Serve `Ingress` with `GradingService`. A farm has two capabilities:

- **Serving**: it has engines, `chips_per_host // inference.tp` per host.
- **Grading**: it has `grading.families`.

Zero engines is a valid farm. `accelerator: "cpu"` has no chips, so its farm serves grading only.
The della CPU worker is just a profile of this runtime:
`profiles/inference-farm-grading-della-cpu-001.json`. There is no separate code path. Every new
setting defaults to today's behaviour, so TPU farms are unchanged.

## What a zero-engine farm does differently (all in the shared code)

| Step | Engines > 0 (today) | Zero engines |
|---|---|---|
| controller `source_ready` | downloads the vLLM base bundle | skipped |
| `Host.prepare('inference')` | caches, HF snapshot, serving env | returns at once |
| `deploy()` | Engine deployments plus Ingress | Ingress only (`engines=[]`) |
| Ingress `/health` | engines healthy | grading ready and not draining |
| `/v1/models`, `served_models` | the model | `[]`, so no trainer or generation client ever picks it |
| GCS writeback | caches and run state | none (`writes_back`): logs stay under `root` |

The grading contract is the same everywhere: the `/skyrl/v1/grading/*` routes, open grading with
admission up to 1.5x the slots (`OPEN_QUEUE_FACTOR`), the grader-hash check, the resource contracts,
the request protocol with its TTL and journal, and leases where relevant. `/status` also reports
the farm's `url`.

**Discovery.** The farm probe accepts `expected_engines == 0` when the farm serves grading. Where
the GCE metadata host does not resolve (della), the farm's self-reported `url` is used and the farm
is marked `direct`. A direct farm is listed at `http://<node>:24800` with no tunnel.
`is_ready` takes `engines == 0` plus grading in place of a serving attestation.

## Host-sized grading

A family with `slots_per_host: 0` is sized from the host at bootstrap (`Config.for_host`):

- The pool is the allowed CPUs minus `grading.service_cpus`, and the job's cgroup `memory.max`
  minus `grading.service_memory_gib`.
- Each family gets `min(pool CPUs // cpus, pool GiB // memory_gib)` slots.
- The resolved profile is written to `runs/<run_id>/profile-resolved.json` and used by the
  controller.

Example: 16 CPUs and 32 GiB give 7 circuit slots (2 CPU / 2 GiB), 1 qubit slot (10 / 12) and 6 AC2
slots (2 / 4, killed at 8). The core pool (`tpu/science/core_pool.py`) admits all families from the
same cores.

## Isolation backends (`grading.isolation`)

`systemd` (the default) is today's `sudo systemd-run` unit. `bwrap` is for no-sudo hosts
(`tpu/science/unit.py`): each launch site runs the exact program argv it gives systemd-run, under a
supervisor with the same numbers.

| systemd-run property | No-sudo supervisor |
|---|---|
| AllowedCPUs, CPUQuota | pinned to the pool cores; widened affinity is pinned back |
| MemoryMax, MemorySwapMax=0, OOMPolicy=stop | tree anon + shmem + swap, sampled every 50 ms and PSS-confirmed; the tree is killed over the limit |
| TasksMax | the tree is killed over the thread limit |
| RuntimeMaxSec, TimeoutStopSec=2 | TERM, then KILL 2 s later |
| KillMode=control-group, owner death | subreaper; leftovers are stopped |
| PrivateTmp (AC2) | bwrap mount namespace |
| clean environment | the same systemd-like environment |

Candidates keep their own bubblewrap sandbox. A no-sudo sandbox also binds `/etc/alternatives`
read-only: RHEL links `ld` through it, and without it qubit builds failed to link.

Qubit's delegated case cgroups become per-case units (`routing_parallel.CaseUnits`, 2 CPU / 2 GiB /
128 tasks each).

Remaining differences:

- Limits act at the next sample, not at allocation.
- TasksMax kills instead of failing the fork.
- The qubit coordinator's 4 GiB sub-limit is covered only by the 12 GiB program limit.
- No per-unit memory peaks are reported.

## Draining short-lived hosts

When the farm stops, the controller calls `Ingress.drain_grading` before Serve shuts down. The stop
comes from SIGTERM to bootstrap or the controller, or at `grading.time_limit_lead_seconds` before
the host's end (`SKYRL_FARM_STOP_AT`, else the Slurm job's end from `squeue`). Draining
(`GradingService.shut_down`) works like this:

- New grades get 429, and capacity reports not ready.
- Queued grades are handed back at once.
- Dispatched grades may finish for `grading.drain_seconds`.
- The rest are cancelled with `error.class: 'draining'`.
- The farm lingers until clients have fetched those verdicts.

The client treats a `draining` cancel as `Lost('draining')`: the grade is resubmitted elsewhere and
not charged to `max_infra_retries`.

`drain_seconds: 0` (the default) keeps the old immediate cancel. Bootstrap keeps its graceful
SIGTERM handler after `ray.init` (Ray's own handler aborted the process) and forwards the stop to
the controller at once. A SIGTERM to bootstrap therefore drains in about 20 s, plus the linger,
and exits about 75 s later.

Slurm signals every process of a job at its time limit, Ray included, and kills it 45 s later
(KillWait). The time-limit lead is therefore the reliable path, and SIGTERM is best effort. A worker
whose time limit is shorter than the lead stops as soon as it is ready.

## Offline runtime (`prepare_grading_root.sh`, run once on della-milan)

```
/scratch/gpfs/ZHUANGL/sk7524/skypilot-slurm-cpu-scratch/
  envs/controller/          Ray 2.58 + farm pins           (profile controller_env)
  code/<bundle sha12>/      farm science bundle = $code    (SCIENCE_WORKER_ROOT; .science venvs,
                            Rust toolchain + vendored crates + router target, inputs, fast proxy)
  farms/<node>/             per-node farm root             (profile root, ${SLURMD_NODENAME})
```

SkyPilot worker setup only exports these paths and runs the ordinary entrypoint:

```
export SCIENCE_WORKER_ROOT=$code SCIENCE_PLACEMENT_HELPER=fast_proxy_v1 PYTHONPATH=$code:$code/tpu
export SKYPILOT_NODE_IPS=<node ip> SKYPILOT_NODE_RANK=0     # SkyPilot sets these
cd $code && python3 -m tpu.swarm.ray_train.bootstrap $code/tpu/swarm/ray_train/profiles/inference-farm-grading-della-cpu-001.json
```

Several farms may share a node (non-exclusive Slurm workers). Discovery finds them without any
registry or state:

- **Ports.** A `job_ports` profile places its ports by its SkyPilot managed job id
  (`config.farm_ports`, used by both the farm and discovery):
  - HTTP is `20000 + id % 4000`.
  - Ray uses a 100-port block at `25000 + id % 75 * 100`: listeners at the base, workers from
    base+10. The della profile's Ray worker range is therefore 90 ports.
  - Both ranges are below della's ephemeral range (32768+) and clear of the TPU ports and of Ray's
    default worker range (10002-19999), which other users' jobs use.
  - A port held by another process fails the start, and managed-job recovery moves the farm.
  - `SKYRL_FARM_PORT` and `SKYRL_FARM_RAY_PORT_BASE` override the placement. TPU farms keep their
    fixed ports.
- **Discovery.** Run `--slurm-pool <pool>`. Each running farm job of that pool is probed directly at
  `http://<node>:farm_ports(job id)[0]`. The node comes from `squeue -n <cluster>`; the SkyPilot
  cluster is the Slurm job of that name. The same probe and the same `/health` plus grader-hash gate
  apply.
- **Per-job root.** The root is `farms/${SLURMD_NODENAME}-${SLURM_JOB_ID}`, so two farms never share
  a run dir or Ray temp dir. Ray's socket folder moves to `/tmp` when HOME is long. Discovery runs on della with the
usual `--name-contains inference-farm`. Every runtime code change needs a new bundle and a new
`code/<sha>`; clients must come from the same tree (the grader hash covers `tpu/science`).
Validate each new code dir on a compute node with known-valid circuit and qubit programs before
adding workers.
