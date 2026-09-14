Placement grading uses the existing workload Ray cluster. On the proposed
v5p-32 layout, host 3 advertises `TPU: 4` and `placement_tpu_host: 4`.
The latter marks grading capacity; there are no custom per-chip resources.

The driver creates one `STRICT_PACK` placement group on that host, with four
bundles. Each bundle reserves 4 CPUs, 16 GiB of logical Ray memory, one TPU,
and one grading token. A small request for Ray's `node:<host IP>` resource
pins every bundle to the chosen grading host. Remote tasks use bundle index
`-1`, allowing Ray to choose an available bundle and physical TPU ID.

The remote task reads `get_accelerator_ids()['TPU']` and requires exactly one
valid physical chip ID. It passes that assignment in the trusted request
to the subprocess. The subprocess has no default-to-zero behavior: it uses
the same ID for `TPU_VISIBLE_CHIPS`, the exposed accelerator device, a
per-chip coordination port, and a disjoint four-core CPU set. v5p exposes
only the assigned VFIO group and the shared VFIO control node. The JAX
runner requires exactly one TPU device before importing candidate code.

Each evaluation has an independent systemd cgroup with 16 GiB RAM,
400% CPU quota, a 300-second overall limit, and a watchdog tied to the Ray
worker's lifetime. Candidate execution has a 180-second limit; trusted CPU
grading has a separate 90-second limit. A per-chip lock prevents overlap
across payload directories and is held through cgroup cleanup. Ray schedules
resources; the sandbox and OS limits enforce isolation.

The four bundles can execute four netlists from one submission or evaluations
from different submissions. Additional tasks queue. The driver removes its
placement groups when the pilot finishes. Ray tasks never initialize JAX in
the reusable worker process; JAX lives in the fresh isolated subprocess.

The v5p Muse pilot profile serves two TP=2 replicas on each of hosts 1 and 2.
Host 0 runs the controller and is reserved for future training. This profile
is a generation-and-grading pilot, not an RL training run.

Build local artifacts without submitting jobs:

```bash
python -m tpu.science.package_placement \
  --profile tpu/swarm/ray_train/profiles/science-placement-v5p-muse-pilot-001.json \
  --output .science/deployment-placement-v5p-ray-pg/probe --probe
python -m tpu.science.package_placement \
  --profile tpu/swarm/ray_train/profiles/science-placement-v5p-muse-pilot-001.json \
  --output .science/deployment-placement-v5p-ray-pg/muse
```

Validation: the focused tests cover configuration, device selection, missing
or ambiguous assignments, independent locks, and propagation of Ray chip 3
despite a conflicting inherited chip-0 environment. A real Ray 2.58 test on
a Slurm CPU allocation uses four logical TPU resources to verify four distinct
assigned IDs, simultaneous execution, and queuing/reuse by a fifth task.
That test does not execute a TPU kernel. The four-chip v5p JAX reference
probe and Muse sampling pilot still require hardware execution.
