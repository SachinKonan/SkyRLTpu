## CPU placement training profile (2026-09-17)

New `science_placement_backend="cpu"` profiles retain four trainer hosts and
four TP4 inference hosts on each v6e-32. All eight hosts advertise
`placement_cpu_host: 16`. Each Ray case requests 4 CPUs, 8 GiB logical memory,
and one CPU grading token, with no TPU resource request. Host-wide file locks
also limit admission to 16 slots across runtime directories. Slots use disjoint
four-CPU sets 16-79. A systemd cgroup enforces 8 GiB total RAM, no swap, four
CPUs and a 300-second lifetime; candidate execution remains 180 seconds and
trusted grading 90 seconds. JAX is forced to CPU inside a namespace with no
accelerator device mounts. The candidate receives 170 seconds including import
and compilation overhead; the prompt lists libraries and shared resource limits.

The topology/reference gate checks both NumPy and JAX reference programs on all
four netlists on all eight hosts before training. Sixteen cases per host means
128 concurrent cases across a slice, not 128 complete four-case submissions.
RAM-backed model caches remain capped at 128 GiB per host. These limits reserve
at most 64 CPUs and 128 GiB per host for candidate execution and grading.

Old profiles default to the TPU backend described below. CPU runs use new IDs,
prompt files and CPU-regraded seed pools; TPU rewards must not be imported into
the new pools. CPU regrading can change legality as well as score.

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
to the subprocess. The subprocess uses that physical ID for the exposed
accelerator device, a per-chip coordination port, and a disjoint four-core
CPU set. The Ray worker checks that its `TPU_VISIBLE_CHIPS` agrees with its
runtime allocation before dispatch. In an isolated v4-64 or v6e subprocess,
libtpu enumerates the sole mounted device as local ordinal 0, so only the
subprocess visibility is translated to 0 and its host chip bounds are
`1,1,1`. This never changes the physical mount or Ray allocation. Bare-host
execution retains Ray-assigned visibility. v6e also disables the host
tpunetd client inside the network namespace. v5p exposes
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

On 2026-09-15, the isolated v4-64 path passed eight actual Ray reference
evaluations (NumPy and JAX seeds on ibm01/04/08/18), with four simultaneous
candidates, all four physical chips, one visible device per subprocess,
matching Ray worker visibility, and malformed-output rejection. Evidence is
in `results/placement/v4-isolation-fix-20260915.json`. The v6e equivalent is
in `results/placement/v6e-isolation-fix-20260915.json`. These are grader
checks, not model-generated results or proof of completed RL training.

The placement branch is rebased onto native-training commit `32620e45`,
which pins Discover to `1d662eb`. Placement clients read `config.ports`
instead of the old hardcoded service ports. The standalone reference probe
uses the native port defaults and checks its full worker range against
Linux ephemeral ports and other Ray workers before startup. Grading
resource registration preserves the native RG-LRU grader path; profiles
cannot combine RG-LRU and placement grading roles.

Fresh artifacts for this base are built under
`.science/deployment-placement-v5p-rebased/`. Their manifests record the
main-repository commit, Discover pin, archive hash, and source-file hashes.
Earlier deployment bundles remain historical and should not be submitted.
