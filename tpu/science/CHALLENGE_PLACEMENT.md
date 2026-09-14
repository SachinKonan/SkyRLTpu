# CPU placement scaffold and baseline reproduction

Work lives in the separate `SkyRLTpu-science-placement` worktree. These are
Slurm CPU pilots; this scaffold is not yet connected to the production Ray v2
executor or to a Qwen generation job. No portfolio or TPU workload was changed.

## Frozen scientific definition

Use the pinned Partcl/TILOS position evaluator:

`cost = wirelength + 0.5 * density + 0.5 * congestion`.

Report per-case components and mean cost. For a suite with every case valid,
reward and ranking raw_score are `max(1e-6, 1/(1+mean(cost)))`; otherwise both
are zero. Never average individually transformed rewards to obtain the suite
reward. Keep correctness and scientific metrics separate. A timeout, exception,
missing case, nonfinite result, or illegal output invalidates the entire suite.

`PLACEMENT_PROMPT.md` is the candidate prompt. `challenge_seed.py` is the complete
NumPy starting implementation. The position-only contract places both hard
macros and soft clusters, with fixed dimensions, orientations and pin offsets.
It does not use the earlier grid placement/fixed-completion contract. The native
binary adapter files remain an earlier, separate experiment.

The prompt lists libraries, thread policy, shape and legality requirements,
resource budgets, exact reward, and numeric schema. `challenge_contract.py`
exports actual ordered pin endpoints and native net weights. This avoids the
upstream Benchmark loader's unit-weight simplification. No instance names,
file paths, or reference placements from other algorithms are given to the
candidate. Initial positions can be illegal. Our seed legalizes them; it is a
greedy heuristic and makes no universal packing guarantee.

This v1 scaffold provides no exact scoring helper inside candidate code.
`challenge_candidate_child.py` executes a candidate in an isolated process;
`check_challenge_isolation.py` checks that the seed produces the same arrays
there as in the direct control run. The import allowlist is a policy check,
not a security boundary. Bubblewrap removes credentials/network and restricts
mounts; process limits, CPU affinity, and the enclosing Slurm memory allocation
bound these pilots. Ray process-tree memory enforcement still needs wiring.

## Resources and measured checks

Pilot profile: four allocated CPU cores, 16 GiB total Slurm job memory, 180 s
candidate cap including imports. Each OpenBLAS/MKL/OpenMP pool is configured
for at most four threads; CPU affinity bounds the whole candidate, including
oversubscription. There is no meaningful independent memory quota per NumPy
or SciPy library: they share the process-tree allowance.

The 16 GiB allowance is deliberately headroom for generated search algorithms,
not a measured minimum. Four numeric inputs use only roughly 0.5–2.4 MB each.
The measured seed takes about 2.4–4.1 s; native scoring takes 1.6–37 s on the
four tested netlists. Host-to-host timings vary. Propose 90 s additional final
grading per case and 1,200 s overall for four discovery cases, pending actual
Ray host calibration. Do not reuse the old 600 s total with this larger search
budget. Public leaderboard reproduction uses different resources and times.

`results/placement/scaffold-pilot/report.json` contains four successful checks
on ibm01, ibm04, ibm08, ibm18: legal output, no hard overlaps, weighted pin-HPWL
parity with the native scorer within float32 accumulation tolerance, and
rejection of NaN, out-of-bounds, and overlapping corrupted outputs.

Use a predeclared discovery subset for feedback. A public 17-case aggregate is
comparable to the challenge table only when all 17 are evaluated, and is not
evidence of generalization to unseen designs. Any cases repeatedly used for
development should be labeled validation; reserve separate designs if making
a generalization claim.

## Upstream implementations

Pinned commits and entrypoints are in `results/placement/baseline-sources.json`.

- Archgen: `.science/archgen`, commit
  `6b3661bad55d81c00fad151d81ec1e74ef0271e0`. Includes an upstream CPU fallback.
  The public README records a package score of 0.961653, distinct from its
  later 0.9507 leaderboard score. Its intended full setup uses CUDA and optional
  Xplace-RA. CPU pilots explicitly disable that unavailable lane and reduce
  public wall-time knobs; candidate source is unchanged.
- Carrotato/AbuPlace: `.science/abuplace`, commit
  `a24087c45588f1873eb2dc18293e407e5477041d`. The intended algorithm requires
  CUDA/Triton/Xplace; its source explicitly has no full CPU replacement path.
  CPU executions exercise its exception handling and legal-output fallback,
  not the published GPU optimizer. Published Xplace binaries target Python
  3.12; our CPU scaffold uses Python 3.11 and Torch 2.5.1+cpu for trusted loading.
- JaneRT: public source not located. The leaderboard alone does not provide
  runnable code. No messages were sent to authors.

`run_placement_baselines.py` runs downloaded entrypoints under namespace/time
limits. It accepts only numeric NPY positions from that process, then loads
fresh canonical inputs in the independent grader. Upstream internal scores
are not trusted as results. Candidate logs, pinned commits, environment knobs,
times, legality errors and scores are recorded. `--reuse-completed` is intended
only for regrading outputs from the same pilot directory after reporting fixes.

The CPU runs are resource-constrained adaptations/fallbacks, not reproductions
of the published GPU leaderboard scores. An illegal fallback remains invalid;
do not silently repair its output and report the repaired score as upstream.
Baseline reproduction permits the dependencies shipped by the baseline
(including CPU Torch and C extensions). Generated candidates have the narrower
library list in the prompt. Thus even with matched CPU/time/memory, this pilot
is not a claim of identical library access or a complete algorithm comparison.

The Archgen 180-second pilot returned a valid ibm01 placement with proxy
0.8244787455 in 131.73 candidate seconds. Its ibm18 output was outside vertical
canvas bounds and received zero reward. Therefore the two-case Archgen suite
is invalid, even though one case improved over the seed. See
`results/placement/baseline-cpu-180-v2/report.json` and its per-case logs.

Carrotato's corrected CPU fallback pilot is in
`results/placement/abuplace-cpu-180-linker-fixed/report.json`: both cases are
legal, with ibm01 cost 1.1004959345 and ibm18 cost 1.7874572277. Candidate work
took about 1.5–1.9 seconds; ibm18 independent loading/validation/scoring took
47 seconds on that host. All GPU trajectories failed and the original code
used its emergency C legalizer. Earlier AbuPlace rows in the combined report
are superseded: their sandbox lacked the system linker symlink used to build
that legalizer. No candidate-source change was required to fix the mount.

The isolated scaffold checks passed on all four cases, producing byte-identical
position arrays to the direct seed. The complete rendered prompt plus seed is
`.science/rendered/placement.txt`; regenerate with
`python tpu/science/render_placement_prompt.py`.

## Reproduce locally

From the worktree root, fetch only pinned scorer source and data:

```bash
python tpu/science/prepare_challenge_sources.py --all
```

This verifies existing files against the pinned raw sources and records their
SHA256 values. It fetches all 17 IBM input pairs without downloading unrelated
large physical-design flows. The tested Python package versions are captured
in `requirements-challenge-pilot.lock`. Install the CPU Torch wheel separately
from its official CPU wheel index.

Run each command within a four-core, 16 GiB Slurm CPU allocation:

```bash
.science/venv/bin/python tpu/science/check_challenge_scaffold.py
.science/venv/bin/python tpu/science/check_challenge_isolation.py
.science/venv/bin/python tpu/science/run_placement_baselines.py \
  --methods archgen abuplace --cases ibm01 ibm18 --seconds 180 \
  --out tpu/science/results/placement/new-baseline-pilot
```

Do not compare a two-case pilot mean to the 17-case published average. A full
GPU reproduction, JaneRT execution, and production Ray/Qwen integration are
not completed by these commands.


## JAX / Ray v2 pilot (2026-09-13)

The placement worktree now has a separate `placement_ray.grade_case` executor.
It reserves four Ray CPUs, 16 GiB logical memory, and a whole-host custom TPU
resource. A systemd cgroup enforces 16 GiB RAM, zero swap, CPUs 16-19, a 400%
CPU quota and a 300-second case lifetime. A separate bubblewrap process sees
only physical TPU chip 0; serving hosts publish no placement TPU resource.
Candidate initialization, JIT and completed output transfer share a hard
180-second cap. Trusted CPU scoring gets 90 seconds. The suite is all-or-nothing
and aggregates costs before applying the reward transform.

The JAX candidate environment pins Python 3.12.12, JAX/jaxlib 0.10.1, libtpu
0.0.41, Optax 0.2.8, NumPy 2.4.6, SciPy 1.17.1, and NetworkX 3.6.1. The trusted
Torch CPU grader remains in a separate Python 3.11.13 environment. TPU jobs use
a cgroup RAM cap rather than RLIMIT_AS, since device mappings are virtual memory.
No Python generated by a model is executed in the Ray controller.

Local checks: all four JAX seed outputs are legal under the canonical grader.
This check used JAX's CPU backend on local Python 3.12.0, and therefore does not
prove TPU execution. Measured solve times were 8.1-10.1 seconds; fresh independent
grading took 6.4-30.3 seconds. Costs were 1.36617, 1.61872, 1.70021, 2.40482; this
seed's smooth search surrogate improves wirelength but worsens overall density
and proxy cost versus the CPU legalizer. It is a functional example, not a
claimed strong baseline. Suite tests cover valid, invalid, missing and duplicate
cases.

Managed preflight 845 and Qwen pilot 846 have been submitted to the existing
v4-64 pool. Qwen generation is gated on four valid JAX reference evaluations on
its own Ray cluster. The first standalone preflight payload contains the earlier
JAX seed with a read-only NumPy conversion bug; the model pilot contains the
corrected explicit copy. Preserve that distinction when interpreting results.
The model driver samples four completions with the unchanged native grid budget,
then grades every candidate across ibm01/04/08/18 and archives results before
requesting graceful shutdown. Submission does not establish runtime success.


Update: standalone preflight 845 is FAILED and cleaned up. It reached host setup
after roughly 20 minutes in the shared API execution queue, then hit uv's default
first-index resolution of `packaging` on the PyTorch CPU index. The Qwen 846,
Muse, and Gemma bundles already include the `unsafe-best-match` correction for
the two explicitly selected package indexes. Preflight 845 never ran JAX or
grading. The attempted cancellation was skipped because the job was terminal.
The local `.science/deployment-placement-001/run_comparison_background.py`
collects Qwen's archive and submits the prepared Muse/Gemma pilots only after
all four TPU Ray reference evaluations are valid. It stops on failed references
or pre-archive job failure, and has an eight-hour monitoring deadline. Its live
state and process ID are in `comparison-state.json`.


## Metadata-free TPU initialization repair

The first model pilot (846) reached Ray grading but all references timed out
inside `jax.devices()` before candidate code executed. The child logs show
libtpu trying metadata queries for ALT, WRAP, and accelerator-type. Peak process
RAM was only about 104-113 MiB, consistent with initialization waiting rather
than placement optimization.

The revised isolated environment sets TPU_SKIP_MDS_QUERY=true and explicitly
supplies accelerator type, host/chip/process bounds, local worker identity,
and non-wrapping topology. It also mounts read-only /etc/hosts and private
/dev/shm. Network isolation, one-chip visibility, cgroup CPU/RAM caps, and the
180-second candidate deadline remain enforced. Initialization start/end events
now distinguish runtime bootstrap from candidate solve time.

Google documents metadata-free TPU container initialization here:
https://docs.cloud.google.com/dataflow/docs/tpu/use-tpus

A local sandbox test confirmed localhost TCP works inside the private network
namespace. Full hardware validation is assigned to managed reference job 849
(science-placement-jax-probe-002), followed by model pilot revision 002.
Do not interpret a submitted job or local namespace test as successful TPU
execution; use the device reports and native grading results.

### Cleanup and startup preflight (2026-09-14)

Job 849 failed before grading because its Unix socket path exceeded 107 bytes.
The probe now uses a short UID/run-hash directory under /tmp and validates the
socket path before starting Ray. Ray 2.58 started and executed a task on worker
264 in 7.5 seconds with this naming scheme; its test processes were cleaned up.

A separate isolated TPU preflight on worker 281 exposed a systemd MEMLOCK limit:
libtpu failed to mmap /dev/accel0 until LimitMEMLOCK=infinity was set for the
individual service. MemoryMax=16G, swap=0, CPUQuota=400%, CPU affinity, and network
isolation remain enforced. With that correction, JAX initialized one v4 chip,
compiled and synchronized an array calculation in 3.66 seconds, and released
the device. This verifies TPU startup and execution, not full placement scoring
or four concurrent chip workers.

All 32 hosts across idle workers 264/281/289/291 were audited. TPU devices were
unowned, placement ports free, and no placement grading units remained. Two live
orphan GCS processes from terminal jobs 804 and 846 were stopped only after
matching PID start times and immutable task IDs. Pool Ray services, cached
models, and benchmark artifacts were preserved. Evidence is under
.science/deployment-placement-clean-003/.
