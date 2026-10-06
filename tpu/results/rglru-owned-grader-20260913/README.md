# Dedicated RG-LRU grader and launch priorities

User priority: finish the current tests, restore the full eight-run Gemma
AC2/circle-packing grid, then expand RG-LRU across models. One initial Muse
RG-LRU run is authorized alongside testing; do not fill spare capacity with
additional RG-LRU runs ahead of Gemma.

Muse job 808 (`native-v5p-muse-rglru-grpo-lr4e5-s1-003`) uses the existing
v5p-32 pool and Ray v2. Host 0 trains (TP1/FSDP4), host 1 exclusively grades,
and hosts 2/3 each serve two TP2 Muse engines, max sequences 16. GRPO, LR 4e-5,
seed 1, 15 iterations, 16 parent groups x 32 rollouts, native completion,
context 22528, phase-one prompt plus thinking 16384, trainer buckets
18432/22528. Existing Muse HF, Orbax and compiler caches are reused.

The grader owns four TPU resources in the workload Ray namespace. It runs
candidate CPU pregates and isolated per-case TPU children, four chips at most
at once, preserving the six-case forward/backward grading contract. The
client finds the named actor in this workload; no external grader URL or
separate grader slice is used. Actor shutdown cancels requests, terminates
child process groups and waits before releasing chips.

Before training, a hardware self-test must accept the shipped seed across
all six cases and pass strict forward/backward verdict translation. It must
also reject a non-Pallas candidate at pregate. Failure aborts the run before
sampling. `grader/self-test.json` and controller events record the result.
A passing test is not assumed merely from successful submission.

Local checks: 18 tests passed (grader transport, chip isolation, timeout,
shutdown, child process cleanup, infrastructure errors and replay failure
handling). Real CPU Ray actor test passed named lookup, four-resource
reservation and verdict transport. Its first attempt used the older worktree
as cwd and failed actor import; rerunning from the deployment worktree passed.
Build validation checks the complete judge package is bundled, the self-test
is wired, the native external transport patches are included, and overlay
hashes match. Each assigned host runs the clean-host gate before bootstrap.

Operational artifacts:
`/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/benchmarks/rglru-owned-grader/`.
