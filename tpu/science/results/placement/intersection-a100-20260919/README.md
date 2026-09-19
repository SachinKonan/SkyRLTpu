# A100 comparison on the shared valid 16-case intersection

Requested 2026-09-19. Cases are the intersection of the frozen Gemma/Qwen/Muse
full17 outputs that passed independent validity checks. This excludes ibm09,
where Qwen exceeded the canvas boundary. The original 17-case report keeps
that failure; this is a separate conditional comparison, not a qualifying
17-case leaderboard score.

Model results from the existing CPU evaluations on the shared A40 host:

| Model | Valid cases | Mean proxy cost |
|---|---:|---:|
| Gemma | 16/16 | 1.0455511436 |
| Muse | 16/16 | 1.0683691762 |
| Qwen | 16/16 | 1.0727658160 |

32 baseline tasks evaluate AbuPlace and ArchGen on those exact same cases.
All 16 cases are re-evaluated on A100; no A40/A100 score mixing within a human
method's mean. Frozen methods, original starting inputs, scorer and 3450-second
external candidate limits match the ongoing A40 comparison. A100 hardware and
available compute differ from model CPU budgets, so this is not equal-compute
algorithm comparison. Every case must pass for a 16-case mean to be published.

Execution: one A100 80 GB, 16 CPUs, 100 GiB host memory and one hour per Slurm
job. QoS gpu-test permits 3 running jobs and 25 submitted jobs per user. The
submission frontend accepts `--partition=gpu-test` and maps it to internal
partition `gputest`. Two array waves stay within submission limits; the second
wave is automatically submitted when at most 8 gpu-test tasks remain queued.

An initial GPU build job (14143567) failed during configuration because Bison
and Cairo development files are absent on compute nodes. Its dependent array
14143577 was cancelled before execution. The extensions are instead cross-built
for sm_80 on the local host, where those development tools exist. The A40-only
cached extensions are not reused as A100 CUDA kernels. The source is copied
from the exact unmodified Xplace tree used by the baseline comparison; the
ibm14 preparation-only optimizer patch is not applied to human baselines.

`plan.json` fixes cases and task-index mapping. `model-reference.json` preserves
reference averages. `submissions.json` records live array IDs once submitted.
`logs/` holds build/job logs. Each method-case directory retains output positions,
component metrics, runtime, validity and failure logs. Per-job private Xplace
copies prevent concurrent runs from sharing mutable logs/cache directories.
The existing local A40 comparison is preserved.

Build completed successfully on the local host. `binary-manifest.json` verifies
sm_80 CUDA images and binary hashes. A100 runtime preflight 14143843 passed on
della-l08g1 (A100-SXM4-80GB, Torch 2.5.1+cu124). First evaluation array: 14143859,
indices 0–15. The detached submitter records/queues the second wave automatically.
`submit-launch.json` identifies its process. The isolated wrapper explicitly
includes the private Xplace library directory in LD_LIBRARY_PATH so installed
ELF dependencies resolve after relocation.

CPU portability recovery: the first two AbuPlace jobs exited with SIGILL (132)
in refinement. A cached liblegalize.so from the A40 host contained AVX-512 mask
instructions, while these NVIDIA A100 nodes have AMD host CPUs. Private-copy
AbuPlace C helpers are now removed and rebuilt on the assigned CPU using the
unchanged upstream -march=native build. Retry 14143969_0 passed refinement and
completed a valid internal trajectory. Two original ArchGen tasks continued.
One newly started ArchGen task (index 5) was rescheduled to free a validation
slot; its partial artifacts and the two failed AbuPlace attempts are retained
under infrastructure-attempts. All 21 temporary pending-job holds were released.

Second wave 14143899 (indices 16–31) is submitted. The remaining cancelled /
interrupted indices 2,4,5,6,8,10,12,14 are automatically resubmitted by
submit_retries.py once QoS submission slots are available. Its PID is in
retry-submit-launch.json, and the resulting array ID goes to retry-submission.json.
These infrastructure retries do not alter the frozen algorithms or case set.
