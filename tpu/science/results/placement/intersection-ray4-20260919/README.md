# Four-GPU Ray allocation queue for the 16-case human comparison

Each Slurm GPU-test array element requests one node, four whole NVIDIA A100
GPUs (40 or 80 GB), 32 CPUs, 64 GiB host RAM and one hour. Three array elements
may run concurrently, for twelve independent method-case evaluations. GPU-test
still enforces its three-job and 25-submitted-job limits.

Each allocation starts its own local Ray instance, with four one-GPU/eight-CPU
workers. The 64 GiB is a shared Slurm allocation limit; Ray's 12 GiB per-worker
reservation is scheduling metadata, not a hard per-worker memory cap. CPU
workers use separate affinity sets. A separate pinned Ray 2.51.1 environment
leaves the CUDA candidate environment unchanged. Each task has its own Xplace
runtime and native helper build. GPU UUIDs preserve assignment inside bwrap.

The GPFS queue is keyed by (method, case). A task lock is held until completion;
persistent allocation ownership prevents reclaiming an orphan until its Slurm
job has ended. Attempts have separate directories. Completed reports, including
invalid outputs, are terminal and are not silently retried for a better score.
Infrastructure interruptions without a completed report may be retried by a
later allocation. Eight allocation elements suffice for 32 tasks if each worker
can execute only one; short tasks may allow subsequent claims earlier.

A worker claims a new task only with at least 3330 seconds left in the actual
Slurm allocation: 3300 seconds total task wall time plus 30 seconds for cleanup.
The task wall limit includes private runtime copying, candidate execution and
independent grading. Candidate subprocess timeout is 3180 seconds; ArchGen's
existing 3150-second internal search target is preserved. The independent grader
retains its 120-second limit within the overall task deadline. Timeouts are
explicit invalid results. No shortened-budget jobs are started near shutdown.
This conservative rule means a GPU finishing after the admission cutoff drains;
it does not promise to fill the tail with another full-budget evaluation.

Completed and already-running single-GPU A100 evaluations are retained via the
legacy manifest, and remaining queued single-GPU tasks were cancelled. Those
legacy evaluations used 16 CPUs and a 3450-second candidate timeout; new tasks
use eight CPUs and a 3300-second total limit. Per-attempt provenance records the
difference, so this is a throughput-oriented comparison, not matched compute.
The case set, frozen algorithms, internal search target and scorer are unchanged.
The model reference remains the previously evaluated CPU solutions. ibm09 is
excluded from this conditional intersection, not erased from the full17 report.

Validation: seven queue tests cover cross-process locking, live-owner exclusion,
recovery after an ended owner, valid/invalid terminal reports, legacy ownership,
full-budget deadline admission, and rejection of dependency-incomplete legacy results. Actual GPU placement and Ray startup are
verified with the first production allocation, before submitting the remainder.

First allocation 14145288 ran four independent tasks on four A100-SXM4-40GB
GPUs; observed Slurm peak RSS was approximately 9.3 GiB total. Inspection found
AbuPlace skipping polish and basin-jump stages due to missing setuptools in
the original CUDA environment. That allocation was cancelled and its attempts
are preserved under infrastructure-attempts/before-setuptools. Earlier legal
AbuPlace outputs are not complete reproductions and are excluded from this
queue's comparison; all sixteen AbuPlace cases are re-evaluated.

A separate candidate environment (.science/venv-cuda-ray4) contains the same
CUDA packages plus setuptools 80.9.0; ninja remains at the original 1.13.2.
Candidate and Ray package versions are frozen in the adjacent requirements
files. Worker startup imports setuptools and torch.utils.cpp_extension before
claiming work. The grading wrapper rejects a completed run that logged a
ModuleNotFoundError, even when it returned legal coordinates.

Production array 14145560 uses indices 0-7%3. Single-GPU ArchGen ibm06 had
started during the old queue cancellation; it was cancelled to free a QoS job
slot and returned to the shared queue. Its partial legacy output is preserved.
Completed ArchGen results remain in use. summarize.py combines these retained
results with atomic done.json pointers from the new workers, and never reports
a qualifying mean while any required case is missing or invalid.
