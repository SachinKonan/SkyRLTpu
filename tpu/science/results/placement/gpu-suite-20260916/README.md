# Four-case GPU placement evaluation

Execution uses A40 GPU 0 on della-vis2. The other GPU is untouched.
No Slurm evaluation jobs remain: job 13984557 was cancelled while pending;
CPU build job 13984543 failed without CUDA tools. The subsequent local A100
build was stopped when A40-only execution was requested.

Xplace completed all four cases. Each raw global-placement array is retained
as raw_positions.npy. positions.npy is the deterministic CPU-legalized array,
independently verified by the pinned task scorer. The published mandatory
starting inputs are in ../xplace-start-v1, with a hashed manifest. New science
placement packages require those inputs. Existing deployed bundles are unchanged.

Human baselines use their original starting inputs and full published pipelines,
not the new Xplace inputs. AbuPlace and Archgen run sequentially on ibm01,
then the remaining supervisor runs both on ibm04, ibm08 and ibm18. Each gets
3450 seconds external candidate time, with independent grading afterward.
Archgen target search time is 3150 seconds to reserve time for shutdown/scoring.
Human outputs are graded as returned, without automatic repair.

Watch */report.json, */candidate.log, humans-a40-driver.log,
humans-a40-remaining.log, and a40-remaining-status.json. Launch PID records
identify both supervisors. These are real baseline runs, not scored smokes.
An invalid result remains invalid and must not be silently omitted from a suite.

The Xplace runtime uses the pinned Archgen route-aware patches. AbuPlace keeps
its own vendored Python source and uses locally compiled Xplace CUDA extension
binaries, as permitted by its binary-installation instructions. Logs must be
checked for fallback paths before calling a result a successful reproduction.

Checks: 19 tests and 2 subtests passed. input-fidelity.json confirms only the
initial_positions arrays changed; fixed objects and every other input field
are preserved. Starting-layout tampering and stale netlists fail validation.
