# Qubit routing relaunch — implementation and live validation

Updated 2026-09-21, 16:02 EDT. Worktree: `SkyRLTpu-qubit-parallel-relaunch`, branch `agent/qubit-parallel-relaunch-20260921`, based on science/farm migration `b67b3eeb` with the deployed first-available admission changes carried in `bd196ef3`.

## Authorized deployment

Three existing v4-64 pool workers, one each for Gemma, Qwen, Muse, with inference farm borrowing. Ten optimizer steps per fresh run. Keep AC2 running and borrowing under its existing settings. Preserve existing qubit results. Gemma must complete a realistic step, checkpoint, and next generation with updated adapter before Qwen and then Muse are admitted. No fleet expansion.

## Implemented

- Four independent, cgroup-limited case processes per program; compile once; exact pinned case ordering and seeds; unchanged 20 × 20 trials and independent verification.
- One 1,900-second deadline including preparation, compilation, routing and verification, with a 2,100-second systemd envelope.
- Program cap: 10 vCPU / 20 GiB. Host maximum: 10 programs, 100 vCPU / 200 GiB. Versioned admission locks also block conflicting legacy graders. Training initially uses **eight** program slots per VM and a 240-GiB runtime reserve beside the existing 128-GiB cache. Compilation/coordinator overhead is included.
- Explicit opt-in through the Ray v2 configuration, request resource contract, CPU reservations, task limits, and disjoint service CPU affinity.
- Published Gemini topology targets in the initial and parent prompts and topology feedback. Scalar reward remains unchanged.
- Failed programs receive reward zero. All 72 case IDs appear in partial feedback; verified results retain SWAP/baseline counts. Failed, unfinished, deadline-unfinished and not-started cases are distinguished. Incomplete topology totals have no Gemini gap claim.
- Zero-baseline cases are verified without attempting to score them individually; only the complete suite receives the scalar reward.
- Full bootstrap archive freezing, exact-source deduplication, production-worker regrade runner, deterministic fresh seed import, and farm-enabled profile builder for all three models.

The focused and integration suites passed **139 tests plus 9 subtests** on CPU allocations. This includes routing/resource/feedback tests and Ray command/runtime/package checks. This is not proof of completed TPU training.

## Frozen inputs

Each source manifest preserves 1,024 draft occurrences, source hashes, GCS object generations, old outcomes, and the historical prompt provenance. No new generations.

| Model | Unique programs | Raw valid before | Unique valid before | Best original bootstrap reward |
|---|---:|---:|---:|---:|
| Gemma | 196 | 494 | 96 | 0.5329396104918068 |
| Qwen | 114 | 725 | 46 | 0.5202905231030451 |
| Muse | 245 | 128 | 48 | 0.5209892511252158 |

Files: `.science/routing-relaunch-20260921/sources-v2/<model>/source-manifest.json` and the adjacent original grade records. These are pre-regrade results.

## Live attempt history

- Worker 718 was idle; workers 717/681 still ran old Gemma 1339 / Muse 1340. No old training job has been cancelled.
- Validation 1436 reached worker 718. It confirmed child cgroup execution and empty descendant lists after cleanup, but exposed the individual zero-baseline scoring bug. This is a failed validation, not an accepted result. Preserved evidence is under `benchmark-v1` and its manifest's GCS results URI. It was cancelled after the corrected immutable artifact was built.
- Corrected validation 1437 used commit `936963e6` and `benchmark-v2`. Its execution request was delayed by the shared SkyPilot long-request queue. A temporary standard request executor on a Slurm CPU node failed its Google API DNS lookup before the payload ran. That attempt was cancelled for retry from a connected host. Neither attempt changed the shared API server or cancelled unrelated jobs.
- Submission, cancellation, exact code/task hashes, and protected-job inventories live under `.science/routing-relaunch-20260921/benchmark-v*/`.

## Corrected live validation

Benchmark **1439** is running on existing v4-64 worker 718, using commit `936963e6`, code SHA `320b8c5bed3b733511f04c394d098d7913efcac610ee2d8c165b6cb519316b52`.

- Winner, four case workers: two complete 72/72 results, 398.093 and 402.709 seconds, reward 0.5329396104918068. Both per-case SWAP vectors equal the archived original.
- Winner, one case worker: 1,398.374 seconds, same reward and all 72 verified cases. Parallel median is 400.401 seconds: 3.49 times faster on this panel.
- Winner, legacy grader: 1,349.922 seconds, same reward and all 72 verified cases.
- Explicit 120-second timeout: returns reward zero after 120.033 seconds, with 14 passed, 4 deadline-unfinished, and 54 not-started rows. All 72 IDs remain in feedback.
- Deliberate compilation error: reward zero, all 72 not-started rows.
- Completed validation units have no surviving child PIDs. Parallel winner peaks were 1.37-1.39 GiB, below the 20-GiB cap. This does not establish maximum host concurrency under training load.
- The historical timeout comparison is still finishing. The full panel must pass before bootstrap regrading.

Regrade bundles for all three models use commit `3e888b8e`, code SHA `e5baf2e25b96cfa407c21a77a8a0d39b392afedce8155649d06f0de41f746e76`, evaluator SHA `e35333cb6e2699aeb1040dafeab1c29c3d062c5befd0e3cc68311830388a95a4`. Writable evidence is separated by model and host even when sharing the immutable code cache. These regrade jobs are not yet submitted.

## Staged launch control

`tpu/science/ops/routing_relaunch_watch.py` advances this bounded campaign: accept the benchmark, regrade Gemma, prepare its immutable seed/profile/bundle, retire only its verified old routing job, launch Gemma, prepare Qwen and Muse, then admit them only after Gemma's first complete cycle. The gate requires successful training metrics, a durable checkpoint, adapter reload, subsequent generation, and observed remote farm generation. It stops on failed new jobs or mismatched artifacts. It never resizes a pool or cancels AC2, circuit or farm jobs. The controller is not yet started at this snapshot.

## Not yet established

Complete bootstrap regrades and new seed pools; any new Gemma/Qwen/Muse training submission; a new optimizer update or farm lease. Keep these gates explicit when reporting progress.
