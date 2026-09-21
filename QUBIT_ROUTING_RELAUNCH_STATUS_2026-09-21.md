# Qubit routing relaunch — implementation and live validation

Updated 2026-09-21, 17:30 EDT. Worktree: `SkyRLTpu-qubit-parallel-relaunch`, branch `agent/qubit-parallel-relaunch-20260921`, based on science/farm migration `b67b3eeb` with the deployed first-available admission changes carried in `bd196ef3`.

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

The focused and integration suites passed **152 tests plus 9 subtests** on CPU allocations. This includes routing/resource/feedback tests and Ray command/runtime/package checks. This is not proof of completed TPU training.

## Frozen inputs

Each source manifest preserves 1,024 draft occurrences, source hashes, GCS object generations, old outcomes, and the historical prompt provenance. No new generations.

| Model | Unique programs | Raw valid before | Unique valid before | Best original bootstrap reward |
|---|---:|---:|---:|---:|
| Gemma | 196 | 494 | 96 | 0.5329396104918068 |
| Qwen | 114 | 725 | 46 | 0.5202905231030451 |
| Muse | 245 | 128 | 48 | 0.5209892511252158 |

Files: `.science/routing-relaunch-20260921/sources-v2/<model>/source-manifest.json` and the adjacent original grade records. These are pre-regrade results.

## Historical attempt record (superseded by the latest status below)

- Worker 718 was idle; workers 724/681 still ran old Gemma 1339 / Muse 1340 at that snapshot. Those jobs were subsequently retired at 17:00 EDT as recorded below.
- Validation 1436 reached worker 718. It confirmed child cgroup execution and empty descendant lists after cleanup, but exposed the individual zero-baseline scoring bug. This is a failed validation, not an accepted result. Preserved evidence is under `benchmark-v1` and its manifest's GCS results URI. It was cancelled after the corrected immutable artifact was built.
- Corrected validation 1437 used commit `936963e6` and `benchmark-v2`. Its execution request was delayed by the shared SkyPilot long-request queue. A temporary standard request executor on a Slurm CPU node failed its Google API DNS lookup before the payload ran. That attempt was cancelled for retry from a connected host. Neither attempt changed the shared API server or cancelled unrelated jobs.
- Submission, cancellation, exact code/task hashes, and protected-job inventories live under `.science/routing-relaunch-20260921/benchmark-v*/`.

## Corrected live validation

Benchmark **1439** succeeded on existing v4-64 worker 718, using commit `936963e6`, code SHA `320b8c5bed3b733511f04c394d098d7913efcac610ee2d8c165b6cb519316b52`.

- Winner, four case workers: two complete 72/72 results, 398.093 and 402.709 seconds, reward 0.5329396104918068. Both per-case SWAP vectors equal the archived original.
- Winner, one case worker: 1,398.374 seconds, same reward and all 72 verified cases. Parallel median is 400.401 seconds: 3.49 times faster on this panel.
- Winner, legacy grader: 1,349.922 seconds, same reward and all 72 verified cases.
- Explicit 120-second timeout: returns reward zero after 120.033 seconds, with 14 passed, 4 deadline-unfinished, and 54 not-started rows. All 72 IDs remain in feedback.
- Deliberate compilation error: reward zero, all 72 not-started rows.
- Completed validation units have no surviving child PIDs. Parallel winner peaks were 1.37-1.39 GiB, below the 20-GiB cap. This does not establish maximum host concurrency under training load.
- Historical timeout: the four-worker path returned reward zero at 1,900.034 seconds with 71 passed and one deadline-unfinished; the one-worker path returned zero at 1,900.005 seconds with 17 passed. Both retained all 72 feedback rows and leaked no child processes. All winner-mode per-case SWAP counts matched. The complete benchmark gate passed.

Regrade bundles for all three models use commit `3e888b8e`, code SHA `e5baf2e25b96cfa407c21a77a8a0d39b392afedce8155649d06f0de41f746e76`, evaluator SHA `e35333cb6e2699aeb1040dafeab1c29c3d062c5befd0e3cc68311830388a95a4`. Writable evidence is separated by model and host even when sharing the immutable code cache. Gemma regrade **1444** has been submitted to existing worker 718. Qwen and Muse regrades remain staged.

## Staged launch control

`tpu/science/ops/routing_relaunch_watch.py` advances this bounded campaign: accept the benchmark, regrade Gemma, prepare its immutable seed/profile/bundle, retire only its verified old routing job, launch Gemma, prepare Qwen and Muse, then admit them only after Gemma's first complete cycle. The gate requires successful training metrics, a durable checkpoint, adapter reload, subsequent generation, and observed remote farm generation. It stops on failed new jobs or mismatched artifacts. It never resizes a pool or cancels AC2, circuit or farm jobs. The controller runs as `skyrl-qubit-routing-relaunch-20260921.service` on `della-vis2`, with state and logs under `.science/routing-relaunch-20260921/rollout-{state.json,watch.log}`. Its phase-transition logger was corrected after the Gemma submission; the recorded job ID prevents duplicate submission.

## Not yet established

Complete bootstrap regrades and new seed pools; any new Gemma/Qwen/Muse training submission; a new optimizer update or farm lease. Keep these gates explicit when reporting progress.

## Regrade setup failure and corrected validation (16:30 EDT)

The rollout service is **held and inactive**; job 1444 may continue evaluating, but its results must not be imported into a training pool. At 16:19 it had 137/196 durable verdicts, including 64 valid, with all eight hosts still evaluating candidates. Two historically valid sources failed before any case with `OSError: [Errno 16] Device or resource busy`.

Commit `70629a51` creates the cgroup hierarchy before spawning compiler descendants, leaves the compile phase its 10-CPU/20-GiB allowance, and narrows the coordinator to 2 CPUs/4 GiB before admitting four 2-CPU/4-GiB case workers. Setup errors are explicitly infrastructure failures, rejected by the regrade runner and seed importer; Ray records the evidence and raises instead of returning a candidate grade. Worker exception tracebacks are retained. The latest focused suite passed 54 tests plus 9 subtests.

A bounded live cgroup probe reproduced errno 16 when the old order enabled controllers while a child remained in the service root. The new order succeeded with the same child workload; root processes were empty, and coordinator/case memory limits were each 4 GiB. This reproduces a mechanism consistent with the original failures; their original verdicts did not contain a syscall traceback. See `.science/routing-relaunch-20260921/cgroup-order-evidence.json`. Relevant kernel constraint: <https://docs.kernel.org/admin-guide/cgroup-v2.html#no-internal-process-constraint>.

Three real production-worker full-suite canaries are live on ranks 0, 2, and 3 of existing worker 718, sharing the same host admission locks and ten-program ceiling with job 1444. Their process identities and immutable code SHA are in `cgroup-canary-receipts.json`; poll with `probe_cgroup_canaries.py`. These checks cover the archived winner and the two formerly valid programs rejected during setup. They have passed setup and begun verifying cases; complete results are still pending.

Prepared replacement regrades for all three models are under `regrade-v3`, code SHA `961a0f90bcdfe2904166ed67e5dcc0da1e67e5690986ea7f3d16ff2edbe52acc`, evaluator SHA `cc7b510448d67a1f9a776fbcf83e187c72ec3b428b6a3fc1f9f00ca18e35fbe1`. They have not been submitted. After the canaries pass, preserve/supersede job 1444, run the corrected regrades under one common evaluator identity, and resume the three-model deployment sequence. Do not mix pre-fix verdicts into the new seeds. No old Gemma/Muse training job or inference farm has been cancelled.

## Corrected regrade resumed (16:40 EDT)

All three production-worker canaries passed all 72 cases, with no surviving child PIDs. Winner reward 0.5329396104918068 and every case count matched the accepted original benchmark (433.799 seconds under concurrent grading load). The two formerly valid programs rejected during cgroup setup now return their original rewards, 0.2464875615016014 and 0.5197193687096632. Complete proof and immutable GCS copies are recorded by `cgroup-fix-accepted.json`.

Only superseded CPU regrade **1444** was cancelled; its durable outcomes are preserved separately and excluded from training. The pre-cancellation snapshot contained 184 outcomes, 107 valid, two infrastructure errors, and 85,766.654 aggregate CPU seconds. These are diagnostic attempt costs, not part of the corrected starting pool. Full records are under `regrade-v2/gemma/verdicts` and the original GCS prefix.

Corrected Gemma regrade **1450** is assigned to existing worker 718, using the `regrade-v3` artifact recorded above. The automatic rollout service is active again and reads `regrade_revision=regrade-v3`; Qwen and Muse will use the same evaluator hash. Existing Gemma 1339 and Muse 1340 have not been cancelled. All three training package preflights succeeded locally.

At the latest farm preflight, discovery reported healthy four-engine Muse 1373, Gemma 1371, and Qwen 1330 farms, with no active requests and leases unleased/expired. Both advertised serving-source hashes match the deployment source exactly. This confirms available compatible source snapshots, not that the new training jobs have leased or generated yet. Snapshot: `farm-preflight.json`. First-cycle gating still requires actual remote generation, an optimizer update, checkpoint publication, adapter reload, and subsequent generation.

## Opportunistic scheduling (16:53 EDT)

The launch watcher can begin Muse's regrade while Qwen's regrade is running if fewer than three nonterminal jobs remain in the existing v4-64 pool. Pending, starting, recovering, and cancelling work all count against this check. It does not retire the old Muse run early or resize the pool. Once Gemma's complete cycle has passed and a slice is free, Qwen training may start while Muse's remaining regrade cases finish. This removes a serial wait while preserving the Gemma-first gate. The 21 watcher tests pass, including capacity and prerequisite checks.

## All three regrades admitted (17:07 EDT)

The old Gemma 1339 and Muse 1340 jobs are now CANCELLED in the controller. Before cancelling, the exact live identities were checked, checkpoint archives were verified, and latest metrics, checkpoint indexes, and PUCT pools were preserved locally and immutably in GCS. Gemma had reached step 4; Muse step 7. Original results and checkpoint archives remain in their source buckets. The three existing workers remain allocated; no pool size change was made.

| Model | Corrected regrade job | Assigned worker | Latest observation |
|---|---:|---:|---|
| Gemma | 1450 | 718 | 180/196 unique verdicts saved; 105 valid; no infrastructure verdicts |
| Qwen | 1451 | 681 | Submitted; its existing SkyPilot execution request has been dispatched |
| Muse | 1452 | 724 | Submitted; its existing SkyPilot execution request has been dispatched |

All three use the same corrected `regrade-v3` evaluator. Each model maps its unique verdicts back to all 1,024 frozen original draft occurrences. Gemma's best saved reward remains 0.5329396104918068; no historically valid program has failed among its saved verdicts at this snapshot. This is incomplete regrading, not completed new training.

The two new execution requests were waiting in the shared API long-request queue. Only those exact existing requests were run through SkyPilot's status- and lock-guarded standard executor on the connected API host. No duplicate managed job was submitted and the shared API was not restarted. Executor records are `qwen-regrade-v3-executor.log` and `muse-regrade-v3-executor.log` under the private evidence directory.

The rollout watcher was stopped during this state transition and is active again. Regrades may run concurrently on all three original slices. Fresh ten-step training still follows the Gemma-first full-cycle gate, including a real remote farm generation. No new training is claimed yet. Preservation and submission receipts: `parallel-regrade-retirement.json`, `parallelize-regrades.log`, `retired/20260921T205932Z/`, and `rollout-state.json` under `.science/routing-relaunch-20260921`.

## Completed Gemma/Qwen regrades and Gemma training launch (17:30 EDT)

| Model | Drafts | Unique evaluated | Valid drafts before → after | Unique valid before → after | Unique timeouts before → after | Retained seeds | Best regraded reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gemma | 1024 | 196 | 494 → 579 | 96 → 111 | 25 → 10 | 111 | 0.5329396105 |
| Qwen | 1024 | 114 | 725 → 725 | 46 → 46 | 0 → 0 | 46 | 0.5202905231 |

Both imports use the common corrected evaluator hash. All exact original sources have verdicts, all retained programs passed the 72-case suite, and seed imports start fresh PUCT statistics and adapter/optimizer state. No infrastructure verdicts were accepted. Qwen's slowest host took 354.687 seconds; Gemma's 2,243.511 seconds, excluding setup. Aggregate evaluator CPU cost was 37,360.050 and 93,752.067 seconds respectively. Full source provenance, per-topology combined-policy scores, separate topology minima, and costs are in `seeds/<model>/{seed-import.json,rank-completion.json}`.

Gemma training **1453**, `qubit-v4-gemma-parallel2-20260921`, has been submitted. Its immutable training bundle is `00e9db115a7e11e562e6990a19a482233f9bf71147f1cd628eb2b22bd32437ba`, built at commit `ba9a5ccc1983487c75e88d5c0277f26dbc2a16eb`; its seed pool hash is `3360bd344a015350e2c84e374020d92538de5ee08ded58e532e68a7446f58453`. The first attempt on worker 681 failed the read-only disk preflight before training: rank 1 had only 19.74 GiB free. The controller selected existing worker 718 for recovery. All eight hosts there had 76–79 GiB free; the recovered job is RUNNING and compiling its runtime. No optimizer update or new farm generation is claimed yet.

Qwen's training bundle/profile is prepared, with seed hash `b7d12ba1ed47a8fab47d5bc8839fb9c32f5e1e4cbe7c7f22d4efdc71a0a95e29`; admission still waits for Gemma's full cycle. Muse regrade 1452 continues on worker 724: latest detailed snapshot 120/245 verdicts, 54 valid, no infrastructure errors. Worker 681's low-space inventory identified completed Qwen checkpoint/upload copies; targeted cleanup verifies durable copies before removing local duplicates and retains restoration receipts. Results, database, and source GCS archives are preserved.

All model recipes remain 16 parents × 32 rollouts, 10 optimizer steps, mean-baseline advantages with importance-sampling loss, two elite parent slots, and top-two child retention per parent. Elite slots exclude initial seeds in the current sampler. Final generated profile diffs explicitly record the parallel evaluator/resources, regraded seeds, fresh paths, and farm borrowing settings.

## Startup repairs and lease discovery (17:45 EDT)

Gemma 1453 continues its recovered startup on worker 718. All four local inference engines loaded the model and are progressing through vLLM input-shape compilation; no completed training step or farm generation is claimed. In the 17:44 snapshot, engines had compiled the 16- and 32-token shapes and were progressing through 64/128. The current startup path waits for these local engines before reserve/startup proceeds. Evidence: `gemma-training-latest.json`.

The shared discovery service had been restarted at 17:14 with its `--farm-pool` selector removed. Its remaining `--farm-name-contains inference-farm` filter excluded the live `farm10-*` jobs, producing an empty advertised farm list. Restored only `--farm-pool tpuswarm-v4-32-central2-smoke`, retained all trainer pools and the name filter, and restarted the discovery service. A fresh tick again found all three healthy farms (Muse 1373, Gemma 1371, Qwen 1330); job 1453 still awaited ingress readiness. Unit backup, before/after digests, and fresh discovery evidence are under `discovery-unit-before-restore.service`, `discovery-selector-restoration.json`, and `discovery-after-restore.json`. No farm job or lease was cancelled.

Worker 681's two low-space hosts were repaired after inventory confirmed the old Qwen job 1338 had succeeded and no active managed job occupied the worker. Only local copies of that finished run's checkpoint archives and transient upload tars were removed. Existing GCS checkpoints were checked by size, generation, and CRC32C; upload tars without an exact durable copy were archived under the campaign's `retired-local-cache` prefix and verified before unlinking. Results and database remain. Free space rose from 24.10 to 31.24 GiB on rank 0 and 19.70 to 41.22 GiB on rank 1. Restoration receipts: `disk-681-rank{0,1}-cleanup.json`. Remaining hosts had already passed the 30-GiB home-space gate.

## Gemma farm lease and live sampling (17:54 EDT)

Gemma 1453 has completed local inference startup, started its trainer/client, and loaded the fresh 111-state step-0 seed pool. Direct read-only borrower and farm status checks confirm the same owner run on farm 1371, a reserved run-scoped lease, four ready engines, and four active remote requests. Borrower and farm runtime identities match exactly (`4970783063bc6b1fff3e541e56ec0d7a0572a27ee357280eadeb00494339eab0`); the borrower has attested adapter SHA256 `d02ddc482047b0d2f7cc0cb216fe952e7cf987f00d250c01f08f48538e46a676`. Evidence: `gemma-lease-readiness.json`.

This establishes actual leased-farm request dispatch. It does not yet establish completed answers, the first optimizer update, or post-update adapter reload/generation. The admission gate continues to require those events. Muse's latest detailed regrade snapshot was 215/245 verdicts, 77 valid, with no infrastructure errors or formerly-valid failures among completed outcomes. Qwen's complete training bundle remains staged.

## Farm throughput and remaining capacity (18:07 EDT)

The same Gemma lease owner was present at both direct farm snapshots, 564.253 seconds apart. All four engine token counters increased without resets: 94,367 generated tokens in total, or 167.24 tokens/s across the farm. Successful engine-request counters increased by 13. These are engine-level measurements, not proof that 13 complete training trajectories or a full 32-rollout group returned. The client remains in its first 16-group sampling batch; no first optimizer step is established. Evidence: `gemma-farm-tokens-{before,after}.json` and `gemma-farm-throughput.json`.

The inference queues are active but memory constrained. The farm snapshot has two running sequences and 26–27 waiting per engine. Local inference logs around 18:02 showed four to five running sequences per engine, high KV-cache occupancy, and roughly 75–91 generated tokens/s per engine. These local samples cover different intervals from the farm measurement and should not be combined into a precise aggregate or a causal speedup estimate. No serving settings were changed during the run.

Muse 1452 remains RUNNING on worker 724: 228/245 durable verdicts, 81 valid, no infrastructure verdicts, and no formerly valid failures among completed outcomes. Gemma 1453 remains RUNNING on worker 718. Qwen training remains staged behind the full-cycle gate. Fresh disk checks on all hosts of workers 681 and 724 pass the 30-GiB preflight threshold; the tightest is worker 681 rank 0 at 31.18 GiB. Evidence: `regrade-muse-latest.json`, `remaining-worker-capacity.json`.

Compilation-cache writeback warnings were narrowed to a create-only upload race with different serialized contents under the same JAX key. For the inspected `jit_sample_with_budget-3bc76a...-cache`, the GCS object exactly matches ranks 6 and 7; ranks 2 and 4 have another size/checksum. The decompressed executable bytes also differ, not merely the four-byte compile-time header. Transfer logs show skipped existing destinations and a 412 precondition response. Thus the exact-byte check correctly rejected a mismatch; this evidence does not establish semantic incompatibility or corruption. Generation continues, and no cache contents, integrity checks, or running code were changed. Evidence: `compile-race-evidence.json`, `gemma-training-work.json`.
