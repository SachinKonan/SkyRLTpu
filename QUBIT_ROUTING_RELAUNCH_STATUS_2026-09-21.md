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

## All bootstrap regrades complete (18:24 EDT)

Muse regrade **1452 SUCCEEDED**: all 245 unique verdicts are durable, with 82 valid programs, no infrastructure errors, and all 48 previously valid unique programs still valid. Raw valid drafts increased from 128 to 203 out of 1,024; unique valid programs increased from 48 to 82; unique timeouts fell from 122 to 88. Every host reports completion and no active evaluations. The eight rank receipts sum exactly to the import's program, valid, and CPU totals and match its evaluator/source identities. The same receipt checks passed for Gemma and Qwen.

All three fresh seed pools are now imported: Gemma 111, Qwen 46, Muse 82. The completed bootstrap comparison is in `QUBIT_ROUTING_BOOTSTRAP_REGRADES_2026-09-21.md`, including each best policy's Q20/Willow/Heron totals and target gaps, independent topology minima, timeout changes, and elapsed/CPU costs. Muse's valid-program median is 10.79 minutes; its slowest host took 74.26 minutes excluding setup. The earlier 5–7-minute grading estimate described Gemma/Qwen, not Muse. Inspected real Muse timeout outcomes retain a full 72-case status list with passed, deadline-unfinished, and not-started cases; all receive zero reward.

Some identical sources already varied between historical evaluations. The report records this explicitly: Muse's former winning source ranged from 0.5196652959 to 0.5209892511 across eight archived copies; its one canonical regrade scored 0.5198302531. No candidate was replayed to recover a higher score. This observation does not prove equivalence for every candidate or explain every changed reward.

Muse's immutable ten-step training artifact is ready, built at `faf04c61dc5fecf21a1504c361223eb765274064`: code SHA `db6dd783dfe475d7ae4bc25dccad82c2dd84d12a0ef49ed01197948854f7c5cf`, seed SHA `19560c6f6f9784a6bc96c4fc5c01dffe36bb57e308fff53400849e28d7546c1a`, task SHA `a8c09fb3594441ef3fb16496772bb1afb80184ab3ff90b1fff53b66620f89e31`. Its reviewed profile diff preserves the model, learning recipe, 16×32 batches, token limits, and caches; changes cover fresh paths/seeds, farm borrowing, evaluator resources, and cache memory reservation.

The watcher is active in `gemma_cycle`. Gemma 1453 is running on worker 718; Qwen and Muse training are prepared but not submitted, awaiting the required full Gemma generation/grading/update/checkpoint/reload cycle. Workers 681 and 724 remain allocated for them. The latest direct farm check shows Gemma's current lease ready with four active requests, Qwen's previous lease expired with zero active requests, and Muse unleased. Their eventual launches still require actual lease/loading/generation evidence; no claim of three running training jobs is made.

## First generated groups and successful production grading (18:38 EDT)

Gemma 1453 has returned four complete local generation groups, 128 rollouts out of the first 512-rollout batch. The first group returned 194,283 output tokens in 1,805.274 seconds. New candidate grading runs on all eight hosts while inference continues. Startup grader checks are explicitly excluded using the initial sampler commit timestamp. The latest detailed verdict snapshot contains 92 newly graded candidates: 87 Rust compilation failures and five valid full-suite results, with no infrastructure failures. These are partial-batch counts, not the final batch validity rate.

The five valid candidates each passed 72 cases and took 327.724–391.352 seconds to grade under simultaneous inference/grading load. Their best reward is 0.5211400967, below the imported best seed's 0.5329396105; no learning improvement is claimed. This establishes completed new generation and successful grading, but not an optimizer update or checkpoint/reload cycle. The high initial rejection count reflects generated Rust errors, including duplicate imports and type/name errors; no source sanitization or reward changes were applied.

Evidence: `gemma-first-group-return.json`, `gemma-training-start.json`, `gemma-training-verdicts.json`, and the timestamped `gemma-progress-timeline.jsonl`. The farm continues generating under the same Gemma lease, but no full remote group has returned in this snapshot. The admission gate still requires completed remote generation and the entire first training cycle. The read-only progress monitor is bounded; the independent systemd rollout watcher remains responsible for later submissions.

## Pre-update search improvement (18:42 EDT)

Gemma's first batch has produced a verified program with reward **0.5395005455**, above the best regraded seed's 0.5329396105. No optimizer update has occurred: this is sampling/search improvement, not evidence of learning from a gradient step. All 72 case identifiers, source hash, integer SWAP counts, and the original combined reward were checked against the pinned manifest before preservation. It is one observed full-suite evaluation, not an independent reproduction.

| Topology | Best starting seed | New combined-policy winner | Change in SWAPs | Gap to Gemini |
|---|---:|---:|---:|---:|
| Q20 | 15,987 | 15,320 | −667 | +1,850 |
| Willow | 34,661 | 32,351 | −2,310 | +870 |
| Heron | 44,892 | 45,256 | +364 | +2,860 |

Source SHA256: `ddf314d5eccc3922e8dd4486abeed4cf763f80285b347ed5345c4418cab9a37c`. Grader occurrence: `460e23275eff41bf9727af88b7a1261f`. Candidate, complete verdict, and run/bundle provenance are preserved locally under `.science/routing-relaunch-20260921/new-best/gemma/<source>/<occurrence>/` and immutably at `gs://sk7524-tinker-tpu-us-central2/routing-relaunch-20260921/live-best/gemma/ddf314d5eccc3922e8dd4486abeed4cf763f80285b347ed5345c4418cab9a37c/460e23275eff41bf9727af88b7a1261f/`. Receipt: `gemma-new-best-latest.json`.

The latest live monitor at this snapshot reports 128 returned rollouts, 106 grades completed, 19 valid, and 22 active evaluations, with zero infrastructure failures. All four returned groups are local; completed farm groups, the remaining batch, and the first optimizer/checkpoint/reload cycle are still pending. Qwen and Muse remain prepared behind the explicit first-cycle gate.


## All three qubit training jobs admitted (19:19 EDT)

The user explicitly requested launching the remaining qubit runs immediately, superseding the earlier Gemma-first-cycle admission gate. The existing watcher was stopped and its state updated under `rollout.lock`; the original state and exact instruction were preserved in `.science/routing-relaunch-20260921/rollout-state-before-parallel-admission.json` and `rollout-state.json`. The phase is now `submitted_all`. This records submission, not completion of first-step validation. Farm management remains with the other agent.

| Model | Job | Existing v4-64 worker | Verified launch state |
|---|---:|---:|---|
| Gemma | 1453 | 718 | Existing running workload continues |
| Qwen | 1466 | 681 | RUNNING; grading dependencies compiling on all eight hosts |
| Muse | 1467 | 724 | RUNNING; grading dependencies compiling on all eight hosts |

All three use the previously reviewed immutable bundles, corrected bootstrap seed pools, 16 parents × 32 rollouts per step, and a ten-step limit. Farm discovery, run-scoped leases, adapter attestation, and local fallback remain enabled. No new model runtime or evaluator code was packaged for this admission change. No farms or unrelated jobs were changed or cancelled; no extra TPU capacity was requested.

Read-only preflight found only Gemma occupying the target pool and verified free disk on all hosts of workers 681 and 724 (minimum 31.06 GiB). Both new jobs were selected onto those retained workers. Their existing `sky.exec` requests were pending in the shared API queue; they were dispatched using SkyPilot's normal request-locking wrapper, and both requests finished SUCCEEDED. Job RUNNING here means the payload has started: host logs show active Rust dependency builds. It does not establish loaded models, farm leases, completed generations, or optimizer steps for Qwen/Muse.

Evidence under `.science/routing-relaunch-20260921/`: `remaining-worker-capacity.json`, `parallel-launch-requests.json`, `launch-dispatch-1466.json`, `launch-dispatch-1467.json`, and `parallel-startup-logs.json`. The earlier `parallel-runtime-startup.json` probe guessed systemd unit names from bundle hashes and is not valid runtime evidence; actual runtime units use per-attempt identifiers and system scope. The host workload logs establish startup instead.


## Qwen RAM-cache admission fix and capacity loss (19:46 EDT)

Qwen 1466 failed before model startup at `mount_cache`: its trainer cache reservation was 128 GiB plus a 240 GiB runtime reserve (368 GiB total). The eight-host audit found no active executor units after failure, and the head had only about 368.97 GiB available at rest; other hosts had substantially more headroom. Existing retired Muse RAM caches had already been reclaimed by the normal admission path. The exact available-memory reading at failure was not logged, so the audit establishes a fragile budget rather than an exact failure-time shortfall.

The replacement profile `qubit-v4-qwen-parallel2-20260921-r2.json` changes only the trainer cache cap to **96 GiB** and the run/root/sick-marker identifiers. The 240 GiB runtime reserve, 128 GiB inference cache, optimizer, 16×32 batches, ten-step limit, farm borrowing, grader configuration, and corrected seed-pool hash are unchanged. GCS inventories show the trainer checkpoint at 39.015 GiB and the two compilation prefixes at 0.237 and 0.232 GiB, leaving substantial space within 96 GiB. Trainer HF restoration excludes model weights. A Slurm CPU validation passed Config validation, seed-pool verification, and the ten-step/cache-budget assertions. The new run ID avoids saved bootstrap-configuration conflicts.

Profile commit: `88df16cbada15b8d9bafa3ed0d87e427`. Immutable code SHA: `74a215b78f2b05f4fc0acb38989ce2db56a2688bf1d495f72207cc82f470912c`. Task SHA: `2f7bd6587bd10f4a9ee51a7ed57d70f72f55c5f14504bdff7bf68f01e5d0087f`. The same seed pool was copied immutably to the fresh destination; its import report records the prior target run.

The failed job **1466 was cancelled** only after the replacement bundle and seeds were uploaded. It had no durable training checkpoints. Replacement **1473** is submitted to the same pool and currently PENDING with no assigned worker. The local rollout registry now maps Qwen to 1473 and preserves the prior job/artifact in `training_replacements`.

A second preflight found worker 681 head disk at 28.04 GiB, below the 30 GiB launch threshold. Planned cleanup of verified retired checkpoint copies was aborted by its ownership guard before any remote deletion: Muse 1467 had acquired worker 681 during recovery. No disk cleanup occurred in this repair. Worker 681 still requires disk-headroom revalidation before a future Qwen assignment.

Provider inventory then showed only one READY/HEALTHY v4-64 node, corresponding to worker 681. Workers 718 and 724, used by Gemma and Muse before recovery, were absent. The pool reports no idle replicas; the 1473 controller is waiting for capacity. This thread did not change pool sizing, remove TPU VMs, cancel Gemma/Muse, or mutate farms. The existing pool provisioning/recovery machinery remains responsible for supplying workers. The reduced cache configuration has not yet passed a live startup because Qwen has no assigned slice; do not report training or farm success for 1473.

Evidence under `.science/routing-relaunch-20260921/`: `qwen-memory-audit.json`, `qwen-cache-budget.json`, `qwen-recovery-preflight.json`, `qwen-recovery-cancel-intent.json`, `qwen-r2-launch-requests.json`, `qwen-recovery-provider.json`, and `training/qwen-r2/`.


## Pool workload scope reaffirmed (19:50 EDT)

The user explicitly directed: "keep this pool focused on qubit routing." The v4-64 pool `tpuswarm-v4-64-central2-qwen35-erdos` is reserved in this workflow for the Gemma, Muse, and Qwen qubit-routing runs and their replacements. Do not place AC2, circuit optimization, or standalone inference farms in this pool. Continue borrowing external inference-farm capacity; farm management belongs to another agent. This is the operational allocation instruction, not a newly installed scheduler admission filter.

A fresh inventory found exactly three nonterminal jobs in the pool: Gemma 1453, Muse 1467, and Qwen replacement 1473; all are qubit-routing jobs. No unrelated jobs required cancellation. GCP showed worker 681's v4-64 node READY/HEALTHY and one replacement v4-64 node CREATING. The prior 718/724 nodes were absent. Qubit recovery/submission remains queued against that changing capacity. Evidence: `.science/routing-relaunch-20260921/qubit-pool-workload-inventory.json` and `v464-loss-latest.json`.


## Adaptive PWC comparisons submitted (20:00 EDT)

The user approved additional Gemma, Muse, and Qwen qubit runs with adaptive PWC at rho=0.5. Existing mean-baseline runs remain submitted and were not cancelled or replaced by these comparisons. All new jobs target `tpuswarm-v4-64-central2-qwen35-erdos`; no pool or farm management settings were changed. Task priority is 100 versus 110 on the existing baselines, so these additional comparisons do not intentionally outrank baseline recoveries.

| Model | PWC job | Controller status | Assigned worker |
|---|---:|---|---|
| Gemma | 1481 | PENDING | Unassigned |
| Muse | 1482 | STARTING | Unassigned |
| Qwen | 1483 | PENDING | Unassigned |

Each comparison uses its model's exact corrected seed pool (Gemma 111, Muse 82, Qwen 46 valid unique seeds), fresh adapter/optimizer/PUCT state, seed 1, ten optimizer steps, 16 parent groups × 32 generations, the same full-suite parallel-v2 grader/timeouts/feedback, native thinking/token limits, learning rate, and importance_sampling loss. Only the advantage estimator changes to `piecewise_valid_entropic_centered_adaptive`, with `TTD_ADV_PIECEWISE_RHO=0.5` and `TTD_ADV_PIECEWISE_INVALID_REWARD=0`. Run IDs, roots, and sick-marker paths are fresh. Farm discovery, run-scoped leases, attestation, and local fallback match the controls. Qwen uses the repaired 96-GiB trainer cache cap and 240-GiB runtime reserve from its r2 control. Compile-cache sources are retained.

The configuration gate formerly admitted only Q20 routing for adaptive PWC. It now also admits full routing with the corrected `parallel-v2` evaluator; the unsupported legacy full-suite path and unrelated science paths remain rejected. `tpu/science/*.py`, the grader, and the estimator arithmetic are unchanged. The packaged training source overlay manifests and every overlay file were compared against each baseline's immutable bundle and matched byte-for-byte.

Validation on Slurm CPU job 14247458: **49 passed**, covering advantage arithmetic/margins/centering/edge cases, supported configuration gates, unchanged settings in the matched profiles, and identical overlay manifests. Source/profile commit: `13f9208d`. Each new bundle and step-zero seed pool was uploaded immutably before its managed-job submission. Artifacts and hashes: `.science/routing-relaunch-20260921/training/pwc-{gemma,muse,qwen}/artifact.json`; submission receipts and final snapshot: `pwc-submission-summary.json` and `pwc-final-status.json`.

Submission and STARTING status alone do not establish model readiness, a live farm lease, completed rollouts, or training steps. Those checks remain pending while capacity is assigned.


## Idle surviving worker repaired (20:10 EDT)

The user's observation that nothing was running was correct: all six qubit jobs were waiting, and GCP showed only worker 681's READY/HEALTHY v4-64 slice. The temporary CREATING replacement seen earlier was no longer present. Worker 681 was idle rather than training. Muse baseline 1467's recovery had started local job 10 there, then failed the head-host clean-storage audit at 28.02 GiB versus the required 30 GiB. Its `current_cluster_name` still reserved 681, so `get_next_cluster_name` excluded the only healthy worker, including when 1467 itself retried. This was a stale reservation after a failed payload, in addition to provider capacity loss.

Under the exact pool's scheduling file lock, all eight hosts were checked: no active executor runtime units and no users of the actual `/dev/accel*` devices. The stale reservation belonged only to 1467; its controller was PENDING/RECOVERING and the local payload had failed. Three retired Muse checkpoint/upload archives on the head were removed only after GCS size/CRC32C/generation verification. The two upload tars without an existing matching checkpoint object were first preserved under immutable `routing-relaunch-20260921/retired-local-cache/worker681-rank0/...` paths. Run databases, unpacked checkpoints, logs, and active-run data were preserved. Free disk rose from 28.00 to 31.60 GiB.

After repeating the idle check, the standard SkyPilot state API cleared only 1467's stale `current_cluster_name`; no job, VM, or farm was cancelled or restarted. The normal controllers immediately selected worker 681 for **Muse PWC 1482**. Its existing `sky.exec` request was dispatched with the normal request lock and finished SUCCEEDED. Fresh worker logs show its new payload installing grading dependencies. Thus one workload has restarted setup, while the other five jobs remain queued. No completed generation or optimizer update is established for 1482 yet. Although task priorities were set on submission, the observed controller scheduling selected PWC ahead of pending baselines; do not claim enforced baseline-first dispatch.

This was a targeted operational repair, not a permanent fix to SkyPilot's failed-recovery reservation lifecycle. A future failed attempt could require the same evidence-driven reconciliation. Evidence: `idle-fleet-diagnosis.json`, `surviving-worker-idle.json`, `idle-scheduler-reservations.json`, `worker681-before-recovery.json`, `disk-681-rank0-qwen-r2-cleanup.json`, `worker681-reservation-repair.json`, `worker681-pwc-dispatch.json`, and `worker681-restarted-workload.json`, all under `.science/routing-relaunch-20260921/`. The first device probe used nonexistent `/dev/vfio/0` and was not accepted as idle evidence; the successful repair checked dynamically discovered `/dev/accel*` on all eight hosts.


## Qwen PWC disk-admission repair (21:20 EDT)

Qwen PWC 1483 failed local job 12 on worker 681 because the root filesystem had 29.45 GiB free, below the unchanged 30-GiB startup gate. This gate runs before RAM-cache admission. The actual boot block device is 100 GiB (`/dev/sda`), with an approximately 97-GiB root filesystem, despite SkyPilot displaying disk=300. No disk or filesystem resizing was attempted.

Under the pool scheduling lock, all eight hosts had no active executor runtime units or TPU-device owners, and only pending/recovering 1483 reserved worker 681. Jobs 1236, 1237, and 1466 were confirmed CANCELLED. Only their reproducible environment/source directories and the cancelled 1466 code bundle's CPU environment, Rust toolchain/cache, and compiled router target were removed. The old CPU readiness marker was invalidated first. All run directories, checkpoints, grading outputs, and logs were preserved. Disk free space rose from 29.40 to 35.18 GiB.

After rechecking ownership and idleness, 1483's stale cluster reservation was cleared using the standard SkyPilot state API. The scheduler selected Gemma PWC 1481 for 681; Qwen PWC 1483 remains queued without a worker. The existing Gemma launch request 929bc149-7736-421d-a822-25930af65022 was dispatched through the normal request lock and succeeded. Its payload advanced to CPU dependency setup; head disk then had 33.46 GiB free. This establishes recovery past the original disk gate, not Qwen training success. No jobs or TPU VMs were cancelled, and worker 727's Muse run and farms were untouched.

Qwen PWC's immutable profile already configures tmpfs model/compilation caches: 96 GiB trainer cap, 128 GiB inference cap, and 240 GiB runtime reserve. HF assets, Orbax base weights, and JAX/vLLM compilation files are restored beneath `<root>/ram`; the code validates tmpfs and rejects swap. Environments, source bundles, logs, grading workspaces, and checkpoint staging still consume disk, so RAM-backed model caches do not eliminate the disk-space gate.

Evidence: `.science/routing-relaunch-20260921/qwen1483-{disk-inspection,cache-inventory,block-device-inventory,cleanup-preflight,cleanup-result,reservation-repair,disk-repair-next-dispatch,disk-fix-verification}.json`. The targeted operational helper is `fix_qwen1483_disk.py` in the same evidence directory.


## Cache replacements submitted and Muse farm failure identified (22:20 EDT)

Gemma PWC 1481 passed the disk gate but failed trainer tmpfs admission with the original 128-GiB cap and 240-GiB runtime reserve. Its failed setup reduced head free disk to 29.66 GiB again, and its pending reservation kept worker 681 idle. A second, broader read-only inventory identified 28 GiB of reproducible installation/source/build directories. Under the pool scheduling lock, all eight hosts were verified idle and only failed 1481 owned the worker. The inventoried head-host installations were removed after checking for process references; CPU readiness markers were invalidated first. All run directories, model/checkpoint caches, logs, and grading results were preserved. Actual free disk rose to 52.63 GiB. Worker 727 and its active Muse job were untouched.

The three non-running old jobs were cancelled and replaced after verifying immutable code objects, task hashes, canonical seed-pool hashes, and absence of optimizer checkpoint archives. The first submission preflight incorrectly compared the raw JSON file hash against the canonical pool identity; it stopped before cancellation. The corrected check uses the same canonical identity as runtime seed validation.

| Old job | New job | Model / method | New configuration |
|---|---|---|---|
| 1453 | 1491 | Gemma mean baseline | trainer tmpfs 96 GiB, runtime reserve 240 GiB |
| 1467 | 1492 | Muse mean baseline | trainer tmpfs 96 GiB, runtime reserve 240 GiB |
| 1481 | 1493 | Gemma adaptive PWC rho=0.5 | trainer tmpfs 96 GiB, runtime reserve 240 GiB |

All three old jobs are confirmed CANCELLED. Qwen 1473/1483 already have the reduced cap and were retained; Muse PWC 1482 is live on 727 and was retained. Model/seed pools, learning settings, evaluator, farm borrowing, and ten-step limits match the previous runs; only trainer cache budget and fresh run/root identifiers change. Profiles are committed at be0f1806. Prior Gemma sampling results remain archived, including reward 0.5395005455. The local rollout registry aliases and replacement history now reference 1491/1492/1493.

The scheduler selected Gemma 1491 for worker 681. Its existing pending sky.exec request 0da364bc-ef6b-44d4-a8d6-686a6100007a was executed through the normal request lock and succeeded. Worker-local logs show fresh payload setup. Queued: Muse baseline 1492, Gemma PWC 1493, Qwen baseline 1473, Qwen PWC 1483. RUNNING does not yet prove a trainer update.

Read-only farm logs resolve Muse 1482's original HTTP error: at 01:06:57 UTC on September 22, old Muse farm job 1474's engine hit `RESOURCE_EXHAUSTED: E0101 RuntimeProgramAllocationFailure` while loading `jit__substitute_placeholder_token` in TPU inference's asynchronous token-substitution path. Its vLLM engine died and `/v1/completions` returned HTTP 500. The borrower marks the lease lost on this failure; concurrent requests then raise BorrowingProtocolError and fall back locally. The borrower log intentionally records only exception class, so the engine traceback was necessary to establish the failure. This is a TPU program-allocation failure, not evidence of a host disk problem or missing farm discovery.

The farm owner has separately replaced 1474 with Muse farm 1490 on worker 182; this thread only inspected it. In the latest Muse 1482 snapshot, all 512 step-zero rollouts have completed locally; no remote completion is recorded, and the old lease is still not reserved/ready. RunBorrower prepares/reacquires at the next sampling phase. Updating discovery URLs or replacing the farm does not revive an already lost lease during the existing phase. This does not establish successful generation on replacement farm 1490.

Evidence under `.science/routing-relaunch-20260921/`: `worker681-install-cleanup-{preflight,result}.json`, `cache96-repair-submit-{preflight,final}.json`, `gemma1491-dispatch.json`, `muse-farm-error-evidence.json`, and `muse1482-farm-status-after-replacement.json`.


## PWC-first dispatch corrected (22:23 EDT)

The user reiterated that PWC is the priority. Gemma baseline 1491 had been selected by the scheduler; letting it take the free worker was incorrect. Baselines 1491 (Gemma), 1492 (Muse), and 1473 (Qwen) were cancelled after confirming no optimizer checkpoint archives. Their artifacts and seed pools remain preserved in `baseline_deferred_for_pwc` in the rollout registry for a later explicitly scheduled baseline phase. They are not queued and should not be described as queued or automatically resuming.

Only three active managed jobs remain in the v4-64 queue: Muse PWC 1482 on worker 727, Gemma PWC replacement 1493 assigned to worker 681, and Qwen PWC 1483 waiting for capacity. All use adaptive PWC rho=0.5 with importance_sampling loss and a ten-step cap. Muse's live job is untouched. Gemma and Qwen use the reduced 96-GiB trainer-cache cap. The normal request-locking wrapper dispatches Gemma 1493's existing sky.exec request 6c6bc9e4-295c-489b-9f03-56dcebfd0a04.

SkyPilot task resource priority did not enforce method ordering in the earlier observed dispatch races. Removing baseline jobs from the active queue is the concrete way this workflow now enforces PWC-first. Evidence: `pwc-priority-baseline-cancel-intent.json`, `pwc-priority-dispatch-current.json`, and `gemma1493-dispatch.json`.


## PWC retry after cancellation teardown (22:28 EDT)

Gemma PWC 1493's first local attempt (job 15) was rejected by the clean-host gate while a private Ray process from the cancelled baseline was still exiting. It had stopped by the subsequent inventory; no process was killed manually. After verifying no active runtime units, TPU owners, or private executor Ray processes on all eight hosts, reproducible old installation caches were also reclaimed from ranks 1-7, preserving recent r2 installations and all results/checkpoints. Free disk is now at least 46.31 GiB on every host (head 49.63 GiB). This addresses the rank-1 near-threshold disk budget as well as the earlier head-host issue.

Under the pool scheduling lock, only 1493's stale reservation was cleared. The scheduler selected 1493 again for 681; existing request 072d0c57-6a00-43b3-a2d9-667d946c9899 was dispatched through the normal request lock. Muse PWC 1482 remains live on 727; Qwen PWC 1483 is the only other queued job. Baselines remain cancelled/deferred. Evidence: `worker681-all-host-install-inventory.json`, `worker681-all-host-cleanup-result.json`, `gemma1493-stale-reservation-cleared.json`, `pwc-second-worker-dispatch.json`.
