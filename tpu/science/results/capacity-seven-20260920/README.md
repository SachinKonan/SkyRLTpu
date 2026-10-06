# Seven capacity launches — 2026-09-20

The user authorized a one-trainer/seven-inference run on `shu-v5p-64-on-demand-machine1`, three additional jobs in the v5p pool, and three in v6e central. Allocation keeps the new Gemma trainers on v5p because Gemma circle packing's v6e backward compilation failed HBM capacity checks.

The dedicated machine runs Gemma circle packing; see [its launch record](../shu-v5p64-gemma-cp26-20260920/README.md). Existing managed jobs and provider pool targets were not cancelled or changed.

## Pool submissions

| Job | Pool | Model/problem | Imported bootstrap states | Trainer TP/FSDP | Inference engines |
|---:|---|---|---:|---|---|
| 1318 | tpuswarm-v5p32-east5a-erdos | qwen ac2 | 291 | 1/4 | 3 x TP4 |
| 1319 | tpuswarm-v5p32-east5a-erdos | gemma ac2 | 160 | 4/1 | 3 x TP4 |
| 1320 | tpuswarm-v5p32-east5a-erdos | gemma qubit | 96 | 4/1 | 3 x TP4 |
| 1321 | tpuswarm-v6e32-central1b | muse ac2 | 440 | 8/2 | 4 x TP4 |
| 1322 | tpuswarm-v6e32-central1b | qwen qubit | 46 | 8/2 | 4 x TP4 |
| 1323 | tpuswarm-v6e32-central1b | muse qubit | 0 | 8/2 | 4 x TP4 |

Five runs import their original bootstrap PUCT step-zero pools, verifying the canonical pool SHA against the source completion marker and validating finite positive scores and code. The qubit imports also use the runtime seed checksum guard. Muse qubit starts a fresh bounded bootstrap: 32 concurrent groups of 16, up to 1,024 drafts, targeting 512 distinct valid programs, no repair, then GRPO.

All seven new runs start fresh model and optimizer state in separate run/cache namespaces. No source optimizer checkpoints are promoted: checkpoint filenames alone proved insufficient to establish successful training, since the failed-train path can also save a checkpoint. Source experiments and all saved artifacts are preserved. The seeded runs start training directly from the imported programs.

Recipes retain mean-baseline GRPO/importance-sampling, Qwen LR 1.5e-4, Gemma/Muse LR 4e-5, 16 groups of 32 training rollouts, 15 epochs, native 16,384 prompt-plus-thinking allowance, and 22,528 total tokens. Qubit optimizes all Q20/Willow/Heron cases with existing weighted reward and per-case feedback. Each qubit grading slot uses 4 CPU and 8 GiB, with 16 slots per host. No task objectives or grading budgets were changed.

`prepare.py` builds profiles, validates seed provenance, and packages current runtime code. `submit.py` checks identities, bundle hashes and destination namespaces, writes durable attempt receipts before CLI dispatch, and stops on uncertain submission. `jobs.json` records source generations and checksums; `submissions.json` records accepted IDs. Package-local immutable bundles and receipts are under `.science/packages/capacity-seven-20260920/`.

`monitor.py` is the four-minute background collector, covering the original 24 jobs, these six jobs, and the direct v5p-64 run. Runtime probes, raw checkpoint journal entries, and controller states are recorded separately in `.science/monitor-20260920/`. Checkpoints must not be reported as verified optimizer updates without additional training evidence. No automatic cancellation or relaunch occurs in this collector.

At dispatch, five new pool jobs acquired workers and Gemma qubit waited for a v5p slot. Submitted/STARTING does not prove service readiness or successful training. No new optimizer step is claimed here.
