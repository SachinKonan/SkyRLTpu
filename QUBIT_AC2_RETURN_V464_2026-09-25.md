# Qubit and AC2 return to Central2 v4-64 — September 25, 2026

The user authorized immediate migration of the three v5p jobs to existing v4-64 capacity, with one state transfer and region-local operation thereafter. This is a hardware migration, not a new optimizer/search restart.

| Run | Cancelled East5 job | New Central2 job | Durable starting checkpoint | Target |
|---|---:|---:|---:|---:|
| Qubit Gen-1 Muse16 on Gemma20 | 1751 | 1755 | Branch step 0; recipient Muse16 adapter | 10 branch updates |
| AC2 Gemma | 1752 | 1756 | 13 | 20 total updates |
| AC2 Muse | 1753 | 1757 | 16 | 20 total updates |

All three originals were confirmed CANCELLED before submission. Existing v4-64 Qubit Qwen 1736 and Circuit Qwen/Muse 1728/1729 were not changed. New priorities are 710, 700, and 690 respectively, in pool `tpuswarm-v4-64-central2-qwen35-erdos`, zone `us-central2-b`.

## Preserved state and migration validation

- GCS server-side copies were pinned to source object generations and verified by byte size and CRC32C. Total one-time East5 to Central2 transfer: 17,985,930,147 bytes (16.75 GiB). The copy receipts cover training archives, latest sampler archives where applicable, client/search/history files, and portable queue/database snapshots. Old superseded checkpoint archives remain at their original source; the exact latest resume checkpoints are staged in Central2.
- AC2 Gemma retains its adapter, all Adam moments/hyperparameters/counters at update 13, and its 571-state search tree (`puct_T=6448`, 208 populated visit/value entries). Best tree reward: 0.9541787586280039.
- AC2 Muse retains its adapter, all Adam moments/hyperparameters/counters at update 16, and its 822-state search tree (`puct_T=7936`, 248 populated visit/value entries). Best tree reward: 0.9429493624651926.
- The AC2 numerical NPZ payloads are byte-identical to the source. Only saved device-placement metadata was removed so the v4 trainer uses its destination mesh. This does not reset or alter optimizer tensors.
- Muse AC2's durable training archive and matching PUCT snapshot reached 16, while its logging counter/metrics stopped at 15 and the copied database lacked the 16 checkpoint registration. The model registration exists. After validating the archive and actual optimizer counters, the new offline database was repaired to register the durable training/sampler checkpoint. The original database is retained under `migration-audit/qubit-ac2-return-v464-20260925/ac2-muse/source-tinker-backup.db`. Logging/life history was preserved without fabricating a missing metrics row.
- Qubit Muse has no successful experimental optimizer update to restore. Its initialized Muse16 adapter and exact 711-state Gemma20 seed pool are retained; the existing branch begins at zero with a fresh optimizer and reset PUCT statistics. All eight native K/V LoRA B tensors were verified to expand exactly to the original Central2 v4 adapter. Every other tensor matched exactly. No donor-model weights were substituted.
- Portable SQLite snapshots passed `PRAGMA quick_check`. The frozen executor retires dead-client pending requests before checkpoint recovery; it does not replay the old interrupted batch. Source database/history objects remain preserved.

## Execution and regional storage

The frozen source bundles are reused with new profiles. Ray v2 remains the executor. Each v4-64 job uses four trainer hosts and four independent TP4 inference engines. Gemma training uses TP4/FSDP4; Muse uses TP8/FSDP2 with eight logical KV heads.

Sampling, loss, learning rate, targets, grading recipe, and seed settings are preserved from each moved run: 16 groups of 32, adaptive piecewise-centered rho 0.5, importance_sampling loss, LR 4e-5. AC2 retains context 22,528 / phase-one 16,384. Qubit retains its current context 18,432 / phase-one 13,824. The v4 trainer uses its established packing/padding layout.

All active-profile and task GCS references point to `gs://sk7524-tinker-tpu-us-central2`, whose actual bucket location was checked as `US-CENTRAL2`. Runtime/CPU bundles, HF weights, Orbax weights, training/sampler checkpoints, and compilation seeds were verified to exist there. Compilation seeds are the existing **v4** caches, not v5p caches. New run-specific compilation output paths remain Central2. Farm borrowing stays disabled as in the moved v5p profiles; each job has its own four local inference engines, avoiding a new cross-region serving dependency.

Trainer and inference model caches remain RAM-backed (96 GiB trainer / 128 GiB inference; 240 GiB reserve). Ports retain the validated isolated range: Ray workers 30000–30999, fixed services 31679–31807, systemd control 31900.

Before submission, all eight hosts of each idle worker 794, 796, 799, and 801 passed TPU-owner, local-job, disk-space, and RAM checks. Minimum free disk was 83 GiB/host. The temporary CPU-only checkpoint inspection files on 794 were removed after validation, reclaiming 25 GiB. No unrelated workloads or model caches were removed.

## Evidence and monitoring

All preparation, immutable bundle hashes, source inventories, transfer receipts, profile diffs, numerical-state proofs, database repair evidence, and submission receipts are in `.science/qubit-ac2-return-v4-20260925/`.

User systemd services on della9:

- `qubit-ac2-v464-migration-progress-20260925.service`: checks control-plane state, live serving, logs, and saved checkpoints every minute.
- `qubit-ac2-v464-migration-dispatch-20260925.service`: dispatches only these jobs' exact native pending SkyPilot requests when the shared API queue stalls; verifies job identity, task/environment equality, worker idleness, and disk space first. It does not create duplicate jobs or restart shared controllers.

Live evidence is written to `latest-progress.json` and each run's `latest-progress.json` / `progress-history.jsonl`. Old run-specific v5p dispatch/progress monitors were stopped after cancellation.

Submission/worker assignment is not proof of serving readiness or a successful v4 optimizer update. Those must be reported from live logs/checkpoints as startup completes.

## Latest launch verification

Snapshot: 2026-09-25T12:56:08.744595+00:00.

| Run | Job | Worker | Sky status | Executor phase |
|---|---:|---:|---|---|
| qubit-muse | 1755 | 794 | RUNNING | cache_setup, environment_setup |
| ac2-gemma | 1756 | 796 | RUNNING | cache_setup |
| ac2-muse | 1757 | 799 | RUNNING | preflight_complete |

All three downloaded profiles matched the staged Central2 cache configuration. Frozen-bundle comparison found only the replaced profile; all 622/634/700 other files respectively were unchanged. No new optimizer update has yet been verified on the destination workers.
