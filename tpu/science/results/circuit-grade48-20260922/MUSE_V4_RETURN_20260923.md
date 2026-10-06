# Muse circuit return to v4-64, September 23

User requested return from v6e-central to the free v4-64 worker. Run identity remains `circuit300-v464-muse-pwc05-three-starts-10step-20260922-r4`.

Cancelled only Muse managed job 1612. Latest durable optimizer and shared search state remain at step 6, best reward 0.5104587089575159. Copied the stopped run's authoritative central1 state back to central2: 1,366 objects inspected, 161 differing/missing objects copied, every destination size and CRC32C verified. Before replacing differing central2 objects, preserved them under `gs://sk7524-tinker-tpu-us-central2/migration-backups/muse-v6e-to-v464-20260923/`. Frozen source generations were rechecked after transfer. No checkpoint objects were deleted from GCS.

New profile `tpu/swarm/ray_train/profiles/circuit300-muse-v464-resume6-to15-20260923.json` derives from the existing v4-64 continuation profile with minimum checkpoint step changed from 10 to 6. Total target is 15, strict checkpoint resume enabled. Same adaptive PWC rho 0.5, 16 groups x 32 rollouts, 300-second placement search, model/seed/optimizer settings and original bootstrap contract. v4-64 uses TP8/FSDP2 on four trainer hosts, process bounds 1,1,4, four local inference engines, 48 grading slots per host, 64-GiB role caches and farm borrowing. Original v4 compile-cache destinations are reused. Bootstrap reuse validation passed against the copied original contract; no reverse-migration code exception was needed.

Built using Slurm CPU job 14336509. Verified local and uploaded immutable bundle hashes before submission.

Free worker 726's head and SSH worker4 initially had only 22 GiB free. With no workloads or TPU owners, reclaimed checkpoint replicas from earlier AC2/qubit runs only after matching GCS size and CRC32C. Also removed disposable upload tar files for adapters whose sampler checkpoint exists durably. Preserved databases, client/search state, all GCS copies, worker allocation and workloads on other workers. Re-ran the packaged startup audit across all eight hosts: all passed. Head now has ~50 GiB free; worker4 ~70 GiB. Retired owned RAM caches are handled by the executor's existing guarded cache-admission path.

Detailed transfer, cleanup, per-host audits, bundle manifest and submission receipt live under `.science/launch/muse-v4-return/`. Submission is not itself proof of successful restore; record live resume evidence below when available.

Submitted replacement **job 1646** at 19:06 EDT. Controller selected the audited **v4-64 worker 726**. It is in the launch path; checkpoint restore and new sampling are not yet verified. Old v6e job 1612 is cancelled. The submission receipt pins resume checkpoint `tinker://model_3d182644/weights/000006` and target 15.
