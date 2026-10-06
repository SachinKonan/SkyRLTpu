# Gemma AC2 recovery — September 23, 2026

Requested outcome: resume existing AC2 Gemma job 1638 on v4-64, preserving its saved step-6 state and 15-step target.

## Findings and repairs

The pool had four READY workers, confirmed against four READY/HEALTHY provider TPU nodes. Workers 726, 761, and 773 were assigned to running circuit/qubit jobs. Worker747 was physically idle across all eight hosts but retained Qwen AC2 job1637's nonterminal reservation after worker-local job9 failed. That attempt failed its clean-host disk check: only 16.29 GiB free versus 30 GiB required.

Under the pool scheduling lock, verified all eight hosts had no TPU owners, runtime/grading systemd units, or private executor processes. The head's last Qwen attempt was FAILED with an end time. No pending/running SkyPilot execution request existed for that worker before repair.

On worker747's head only, removed six CRC-verified local duplicates of old AC2 Qwen checkpoint archives already in central2 GCS, removed regeneratable downloaded code cache, and ran `uv cache prune --ci`. Checkpoints in GCS, client state, databases, and logs were retained. Head free space increased to 35.09 GiB. Other hosts already had more than 47 GiB free.

Raised only Gemma AC2 job1638's persisted DAG/queue priority from 100 to 120. Cleared only Qwen AC2 job1637's verified stale worker binding using SkyPilot state APIs, preserving its queued job and checkpoints. No pool worker or unrelated workload was cancelled.

## Remaining capacity blocker

After release, the existing Gemma circuit recovery job1635 claimed worker747 and registered SkyPilot execution request `ce8fd5e0-af46-4666-9e91-2b3985f497ac`. Gemma AC21638 remained pending. The priority change did not prevent the existing recovery controllers from racing for that worker; it is not a guarantee of the next assignment.

Asked the user whether to defer circuit job1635 and give worker747 to AC2, because circuit work is owned by the other thread. No action against circuit1635 has been taken in this recovery work.

Evidence: `.science/ac2-gemma-recovery-20260923/{before.json,cleanup.json,priority-before.json,repair-result.json,resume-checks.json,repair.log}`.
