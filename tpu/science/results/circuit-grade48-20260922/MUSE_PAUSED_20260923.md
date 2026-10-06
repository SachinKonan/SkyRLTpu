# Muse circuit run paused — September 23, 2026

User superseded the request to migrate immediately to v6e-central: cancel the existing v4-64 queue entry for now and note a possible later move.

- Job: 1594, `circuit300-v464-muse-pwc05-three-starts-10step-20260922-r4`.
- Source pool: `tpuswarm-v4-64-central2-qwen35-erdos`; former worker 727 became unhealthy. Cancellation requested while job was PENDING, with no allocated replacement.
- Latest verified durable checkpoint: **6**; best reward **0.5104587089575159**. Eventual requested target: **15 total steps**.
- State remains at `gs://sk7524-tinker-tpu-us-central2/ray-training/circuit300-v464-muse-pwc05-three-starts-10step-20260922-r4/`.
- Checkpoint: `tinker://model_3d182644/weights/000006`; sampler `tinker://model_3d182644/000006`.
- Original bootstrap, sampler pool, metrics, and trainer state are preserved. No migration copy or destination job was started.

Muse was removed from `.science/launch/circuit15/plan.json` and the watcher was restarted with Qwen and Gemma only. The previous plan is preserved as `plan-before-muse-pause.json`. Thus the v4 continuation cannot automatically resubmit Muse.

Possible later destination: `tpuswarm-v6e32-central1b` in `us-central1-b`, with state copied to the central1 bucket. This is a deferred option, not a submitted migration. Before moving, inspect free workers and their host state, configure v6e topology, verify cross-topology checkpoint restoration, explicitly validate bootstrap contract reuse for hardware/bucket changes, and require a checkpoint of at least 6. Preserve adaptive PWC rho 0.5, 17-case grading, cached starts and existing sampling settings. Current bootstrap reuse validator does not automatically permit accelerator/bucket/topology changes; handle that explicitly rather than silently altering the contract.

Controller verification after cancellation: job 1594 is **CANCELLED**. Continuation service is active and reports only Qwen 1602 and Gemma 1591, both RUNNING.

Superseded by subsequent user authorization: migration was completed and v6e-central job **1609** submitted. See `MUSE_V6E_CENTRAL_RESUME_20260923.md` for copied-state evidence and actual destination worker.
