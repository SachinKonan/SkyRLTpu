# Circuit continuation to 15 total steps — September 23, 2026

User requested all three circuit-placement models continue to 15 total optimizer steps.

| Model | Existing job | Durable checkpoint verified during staging |
|---|---:|---:|
| Qwen | 1602 | 4 |
| Gemma | 1591 | 3 |
| Muse | 1594 | 4 |

The existing clients retain their startup limit of 10. Their in-flight work is not interrupted. The user systemd service `circuit-continue15-20260923.service` checks the controller every 60 seconds and submits each continuation after its predecessor succeeds, no other job for that run is active, and GCS reports a checkpoint at least at step 10. A failed predecessor is reported and is not silently bypassed.

Continuation profiles are the corresponding `grade48-cache64-continue15.json` files in `tpu/swarm/ray_train/profiles/`. The only configuration differences are `NUM_EPOCHS=15` and `resume_min_checkpoint_step=10`. They preserve run IDs, bootstrap contracts, model checkpoints, sampler state, grading settings, and inference-farm configuration. Existing bootstrap contracts validated successfully for all three. Bundles were uploaded and their downloaded SHA256 hashes verified.

Pool: `tpuswarm-v4-64-central2-qwen35-erdos`. The continuation uses the existing durable run namespace in `gs://sk7524-tinker-tpu-us-central2/ray-training/` and trains steps 11–15 after restoring step 10.

Operational files: `.science/launch/circuit15/` contains the plan, build output, manifests, staged verification, watcher, and log. The watcher writes an exclusive submission receipt before launching, preventing automatic duplicate submissions if launch acknowledgement is uncertain. An uncertain receipt requires manual reconciliation. Successful submissions create `<model>-submission.json` with the new job ID. The service exits once all submission receipts exist; it does not certify successful completion of step 15.

Inspect:

```bash
systemctl --user status circuit-continue15-20260923.service
tail -n 20 .science/launch/circuit15/watch.log
```

No current training jobs were cancelled and no workers were released.
