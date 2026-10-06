# Circuit Gen-1 correction: carry recipient weights, reset optimizer

The first Gen-1 submissions (Qwen 1701 and Muse 1702) incorrectly initialized fresh adapters. They are invalid for the intended weight-continuation experiment. Cancellation was explicitly requested and both controller logs now confirm cancellation. Their GCS results remain intact. The previous automatic submission service was disabled; original Muse Gen-0 job 1646 was left untouched.

The corrected branches use the existing `ensemble.py` initialization path, `TTD_INIT_STATE_PATH_QWEN` / `TTD_INIT_STATE_PATH_MUSE`, which invokes `create_training_client_from_state_async` before step zero. There is no recipient checkpoint journal at initialization, so preemption-resume does not supersede this initialization path. Subsequent checkpoints resume normally with optimizer state within this new generation.

| Recipient | Source weights | Optimizer | Search pool | New steps |
|---|---|---|---|---:|
| Qwen | Gen-0 step 15, `tinker://model_a688983a/weights/final` | Fresh | Gemma step 15, 567 programs | 10 |
| Muse | Gen-0 step 14, `tinker://model_52a709a0/weights/000014` | Fresh | Same Gemma pool | 10 |

Muse's original run has not saved step 15; this branch deliberately uses its latest durable step 14 without cancelling or modifying that original job.

## Guaranteeing the optimizer reset

The local Tinker API's `load_weights` path invokes the backend's `load_checkpoint`, which restores `optimizer_state.npz` whenever present. The weights-only SDK method alone therefore does not establish a fresh optimizer in this deployment.

`tpu/science/circuit_weight_handoff.py` creates a new archive in the new run's storage namespace, removes `optimizer_state.npz` and `optimizer_layouts` metadata, and checks that `lora_weights.npz` is byte-identical by SHA256 before and after. Original donor archives are not modified. Without optimizer payload the backend leaves the newly initialized optimizer untouched. A minimal SQLite registry carries only the source model, its session and the chosen completed checkpoint; no old pending/completed training requests or optimizer client history are imported.

Validation covers byte-preserving weights extraction, optimizer omission, selected-checkpoint registration and exclusion of old requests. Actual artifact processing verifies source and destination SQLite integrity, base-model/rank compatibility, checkpoint completion and retained tensor checksum. Cloud uploads are create-only, CRC32C checked, and region-local (us-central2). Final bundles retain the Ray v2 executor, local inference, farm borrowing, adaptive PWC rho 0.5, importance_sampling loss and IBM17 grading recipe.

Artifacts and final submission receipts live in `.science/launch/circuit-gen1-weights/`. Packaging and archive processing run on Slurm CPU job 14400440 (replacing packaging-only job 14400264 to use faster lossless gzip compression). `*-handoff.json` pins source GCS generations, checkpoint paths and tensor hashes. `*-staged.json` pins destination objects. `*-submission.json` is the authority for successful replacement submission; preparation alone is not proof of launch or runtime restoration.

New run IDs:
- `circuit300-gen1-qwen-weights-on-gemma15-pwc05-10step-20260924`
- `circuit300-gen1-muse-weights-on-gemma15-pwc05-10step-20260924`

## Submitted replacements

- **1715:** Qwen weights continuation from Gen-0 step 15.
- **1716:** Muse weights continuation from Gen-0 step 14.

Both were submitted successfully to `tpuswarm-v4-64-central2-qwen35-erdos`, retaining the cancelled predecessors' priorities (160 and 125). Source LoRA bytes verified unchanged, optimizer payloads omitted, both bundles and region-local uploads verified, two focused tests passed. Old jobs 1701 and 1702 were confirmed CANCELLED before submission. Job 1646 remains separate and untouched.

Use `.science/launch/circuit-gen1-weights/status.py` for these replacements; the earlier Gen-1 status scripts refer to cancelled experiment IDs. Startup and actual initial weight loading must still be verified from the new jobs' runtime logs before claiming an executed weight continuation.

## Verified runtime and periodic monitoring (September 25)

The replacement submissions are now jobs **1728 (Qwen, worker 775)** and **1729 (Muse, worker 790)**; jobs 1715/1716 were cancelled before placement. Direct worker logs verify `weights initialized from tinker://model_a688983a/weights/final (fresh optimizer)` for Qwen and `weights initialized from tinker://model_52a709a0/weights/000014 (fresh optimizer)` for Muse. Both have completed optimizer steps and surpassed the inherited Gemma best. Snapshot: Qwen step 4, best 0.5194456909838393; Muse step 3, best 0.5183725808122623. Direct serving checks show four registered local engines and a ready four-engine borrowed farm each, with completed requests on both routes.

Read-only periodic service installed on `della-vis2`: `circuit-gen1-progress.timer` invokes `circuit-gen1-progress.service` every five minutes. First poll completed successfully; timer is active. It reads checkpoints, metrics and live controller records and writes `.science/launch/circuit-gen1-weights/monitor-status.json` and `monitor-history.jsonl`. It follows run IDs across resubmissions and performs no job cancellation, submission or resource mutation. A terminal controller record alone does not establish ten completed training steps. The completion indicator requires step-10 metrics, a checkpoint at least 10 and an existing nonempty corresponding archive. The monitoring task remains in progress until the experiment's completion is verified.
