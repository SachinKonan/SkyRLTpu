# Gemma circuit checkpoint disk recovery, September 23

Run: `circuit300-v464-gemma-pwc05-three-starts-10step-20260922`.

Job 1591 completed optimizer step 10, then failed while packing the additional final checkpoint (`OSError: [Errno 28] No space left on device`). GCS contains the numbered step-10 optimizer archive (2,938,939,279 bytes), sampling archive (979,646,449 bytes), client checkpoint record and search state. Best recorded reward: 0.5149075514728466. This is not a missing step-10 checkpoint; the failed operation was the extra final save.

Worker 744 has eight roughly 97-GiB root filesystems despite the submitted YAML requesting disk_size 300. Existing pool worker disks do not acquire that requested size. The physical trainer host `t1v-n-c942d444-w-7` (SSH alias worker6; do not assume physical and logical rank equality) had 42 GiB of local checkpoint replicas and only 1.8 GiB free. All eight hosts had no TPU owners at the read-only audit.

Cancelled only managed job 1591, preserving worker 744. Reclaim older numbered local Gemma checkpoint replicas after checking size and CRC32C against GCS, retaining steps 9, 10 and final. No GCS objects are deleted. Remove disposable inference upload tar files for Gemma steps 2–8 from the client host; these are transport copies of durable sampler checkpoints. Preserve all databases, bootstrap/search state, recent adapters and unrelated runs.

Resume using the already-staged grade48/cache64 continuation profile: same run ID, strict checkpoint resume with minimum step 10, 15 total epochs. Scientific settings remain unchanged. This cleanup provides room for the five remaining steps; it is not a general automatic checkpoint-retention implementation. Growing local archives remain a concern for longer future continuations.

Submission receipt and exact per-host packaged startup audits are under `.science/launch/circuit15/gemma-submission.json` and `gemma-clean-audit-*.json`. Do not treat submission or SkyPilot RUNNING alone as evidence that checkpoint restore and sampling have succeeded.

Replacement job **1635** was submitted at 13:25 EDT and selected worker **744**. All eight copies of the packaged clean-host audit passed. SkyPilot dispatched it after 359 seconds. At 13:34 EDT all eight Ray executor hosts were in cache setup with topology probes complete and no reported cache errors. Startup verification is still in progress.

Cleanup logs record 28,409,746,545 bytes of checksum-verified checkpoint replicas removed in the final cleanup pass, plus 6,856,980,480 bytes of disposable adapter upload archives on the client host. Trainer and client filesystems each had approximately 31 GiB free before launch. GCS archives and search results remain intact.


Verified at approximately 13:43 EDT: client restored `tinker://model_b29c9399/weights/000010`, PUCT snapshot step 10 (413 states), and reported `Resume optimizer batch=10 search snapshot=10`. It started zero-indexed step 10 sampling (the batch for the 11th update), 16 groups × 32 rollouts. `/status` showed no fatal error, four registered local replicas, four active local group requests and four active remote group requests. Farm lease was ready at `http://10.202.0.113:24800` with four engines. Eight further group requests were queued. No step-11 checkpoint is claimed yet. See `gemma-disk-recovery/resume-evidence.txt`.
