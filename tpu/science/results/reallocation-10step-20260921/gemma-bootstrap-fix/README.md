# Gemma RG-LRU bootstrap reuse repair

The user authorized cancellation and resubmission of job 1337 after fixing its
bootstrap configuration rejection. Job 1337 was verified CANCELLED before
replacement job **1343** was submitted to `tpuswarm-v6e32-east5b-qwen35` at
priority 110. No other job was cancelled.

The canonical run remains `fresh-v6e-gemma-rglru-grpo-lr4e5-s1-20260919-fix1`.
It retains the same cloud namespace, model, training recipe, and completed seed
pool. `NUM_EPOCHS=10` is the total optimizer-step cap. The preflight found no
optimizer checkpoints yet.

## Cause and fix

The old startup check compared the entire saved bootstrap configuration with the
current training configuration. New compile-cache destinations, external farm
controls, and the 15-to-10 step cap caused a mismatch. Adding the farm-borrowing
wrapper also changed the bootstrap implementation hash.

Commit `306db916` adds explicit completed-bootstrap reuse with two SHA256 pins:

- Contract: `580278defc02e091e8d73ad056e86926747b8bbb323a916947879b7d66d74028`
- Pool: `38db6765f99decd5c642fccb0731231b7445a29f2d67a684036a46df396a65c5`

The stored contract content, recorded contract checksum, completion record, and
actual promoted pool must agree with those pins. Missing completion never triggers
regeneration. Only compile writeback destinations, external-pool controls, and
`NUM_EPOCHS` may differ; all other normalized configuration remains exact.
Unpinned bootstrap runs retain the existing strict checks. The original contract,
completion record, and promoted pool are preserved unchanged.

Cloud verification found **1,024 drafted, 120 valid, 29 retained, zero optimizer
steps**. Separate generation-pinned cloud snapshots were created and verified
before cancellation. Their locations are recorded in `preflight.json`.

## Validation and deployment

- Focused suite: **30 passed, 7 subtests passed**, Slurm job 14216114.
- Broader initial bootstrap suite: 52 passed, 5 failed, 7 subtests passed. The five
  failures were missing `examples.circle_packing` / `examples.ac_inequalities`
  imports in the local test environment. This broader suite was not fully green.
- The actual cloud-restored contract, completion record, and pool passed the new
  `Host.bootstrap_status()` on a CPU node before packaging; see `verification.jsonl`.
- Immutable bundle: `5d9638d7ebe5fa15b43546f5d3cbc0545bbc881e0fa4fbba595a4090857ec382`.
- New profile: `fresh-v6e-gemma-rglru-grpo-lr4e5-s1-20260919-fix1-bootstrap-reuse-20260921.json`.
- `upload.json` records the verified GCS object generation. `cancel-verified.json`
  and `submission.json` record the exact old and new jobs.
- This bundle also includes reliability commit `6f743060`: bounded initial farm
  admission, cancellable-acquire support where the farm advertises it, and the
  revised handling of transient status failures and rejected generation requests.

`live.json` and `placement.json` record job 1343 STARTING on worker 5590. At the
last check the new bundle had not reached the worker and there were no new
bootstrap events (`startup.json`). The successful reuse check was performed
locally on the cloud-restored artifacts; TPU startup acceptance and a completed
optimizer step remain unverified.
