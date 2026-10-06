# Gemma qubit Central2 compilation-cache repair — 2026-09-24

The user authorized correcting qubit Gemma job 1603's cross-region compile-cache paths and separately cancelling AC2 farm job 1619.

- **1619: CANCELLED**, verified in controller state. Shared pool retained.
- **1603: CANCELLED** before launching a same-run successor, preventing concurrent writers.
- **1675: PENDING** in `tpuswarm-v4-64-central2-qwen35-erdos`, with the same runtime run ID and target of 25 total optimizer steps. This is a submitted continuation, not a claim of resumed training.
- Durable checkpoint **22**, matching metrics, nonempty PUCT state, optimizer archive and completed checkpoint registry were verified before replacement. The original Central2 run-state location is retained.

Only `cache.trainer_compile` and `cache.inference_compile` changed inside the deployed profile. Both now use:

`gs://sk7524-tinker-tpu-us-central2/qubit-v4-continue25-20260923/qubit-v4-gemma-parallel2-pwc-rho05-20260921-r2/{trainer,inference}`

All 2,942 existing compile objects (1,832,053,065 bytes) were copied once from the previous East5 prefixes using server-side rewrite, with source generations pinned and destination size/CRC32C verified. This one-time cross-region staging preserves the warm compilation cache; future restore and writeback use Central2. No model weights were copied and no old cache/state objects deleted.

New immutable bundle SHA256:
`93834ba54c1ae09ce9930b0a281697ad65d3d4c2a735ca49dfb805f4e5f36a98`

Archive comparison verified that every other file is byte-identical to job 1603's deployed bundle. The controller's stored task for 1675 references this new URI/hash. All ten explicit GCS URIs in the new task/profile are Central2-local. Source profiles (normal and continue25) were also corrected to avoid reintroducing the wrong paths.

Private operational receipts, scripts and hashes: `.science/routing-relaunch-20260921/gemma-central2-cache-fix-20260924/`. This directory includes resume proof, copy receipts, cancellation verification, submission receipt and final controller verification.

No AC2 training job, unrelated farm or TPU pool was modified. Runtime resume verification awaits worker allocation.
