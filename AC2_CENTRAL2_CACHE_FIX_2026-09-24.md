# AC2 regional compilation-seed repair — September 24, 2026

The user authorized cancelling AC2 Gemma 1638 and Qwen 1637, correcting all cross-region storage sources, and requeuing both continuations.

| Model | Cancelled job | Replacement | Verified saved checkpoint | Target | Preserved queue priority |
|---|---:|---:|---:|---:|---:|
| Gemma | 1638 | 1686 | 8 | 15 | 120 |
| Qwen | 1637 | 1687 | 9 | 15 | 119 |

Both replacements were accepted into `tpuswarm-v4-64-central2-qwen35-erdos` in us-central2-b. Submission and queue state are not proof of resumed training; runtime optimizer/search-state restore remains to be checked after allocation.

## Exact change

Only `cache.trainer_compile_seed` and `cache.inference_compile_seed` changed in each selected profile inside its immutable bundle. Both now use:

`gs://sk7524-tinker-tpu-us-central2/regional-compile-seeds-20260924/ac2-{gemma,qwen}/{trainer,inference}`

Every other archived file was compared byte for byte with the predecessor's deployed bundle. The same run ID, durable state location, Ray v2 executor, model, LoRA, optimizer, search pool, sampling, loss, grading configuration and 15-step target remain. Local canonical profile JSONs were written from the corrected deployed profiles. Historical deployment receipts were retained.

Gemma's 2,942 compile objects (1,832,053,065 bytes) were staged from the already verified **Central2** copy made for the qubit repair, avoiding another cross-region copy. Qwen's 2,146 objects (1,472,969,653 bytes) were copied once from the old East5 seeds. Copies pinned source generations, rejected destination conflicts and verified destination size and CRC32C. Subsequent seed reads use Central2. No model weights were copied or checkpoints deleted.

## Resume and concurrency checks

Before submission, each predecessor was confirmed CANCELLED. No other nonterminal job owned its runtime run ID. The latest checkpoint ledger entry had matching metrics, a nonempty corresponding PUCT snapshot object, and an existing optimizer archive. The downloaded portable registry passed SQLite `quick_check` and contained the matching COMPLETED training checkpoint. Saved checkpoints were Gemma 8 and Qwen 9.

Submission intents, receipts and locks prevent blind duplicate resubmission. The controller's persisted task was checked against the new bundle URI/SHA. The prior queue priorities were restored through SkyPilot's persisted-job update function, preserving the task contents; the old submitted resource YAML had priority 100, while the live job priorities had previously been raised to 120/119.

## Regional verification

The post-submission audit reopens every nonterminal AC2/qubit job's submitted bundle, extracts the actual selected profile, enumerates task/profile GCS buckets and compares their actual GCS locations with the configured TPU zone. Results are in `.science/ac2-central2-cache-fix-20260924/all-active-regional-verification.json`.

This verifies configured GCS reads and writes, including runtime bundles, HF/Orbax weights, compilation seeds/destinations and bucket-derived run-state locations. Farm HTTP traffic is separate: discovery/leases are dynamic and must be checked at runtime; no claim is made that every possible future network connection is confined to the region.

Operational evidence: `.science/ac2-central2-cache-fix-20260924/`, including per-model manifests, archive equivalence, resume proof, copy receipts, submission receipts, and controller verification. AC2 farm 1619 and predecessor qubit job 1603 remain cancelled; no worker was released or pool resized.
