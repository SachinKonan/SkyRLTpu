# Qubit routing bootstrap regrade comparison — 2026-09-21

Complete imports are reported below; unfinished models are explicitly pending. These are bootstrap results, not learning improvements from the new training runs.

All models reuse their own original 1,024 draft occurrences, deduplicate exact source, and evaluate each unique program once with the same `parallel-v2` evaluator. Each valid program passes the complete 72-case suite with unchanged reward, trial counts and seeds. Four case workers share one 1,900-second candidate deadline. No duplicate padding or best-of-replay selection is used.

Evaluator SHA256: `cc7b510448d67a1f9a776fbcf83e187c72ec3b428b6a3fc1f9f00ca18e35fbe1`. Fresh training uses the regraded seed pool with reset PUCT statistics, adapter and optimizer. Historical drafts retain their original prompt provenance; new training prompts include the Gemini targets.

| Model | Unique programs | Valid drafts before → after / 1,024 | Unique valid before → after | Newly valid unique | Unique timeouts before → after | Retained seeds | Best reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gemma | 196 | 494 → 579 | 96 → 111 | 15 | 25 → 10 | 111 | 0.5329396105 |
| Qwen | 114 | 725 → 725 | 46 → 46 | 0 | 0 → 0 | 46 | 0.5202905231 |
| Muse | Pending | Pending | Pending | Pending | Pending | Pending | Pending |

## Best single-policy scores

Each row is one complete verified policy, selected by the unchanged combined reward. Positive target gaps mean more SWAPs than the supplied Gemini reference; lower is better. Independent topology minima below must not be combined into a fictitious policy.

| Model | Q20 SWAPs (gap) | Willow SWAPs (gap) | Heron SWAPs (gap) | Policy source SHA256 |
|---|---:|---:|---:|---|
| Gemma | 15,987 (+2,517) | 34,661 (+3,180) | 44,892 (+2,496) | `1e55c0c740b7bd723ac63f109961d94b98a95841971cc2e6f84a8c2e2f976946` |
| Qwen | 17,517 (+4,047) | 37,393 (+5,912) | 45,952 (+3,556) | `f7a7e1d15c4bba7fe287818e3a7eb15ee71014b20b7691a335fa7eebb4f0ed17` |

## Measured grading cost

Slowest-host elapsed time measures its complete regrade workload, excluding environment setup. Summed program wall time is not elapsed campaign time. CPU cost covers the accepted regrade attempt; earlier failed validation/regrade attempts are recorded separately in the relaunch status and are not silently included in these throughput figures.

| Model | Slowest host elapsed | Summed program wall | Evaluator CPU | Median valid program | Raw timeout occurrences before → after |
|---|---:|---:|---:|---:|---:|
| Gemma | 37.39 min | 19.60 h | 26.04 CPU h | 5.86 min | 139 → 54 |
| Qwen | 5.91 min | 4.18 h | 10.38 CPU h | 5.42 min | 0 → 0 |

## Gemma provenance and topology minima

Target run: `qubit-v4-gemma-parallel2-20260921`. Seed pool SHA256: `3360bd344a015350e2c84e374020d92538de5ee08ded58e532e68a7446f58453`. Source manifest SHA256: `7c9910d4982cf66dc534763a84edc94a1196d37eafe887c756dd7566133bcf79`.

| Topology | Independent minimum SWAPs | Source SHA256 |
|---|---:|---|
| q20 | 15,987 | `1e55c0c740b7bd723ac63f109961d94b98a95841971cc2e6f84a8c2e2f976946` |
| willow | 34,661 | `1e55c0c740b7bd723ac63f109961d94b98a95841971cc2e6f84a8c2e2f976946` |
| heron_fez | 44,892 | `1e55c0c740b7bd723ac63f109961d94b98a95841971cc2e6f84a8c2e2f976946` |

Detailed evidence: `.science/routing-relaunch-20260921/seeds/gemma/seed-import.json`, `rank-completion.json`, and the immutable GCS verdicts named in `regrade-v3/gemma/manifest.json`.

## Qwen provenance and topology minima

Target run: `qubit-v4-qwen-parallel2-20260921`. Seed pool SHA256: `b7d12ba1ed47a8fab47d5bc8839fb9c32f5e1e4cbe7c7f22d4efdc71a0a95e29`. Source manifest SHA256: `65a6f7b7f4e6c42aa1273fdcbe2ab9b7fc2e82aebf9f45bbcf256bdf51c7c66d`.

| Topology | Independent minimum SWAPs | Source SHA256 |
|---|---:|---|
| q20 | 17,517 | `f7a7e1d15c4bba7fe287818e3a7eb15ee71014b20b7691a335fa7eebb4f0ed17` |
| willow | 36,969 | `00b0921e1235136f6815c9cf80728a4e6a7f227121ca109b162950e7ee5d5810` |
| heron_fez | 45,952 | `f7a7e1d15c4bba7fe287818e3a7eb15ee71014b20b7691a335fa7eebb4f0ed17` |

Detailed evidence: `.science/routing-relaunch-20260921/seeds/qwen/seed-import.json`, `rank-completion.json`, and the immutable GCS verdicts named in `regrade-v3/qwen/manifest.json`.
