# Qubit routing bootstrap regrade comparison — 2026-09-21

All three bootstrap regrades and seed imports are complete. These are bootstrap results, not learning improvements from the new training runs.

All models reuse their own original 1,024 draft occurrences, deduplicate exact source, and evaluate each unique program once with the same `parallel-v2` evaluator. Each valid program passes the complete 72-case suite with unchanged reward formula, trial counts and seeds. Four case workers share one 1,900-second candidate deadline. No duplicate padding or best-of-replay selection is used.

Evaluator SHA256: `cc7b510448d67a1f9a776fbcf83e187c72ec3b428b6a3fc1f9f00ca18e35fbe1`. Fresh training uses the regraded seed pool with reset PUCT statistics, adapter and optimizer. Historical drafts retain their original prompt provenance; new training prompts include the Gemini targets.

| Model | Unique programs | Valid drafts before → after / 1,024 | Unique valid before → after | Newly valid unique | Unique timeouts before → after | Retained seeds | Best reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gemma | 196 | 494 → 579 | 96 → 111 | 15 | 25 → 10 | 111 | 0.5329396105 |
| Qwen | 114 | 725 → 725 | 46 → 46 | 0 | 0 → 0 | 46 | 0.5202905231 |
| Muse | 245 | 128 → 203 | 48 → 82 | 34 | 122 → 88 | 82 | 0.5207190348 |

## Best single-policy scores

Each row is one complete verified policy, selected by the unchanged combined reward. Positive target gaps mean more SWAPs than the supplied Gemini reference; lower is better. Independent topology minima below must not be combined into a fictitious policy.

| Model | Q20 SWAPs (gap) | Willow SWAPs (gap) | Heron SWAPs (gap) | Policy source SHA256 |
|---|---:|---:|---:|---|
| Gemma | 15,987 (+2,517) | 34,661 (+3,180) | 44,892 (+2,496) | `1e55c0c740b7bd723ac63f109961d94b98a95841971cc2e6f84a8c2e2f976946` |
| Qwen | 17,517 (+4,047) | 37,393 (+5,912) | 45,952 (+3,556) | `f7a7e1d15c4bba7fe287818e3a7eb15ee71014b20b7691a335fa7eebb4f0ed17` |
| Muse | 17,213 (+3,743) | 37,261 (+5,780) | 46,078 (+3,682) | `1ba96846d65f9c47798211b0a4a1c268e784e2feab1ea696b24fb492379c8390` |

## Measured grading cost

Slowest-host elapsed time measures its complete regrade workload, excluding environment setup. Summed program wall time is not elapsed campaign time. CPU cost covers the accepted regrade attempt; earlier failed validation/regrade attempts are recorded separately in the relaunch status and are not silently included in these throughput figures.

| Model | Slowest host elapsed | Summed program wall | Evaluator CPU | Median valid program | Raw timeout occurrences before → after |
|---|---:|---:|---:|---:|---:|
| Gemma | 37.39 min | 19.60 h | 26.04 CPU h | 5.86 min | 139 → 54 |
| Qwen | 5.91 min | 4.18 h | 10.38 CPU h | 5.42 min | 0 → 0 |
| Muse | 74.26 min | 61.43 h | 25.40 CPU h | 10.79 min | 304 → 229 |

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

## Muse provenance and topology minima

Target run: `qubit-v4-muse-parallel2-20260921`. Seed pool SHA256: `19560c6f6f9784a6bc96c4fc5c01dffe36bb57e308fff53400849e28d7546c1a`. Source manifest SHA256: `d7a95891efac9371907a689373b73923add298174f9fde04fdc6deb7e3dd76c7`.

| Topology | Independent minimum SWAPs | Source SHA256 |
|---|---:|---|
| q20 | 16,120 | `a9811a020b92106ef5214404ad0ab5d1cbe641066eeb1e2d49f929aff30b660a` |
| willow | 36,977 | `4e3fa61004d4f9cb43e29da401074a862ba3c40b795c3a9536bbd17758d979c0` |
| heron_fez | 45,926 | `0d159e8e7d750105f9944cec6b57cf35bdd135598680dee57b5bee27461a57c3` |

Detailed evidence: `.science/routing-relaunch-20260921/seeds/muse/seed-import.json`, `rank-completion.json`, and the immutable GCS verdicts named in `regrade-v3/muse/manifest.json`.

## Historical reward variation

Exact-source duplicates were not always deterministic in the original archive. Three Gemma sources and one Muse source had multiple distinct valid rewards before this regrade. Qwen had no such observed variation. This establishes pre-existing variation; it does not attribute every changed score to a specific cause or prove full serial/parallel equivalence for every source.

Muse’s original winning source appeared eight times, with valid rewards from 0.5196652959 to 0.5209892511. Its single canonical regrade scored 0.5198302531. The regrade does not repeat candidates to recover an earlier maximum. Gemma’s and Qwen’s original winning sources retained their exact original best rewards.

Source IDs, occurrence counts, and ranges: `.science/routing-relaunch-20260921/historical-reward-variation.json`.
