# Qubit routing comparison — September 23, 2026

Cached references only. Current run results use one best saved valid program per model, evaluated over 72 cases (24 per topology). No candidate replay or new grading was performed. Lower SWAP counts are better.

| Method | Q20 | Willow | Heron | Total SWAPs | Mean per case | Weighted reward |
|---|---:|---:|---:|---:|---:|---:|
| SABRE | 22,714 | 38,352 | 50,186 | 111,252 | 1545.17 | 0.500000000 |
| LightSABRE | 20,063 | 36,802 | 45,827 | 102,692 | 1426.28 | 0.518785493 |
| SimpleTES 20B | 14,180 | 32,135 | 43,032 | 89,347 | 1240.93 | 0.548415609 |
| SimpleTES 120B | 15,147 | 32,258 | 42,274 | 89,679 | 1245.54 | 0.548872118 |
| Gemini target | 13,470 | 31,481 | 42,396 | 87,347 | 1213.15 | 0.553413441 |
| Qwen | 13,905 | 32,293 | 42,519 | 88,717 | 1232.18 | 0.549902427 |
| Gemma | 13,561 | 31,847 | 41,950 | 87,358 | 1213.31 | 0.553519234 |
| Muse | 16,564 | 36,877 | 45,035 | 98,476 | 1367.72 | 0.525516995 |

Rewards use weighted added-gate costs with coefficients Q20=0.2, Willow=0.4, Heron=0.4; the unweighted total can rank methods differently. SimpleTES topology aggregates come from our cached published tables, including GPT-OSS-20B/120B and Gemini. SABRE/LightSABRE have cached case counts. Per-case data for SimpleTES is unavailable in this comparison.

Our saved grades have not been independently replayed with repeated seeds; small margins do not establish a robust SOTA improvement. This is a comparison against cached references, not an exhaustive global SOTA survey.

Current snapshots: Qwen step15, Gemma step13, Muse step13. See `comparison.json` for provenance and `per-case.csv` for all72 cases.
