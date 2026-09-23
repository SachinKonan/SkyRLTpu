# Qubit routing: current best programs versus cached baselines — September 22, 2026

One best saved program per active PWC model, selected by its full-suite weighted reward. All 72 saved case counts are present and their recomputed rewards match. These are observed valid training evaluations, not independent reruns or confidence intervals. Lower SWAP counts are better; added CNOTs equal three times SWAPs.

## Aggregate comparison

| Method | Q20 | Willow | Heron | Total SWAPs | Mean per case | Weighted mean per case | Reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| SABRE | 22,714 | 38,352 | 50,186 | 111,252 | 1545.17 | 1664.92 | 0.500000000 |
| LightSABRE | 20,063 | 36,802 | 45,827 | 102,692 | 1426.28 | 1544.34 | 0.518785493 |
| SimpleTES 20B | 14,180 | 32,135 | 43,032 | 89,347 | 1240.93 | 1370.95 | 0.548415609 |
| SimpleTES 120B | 15,147 | 32,258 | 42,274 | 89,679 | 1245.54 | 1368.43 | 0.548872118 |
| Gemini target | 13,470 | 31,481 | 42,396 | 87,347 | 1213.15 | 1343.53 | 0.553413441 |
| Qwen | 14,635 | 32,255 | 43,434 | 90,324 | 1254.50 | 1383.44 | 0.546168293 |
| Gemma | 15,054 | 31,834 | 44,482 | 91,370 | 1269.03 | 1397.38 | 0.543681764 |
| Muse | 16,604 | 36,685 | 45,367 | 98,656 | 1370.22 | 1505.90 | 0.525075033 |

Each topology has 24 cases. The ordinary average is total SWAPs / 72. The weighted average uses the pinned per-case coefficients Q20=0.2, Willow=0.4, Heron=0.4, divided by their sum (24). Reward is B/(B+C), using weighted baseline/candidate added-CNOT totals; it is not the average of per-case rewards.

SABRE and LightSABRE have cached case-level data. SimpleTES 20B/120B numbers are cached published topology aggregates, not local reruns; no per-case SimpleTES records were found. Gemini is an aggregate target stored in our routing resource contract. Recomputed rewards for these reference aggregates are derived here, not claimed as published rewards.

## Current programs and pairwise wins

| Model | Job | Snapshot | Reward | W/T/L vs SABRE | W/T/L vs LightSABRE |
|---|---:|---|---:|---|---|
| Qwen | 1483 | puct_sampler_step_000009.json | 0.546168293 | 46/3/23 | 36/3/33 |
| Gemma | 1493 | puct_sampler_step_000004.json | 0.543681764 | 51/1/20 | 37/6/29 |
| Muse | 1482 | puct_sampler_step_000008.json | 0.525075033 | 48/4/20 | 20/3/49 |

Source hashes, state IDs, collection timestamps, saved diagnostics, baseline hashes, and per-case deltas are in `comparison.json`; complete machine-readable counts are in `per-case.csv`. Candidate sources are archived alongside them.

## Historical distinction

The September 19 independent Gemma replay scored 0.545904148 with 90,556 SWAPs (Q20 14,922; Willow 32,325; Heron 43,309), better than current Gemma PWC. It is a different saved program and one nondeterministic replay, not evidence that the current Gemma run reached that result. Its full per-case counts are in `gemma-historical-per-case.csv`. The replay report documents its difference from its original saved grade. Qwen and Muse current bests exceed their respective September 19 replay rewards. This comparison does not claim an exhaustive search of every past run.

## Q20: all 24 cases

| Circuit | SABRE | LightSABRE | Qwen | Gemma | Muse |
|---|---:|---:|---:|---:|---:|
| 4gt13_92 | 0 | 0 | 2 | 2 | 2 |
| 4mod5-v1_22 | 0 | 0 | 1 | 1 | 1 |
| 9symml_195 | 5,756 | 5,283 | 3,598 | 3,675 | 3,691 |
| adr4_197 | 538 | 334 | 310 | 327 | 380 |
| alu-v0_27 | 1 | 1 | 2 | 2 | 3 |
| co14_215 | 2,994 | 2,548 | 1,972 | 2,111 | 2,438 |
| cycle10_2_110 | 874 | 747 | 525 | 510 | 705 |
| decod24-v2_43 | 0 | 0 | 0 | 0 | 0 |
| ising_model_10 | 0 | 0 | 6 | 6 | 4 |
| ising_model_13 | 0 | 0 | 9 | 6 | 6 |
| ising_model_16 | 0 | 0 | 15 | 11 | 11 |
| misex1_241 | 507 | 352 | 366 | 372 | 371 |
| mod5mils_65 | 0 | 1 | 4 | 1 | 1 |
| qft_10 | 18 | 12 | 19 | 14 | 13 |
| qft_16 | 62 | 44 | 58 | 55 | 49 |
| radd_250 | 425 | 344 | 259 | 312 | 371 |
| rd73_252 | 711 | 660 | 475 | 477 | 608 |
| rd84_142 | 35 | 34 | 44 | 34 | 42 |
| rd84_253 | 2,049 | 1,833 | 1,392 | 1,379 | 1,737 |
| sqn_258 | 1,448 | 1,215 | 958 | 899 | 1,105 |
| square_root_7 | 866 | 695 | 556 | 572 | 694 |
| sym6_145 | 424 | 351 | 204 | 240 | 342 |
| sym9_193 | 5,551 | 5,283 | 3,598 | 3,754 | 3,691 |
| z4_268 | 455 | 326 | 262 | 294 | 339 |

## Willow: all 24 cases

| Circuit | SABRE | LightSABRE | Qwen | Gemma | Muse |
|---|---:|---:|---:|---:|---:|
| 4gt13_92 | 15 | 12 | 12 | 12 | 13 |
| 4mod5-v1_22 | 5 | 4 | 5 | 4 | 4 |
| 9symml_195 | 8,907 | 8,686 | 7,503 | 7,423 | 8,573 |
| adr4_197 | 901 | 836 | 721 | 721 | 833 |
| alu-v0_27 | 7 | 6 | 8 | 8 | 8 |
| co14_215 | 4,713 | 4,521 | 4,107 | 4,070 | 4,468 |
| cycle10_2_110 | 1,564 | 1,457 | 1,294 | 1,282 | 1,459 |
| decod24-v2_43 | 12 | 11 | 9 | 9 | 12 |
| ising_model_10 | 0 | 0 | 14 | 12 | 10 |
| ising_model_13 | 9 | 0 | 18 | 18 | 15 |
| ising_model_16 | 5 | 0 | 37 | 21 | 28 |
| misex1_241 | 1,273 | 1,129 | 978 | 966 | 1,152 |
| mod5mils_65 | 7 | 6 | 8 | 6 | 7 |
| qft_10 | 31 | 19 | 33 | 24 | 23 |
| qft_16 | 87 | 65 | 93 | 79 | 72 |
| radd_250 | 851 | 783 | 669 | 659 | 787 |
| rd73_252 | 1,327 | 1,282 | 1,108 | 1,080 | 1,284 |
| rd84_142 | 93 | 70 | 80 | 77 | 88 |
| rd84_253 | 3,517 | 3,381 | 2,955 | 2,901 | 3,389 |
| sqn_258 | 2,512 | 2,441 | 2,109 | 2,072 | 2,448 |
| square_root_7 | 1,882 | 1,779 | 1,591 | 1,557 | 1,813 |
| sym6_145 | 938 | 896 | 765 | 745 | 897 |
| sym9_193 | 8,907 | 8,686 | 7,503 | 7,446 | 8,573 |
| z4_268 | 789 | 732 | 635 | 642 | 729 |

## Heron: all 24 cases

| Circuit | SABRE | LightSABRE | Qwen | Gemma | Muse |
|---|---:|---:|---:|---:|---:|
| 4gt13_92 | 15 | 15 | 17 | 18 | 19 |
| 4mod5-v1_22 | 4 | 4 | 7 | 5 | 5 |
| 9symml_195 | 11,762 | 10,993 | 10,071 | 10,448 | 10,696 |
| adr4_197 | 1,239 | 1,050 | 964 | 1,022 | 1,061 |
| alu-v0_27 | 12 | 8 | 7 | 10 | 10 |
| co14_215 | 5,971 | 5,268 | 5,367 | 5,355 | 5,241 |
| cycle10_2_110 | 1,780 | 1,703 | 1,747 | 1,784 | 1,724 |
| decod24-v2_43 | 12 | 11 | 11 | 10 | 12 |
| ising_model_10 | 0 | 0 | 45 | 13 | 21 |
| ising_model_13 | 9 | 0 | 48 | 43 | 36 |
| ising_model_16 | 0 | 0 | 65 | 65 | 45 |
| misex1_241 | 1,515 | 1,346 | 1,436 | 1,384 | 1,367 |
| mod5mils_65 | 9 | 7 | 9 | 10 | 12 |
| qft_10 | 46 | 32 | 53 | 50 | 51 |
| qft_16 | 185 | 98 | 148 | 160 | 167 |
| radd_250 | 1,069 | 934 | 884 | 923 | 955 |
| rd73_252 | 1,785 | 1,580 | 1,462 | 1,522 | 1,597 |
| rd84_142 | 103 | 93 | 119 | 112 | 135 |
| rd84_253 | 4,733 | 4,326 | 3,967 | 4,187 | 4,261 |
| sqn_258 | 3,458 | 3,124 | 2,797 | 2,908 | 2,993 |
| square_root_7 | 2,632 | 2,322 | 2,381 | 2,244 | 2,341 |
| sym6_145 | 1,057 | 1,024 | 930 | 960 | 1,033 |
| sym9_193 | 11,762 | 10,993 | 10,071 | 10,381 | 10,696 |
| z4_268 | 1,028 | 896 | 828 | 868 | 889 |

## Cached sources

- SABRE: `tpu/science/manifests/routing-v1.json`, hash-matched to the cached SimpleTES suite.
- LightSABRE: cached SimpleTES `python/qiskit_lightsabre_20x20.json` (20 layout trials, 20 swap trials, max_iterations=4).
- SimpleTES: cached Table 2 text in `SkyRLTpu-hybrid-inference-migration/.science/reallocation-10step/best-now/simpletes-tables.txt`; also recorded in `tpu/science/results/full-topology-replay-20260919/README.md`.
- Historical replay: `tpu/science/results/full-topology-replay-20260919/summary.json` and its methodology README.

No generation, grading, training, or baseline reruns were launched.
