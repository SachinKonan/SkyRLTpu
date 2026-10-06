# Qubit routing: current best programs versus cached baselines — September 22, 2026

One best saved program per PWC model, selected by its full-suite weighted reward. All 72 saved case counts are present and their recomputed rewards match. These are observed valid training evaluations, not independent reruns or confidence intervals. Lower SWAP counts are better; added CNOTs equal three times SWAPs.

## Aggregate comparison

| Method | Q20 | Willow | Heron | Total SWAPs | Mean per case | Weighted mean per case | Reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| SABRE | 22,714 | 38,352 | 50,186 | 111,252 | 1545.17 | 1664.92 | 0.500000000 |
| LightSABRE | 20,063 | 36,802 | 45,827 | 102,692 | 1426.28 | 1544.34 | 0.518785493 |
| SimpleTES 20B | 14,180 | 32,135 | 43,032 | 89,347 | 1240.93 | 1370.95 | 0.548415609 |
| SimpleTES 120B | 15,147 | 32,258 | 42,274 | 89,679 | 1245.54 | 1368.43 | 0.548872118 |
| SimpleTES Gemini | 13,470 | 31,481 | 42,396 | 87,347 | 1213.15 | 1343.53 | 0.553413441 |
| Qwen | 13,472 | 32,218 | 43,246 | 88,936 | 1235.22 | 1370.00 | 0.548587276 |
| Gemma | 14,329 | 32,012 | 42,040 | 88,381 | 1227.51 | 1353.61 | 0.551566300 |
| Muse | 16,565 | 36,793 | 45,195 | 98,553 | 1368.79 | 1504.51 | 0.525305589 |

Each topology has 24 cases. The ordinary average is total SWAPs / 72. The weighted average uses the pinned per-case coefficients Q20=0.2, Willow=0.4, Heron=0.4, divided by their sum (24). Reward is B/(B+C), using weighted baseline/candidate added-CNOT totals; it is not the average of per-case rewards.

SABRE and LightSABRE have cached case-level data. SimpleTES 20B/120B numbers are cached published topology aggregates, not local reruns; no per-case SimpleTES records were found. The Gemini totals were already cached in `routing_resources.py::GEMINI_TARGETS`; they also match SimpleTES v2 Supplementary Table 5, verified against the paper and archived in `simpletes-v2-tables.json`. Recomputed rewards for these reference aggregates are derived here, not claimed as published rewards.

## Current programs and pairwise wins

| Model | Job | Snapshot | Reward | W/T/L vs SABRE | W/T/L vs LightSABRE |
|---|---:|---|---:|---|---|
| Qwen | 1590 | puct_sampler_step_000011.json | 0.548587276 | 49/5/18 | 38/3/31 |
| Gemma | 1493 | puct_sampler_step_000007.json | 0.551566300 | 59/9/4 | 42/15/15 |
| Muse | 1589 | puct_sampler_step_000010.json | 0.525305589 | 48/4/20 | 20/3/49 |

Source hashes, state IDs, collection timestamps, saved diagnostics, baseline hashes, and per-case deltas are in `comparison.json`; complete machine-readable counts are in `per-case.csv`. Candidate sources are archived alongside them.

## Historical distinction

The September 19 independent Gemma replay scored 0.545904148 with 90,556 SWAPs (Q20 14,922; Willow 32,325; Heron 43,309), below the latest Gemma PWC result. It is a different saved program and one nondeterministic replay, not evidence that the current Gemma run reached that result. The replay report documents its difference from its original saved grade. All three current bests exceed their respective September 19 replay rewards. This comparison does not claim an exhaustive search of every past run.

## Q20: all 24 cases

| Circuit | SABRE | LightSABRE | Qwen | Gemma | Muse |
|---|---:|---:|---:|---:|---:|
| 4gt13_92 | 0 | 0 | 1 | 2 | 2 |
| 4mod5-v1_22 | 0 | 0 | 2 | 0 | 1 |
| 9symml_195 | 5,756 | 5,283 | 3,126 | 3,137 | 3,691 |
| adr4_197 | 538 | 334 | 312 | 328 | 380 |
| alu-v0_27 | 1 | 1 | 2 | 2 | 3 |
| co14_215 | 2,994 | 2,548 | 2,131 | 2,217 | 2,428 |
| cycle10_2_110 | 874 | 747 | 491 | 546 | 705 |
| decod24-v2_43 | 0 | 0 | 0 | 0 | 0 |
| ising_model_10 | 0 | 0 | 7 | 0 | 4 |
| ising_model_13 | 0 | 0 | 7 | 0 | 6 |
| ising_model_16 | 0 | 0 | 14 | 0 | 11 |
| misex1_241 | 507 | 352 | 373 | 334 | 371 |
| mod5mils_65 | 0 | 1 | 3 | 1 | 1 |
| qft_10 | 18 | 12 | 16 | 17 | 13 |
| qft_16 | 62 | 44 | 55 | 57 | 49 |
| radd_250 | 425 | 344 | 294 | 318 | 371 |
| rd73_252 | 711 | 660 | 454 | 452 | 608 |
| rd84_142 | 35 | 34 | 42 | 39 | 42 |
| rd84_253 | 2,049 | 1,833 | 1,288 | 1,344 | 1,708 |
| sqn_258 | 1,448 | 1,215 | 921 | 978 | 1,105 |
| square_root_7 | 866 | 695 | 345 | 506 | 694 |
| sym6_145 | 424 | 351 | 192 | 217 | 342 |
| sym9_193 | 5,551 | 5,283 | 3,126 | 3,545 | 3,691 |
| z4_268 | 455 | 326 | 270 | 289 | 339 |

## Willow: all 24 cases

| Circuit | SABRE | LightSABRE | Qwen | Gemma | Muse |
|---|---:|---:|---:|---:|---:|
| 4gt13_92 | 15 | 12 | 13 | 12 | 13 |
| 4mod5-v1_22 | 5 | 4 | 3 | 3 | 4 |
| 9symml_195 | 8,907 | 8,686 | 7,517 | 7,485 | 8,606 |
| adr4_197 | 901 | 836 | 731 | 731 | 841 |
| alu-v0_27 | 7 | 6 | 7 | 7 | 8 |
| co14_215 | 4,713 | 4,521 | 4,080 | 4,123 | 4,523 |
| cycle10_2_110 | 1,564 | 1,457 | 1,277 | 1,254 | 1,459 |
| decod24-v2_43 | 12 | 11 | 9 | 9 | 12 |
| ising_model_10 | 0 | 0 | 15 | 0 | 10 |
| ising_model_13 | 9 | 0 | 18 | 0 | 15 |
| ising_model_16 | 5 | 0 | 30 | 0 | 23 |
| misex1_241 | 1,273 | 1,129 | 988 | 971 | 1,155 |
| mod5mils_65 | 7 | 6 | 7 | 6 | 7 |
| qft_10 | 31 | 19 | 29 | 28 | 24 |
| qft_16 | 87 | 65 | 89 | 80 | 77 |
| radd_250 | 851 | 783 | 680 | 673 | 780 |
| rd73_252 | 1,327 | 1,282 | 1,098 | 1,089 | 1,276 |
| rd84_142 | 93 | 70 | 74 | 78 | 87 |
| rd84_253 | 3,517 | 3,381 | 2,965 | 2,921 | 3,400 |
| sqn_258 | 2,512 | 2,441 | 2,129 | 2,108 | 2,446 |
| square_root_7 | 1,882 | 1,779 | 1,574 | 1,575 | 1,786 |
| sym6_145 | 938 | 896 | 737 | 740 | 897 |
| sym9_193 | 8,907 | 8,686 | 7,517 | 7,485 | 8,606 |
| z4_268 | 789 | 732 | 631 | 634 | 738 |

## Heron: all 24 cases

| Circuit | SABRE | LightSABRE | Qwen | Gemma | Muse |
|---|---:|---:|---:|---:|---:|
| 4gt13_92 | 15 | 15 | 15 | 13 | 19 |
| 4mod5-v1_22 | 4 | 4 | 6 | 3 | 5 |
| 9symml_195 | 11,762 | 10,993 | 10,069 | 9,678 | 10,638 |
| adr4_197 | 1,239 | 1,050 | 980 | 938 | 1,061 |
| alu-v0_27 | 12 | 8 | 8 | 10 | 10 |
| co14_215 | 5,971 | 5,268 | 5,197 | 5,315 | 5,242 |
| cycle10_2_110 | 1,780 | 1,703 | 1,735 | 1,732 | 1,723 |
| decod24-v2_43 | 12 | 11 | 12 | 10 | 12 |
| ising_model_10 | 0 | 0 | 33 | 0 | 21 |
| ising_model_13 | 9 | 0 | 53 | 0 | 36 |
| ising_model_16 | 0 | 0 | 57 | 0 | 61 |
| misex1_241 | 1,515 | 1,346 | 1,411 | 1,335 | 1,367 |
| mod5mils_65 | 9 | 7 | 8 | 7 | 12 |
| qft_10 | 46 | 32 | 54 | 42 | 51 |
| qft_16 | 185 | 98 | 153 | 126 | 169 |
| radd_250 | 1,069 | 934 | 893 | 850 | 948 |
| rd73_252 | 1,785 | 1,580 | 1,468 | 1,481 | 1,606 |
| rd84_142 | 103 | 93 | 111 | 101 | 135 |
| rd84_253 | 4,733 | 4,326 | 4,016 | 3,917 | 4,215 |
| sqn_258 | 3,458 | 3,124 | 2,844 | 2,762 | 3,008 |
| square_root_7 | 2,632 | 2,322 | 2,233 | 2,187 | 2,296 |
| sym6_145 | 1,057 | 1,024 | 977 | 982 | 1,033 |
| sym9_193 | 11,762 | 10,993 | 10,069 | 9,698 | 10,638 |
| z4_268 | 1,028 | 896 | 844 | 853 | 889 |

## Cached sources

- SABRE: `tpu/science/manifests/routing-v1.json`, hash-matched to the cached SimpleTES suite.
- LightSABRE: cached SimpleTES `python/qiskit_lightsabre_20x20.json` (20 layout trials, 20 swap trials, max_iterations=4).
- SimpleTES: cached Table 2 text in `SkyRLTpu-hybrid-inference-migration/.science/reallocation-10step/best-now/simpletes-tables.txt`; also recorded in `tpu/science/results/full-topology-replay-20260919/README.md`.
- Historical replay: `tpu/science/results/full-topology-replay-20260919/summary.json` and its methodology README.

No generation, grading, training, or baseline reruns were launched.

## Interpretation of this snapshot

Gemma has 88,381 total SWAPs: 20.56% fewer than SABRE, 13.94% fewer than LightSABRE, 1.08% fewer than SimpleTES GPT-OSS-20B, and 1.45% fewer than SimpleTES GPT-OSS-120B. It still has 1.18% more than SimpleTES Gemini. Gemma beats both GPT-OSS references on Willow and Heron; its Heron count also beats the Gemini reference. Qwen is two SWAPs above Gemini on Q20 and has fewer unweighted total SWAPs than either GPT-OSS reference, although its weighted reward remains slightly below the 120B reference. Muse beats SABRE and LightSABRE on topology totals but trails all SimpleTES variants overall.

These are best-of-search saved evaluations. They do not establish a global SOTA, a compute-matched advantage, or reproducibility across repeated replays. No baseline or candidate was re-executed for this report.
