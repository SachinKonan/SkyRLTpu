# Qwen valid versus invalid gradient pilot

Completed groups: 6 / 6. Full run complete: True.

SkyPilot job **326 SUCCEEDED** at 2026-09-07 01:56:34 UTC. All **192 selected
samples** completed without exclusion. The [completion audit](verification.json)
verifies input identity, sample counts, gradient geometry, unchanged parameters
across all 25 live exports, and matching metadata for all 24 cached partitions.
The raw result is saved at
`gs://sk7524-tinker-tpu-us-central2/gradient-conflict/20260906-qwen-v4-64-v1/result.json`.
The [deployment manifest](manifest-v3.json) records the code used.

[View plot](gradient-conflict.png) · [Download SVG](gradient-conflict.svg)

Frozen checkpoint: `tinker://model_4ee1d2d2/weights/000003`; rank-32 LoRA, tpu-v4-64 TP8/FSDP2.

These are text-reconstructed, signed policy gradients at one checkpoint. Historical sampling logprobs, original token IDs, and phase-two action masks were unavailable. The six selected groups describe the observed validity range, rather than a random sample of all training batches.

| Group | Valid / total | Valid sum norm | Invalid sum norm | Invalid / valid norm | Cosine | Random-split cosine | Cancellation ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| step000000-seed02 | 3 / 32 | 1.1917 | 0.68123 | 0.57162 | -0.36608 | 0.027437 | 0.60639 |
| step000000-seed00 | 7 / 32 | 2.0648 | 1.4553 | 0.70483 | -0.48158 | 0.042205 | 0.53049 |
| step000002-seed02 | 10 / 32 | 1.0471 | 0.92479 | 0.88317 | -0.29606 | -0.13957 | 0.59537 |
| step000001-seed08 | 12 / 32 | 1.265 | 1.0882 | 0.86026 | -0.43191 | -0.22386 | 0.53674 |
| step000002-seed01 | 14 / 32 | 1.0709 | 1.1636 | 1.0865 | -0.28078 | 0.074667 | 0.60059 |
| step000001-seed04 | 18 / 32 | 1.1907 | 1.7076 | 1.4341 | -0.31423 | -0.036643 | 0.60314 |

Sum norms measure population contributions. Mean-gradient norms, token counts, per-leaf comparisons, and signed dot products with the combined gradient are retained in `result.json`.

Negative cosine in 6 / 6 nonzero comparisons; median cosine -0.34016 (range -0.48158 to -0.28078).

The combined gradient has a negative dot product with the valid gradient in 0 / 6 groups, and with the invalid gradient in 0 / 6 groups. A sufficiently small SGD step along the negative combined gradient would increase the corresponding subset loss when this dot product is negative.

All selected valid samples have positive original advantages; all selected invalid samples have negative advantages. These are signed training contributions: rewarding valid responses versus suppressing invalid responses. Their cosine is not the cosine of two unsigned likelihood gradients. The random partition is one matched-size control per group, not a statistical null distribution.

Independent full-group reconstruction relative error: 0.00544514.

Parameter fingerprint across drains: `9e6aff9c4637970f4167465b9df2a4f07fcce67293d043027a7a4e9210890dd9`.

A negative cosine demonstrates local opposition of signed loss gradients. It does not by itself establish harm to task reward or predict an Adam update. No model update was applied.
