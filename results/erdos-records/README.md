# Erdős minimum-overlap records

Construction artifacts from the league runs, with the independent verifier used to
check them. Previously these lived in `runs/ttd_league/records/`, which is
**gitignored** — the world-record artifact was never in version control.

## The record

`record_ctrl_0380861884.json` — **C₅ = 0.38086188411423155**, a 600-point construction
found by the qwen member of the λ=0 control arm (`league1b`) at step 9.

| source | C₅ |
|---|---|
| AlphaEvolve (2025) | 0.380924 |
| TTT-Discover (2026) | 0.380876 |
| SimpleTES | 0.380868 |
| **this work** | **0.380861884114232** |

Provenance: the full 602-state tree containing it is at
`gs://sk7524-tinker-tpu-us-east5/skyrl-runs/league1b/tinker_log/league1b-ctrl-qwen-gemma-erdos/puct_sampler_step_000010.json`.

## Verifying

```bash
uv run --isolated --extra dev python verify_erdos.py record_*.json
```

`verify_erdos.py` replicates the grader exactly — finiteness, the [0,1] box, the
`(n/2)/sum(h)` rescale, then `max(np.correlate(h, 1-h, "full") * dx)` — and requires
`repr(recomputed) == value_full_precision`, i.e. bit-identical, not merely close.

This check is not redundant with the grader. The env's `evaluate_erdos_solution`
returns the model's **self-reported** `c5_bound` after an `np.isclose(atol=1e-4)`
agreement test, and 1e-4 is roughly 150× the margin that separates these records — so
a construction can pass grading while its logged score is optimistic.

## Verification status (2026-08-09)

| artifact | logged | recomputed | status |
|---|---|---|---|
| `record_ctrl_0380861884.json` | 0.38086188411423155 | same | **VALID** |
| `record_ctrl_0380863708.json` | 0.3808637078451589 | same | VALID |
| `record_ctrl_0380865191.json` | 0.38086519135259295 | same | VALID |
| `record_0380868691.json` | 0.3808686911676384 | same | VALID |
| `record_0380868693.json` | 0.38086869298966536 | same | VALID |
| `record_0380687168.json` | 0.38086871677249934 | same | VALID |
| `record_0380868148.json` | 0.3808688147729652 | same | VALID |
| `record_0380868949.json` | 0.38086894864063414 | same | VALID |
| `record_0380870083.json` | 0.3808700827874527 | same | VALID |
| `record_0380871648_step3.json` | 0.38087164821937486 | same | VALID |
| `record_0380872_step4.json` | −0.3808722900695233 | 0.3808722900695233 | sign convention; magnitude exact |
| `record_0380868561.json` | 0.380868561120077 | 0.3808699278668183 | **INVALID — overclaims by 1.37e-06** |

Two notes on the failures:

- `record_0380868561.json` is a real overclaim, not a rounding artifact. Its logged
  value would place it near SimpleTES; the construction does not support that. It was
  briefly cited as the λ=0.1 arm's best result — the correct figure for that arm is
  `record_0380868691.json` at 0.3808686911676384.
- `record_0380872_step4.json` stores the negated value (the tree's internal convention
  is `value = −C₅`). The magnitude verifies bit-exactly.

Large tree dumps (`tree_step*.json`, `record-tree-*.json`, 3–6 MB each) are deliberately
**not** committed; they live in GCS under the run directories above.

## CORRECTION 2026-08-20 — these are NOT world records

Our best independently verified C5 is **0.3808616082566059** (g-tsw-n). The real
state of the art is BETTER:

| target | C5 | our gap |
|---|---|---|
| live leaderboard | 0.38085857 | we are 3.04e-06 behind |
| public exact, sub-ULP repair | 0.3808590568145 | 2.55e-06 behind |
| public exact, admissible | 0.3808594223653 | 2.19e-06 behind |
| TTT-Discover (paper) | 0.3808753 | we are 1.37e-05 ahead |
| SimpleTES | 0.380868561 | 6.95e-06 ahead |
| AlphaEvolve | 0.380924 | 6.24e-05 ahead |

We beat the published TTT-Discover value, SimpleTES and AlphaEvolve; we do NOT
beat the live leaderboard or the machine-verifiable exact bounds. The earlier
"NEW RECORD" language compared only against paper baselines and our own prior
runs. Every artifact from 2026-08-19/20 carries a `correction` field.

### The evaluator inconsistency behind the inflated numbers

`examples/erdos_min_overlap/env.py`:
- `verify_c5_solution` rescales a LOCAL copy of h when sum(h) != n/2;
- it compares recomputed vs model-reported with `np.isclose(..., atol=1e-4)`;
- `evaluate_erdos_solution` then returns the MODEL-REPORTED `c5_bound`, not the
  recomputed value;
- the archive keeps the original h, not the normalized h that was scored.

atol=1e-4 is ~2x the entire span between the state of the art and AlphaEvolve,
so sub-tolerance overclaims are recorded as truth. Measured here: tsw-n +5.67e-05
(sum(h)=500.119 vs 500), g-grpo-n +8.93e-05. Confirmed bit-for-bit that the
un-normalized correlate reproduces the logged value.

**Fix for the next generation** (do not hot-patch running arms -- it changes the
objective mid-flight): project h onto the capped simplex at search time, score
and ARCHIVE the projected vector, and at certification reject infeasible vectors
rather than silently rescaling, with an independent high-precision recompute.
