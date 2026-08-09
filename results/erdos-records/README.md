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
