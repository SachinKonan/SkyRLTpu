# Gradient-conflict pilot status

**Superseded on 2026-09-06:** v4-32 job 314 was cancelled at the user's request
to move the experiment to v4-64. The replacement is SkyPilot **job 317**, using
`tpu/gradient_conflict/qwen_v4_64.yaml` in `tpuswarm-v4-64-central2-qwen35-erdos`.
Its artifacts live under
`gs://sk7524-tinker-tpu-us-central2/gradient-conflict/20260906-qwen-v4-64-v1/`.
The status below is the historical v4-32 attempt.

Last checked: 2026-09-06 21:44 UTC.

**Awaiting TPU capacity. No completed valid-versus-invalid cosine measurement
has been published.** SkyPilot job **314**, `qwen-v4-32-gradient-conflict-v1-r5`,
is `PENDING` in `tpuswarm-v4-32-central2-smoke`.

The pool reports `NO_REPLICA`, 0/4 ready workers. GCP reported the slice used by
the previous attempt as `READY` / `UNHEALTHY_MAINTENANCE`; the pool's replacement
v4-32 queued resources report `WAITING_FOR_RESOURCES`.

## Experiment

- Qwen3.5-27B, frozen rank-32 LoRA checkpoint
  `tinker://model_4ee1d2d2/weights/000003`.
- Six selected groups of 32 archived completions each, spanning 9.375%–56.25%
  evaluator validity. None of the selected completions was truncated or excluded.
- Sum and mean gradient norms, valid/invalid cosine, matched-size random split,
  cancellation, per-leaf comparisons, and combined-gradient dot products.
- Text-reconstructed, signed advantage-weighted gradients in trainable LoRA
  space. Original sampled token IDs, behavior log probabilities, termination
  tokens, and phase-two action masks were unavailable; this is not an exact
  replay of historical optimizer steps.
- No optimizer updates. A fresh two-sample export validates the restored model;
  parameter fingerprints must stay identical across all gradient drains.

## Evidence obtained

Job 313 successfully restored the trained checkpoint, passed the two-sample
export check, and computed all partitions of the first group. The 26-sample
partition's gradient SHA-256 exactly reproduced the earlier attempt's export.
The parameter SHA-256 was
`9e6aff9c4637970f4167465b9df2a4f07fcce67293d043027a7a4e9210890dd9`.

GCP maintenance interrupted the independent full-group reconstruction pass
(last observed progress: 20/32 samples), before that version published results.
Job 313 was cancelled while recovering and replaced by job 314. Revision 5
saves each scoring chunk and completed gradient partition to GCS, and publishes
the first comparison before the reconstruction check. Subsequent retries can
reuse these artifacts after checksum and parameter-fingerprint verification.

Local validation: **64 tests passed** across the gradient diagnostic, v4-64
configuration, and pool setup safety tests. A live GCS check passed missing-object
handling and exact score save/download recovery. Python syntax, shell syntax,
and `git diff --check` passed. The complete six-group TPU experiment remains
unverified until capacity returns and the job finishes.

## Artifacts and follow-up

GCS prefix:
`gs://sk7524-tinker-tpu-us-central2/gradient-conflict/20260906-qwen-v1/`

- `manifest-v5.json`: deployed overlay file hashes and source commit.
- `pilot.json.gz`: reproducible selected inputs and original mixed-group advantages.
- `cache-v5/<input-sha256>/`: resumable scores and gradient partitions.
- `progress.json`: partial measurements, including reconstruction status.
- `result.json`: written by the wrapper only after successful completion.

Monitor job 314 with the repository's SkyPilot environment. After it succeeds,
download `result.json`, run `tpu/gradient_conflict/report.py`, and inspect the
norm/cosine table and plots. Do not interpret cached preflight norms or this
status file as an answer to the gradient-conflict question.

Implementation and methodology: `tpu/gradient_conflict/README.md`.
