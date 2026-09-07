# Erdős v5p-32 cells: experiment status

Status as of **2026-09-07 15:20Z**. Pool `tpuswarm-v5p32-east5a-erdos`, 14 READY
workers, 13 of them running our cells (one is the other session's probe, job 370).

Every cell is one tpu-v5p-32 slice: rank 0 runs the Tinker trainer, the TTD client
and the grader Ray head; ranks 1–3 serve vLLM. All cells use 16 groups × 32
rollouts per step, 2 elite slots, 15 steps. One step takes about 2–2.5 h.

**Score** is the negated Erdős C5 objective (`pool/best_value`), lower is better.
The starting seed for gen 0 is the tree seed; the gen-0 qwen GRPO run finished
at **0.380859355**, which is the seed for gen 1. **Val** is the kept-rollout
correctness for that step; **tok** is generated tokens per turn.

## Objectives under test

| Objective | Estimator env | Advantage per rollout | What it isolates |
|---|---|---|---|
| GRPO | `mean_baseline` | `r − mean(r)` over the whole group, invalid rollouts have r = 0 | Baseline. The invalid lower tail carries negative advantage, so GRPO trains a **validity** signal plus a mild ranking among valid rollouts. |
| TTD | `entropic_adaptive_beta` | exp-tilted weights over the group with a one-bit β, invalid rollouts pinned at −1 | Sharp ranking: almost all positive mass on the group winner. Same validity push as GRPO in sign, different scale. |
| piecewise LOO | `piecewise_valid_entropic` | GRPO term + α·(w_i − 1) with w_i = e^{βr_i} / leave-one-out mean over valid | Adds the TTD-style ranking only **inside the valid cluster**. The LOO mean gives the bonus a positive mean over valid rollouts (Jensen), so it carries a hidden validity push and a hot winner (adv max 6–9). |
| piecewise centered | `piecewise_valid_entropic_centered` | GRPO term + α·(w_i − 1) with w_i = e^{βr_i} / plain mean over valid | Same ranking term but **zero-mean over the valid set**. Separates "ranking inside valid" from the extra validity push of the LOO form. Winner is cooler (≈4 vs 5.6). |
| GRPO whitened, matched validity | `mean_baseline_std` + oversampling | GRPO with per-group std normalisation, trained on **exactly 32 valid rollouts** per group (draw up to 64, keep first 32 valid) | Removes the validity term entirely, so GRPO and TTD are compared on ranking alone. |
| TTD, matched validity | `entropic_adaptive_beta` + oversampling | TTD on the same 32-valid groups | The other arm of the matched comparison. |

Learning rates: qwen 1.5e-4, gemma and muse 4e-5 (per-model tuned earlier).

## Gen 0: objective pairs per model (mixed-validity batches)

Question: on the same model, same seed, same lr, which objective descends fastest,
and does the answer depend on the model?

| Cell | Job | Model | Objective | Step | Best | Val | Tok | State |
|---|---|---|---|---|---|---|---|---|
| stageC-lr-n | done | qwen | GRPO | 15/15 | 0.380859355 | 0.95 | 13173 | finished |
| stageC-tlr-n | done | qwen | TTD | 14/15 | 0.380864756 | 0.79 | 13341 | finished (14 banked) |
| stageC-pw-n | 330 | qwen | piecewise LOO | 13/15 | 0.380862332 | 0.73 | 3819 | running |
| stageC-pwc-n | 340 | qwen | piecewise centered | 7/15 | 0.380868316 | 0.70 | 5065 | running |
| stageB2-g-grpo-n | done | gemma | GRPO | 15/15 | 0.380911409 | 0.75 | 6031 | finished |
| stageB2-g-ttd-n | 282 | gemma | TTD | 15/15 | 0.380900374 | 0.68 | 5900 | finished |
| stageB2-g-pw-n | 294 | gemma | piecewise LOO | 15/15 | 0.380865312 | 0.75 | 5977 | finished |
| stageB2-g-pwc-n | 341 | gemma | piecewise centered | 8/15 | 0.380897315 | 0.79 | 5952 | running |
| stageB-m-grpo-n | 332 | muse | GRPO | 15/15 | 0.380866934 | 0.99 | 6438 | finished |
| stageB-m-ttd-n | 333 | muse | TTD | 6/15 | 0.380876653 | 0.86 | 13496 | running, resumed from step 6 |
| stageB-m-pw-n | 334 | muse | piecewise LOO | 7/15 | 0.380866373 | 0.87 | 13183 | running |
| stageB-m-pwc-n | 342 | muse | piecewise centered | 6/15 | 0.380943085 | 0.79 | 12886 | running |

Reading so far:

- **qwen**: GRPO wins (0.380859355). Piecewise LOO trails by 3.0e-6 with two steps
  left and beats TTD by 2.4e-6. Piecewise runs collapsed generation length
  (10k → 4k tokens by step 6) while GRPO/TTD held 13k; winners are not shorter
  than other valid rollouts, so the shrink is policy-wide. Diagnosis: lr 1.5e-4
  with a winner advantage of 6–9 is too hot; gemma at 4e-5 has the same advantage
  magnitudes and no collapse.
- **gemma**: piecewise LOO (0.380865312) beats TTD by 3.5e-5 and GRPO by 4.6e-5.
  TTD beats GRPO by 1.1e-5.
- **muse**: piecewise LOO is ahead of the finished GRPO line with 8 steps in hand
  (0.380866373 at step 7 vs GRPO 0.380866934 at step 15). TTD trails piecewise by
  ~1e-5 at matched steps.
- **centered vs LOO**: centered is behind LOO at every matched step on all three
  models (qwen −2.8e-7 at step 7, gemma −1.9e-5 at step 8, muse −7e-5 at step 6).
  On qwen it keeps length healthier (5065 vs 3827 tokens) but is following the
  same collapse one step later.

## Matched-validity pair (qwen, 32 valid rollouts per group)

Question: once the invalid lower tail is removed from both gradients, does GRPO
still descend faster than TTD, or was GRPO's qwen win coming from the validity term?

| Cell | Job | Model | Objective | Step | Best | Raw validity | Tok | State |
|---|---|---|---|---|---|---|---|---|
| stageC-v32-grpo-n | 290 | qwen | GRPO whitened, 32 valid | 4/15 | 0.380871997 | 0.60 | 11838 | running |
| stageC-v32-ttd-n | 291 | qwen | TTD, 32 valid | 4/15 | 0.380880705 | 0.70 | 11385 | running, step 5 due |

Raw validity is the fraction of drawn rollouts that were valid before filtering
(kept validity is 1.0 by construction). GRPO leads by 8.7e-6 at step 4 (was
1.2e-5 at step 3). TTD keeps raw validity 10 points higher even though neither
objective sees an invalid rollout. Both arms lost length at step 2, so that drop is
the sampler, not the objective.

## Gen 1: meta-tree arms from the qwen GRPO seed

Question: starting from the 48-state seed bank of the finished qwen GRPO tree
(best 0.380859355), how much further can each model/objective push, and does
**carrying** the gen-0 model weights beat starting **fresh** from the base model?
Cells stop early on a flatline: five consecutive steps with gain below 1e-12
after step 4 (the earlier 1e-9 / 3-step rule was rejected on 2026-09-07).

| Cell | Job | Model | Objective | Weights | Step | Best | Val | Tok | State |
|---|---|---|---|---|---|---|---|---|---|
| meta-wt16-carry-g0-muse | 362 | muse | GRPO | carried from muse GRPO step 9 | 12/15 | **0.380858726** | 1.00 | 4421 | running, new flatline rule |
| meta-wt16-fresh-g0-qwen | 363 | qwen | GRPO | fresh | 13/15 | 0.380858844 | 0.96 | 11601 | running, flat since step 7 |
| meta-wt16-carry-g0-gemma-ttd | 331 | gemma | TTD | carried from gemma TTD | 7/15 | 0.380858788 | 0.68 | 5857 | running, old flatline rule |
| meta-wt16-fresh-g0-gemma-ttd | 336 | gemma | TTD | fresh | 10/15 | 0.380858892 | 0.96 | 5979 | running, old flatline rule |
| meta-wt16-fresh-g0-muse-lr4e5 | 335 | muse | GRPO | fresh | 8/15 | 0.380858919 | 0.57 | 10767 | running, flat 2 steps, old rule |

Reading so far: the whole gen-1 basin spans 2e-7. Carried weights lead on both
models (muse carry is the overall best at 0.380858726; gemma carry 0.380858788
vs fresh 0.380858892). Gains are now 1e-9 per step or less, so the tree is near
the floor of this basin.

## Overall best values

| Rank | Value | Cell |
|---|---|---|
| 1 | 0.380858726 | meta-wt16-carry-g0-muse @12 |
| 2 | 0.380858788 | meta-wt16-carry-g0-gemma-ttd @7 |
| 3 | 0.380858844 | meta-wt16-fresh-g0-qwen @7 |
| 4 | 0.380858892 | meta-wt16-fresh-g0-gemma-ttd @10 |
| 5 | 0.380858919 | meta-wt16-fresh-g0-muse-lr4e5 @6 |
| 6 | 0.380859355 | stageC-lr-n (qwen GRPO, gen-0 seed) @15 |

## Open follow-ups

- qwen piecewise at lr 4e-5 or α = 0.25, to test the step-size explanation.
- LOO at α ≈ 0.65 (scale-matched to centered), to separate scale from zero-mean.
- Scale-matched TTD in the matched-validity pair.
- Gen 2 on top of the best gen-1 child (muse carry).
- Bundle fixes: the flatline break must not write the batch=15 "final" checkpoint
  row; `cell_probe.sh` should ignore a CONVERGED marker under a changed rule.

Metrics live in `gs://sk7524-tinker-tpu-us-east5/skyrl-runs/<cell>/tinker_log/<cell>/metrics.jsonl`.
