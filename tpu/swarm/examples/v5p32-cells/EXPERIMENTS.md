# Erdős v5p-32 cells: experiment status

Status as of **2026-09-07 16:55Z**. Pool `tpuswarm-v5p32-east5a-erdos`, 19 READY
workers; 18 of ours running or recovering, one is the other session's Ray v2 probe
(job 370), none idle.

Every cell is one tpu-v5p-32 slice: rank 0 runs the Tinker trainer, the TTD client
and the grader Ray head; ranks 1–3 serve vLLM. All cells use 16 groups × 32
rollouts per step, 2 elite slots, 15 steps. One step takes about 2–2.5 h, or 4–5 h
for the matched-validity cells (oversampling). Bring-up is 20–40 min.

**Score** is the negated Erdős C5 objective (`pool/best_value`), lower is better.
The gen-0 qwen GRPO run finished at **0.380859355**; the gen-1 "wt16" arms start
from an older 48-state qwen seed (best 0.3808618). **Val** is the kept-rollout
correctness for that step; **tok** is generated tokens per turn.

## Objectives under test

| Objective | Estimator env | Advantage per rollout | What it isolates |
|---|---|---|---|
| GRPO | `mean_baseline` | `r − mean(r)` over the whole group, invalid rollouts have r = 0 | Baseline. The invalid lower tail carries negative advantage, so GRPO trains a **validity** signal plus a mild ranking among valid rollouts. |
| TTD | `entropic_adaptive_beta` | exp-tilted weights over the group with a one-bit β, invalid rollouts pinned at −1 | Sharp ranking: almost all positive mass on the group winner. Same validity push as GRPO in sign, different scale. |
| piecewise LOO | `piecewise_valid_entropic` | GRPO term + α·(w_i − 1) with w_i = e^{βr_i} / leave-one-out mean over valid | Adds the TTD-style ranking only **inside the valid cluster**. The LOO mean gives the bonus a positive mean over valid rollouts (Jensen), so it carries a hidden validity push and a hot winner (adv max 6–9). **Bug found 2026-09-07 (see below): unbounded when exactly 2 rollouts are valid; fixed in bundle v23.** |
| piecewise centered | `piecewise_valid_entropic_centered` | GRPO term + α·(w_i − 1) with w_i = e^{βr_i} / plain mean over valid | Same ranking term but **zero-mean over the valid set**. Separates "ranking inside valid" from the extra validity push of the LOO form. Winner is cooler (≈4 vs 5.6) and bounded by nv. |
| GRPO whitened, matched validity | `mean_baseline_std` + oversampling | GRPO with per-group std normalisation, trained on **exactly 32 valid rollouts** per group (draw up to 64, keep first 32 valid) | Removes the validity term entirely, so GRPO and TTD are compared on ranking alone. |
| TTD, matched validity | `entropic_adaptive_beta` + oversampling | TTD on the same 32-valid groups | The other arm of the matched comparison. |
| LoRA mix (staged) | `TUNIX_LORA_MIX_GAMMA` on the trainer | GRPO on weights W = base + γ·ΔW_new + (1−γ)·ΔW_old, γ learnable per adapter | Interpolates "fresh" (γ = 1) and "carry" (γ = 0) and lets the optimizer pick γ. |

Learning rates: qwen 1.5e-4, gemma and muse 4e-5 (per-model tuned earlier).

## Gen 0: objective pairs per model (mixed-validity batches)

Question: on the same model, same seed, same lr, which objective descends fastest,
and does the answer depend on the model?

| Cell | Job | Model | Objective | Step | Best | Val | Tok | State |
|---|---|---|---|---|---|---|---|---|
| stageC-lr-n | done | qwen | GRPO | 15/15 | 0.380859355 | 0.95 | 13173 | finished |
| stageC-tlr-n | done | qwen | TTD | 14/15 | 0.380864756 | 0.79 | 13341 | finished (14 banked) |
| stageC-pw-n | 330 | qwen | piecewise LOO | 14/15 | 0.380862309 | 0.77 | 3848 | running; **step 14 hit the LOO blow-up (adv max 6.25e10)**, step 15 will show the damage |
| stageC-pwc-n | 340 | qwen | piecewise centered | 7/15 | 0.380868316 | 0.70 | 5065 | running |
| stageB2-g-grpo-n | done | gemma | GRPO | 15/15 | 0.380911409 | 0.75 | 6031 | finished |
| stageB2-g-ttd-n | 282 | gemma | TTD | 15/15 | 0.380900374 | 0.68 | 5900 | finished |
| stageB2-g-pw-n | 294 | gemma | piecewise LOO | 15/15 | 0.380865312 | 0.75 | 5977 | finished (no blow-up seen in its 15 steps) |
| stageB2-g-pwc-n | 341 | gemma | piecewise centered | 8/15 | 0.380897315 | 0.79 | 5952 | running |
| stageB-m-grpo-n | 332 | muse | GRPO | 15/15 | 0.380866934 | 0.99 | 6438 | finished |
| stageB-m-ttd-n | 333 | muse | TTD | 7/15 | 0.380872316 | 0.76 | 13638 | running (resumed from step 6) |
| stageB-m-pw-n | 398 | muse | piecewise LOO | 8/15 | 0.380864268 | 0.82 | 13207 | relaunched 16:59Z on v23 (LOO fix), resumes from step 8 |
| stageB-m-pwc-n | 342 | muse | piecewise centered | 7/15 | 0.380917491 | 0.79 | 13056 | running |

Reading so far:

- **qwen**: GRPO wins (0.380859355). Piecewise LOO trails by 3.0e-6 at step 14 and
  beats TTD by 2.4e-6. Piecewise runs collapsed generation length (10k → 4k tokens
  by step 6) while GRPO/TTD held 13k; winners are not shorter than other valid
  rollouts, so the shrink is policy-wide. Diagnosis: lr 1.5e-4 with a winner
  advantage of 6–9 is too hot; gemma at 4e-5 has the same magnitudes and no collapse.
- **gemma**: piecewise LOO (0.380865312) beats TTD by 3.5e-5 and GRPO by 4.6e-5.
  TTD beats GRPO by 1.1e-5.
- **muse**: piecewise LOO at step 8 (0.380864268) is already 2.7e-6 below the
  finished GRPO line with 7 steps in hand. TTD trails piecewise by ~8e-6 at matched
  steps.
- **centered vs LOO**: centered is behind LOO at every matched step on all three
  models (qwen −2.8e-7 at step 7, gemma −1.9e-5 at step 8, muse −5e-5 at step 7).
  On qwen it keeps length healthier but is following the same collapse one step
  later.

### LOO piecewise blow-up (found 2026-09-07 16:50Z)

`stageC-pw-n` step 14 logged advantage max 6.25e10 (mean 1.95e9): one group had
exactly two valid rollouts 2.5e-5 apart. With two samples the one-bit target
KL ≥ ln2 is unreachable, `_solve_one_bit_beta` returns β_max = 1e6, and the
winner's leave-one-out weight is e^{β_max·gap}. The centered form is bounded by nv
and cannot do this. Fix (discover commit 530c4ea, bundle **v23**): nv = 2 reduces
to GRPO, and w is clamped to nv. 334 and 394 were relaunched on v23 as 398 and 397 (2026-09-07 16:59Z); only 330
(one step left) still runs the unfixed estimator.

## Matched validity, exactly 32 valid rollouts per group

Question: once the invalid lower tail is removed from both gradients, does GRPO
still descend faster than TTD, or was GRPO's qwen win coming from the validity term?

| Cell | Job | Model | Objective | Step | Best | Raw validity | Tok | State |
|---|---|---|---|---|---|---|---|---|
| stageC-v32-grpo-n | 290 | qwen | GRPO whitened | 4/15 | 0.380871997 | 0.60 | 11838 | running, step 5 due ~18:30Z |
| stageC-v32-ttd-n | 291 | qwen | TTD | 4/15 | 0.380880705 | 0.70 | 11385 | running, step 5 due ~17:00Z |
| stageB2-g-v32-grpo-n | 390 | gemma | GRPO whitened | 0/15 | – | – | – | launched 16:29Z, engine bring-up |
| stageB2-g-v32-ttd-n | 391 | gemma | TTD | 0/15 | – | – | – | launched 16:29Z, worker 180 preempted at 16:55Z, RECOVERING |
| stageB-m-v32-grpo-n | 392 | muse | GRPO whitened | 0/15 | – | – | – | launched 16:29Z, engine bring-up |
| stageB-m-v32-ttd-n | 393 | muse | TTD | 0/15 | – | – | – | launched 16:29Z, engine bring-up |

Raw validity is the fraction of drawn rollouts that were valid before filtering
(kept validity is 1.0 by construction). On qwen GRPO leads by 8.7e-6 at step 4 (was
1.2e-5 at step 3). TTD keeps raw validity 10 points higher even though neither
objective sees an invalid rollout. Both arms lost length at step 2, so that drop is
the sampler, not the objective. Gemma and muse pairs (lr 4e-5, bundle v21) were
launched 2026-09-07 16:29Z.

## Gen 1: meta-tree arms

Question: starting from a 48-state seed bank of a finished tree, how much further
can each model/objective push, and does **carrying** the gen-0 model weights beat
starting **fresh** from the base model? Cells stop early on a flatline: five
consecutive steps with gain below 1e-12 after step 4 (the earlier 1e-9 / 3-step
rule was rejected on 2026-09-07; 331/335/336 still run under it).

### From the qwen GRPO tree (wt16 seed)

| Cell | Job | Model | Objective | Weights | Step | Best | Val | Tok | State |
|---|---|---|---|---|---|---|---|---|---|
| meta-wt16-carry-g0-muse | 362 | muse | GRPO | carried from muse GRPO step 9 | 12/15 | **0.380858726** | 1.00 | 4421 | running, new flatline rule |
| meta-wt16-fresh-g0-qwen | 363 | qwen | GRPO | fresh | 14/15 | 0.380858844 | 0.97 | 11516 | finished 16:14Z: flatline stop, flat since step 7 |
| meta-wt16-carry-g0-gemma-ttd | 331 | gemma | TTD | carried from gemma TTD (8×32 run) | 8/15 | 0.380858787 | 0.70 | 5912 | running, old rule, gaining ~1e-9/step |
| meta-wt16-fresh-g0-gemma-ttd | 336 | gemma | TTD | fresh | 11/15 | 0.380858891 | 0.96 | 6015 | running, old rule |
| meta-wt16-fresh-g0-muse-lr4e5 | 399 | muse | GRPO | fresh | 9/15 | 0.380858919 | 0.62 | 10923 | 335 stopped 17:23Z on the old rule (3 zero gains); relaunched 17:32Z under the relaxed rule, resumes from step 9 |

Reading: the basin spans 2e-7. Carried weights lead on both models (muse carry is
the overall best at 0.380858726; gemma carry 0.380858787 vs fresh 0.380858891).
Gains are 1e-9 per step or less, so the tree is near the floor of this basin.

### Cross-model tree carry (launched 2026-09-07 18:30Z, bundle v23)

Question: is a tree found by one model a good starting basin for a *different*
model? The qwen-tree row above already answers it for gemma and muse; these fill
the matrix. Each member carries its own gen-0 weights and keeps the objective that
produced them (qwen GRPO 1.5e-4, gemma piecewise LOO 4e-5, muse GRPO 4e-5). The
same-model recarry cells (397 gemma on gemma, 395 muse on muse) were cancelled with
nothing banked; the user preferred model mixing.

| Seed tree | qwen member | gemma member | muse member |
|---|---|---|---|
| qwen (wt16, best 0.3808618) | fresh only, 363 done | carried, 331 | carried, 362 |
| gemma (stageB2-g-pw-n s15, 0.380865312) | **403** meta-gtree48-carry-g0-qwen | cancelled (397) | **405** meta-gtree48-carry-g0-muse |
| muse (stageB-m-grpo-n s15, 0.380866934) | **402** meta-mtree48-carry-g0-qwen | **404** meta-mtree48-carry-g0-gemma-pw | cancelled (395) |

Carried weights: qwen `tinker://model_6a19f0fb/weights/000015`, gemma
`tinker://model_92a979f3/weights/000015`, muse `tinker://model_d789e3e9/weights/000015`.
Seeds built with `tpu/meta/build_meta_seed.py --op winner-top16 --k 48` and uploaded as
`puct_sampler_step_000000.json`; `META_SEED_ONLY=1` guards the first launch. At launch
402 and 403 placed (workers 174, 190), 404 was starting, 405 pending on capacity.

### Learnable carried/fresh LoRA mix (qwen, Ray v2 executor, staged, NOT launched)

`skyrl/backends/lora_mix.py`, bundles v22/v23, profiles
`tpu/swarm/ray_train/profiles/qwen_v5p_32_{mix090,mix050,carry}.json`:

    W = W_base + (alpha/r) * [ gamma * B_new A_new + (1 - gamma) * B_old A_old ]

One learnable gamma per adapter (per projection per layer), Adam lr 0.02, clamped to
[0, 1]; the old half is the carried qwen GRPO step-15 LoRA
(`tinker://model_6a19f0fb/weights/000015`) and never trains; the fresh half starts at
B = 0. Both halves live in one rank-64 adapter (alpha doubled so the qwix scale is
unchanged), exported to vLLM as an ordinary rank-64 PEFT adapter. gamma = 1 is
"fresh" (meta-wt16-fresh-g0-qwen, 0.380858844), gamma = 0 is "carry". Cells:
gamma_0 = 0.9, gamma_0 = 0.5, and a plain carry control (a qwen carry arm did not
exist). All three use the wt16 seed. gamma statistics are logged per optimizer step
as `lora_mix/gamma_{mean,min,max,std}`.

Ray v2 staging per run id under `gs://sk7524-tinker-tpu-us-east5/ray-training/<run_id>/`:
`client/tinker_log/<run_id>/puct_sampler_step_000000.json` (seed),
`checkpoints/model_6a19f0fb/{000015,sampler_weights/000015}.tar.gz` (carried weights),
`tinker-backup.db` (registry rows so `create_training_client_from_state` resolves).
Task yamls: `python -m tpu.swarm.ray_train.build <profile> --output <dir> --upload`.

Why gemma/muse cells stay on the legacy launcher: the Ray v2 executor is qwen-only
today (serving env pins PyPI vllm-tpu 0.23.0 without the muse modeling code; the
client member spec is hardcoded to `Qwen/Qwen3.5-27B:qwen3:qwen`; gemma untested).
Its preflight also refuses workers that still hold warm engines from finished
legacy cells, so every Ray v2 placement on this pool needs an audited retirement
manifest first.

## Overall best values

| Rank | Value | Cell |
|---|---|---|
| 1 | 0.380858726 | meta-wt16-carry-g0-muse @12 |
| 2 | 0.380858787 | meta-wt16-carry-g0-gemma-ttd @8 |
| 3 | 0.380858844 | meta-wt16-fresh-g0-qwen @7 (final) |
| 4 | 0.380858891 | meta-wt16-fresh-g0-gemma-ttd @11 |
| 5 | 0.380858919 | meta-wt16-fresh-g0-muse-lr4e5 @6 |
| 6 | 0.380859355 | stageC-lr-n (qwen GRPO, gen-0) @15 |

## Bundles

| Bundle | Contents |
|---|---|
| v20 | cell scripts through stale-engine eviction; all wt16 gen-1 arms and 290/291/330–336 |
| v21 | + centered piecewise estimator; 340–342, 390–395 |
| v22 | + LoRA mix backend (Ray v2 base bundle for the mix profiles) |
| v23 | + LOO nv = 2 fix and weight clamp (sha256 38c2a12f…); 398, 402–405 |

## Open follow-ups

- Point the Ray v2 mix profiles at v23 (they pin v22, which lacks the LOO fix; the
  mix cells use GRPO so it does not affect them).
- qwen piecewise at lr 4e-5 or α = 0.25, to test the step-size explanation.
- LOO at α ≈ 0.65 (scale-matched to centered), to separate scale from zero-mean.
- Scale-matched TTD in the matched-validity pair.
- Bundle fixes: the flatline break must not write the batch=15 "final" checkpoint
  row; `cell_probe.sh` should ignore a CONVERGED marker under a changed rule.

Metrics live in `gs://sk7524-tinker-tpu-us-east5/skyrl-runs/<cell>/tinker_log/<cell>/metrics.jsonl`.
