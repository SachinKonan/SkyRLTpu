# Erdős v5p-32 cells: experiment status

Status as of **2026-09-07 22:36Z**. Pool `tpuswarm-v5p32-east5a-erdos`, 19 READY
workers, all in use (18 ours + the other session's job 384); 29 more provisioning
on spot. All of ours are placed.

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
| piecewise LOO | `piecewise_valid_entropic` | GRPO term + α·(w_i − 1) with w_i = e^{βr_i} / leave-one-out mean over valid | Adds the TTD-style ranking only **inside the valid cluster**. The LOO mean gives the bonus a positive mean over valid rollouts (Jensen), so it carries a hidden validity push and a hot winner (adv max 6–9). **Bug found 2026-09-07: unbounded when exactly 2 rollouts are valid; fixed in bundle v23 (see below).** |
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
| stageC-pw-n | 330 | qwen | piecewise LOO | 15/15 | 0.380862309 | 0.73 | 3312 | finished 18:47Z; step 14 hit the LOO blow-up, no gain after |
| stageC-pwc-n | 340 | qwen | piecewise centered | 10/15 | **0.380857842** | 0.76 | 4691 | running; **overall best**, three descending steps 8–10 |
| stageB2-g-grpo-n | done | gemma | GRPO | 15/15 | 0.380911409 | 0.75 | 6031 | finished |
| stageB2-g-ttd-n | 282 | gemma | TTD | 15/15 | 0.380900374 | 0.68 | 5900 | finished |
| stageB2-g-pw-n | 294 | gemma | piecewise LOO | 15/15 | 0.380865312 | 0.75 | 5977 | finished |
| stageB2-g-pwc-n | 341 | gemma | piecewise centered | 11/15 | 0.380865377 | 0.87 | 5919 | running; 6.5e-8 from the LOO final, three large steps 9–11 |
| stageB-m-grpo-n | 332 | muse | GRPO | 15/15 | 0.380866934 | 0.99 | 6438 | finished |
| stageB-m-ttd-n | 333 | muse | TTD | 10/15 | 0.380866999 | 0.39 | 13446 | running; validity collapsed 0.86 → 0.39 over steps 6–10 |
| stageB-m-pw-n | 398 | muse | piecewise LOO | 9/15 | 0.380860445 | 0.84 | 13248 | running on v23 (resumed from 8 at 16:59Z) |
| stageB-m-pwc-n | 342 | muse | piecewise centered | 10/15 | 0.380892892 | 0.79 | 13149 | running |

Reading:

- **qwen**: centered piecewise leads everything. At step 10 it is 1.5e-6 below the
  GRPO final and 4.5e-6 below the LOO final, with five steps left. It was behind
  LOO through step 7, then banked 9.2e-6, 9.7e-7 and 3.1e-7 in three consecutive
  steps. LOO collapsed generation length (10k → 3.3k tokens); centered is drifting
  the same way more slowly (6k → 4.7k). GRPO > LOO > TTD among the finished runs.
- **gemma**: LOO (0.380865312) beat TTD by 3.5e-5 and GRPO by 4.6e-5. Centered
  trailed LOO by 1.9e-5 at step 8, then banked 1.4e-5, 7.6e-6 and 7.6e-6 and now
  leads LOO at matched step by 5.3e-6, with the highest validity of any gemma cell
  (0.87). It should pass the LOO final next bank.
- **muse**: LOO at step 9 (0.380860445) is 6.5e-6 below the finished GRPO line.
  TTD matched the GRPO final at step 10 but its validity fell from 0.86 to 0.39 in
  four steps (the β-at-noise-floor pathology). Centered is last, 3.2e-5 behind LOO,
  though its steps 9–10 were its largest so far.
- **Both piecewise forms beat GRPO and TTD on gemma and muse.** On qwen at the hot
  lr, LOO lost to GRPO while centered beats it. Net: the within-valid ranking term
  is the useful ingredient; the LOO form's extra validity push and hotter winner
  cost it on qwen, and its nv = 2 blow-up is a liability.

### LOO piecewise blow-up (found 2026-09-07 16:50Z)

`stageC-pw-n` step 14 logged advantage max 6.25e10 (mean 1.95e9): one group had
exactly two valid rollouts 2.5e-5 apart. With two samples the one-bit target
KL ≥ ln2 is unreachable, `_solve_one_bit_beta` returns β_max = 1e6, and the
winner's leave-one-out weight is e^{β_max·gap}. The centered form is bounded by nv
and cannot do this. Fix (discover commit 530c4ea, bundle **v23**): nv = 2 reduces
to GRPO, and w is clamped to nv. 334 and 394 were relaunched on v23 as 398 and 397;
330 finished on the unfixed estimator with no gain after the blow-up.

## Matched validity, exactly 32 valid rollouts per group

Question: once the invalid lower tail is removed from both gradients, does GRPO
still descend faster than TTD, or was GRPO's qwen win coming from the validity term?

| Cell | Job | Model | Objective | Step | Best | Raw validity | Tok | State |
|---|---|---|---|---|---|---|---|---|
| stageC-v32-grpo-n | 290 | qwen | GRPO whitened | 5/15 | 0.380871592 | 0.54 | 11680 | running, step 6 due ~23:00Z |
| stageC-v32-ttd-n | 291 | qwen | TTD | 6/15 | 0.380866576 | 0.58 | 10890 | running, **leads by 5.0e-6** |
| stageB2-g-v32-grpo-n | 390 | gemma | GRPO whitened | 2/15 | 0.381146586 | 0.83 | 6073 | running |
| stageB2-g-v32-ttd-n | 391 | gemma | TTD | 2/15 | 0.380976050 | 0.78 | 6127 | running, leads by 1.7e-4 |
| stageB-m-v32-grpo-n | 392 | muse | GRPO whitened | 1/15 | 0.381062699 | 0.78 | 14302 | running |
| stageB-m-v32-ttd-n | 393 | muse | TTD | 1/15 | 0.381036950 | 0.76 | 14333 | running |

Raw validity is the fraction of drawn rollouts that were valid before filtering
(kept validity is 1.0 by construction). On qwen GRPO led by 1.2e-5 at step 3 and
8.7e-6 at step 4, gained only 4e-7 at step 5, and TTD's step 6 overtook it by
5.0e-6. Both raw validities are falling (0.60 → 0.54, 0.70 → 0.58). On gemma TTD
leads from step 1. Provisional reading: with the validity term removed, TTD's
sharper ranking wins, so GRPO's mixed-batch win on qwen came from the validity
term, and gemma/muse agree with that. Steps take 4–5 h here.

## Gen 1: meta-tree arms

Question: starting from a 48-state seed bank of a finished tree, how much further
can each model/objective push, and does **carrying** the gen-0 model weights beat
starting **fresh** from the base model? Cells stop early on a flatline: five
consecutive steps with gain below 1e-12 after step 4 (the earlier 1e-9 / 3-step
rule was rejected on 2026-09-07; 331 and 336 still run under it).

### From the qwen GRPO tree (wt16 seed)

| Cell | Job | Model | Objective | Weights | Step | Best | Val | Tok | State |
|---|---|---|---|---|---|---|---|---|---|
| meta-wt16-carry-g0-muse | 362 | muse | GRPO | carried from muse GRPO step 9 | 15/15 | 0.380858715 | 0.99 | 4090 | finished 22:43Z, best gen-1 arm |
| meta-wt16-fresh-g0-qwen | 363 | qwen | GRPO | fresh | 14/15 | 0.380858844 | 0.97 | 11516 | finished 16:14Z: flatline stop (5 zero gains), flat since step 7. A qwen CARRY on its own tree was not run (user: reuses the experiment); it exists only as the Ray v2 mix control |
| meta-wt16-carry-g0-gemma-ttd | 331 | gemma | TTD | carried from gemma TTD (8×32 run) | 10/15 | 0.380858785 | 0.77 | 5997 | running, old rule, ~1e-9/step |
| meta-wt16-fresh-g0-gemma-ttd | 336 | gemma | TTD | fresh | 14/15 | 0.380858858 | 0.94 | 5992 | last step running |
| meta-wt16-fresh-g0-muse-lr4e5 | 399 | muse | GRPO | fresh | 11/15 | 0.380858919 | 0.68 | 10969 | 335 stopped 17:23Z on the old rule; relaunched 17:32Z, flat since step 7, stops at 13 if no gain |

Reading: the basin spans 2e-7. Carried weights lead on both models (muse carry
0.380858717 vs fresh 0.380858919; gemma carry 0.380858785 vs fresh 0.380858858).
Gains are 1e-9 per step or less. **The qwen centered gen-0 run (0.380857842) is now
8.8e-7 below this whole basin** without any reseed.

### Cross-model tree carry (launched 2026-09-07 18:30Z, bundle v23)

Question: is a tree found by one model a good starting basin for a *different*
model? Each member carries its own gen-0 weights and keeps the objective that
produced them (qwen GRPO 1.5e-4, gemma piecewise LOO 4e-5, muse GRPO 4e-5). The
same-model recarry cells (397 gemma on gemma, 395 muse on muse) were cancelled with
nothing banked; the user preferred model mixing.

| Seed tree | qwen member | gemma member | muse member |
|---|---|---|---|
| qwen (wt16, best 0.3808618) | fresh only, 363 done | carried, 331 | carried, 362 |
| gemma (stageB2-g-pw-n s15, 0.380865312) | **403**: step 1 0.380864726, val 0.75 | cancelled (397) | **405**: step 1 0.380865312 (= seed), val 0.75 |
| muse (stageB-m-grpo-n s15, 0.380866934) | **402**: step 1 0.380866919, val 0.80 | **410** (404 failed: head disk 54 GB < 56 GB needed; pruned 15 GB of old tarballs, relaunched 23:10Z) | cancelled (395) |

First-step reading: qwen improved the gemma tree by 5.9e-7 in one step, barely
moved the muse tree (1.5e-8), and muse did nothing to the gemma tree. Carried
weights: qwen `tinker://model_6a19f0fb/weights/000015`, gemma
`tinker://model_92a979f3/weights/000015`, muse `tinker://model_d789e3e9/weights/000015`.
Seeds built with `tpu/meta/build_meta_seed.py --op winner-top16 --k 48` and uploaded
as `puct_sampler_step_000000.json`; `META_SEED_ONLY=1` guards the first launch.

### Learnable carried/fresh LoRA mix (qwen, Ray v2 executor, staged, NOT launched)

`skyrl/backends/lora_mix.py`, bundle v22 (v23 also carries it), profiles
`tpu/swarm/ray_train/profiles/qwen_v5p_32_{mix090,mix050,carry}.json`:

    W = W_base + (alpha/r) * [ gamma * B_new A_new + (1 - gamma) * B_old A_old ]

One learnable gamma per adapter (per projection per layer), Adam lr 0.02, clamped to
[0, 1]; the old half is the carried qwen GRPO step-15 LoRA
(`tinker://model_6a19f0fb/weights/000015`) and never trains; the fresh half starts at
B = 0. Both halves live in one rank-64 adapter (alpha doubled so the qwix scale is
unchanged), exported to vLLM as an ordinary rank-64 PEFT adapter. gamma = 1 is
"fresh" (meta-wt16-fresh-g0-qwen, 0.380858844), gamma = 0 is "carry". Cells:
gamma_0 = 0.9, gamma_0 = 0.5, and a plain carry control. All three use the wt16
seed. gamma statistics are logged per optimizer step as `lora_mix/gamma_{mean,min,max,std}`.
CPU-tested only (12 tests + existing backend suite); never run on a TPU.

Ray v2 staging per run id under `gs://sk7524-tinker-tpu-us-east5/ray-training/<run_id>/`:
`client/tinker_log/<run_id>/puct_sampler_step_000000.json` (seed),
`checkpoints/model_6a19f0fb/{000015,sampler_weights/000015}.tar.gz` (carried weights),
`tinker-backup.db` (registry rows so `create_training_client_from_state` resolves).
Task yamls: `python -m tpu.swarm.ray_train.build <profile> --output <dir> --upload`.
Blocked on: Ray v2 preflight refuses workers holding warm engines from finished
legacy cells (needs the other session's retirement manifests); the executor is
qwen-only; the trainer-env passthrough in `ray_train/config.py` + `commands.py` is
uncommitted because those files also carry the other session's WIP.

## Overall best values

| Rank | Value | Cell |
|---|---|---|
| 1 | 0.380857842 | stageC-pwc-n (qwen centered, gen 0) @10, running |
| 2 | 0.380858715 | meta-wt16-carry-g0-muse @15 (final) |
| 3 | 0.380858785 | meta-wt16-carry-g0-gemma-ttd @10 |
| 4 | 0.380858844 | meta-wt16-fresh-g0-qwen @7 (final) |
| 5 | 0.380858858 | meta-wt16-fresh-g0-gemma-ttd @12 |
| 6 | 0.380858919 | meta-wt16-fresh-g0-muse-lr4e5 @6 |
| 7 | 0.380859355 | stageC-lr-n (qwen GRPO, gen-0) @15 |

## Bundles

| Bundle | Contents |
|---|---|
| v20 | cell scripts through stale-engine eviction; wt16 gen-1 arms, 290/291, 331–336 |
| v21 | + centered piecewise estimator; 340–342, 390–393 |
| v22 | + LoRA mix backend (Ray v2 base bundle for the mix profiles) |
| v23 | + LOO nv = 2 fix and weight clamp (sha256 38c2a12f…); 398, 402–405 |

## Open follow-ups

- Point the Ray v2 mix profiles at v23 (they pin v22; the mix cells use GRPO so the
  LOO fix does not affect them).
- Centered piecewise on qwen at lr 4e-5 (is the hot lr what makes it win there?),
  and a gen-1 arm seeded from the stageC-pwc-n tree once it finishes.
- LOO at α ≈ 0.65 (scale-matched to centered), to separate scale from zero-mean.
- Scale-matched TTD in the matched-validity pair.
- Bundle fixes: the flatline break must not write the batch=15 "final" checkpoint
  row; `cell_probe.sh` should ignore a CONVERGED marker under a changed rule.

Metrics live in `gs://sk7524-tinker-tpu-us-east5/skyrl-runs/<cell>/tinker_log/<cell>/metrics.jsonl`.
