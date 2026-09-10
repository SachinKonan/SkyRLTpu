# Erdős v5p-32 cells: experiment status

Status as of **2026-09-09 15:15Z**. A spot preemption wave at 03:39Z took 17 of 19
workers; the SkyPilot API server and pool controller were restarted twice since
(~10:30 and ~18:08 local), which killed eight job controllers (resubmitted). Overnight, per the user's gpt-oss-first call, every waiting cell was cancelled so the
gpt-oss job could take the first slice; gpt-oss 456 (fixed bundle) is starting on
worker 198, gemma-on-muse-tree 410 is starting on worker 189, and our other 15 cells
were requeued at 00:41Z as 457–471, then cancelled again at 00:52Z (gpt-oss-only hold). Capacity returned 09:00–11:30Z
(8 workers). **12:47Z: user relaunched the six matched-validity cells (483–489,
all placed on their own workers). gpt-oss 486 (tracer fix, bundle gptoss-v9) proved
LoRA install + 18k-row forward/backward on worker 213, sampled slowly at 32 seqs, and
was replaced at 14:2xZ by 502 (GRPO, 128 seqs) on 213 and 503 (TTD) on 242. Priority
per the user 14:20Z: gpt-oss GRPO + TTD first, everything else after; 485 was cancelled
for 503. Preemptions 14:11Z (212), 14:49Z (220), 14:51Z (236) knocked out three cells. Still cancelled and resumable from GCS: the three centered cells, muse
LOO, gemma carry gen-1, muse fresh gen-1, and the four cross-model arms.** Pool: 1 READY,
1 more coming up, 46 provisioning against refused spot capacity. Nothing is lost: every cell
resumes from its last banked step in GCS. Job ids below are the CURRENT ones.

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
| stageC-pwc-n | 457 | qwen | piecewise centered | 12/15 | **0.380857586** | 0.66 | 4967 | waiting; **overall best**, five descending steps 8–12 (was 340/420) |
| stageB2-g-grpo-n | done | gemma | GRPO | 15/15 | 0.380911409 | 0.75 | 6031 | finished |
| stageB2-g-ttd-n | 282 | gemma | TTD | 15/15 | 0.380900374 | 0.68 | 5900 | finished |
| stageB2-g-pw-n | 294 | gemma | piecewise LOO | 15/15 | 0.380865312 | 0.75 | 5977 | finished |
| stageB2-g-pwc-n | 458 | gemma | piecewise centered | 13/15 | 0.380863427 | 0.80 | 5910 | waiting; passed the gemma LOO final at step 12 (was 341/421) |
| stageB-m-grpo-n | 332 | muse | GRPO | 15/15 | 0.380866934 | 0.99 | 6438 | finished |
| stageB-m-ttd-n | 333 | muse | TTD | 13/15 | 0.380865601 | 0.18 | 13255 | **cancelled 03:58Z by the user** (degenerate: validity 0.86 → 0.18 over steps 6–13; step 13 logged advantage max 6.25e10, not reproduced offline from the archived rewards, open). Counts as the muse TTD result. |
| stageB-m-pw-n | 461 | muse | piecewise LOO | 11/15 | 0.380860445 | 0.93 | 13067 | waiting; best muse objective (was 398/424) |
| stageB-m-pwc-n | 464 | muse | piecewise centered | 12/15 | 0.380876406 | 0.71 | 13147 | waiting; four consecutive gains 9–12 (was 342/427) |

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
| stageC-v32-grpo-n | 484 | qwen | GRPO whitened | 6/15 | 0.380870379 | 0.57 | 12109 | waiting (was 290/423) |
| stageC-v32-ttd-n | 483 | qwen | TTD | 6/15 | 0.380866576 | 0.58 | 10890 | waiting; **leads GRPO by 3.8e-6 at equal step** (was 291/422) |
| stageB2-g-v32-grpo-n | 487 | gemma | GRPO whitened | 11/15 | 0.380870881 | 0.78 | 5979 | waiting; the orphaned 390 cell kept training on its VM to step 11 |
| stageB2-g-v32-ttd-n | 485 | gemma | TTD | 4/15 | 0.380910353 | 0.78 | 6000 | **cancelled 14:51Z by the user to give worker 242 to gpt-oss TTD (503)**; resumes from step 4 when resubmitted (was 391/425/463) |
| stageB-m-v32-grpo-n | 489 | muse | GRPO whitened | 4/15 | 0.380898666 | 0.68 | 11116 | preempted 14:51Z, its recovery grabbed worker 242, **cancelled 15:08Z for gpt-oss TTD**; resumes from step 4 |
| stageB-m-v32-ttd-n | 488 | muse | TTD | 2/15 | 0.380913341 | 0.74 | 12642 | preempted 14:49Z, its recovery grabbed worker 242, **cancelled 15:06Z for gpt-oss TTD**; resumes from step 2 |

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
| meta-wt16-carry-g0-gemma-ttd | 467 | gemma | TTD | carried from gemma TTD (8×32 run) | 12/15 | 0.380858785 | 0.83 | 5984 | waiting; 331 stopped 03:29Z on the old rule, relaunched under the relaxed rule, resumes from step 12 |
| meta-wt16-fresh-g0-gemma-ttd | 336 | gemma | TTD | fresh | 15/15 | 0.380858858 | 0.95 | 6018 | finished 00:56Z, flat over its last four steps |
| meta-wt16-fresh-g0-muse-lr4e5 | 471 | muse | GRPO | fresh | 13/15 | 0.380858919 | 0.65 | 11480 | waiting; flat since step 7 (7 steps), stops after one more (was 335/399/428) |

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
| gemma (stageB2-g-pw-n s15, 0.380865312) | **468** (was 403): step 4 0.380864695 (one real step, then flat) | cancelled (397) | **470** (was 405): step 3, still exactly at the seed |
| muse (stageB-m-grpo-n s15, 0.380866934) | **469** (was 402): step 3 0.380866881, flatlined (adv max 0.02, val 0.95) | **410**: step 1 0.380866800 (404 failed on head disk, pruned, relaunched) | cancelled (395) |

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

### Stage F: gen-0 CONTEXT-mixed refinement (staged 2026-09-09, NOT launched)

Question: the third mixing channel. Weights (LoRA mix) and data (tree carry) have
been tried or staged; here each model starts a **fresh** PUCT tree with fresh
weights, and only the *prompt* carries the other models' best gen-0 solutions
(text summary of what they did, C₅, n) while the sandbox pre-imports their
constructions as `reference_constructions[label]`. The prompt's record line is also
corrected (it used to say 0.38092; the published record is 0.380875323). Full
analysis of what each model found and did: `tpu/results/erdos-crossmodel-analysis/ANALYSIS.md`.

| Cell | Job | Model | Objective | Exemplars seen | Compare against | State |
|---|---|---|---|---|---|---|
| stageF-q-ctx-n | - | qwen | centered piecewise, lr 1.5e-4 | all four: qwen 0.380857586 (n=500), muse 0.380860445 (n=512), gemma 0.380863196 (n=536), gpt-oss-120b 0.380887659 (n=144) | stageC-pwc-n 0.380857586 @11 | yaml ready, bundle v25 |
| stageF-g-ctx-n | - | gemma | centered piecewise, lr 4e-5 | all four | stageB2-g-pwc-n 0.380863427 @12 | yaml ready, bundle v25 |
| stageF-m-ctx-n | - | muse | piecewise LOO, lr 4e-5 | all four | stageB-m-pw-n 0.380860445 @8 | yaml ready, bundle v25 |
| gptoss120b pwc ctx | - | gpt-oss-120b | centered piecewise (Ray v2) | all four (code mode) | 502/511 | profile `ray_train/profiles/gptoss120b_v5p_32_pwc_ctx.json`; needs the gpt-oss bundle rebuilt with the exemplar env.py first |

**Weights-carry arms (user, 2026-09-09 evening):** same fresh tree and prompt, but each
member's LoRA starts from its own best gen-0 weights with a fresh optimizer
(`META_INIT_STATE_PATH`, weights only, like the wt16 carry arms); centered piecewise for
all three. The fresh-weights cells above are the controls that separate "the prompt
helps" from "the carried weights help".

| Cell | Job | Model | Objective | Weights from | Compare against | State |
|---|---|---|---|---|---|---|
| stageF-q-ctxw-n | **586** | qwen | centered piecewise, lr 1.5e-4 | stageC-pwc-n step 12 (`model_d09cea07/weights/000012`) | stageF-q-ctx-n, stageC-pwc-n | launched 2026-09-09 23:35Z on w362, bundle v28; **worker preempted ~00:05Z during engine bring-up (nothing banked), PENDING in the recovery loop for ~14 h (the 353 selection at 03:40Z did not stick); placed on worker 396 ~14:40Z 2026-09-10, bring-up from scratch (nothing was banked)** |
| stageF-g-ctxw-n | **587** | gemma | centered piecewise, lr 4e-5 | stageB2-g-pwc-n step 13 (`model_35ca8a17/weights/000013`) | stageF-g-ctx-n, stageB2-g-pwc-n | launched 2026-09-09 23:35Z on w380, bundle v28; survived both waves; CELL-UP 00:1xZ, weights-carry from model_35ca8a17/000013 confirmed; **step 1 banked ~03:30Z: 0.380857584 (val 0.89, 5.0k tok) = the qwen reference adopted and refined by 1.3e-9 in one step** (step 0 seeds were 0.4859); step-2 weights (`model_9dc27641/000002`) and snapshot 2 durable in GCS; **worker 380 preempted ~05:00Z (pool 2/48 READY), recovering; resumes from the banked lineage** |
| stageF-m-ctxw-n | **588** | muse | **centered** piecewise, lr 4e-5 | stageB-m-pw-n step 11 (`model_df2f51fc/weights/000011`, LOO-trained) | stageF-m-ctx-n, stageB-m-pw-n, stageB-m-pwc-n | launched 2026-09-09 23:35Z on w381, bundle v28; **CELL-UP 00:1xZ, weights-carry from model_df2f51fc/000011 confirmed in the log; worker 381 preempted in the second wave ~01:25Z (pool 10 → 2 READY) before step 1 banked, PENDING in the recovery loop**; muse's gen-0 centered cell trailed LOO by 1.6e-5, so this arm also tests whether carried weights close that gap |

The fresh-weights controls (stageF-{q,g,m}-ctx-n) remain unlaunched.

Spot churn log (us-east5-a, 2026-09-10): 586 was placed on 396 at ~14:40Z and preempted at
~15:15Z (7 min RUNNING); 587 was placed on 400 at ~15:10Z and preempted at ~15:23Z (13 min).
Neither reached CELL-UP. New workers come READY and are reclaimed within minutes; pool
oscillating 2–7/48. No cell error in any log; 587's step-2 lineage stays durable in GCS.

Adoption in 587's tree (snapshots 1–2): the 16 seeds (timestep −1, random, 0.4859) were never
built on; **all 31 step-0 states and all 32 step-1 states load `reference_constructions`
(59/63 name `qwen-centered-n500`), every generated state is n = 500**, none use numba/FFT/
multi-grid. The reference was adopted in the FIRST batch and polished by 1.3e-9 there; step 1
added no further gain (0.380857584 → 0.380857584). The metrics row for batch 0 shows the
pre-ingestion buffer (0.4859), which is why the jump looked like a step-1 event. Gemma copied
qwen's LSE + SLSQP recipe from the text summary and added an analytic gradient. The prompt's
"best starting point is …" sentence steers every rollout to the single best reference;
diversity across the four references is not used yet. The metric that matters from here is
the gain below 0.380857586; step 2's snapshot is the first evidence.

**Cancelled 2026-09-10** on the user's "cancel everything in queue" instruction (relayed by the
pool-managing session so the qwen multi-LoRA 615 could take the only free worker): 588 at
~15:37Z, 586 at 17:39Z, 587 at 17:45Z; none held a worker at the time, 587's step-2 lineage is
durable. User's order afterwards: context-mixing cells before the rest; the three yamls were
handed to that session for resubmission (same GCS_RUN, so 587 resumes from step 2).

Env (discover `examples/erdos_min_overlap/env.py`, via `EXTRA_TTD_ENV`):
`TTD_EXEMPLARS_PATH=examples/erdos_min_overlap/exemplars/erdos_gen0_exemplars.json`
(relative to the discover root in the bundle), `TTD_EXEMPLARS_MAX=4` (every model
sees all four solutions, its own included; summaries carry the method only, no
objective/tree/step provenance), `TTD_EXEMPLARS_MODE=summary` (~800 tokens; `code`
adds 2500-char excerpts, ~3400 tokens, for gpt-oss). Off when the path is unset.

## gpt-oss 120B on the Ray v2 executor (2 trainer hosts + 2 engine hosts per v5p-32)

Goal: test the objective ranking (GRPO vs TTD vs centered piecewise) on a second model family. gpt-oss-120b (MXFP4 experts) needs two hosts for the MaxText trainer (tp 4, fsdp 2) and two for vLLM (tp 4), so this is the first Ray v2 topology with `trainer.hosts=2`. Profiles: `tpu/swarm/ray_train/profiles/gptoss120b_v5p_32_{grpo,ttd,pwc}.json` (worktree `SkyRLTpu-gptoss`, branch `agent/tunix-multihost-gptoss`). Executor defaults were first made identical to the legacy v5p cell launcher (prefix caching, 8192 batched tokens, seq buckets, direct routing, exact package pins incl. jax 0.11.1 trainer / 0.10.1 engine).

| job | profile | worker | status | note |
|---|---|---|---|---|
| 411 | grpo | 127 | FAILED 00:31Z | executor bug: orbax CHECKPOINT_COMPLETE preflight probed the marker path with `/**` (a file can never match). Teardown also published 712 stale Qwen compile entries from the hosts' leftover tmpfs into the new gpt-oss compile prefixes (`-ray-v1`, abandoned; delete when convenient). |
| 412 | grpo | 127 | FAILED 02:17Z | marker fixed, tmpfs grown to 200G, 77 GB orbax restored, trainer venv built; then `trainer-flce-contract` failed: the gpt-oss MaxText fork (d388c547) spells the vLLM guard as a tuple and adds an expert_indices return, so the exact-block FLCE patcher failed closed. |
| 413 | grpo | 127 | FAILED 02:22Z | all 4 hosts prepared, trainer started; rank-1 model load died `No module named 'drjax'` (the d388 fork's DiLoCo helper imports it; legacy installs `drjax>=0.1.4`, the Ray v2 pins read off a Qwen cell did not). Preset now pins drjax==0.2.1; import check covers model creation. |
| 414 | grpo | 127 | FAILED 02:43Z | model loaded (29.2 GB/chip, correctly sharded fsdp2 x tp4), both engines up, first create_model: HBM `RESOURCE_EXHAUSTED` in qwix's LoRA install (jit_scan wants 29.19G). |
| 415 | grpo | 127 | FAILED 03:00Z | base state released before the qwix copy; same error with telemetry: 30 GB in use of 102.8 on both trainer hosts. |
| 416 | grpo | 127 | FAILED 03:14Z | allocator dump: qwix's eager tracing forward through MaxText's layer scan holds THREE whole-model copies (peak 89.1 GB) then wants a 4th 29.2 GB as scoped memory. |
| 417 | grpo | 127 | FAILED 03:35Z | same forward under nnx.jit: XLA needs 104 GB of temporaries (scan re-emits the parameter stack). |
| - | grpo | held | ready (bundle gptoss-v5) | fix: qwix trace under `nnx.eval_shape` (no compute), base arrays reused by identity, only LoRA factors initialised on their logical sharding (`qwix_init_mode="abstract"`, commit 7c8ff264; CPU-verified equal to eager on 1 and 4 devices). Held since 03:45Z: spot preemption wave took the pool to 2/52 READY; user to prioritise vs queued legacy cells. |
| - | ttd / pwc | - | ready | yamls built; profiles retire 336 (177) and 331 (178). |

Fixes (branch agent/tunix-multihost-gptoss): `0b5bf349` marker/eviction/compile scoping/tmpfs 200G; `72a1a5c3` FLCE patch for the gpt-oss fork; `cf3f1a6c` drjax pin + import check; `cb393072`/`7c8ff264` create_model memory (base state released early; abstract qwix init); profiles on bundle gptoss-v5 (`a9d9b1d5`). Ray v2 tests: 166 pass; MaxText CPU tests pass.

**Pool event 03:39Z-:** spot preemption wave, erdos pool 2/52 READY at 03:45Z; 340/341/342/402 RECOVERING, 333/391/393/398/399/403/405/410/418 PENDING, only 390/392 RUNNING. Recovery is automatic (resume from last checkpoint).

### gpt-oss 120B: 2026-09-09 update (Ray v2, moving to v6e-32 asia)

| job | where | status | note |
|---|---|---|---|
| 437 | v5p worker 198 | FAILED 00:39Z | all setup + model load + engines OK (bundle v5); abstract LoRA init tripped on a tuple-valued variable (fixed cb -> 16f3cbbf, bundle v6). |
| 456 | v5p | PENDING since 01:07Z | bundle v6. Ran 4 min on 198 at 01:02Z, then the POOL CONTROLLER deleted the replica after API-server probe timeouts (not a preemption). v5p pool: 0 VMs, 46 launches waiting on GCP. |
| 472 | v6e east5b (legacy cell) | CANCELLED | legacy launcher path; superseded by the Ray v2 v6e-32 profile. |
| 456 (cont.) | v5p worker 212 | FAILED 09:19Z | bundle v6. Model load + engines OK; create_model raised `unexpected new leaf from qwix trace: ('adapter','base','decoder','hidden_states') (tuple)`: with num_vocab_tiling>1 the decoder sows hidden_states as an nnx.Intermediate during qwix's LoRA-install forward and the abstract tracer refused it. Fixed 69758d0b (pop nnx.Intermediate in both install paths, CPU regression test), bundle gptoss-v9. |
| 474/475 | v6e-32 asia (Ray v2) | FAILED / CANCELLED | 474: SLICE_FAILURE from a non-contiguous trainer block (executor ranks follow SKYPILOT_NODE_IPS, not the physical 4x8 torus) -> topology probe + `select_v6e_32_topology.py` (f71c9b31). 475: single-host tp4 engines OOM on 32 GB chips -> pair engines tp4/pp2 on the executor's own Ray cluster (52c8f0e9, bundle v8). |
| 477 | v6e-32 east5b replica 207 | FAILED 10:08Z | pair engines: topology probe chose train ranks [0,4,3,7], engine pairs .207+.206 / .210+.208; 120B loaded on 16 chips 09:08Z; job ended 1h later on rank 6 exit 1 with no trace in the SkyPilot log (replica torn down before diagnosis). Same tracer as 456 -> same hidden_states leaf is the only known failure on that path. First 477 attempt was lost to a head-VM sshd failure (cluster INIT teardown). |
| 479 | v6e-32 asia | FAILED | asia copy of bundle v8 was stale (crc32c VGUznA== vs east Om5OhA==); bundle v9 was copied to both buckets and verified identical (Bj3F9Q==). |
| 482 | v5p worker 200 | FAILED 12:55Z | bundle v9; setup, caches and engines started, then libtpu failed the two-host trainer mesh: `Mesh build was incomplete, unassigned nodes tpu1711:pe0:0..3` (SLICE_FAILURE_INIT_ERROR). Sky ranks 0/1 were two layers apart on the 2x2x4 stack; 437/456 only worked because their ranks were adjacent. Fixed ae788ab5: v5p-32 probes all four hosts and trains on the layer next to rank 0 (same pattern as the v6e-32 block probe). |
| 486 | v5p worker 213 | CANCELLED 14:36Z | bundle v9 + z-probe. FIRST RUN THROUGH THE WHOLE CHAIN: probe picked ranks [0,3]; 120B loaded on trainer (29.2 GB/chip) and engines (27 s from RAM); abstract LoRA install +0.02 GB (31.9 GB/chip after create_model); warm-up fb [2,18432] 36 s (29 s compile). Sampling was slow: 840 tok/s per engine at 32 seqs, 4-7% KV use, 321/512 phase-1 completions after 50 min. Cancelled by user decision to relaunch from scratch with 128 seqs + bf16 experts. |
| 502 | v5p | queued 14:37Z | GRPO-002, bundle v10: engines at 128 seqs, experts kept as unscaled bf16 (fork skyrl/v5p-bf16-experts acfe1f05: v5p has no fp8 MXU, fp8 tiles were upcast in every grouped matmul). |
| 503 | v5p worker 242 | FAILED 15:01Z (preflight) | TTD-001. Worker 242 still ran gemma cell 485 after its sky cancel (tmux-detached cell worker, tinker API, gemma vLLM engine holding the TPU); executor refused: `existing inference PID 35080 is not explicitly retired`. Peer session cleaned 242 at 15:17Z; 485/488 task ids added to the profile's retired list. |
| 511 | v5p worker 242 | RUNNING 15:16Z -> PENDING 15:28Z | TTD-001 relaunch (bf16, 128). Probe picked ranks [0,2] on 242; 120B loaded 15:21Z; then the v5p pool was wiped. |
| 502 (15:27Z) | v5p | PENDING (auto-recover) | Step-1 sampling was half done (first wave of 128/engine finished 15:12Z, no KV preemption, KV peak 66%) when a spot preemption wave took the whole v5p pool (0/48 workers, 2 VMs PREEMPTED). Resumes from step 0 when v5p returns. |
| 516 / 517 | v6e-32 east5b | queued 15:33Z | Per the goal (v5p empty -> east): GRPO-002 and TTD-001 on v6e (4 trainer hosts tp8 x fsdp2, pair engines tp4/pp2, fp8 experts, 32 seqs, bundle v10). New profile `gptoss120b_v6e_32_ttd.json`. The v5p copies (502/511) stay queued as a hedge; cancel whichever pair is behind once one pair is past step 1. |
| 502 (15:02Z) | v5p worker 213 | RUNNING | engines 1,870 tok/s each at 128 running (2.2-2.4x over 486's 840 at 32), KV 31%; decode step ~63 ms vs ~40 ms at 32 seqs, so bf16-vs-fp8 is confounded with batch. |
| 486 (orig) | v5p | superseded | bundle v9 + z-probe (`gptoss120b_v5p_32_grpo.json`, 2 trainer hosts tp4 x fsdp2, 2 engine hosts tp4): first run past both fixes. v6e east/asia yamls rebuilt on v9 as fallbacks; one gpt-oss run at a time, v5p first. |

Executor parity was verified against live legacy cell 390 (gemma) from /proc: engine command identical, client env identical apart from the cell's experiment vars, trainer fixed on three v4-64 leftovers (322f25a3). v4 profiles removed; v5p qwen profiles inherit the legacy defaults (d69a67b6). SkyPilot API server: probe-timeout replica teardowns come from 8 long workers vs 46 in-flight launches; restart sizing is the fix (user's call).

## Overall best values

| Rank | Value | Cell |
|---|---|---|
| 1 | 0.380857586 | stageC-pwc-n (qwen centered, gen 0) @12, waiting |
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
| v24 | + Erdős context-mixing prompt (exemplar library, honest record line; discover 6787dc3). Built from the ctxmix worktree at ba5d6665: gen 1788971556711566, sha256 3217a6c0…; superseded by v25 before any launch |
| v25 | prompt shows all four solutions to every model, method-only summaries (discover 47e22fc). Built at e5a53f33: gen 1788994829856576, sha256 4874f28f…; superseded by v26 before any launch |
| v26 | + one-line shape summary per construction in the prompt, `TTD_EXEMPLARS_INLINE_VALUES` opt-in (discover 41a6493). Built at 21c9884b: gen 1788995086374211, sha256 e2baf0ac…; superseded by v27 before any launch |
| v27 | shape line off by default: same design as the original prompt, constructions sandbox-only (discover c5984a2). Built at 8e01e9fb: gen 1788995388522100, sha256 f1dc2baa…; superseded by v28 before any launch |
| v28 | fresh-seed prompts point at the best reference construction instead of the random `initial_h_values`; step-0 "lower is better" fix (discover 1900881); Stage F cells. Built at f23123b7: gen 1788995718280416, sha256 65993360… |

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
