# Erdős v5p-32 cells: experiment status

Status as of **2026-09-09 17:50Z**. A spot preemption wave at 03:39Z took 17 of 19
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
for 503. Preemptions 14:11Z (212), 14:49Z (220), 14:51Z (236) knocked out three cells; their recoveries kept grabbing the worker reserved for gpt-oss TTD, so they were cancelled. A zone-wide reclaim at 15:29Z took every v5p slice; capacity returned 17:47Z (10 workers), all five waiting jobs re-placed, and the three pulled matched cells were relaunched as 557–559. gpt-oss now runs as 560 (GRPO) and 561 (TTD). Still cancelled and resumable from GCS: the three centered cells, muse
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
| stageB2-g-v32-ttd-n | 557 | gemma | TTD | 4/15 | 0.380910353 | 0.78 | 6000 | **cancelled 14:51Z by the user to give worker 242 to gpt-oss TTD (503)**; resumes from step 4 when resubmitted (was 391/425/463) |
| stageB-m-v32-grpo-n | 559 | muse | GRPO whitened | 4/15 | 0.380898666 | 0.68 | 11116 | preempted 14:51Z, its recovery grabbed worker 242, **cancelled 15:08Z for gpt-oss TTD**; resumes from step 4 |
| stageB-m-v32-ttd-n | 558 | muse | TTD | 2/15 | 0.380913341 | 0.74 | 12642 | preempted 14:49Z, its recovery grabbed worker 242, **cancelled 15:06Z for gpt-oss TTD**; resumes from step 2 |

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
| 516 | v6e-32 east5b worker 322 | FAILED 15:38Z | worker deleted 8 min after placement (east5b churn: 12 READY -> 2 -> 7 in 20 min); no log. Resubmitted as 520. |
| 520 / 517 | v6e-32 east5b | PENDING since 15:49Z | 517 reached trainer start on worker 352 (probe block [0,2,5,6], pairs assigned) then lost the worker; 520 lost worker 351 while STARTING. east5b pool: 1 READY worker, many replicas FAILED_CLEANUP. Both auto-recover. |
| 523 / 524 | v6e-32 asia | queued 15:55Z | GRPO asia-001 / TTD asia-001 (new `gptoss120b_v6e_32_ttd_asia.json`), bundle v10 asia copy verified. Six gpt-oss jobs now hedged across v5p/east/asia; cancel the rest once one pair passes step 1. |
| 524 | v6e-32 asia worker 74 | FAILED 16:00Z | 8-rank topology probe aborted with SLICE_FAILURE_SW_INJECT_ERROR (a peer host's probe died). Relaunched as 525. |
| 523 / 517 / 520 | asia 69 / east 366 / east 364 | CANCELLED 16:13Z | Trainers loaded 120B, but every pair engine deadlocked after taking its placement group: vLLM forked its engine core after the server's ray.init (3 threads asleep on a futex, 16 TPUs reserved, 0 used). Fix 08eda12b (spawned engine core with parallel_config.ray_runtime_env), bundle v11. |
| 525 | asia worker 74 | FAILED 16:11Z | same probe slice failure as 524 on the same worker: worker 74 is bad. |
| 526 / 527 / 529 | east GRPO / east TTD / asia TTD | CANCELLED 16:33Z | bundle v11: spawned engine core now creates its Ray workers on both pair hosts, but its Ray job carried no runtime env, so workers ran the default interpreter: `No module named tpu_inference` (527 engines restarted 3x). Fix 159a313f: runtime env handed to the child via RAY_JOB_CONFIG_JSON_ENV_VAR; bundle v12. |
| 528 / 530 | asia worker 74 | FAILED / CANCELLED | worker 74's host w-2 fails libtpu `index_on_host() == i (0 vs. 1)` when a process grid is declared; every 2x2 block with the head includes it. Needs a host reset/replacement. |
| 531 | east worker 369 | FAILED (bootstrap) | newer uv refuses to replace an existing venv ("already exists"); fixed 036821bc with UV_VENV_CLEAR=1 (executor archive). |
| 532 | east worker 364 | CANCELLED 16:50Z | bundle v12: pair-engine Ray workers came up under the serving venv on BOTH hosts (first time), then died in the lora_filesystem_resolver plugin: VLLM_LORA_RESOLVER_CACHE_DIR not forwarded. Also found that adapters are loaded from a local path per worker, so the partner host never had them. Fix 2ee10eb8 (forward LoRA env; replicate adapter dirs to partner hosts via pinned Ray tasks), bundle v13. |
| 533 / 534 | asia workers 74 / 69 | CANCELLED | 533 landed on bad worker 74 (user asked to kill + sweep it: done, no chip holders, executor leftovers removed); 534 cancelled for v13. |
| 536 / 537 | east | CANCELLED 16:55Z | user: v6e gpt-oss runs go to the asia pool only. |
| 538 / 542 / 547 | asia worker 74 | FAILED / CANCELLED | bad worker kept receiving idle placements; released 17:32Z by deleting its VM at the user's request (host w-2 chip-order fault survived a full process sweep and a permuted TPU_VISIBLE_CHIPS test). |
| 540 | asia worker 69 | CANCELLED (v13) | pair workers now ran under the serving venv on both hosts; partner worker lacked VLLM_LORA_RESOLVER_CACHE_DIR because vLLM copies only registered VLLM_* + platform additional_env_vars to Ray workers -> fork fix (08ae1669), bundle v14. |
| 548 | asia worker 65 | FAILED (bootstrap) | port 19679 held by the cancelled 543's executor gcs_server; bootstrap now evicts a same-user Ray listener (625809d2). |
| 550 | asia worker 69 | CANCELLED (v14) | partner worker: resolver dir "must be set to a valid directory" (dir only on the engine head) -> server mkdirs it on partner hosts (625809d2), bundle v15. |
| 553 | asia worker 65 | FAILED 17:36Z | block probe: worker 65's host w-1 has the same chip-order quirk as 74's w-2. Selectors now emit every contiguous block and the controller falls back (a41f731b). |
| 554 | asia worker 69 | CANCELLED 17:40Z | would have stalled at readiness: API server runs on the block's first host (rank 3), client/readiness polled the head. Fixed in a41f731b (api_host routing). |
| 562 | east worker 385 (VM 7nvc, head 136.83.49.36) | FAILED 20:47:56Z | Topology stage. Full 8-host probe passed; the first candidate block [0,5,3,4] died when the head host w-0 hit `SLICE_FAILURE_SW_INJECT_ERROR` (libtpu killed its controller because a TPUworker in the slice was anomalous) and w-7 then aborted via the coordination service; w-1/w-2 had finished. The fallback then rejected candidates 2 and 3 with `owned process already active: topology-subset`: `checked_get` raises on the first failing host while the peers' probes are still blocked in mesh formation, so the next candidate's probe on those hosts is refused. Executor bug, fixed in the worktree (`Host.stop_process` + `Controller.abandon_probe` kills leftover probes and waits before the next candidate; test `test_ray_train_block_fallback.py`). Relaunch after the rebuild. |
| 598 -> 601 -> 603 (GRPO), 599 -> 602 (TTD), 600 (PWC) | v5p | 15:11Z-15:51Z | 600 placed 15:11Z (worker 397; API host rank 3, so the new API-host DB path is in use), sampling step 0 since 15:27Z at ~1,815 tok/s. 598 placed 15:21Z, VM reclaimed 15:24Z (wave of 4), recovery landed on worker d21 where the peer's cancelled cell 483 (stageC-v32-ttd-n) still runs Qwen engines + Tinker API; the preflight refused (`existing inference PID ... is not explicitly retired`) and SkyPilot marked the job FAILED (exit 1 is not retried). Nothing on d21 was touched; the peer session was asked to clean it and 483's task id was added to the v5p gpt-oss `retired_task_ids` (d73342d3). Pending 599/601 replaced by 602 (TTD, resumes checkpoint 000001) and 603 (GRPO) on the updated yamls. |
| 596 | v5p worker 392 (12:34Z-12:44Z) | FAILED at resume | First checkpoint-resume attempt. The executor restored the client dir (global_step 1, tree snapshot with 48 states, checkpoints.jsonl pointing at model_8b4c841f/000001), but the fresh Tinker server answered `Model not found` and the ensemble refused an inconsistent resume. Root cause in the executor: `_sync_run` backed the sqlite registry up with `Connection.backup`, whose destination inherits WAL mode, so every uploaded `tinker-backup.db` was an empty main file with all rows in `tinker-backup.db-wal` (never uploaded). Restore therefore installed an empty registry, and 596's final writeback overwrote the bucket copy with another empty one. Second bug: the DB was restored to and backed up from rank 0, but the API server runs on `train_ranks[0]` (rank 3 on 578). Fix 1f9bfa6a: `registry.backup_database` folds the WAL (journal_mode=DELETE) before upload; restore/backup target the API host; after `services_ready` the controller re-registers every checkpoint named in the client's checkpoints.jsonl whose tarball the GCS mirror holds (event `checkpoints_registered`), so resume self-heals even from an empty backup. Tarballs 000001 (15.7 GB weights, 5.2 GB sampler) are intact in GCS. Relaunched as 598 (GRPO) / 599 (TTD, should resume at step 1) / 600 (PWC) at 12:51Z; 595/597 cancelled. |
| 595 / 596 / 597 | v5p (bundle v18) | launched 04:12Z | Fixes from the trajectory audit: cvxpy + cvxopt/ecos/scs/osqp/clarabel added to the grader (controller) venv (bootstrap identity grading-v4) and `last_codeblock_postprocess` prefers the last block defining `run()` (discover 0f08843; test `test_last_codeblock_run.py`). 577/578/579 cancelled (577 reclaimed twice, 579 reclaimed 04:11Z after 29 min, 578 pending since 01:26Z). Same run ids, so 596 (TTD) resumes from checkpoint 000001 and 595/597 start from scratch. |
| 578 step-0 trajectories | | 04:10Z | 512 trajectories (16 seeds x 32) in `member_gptoss/trajectories/gptoss_step_000000.jsonl.gz`; every response has the harmony analysis channel then a final channel (mean 60k chars). Outcomes: 200 valid C5 bounds (best 0.381302, median 0.433), 107 `Invalid solution.`, 70 sandbox failures `No module named 'cvxpy'` although the prompt lists cvxpy as allowed, 24 `name 'run' is not defined` (last-code-block extraction picked a snippet without `run`), 13 `evaluate_erdos_solution` referenced, 3 timeouts at 1,105 s, ~20 assorted syntax/import errors. So ~14% of samples are lost to a sandbox gap (cvxpy missing) and ~5% to code extraction; both are cheap wins before the next relaunch. |
| 578 step-1 metrics | | 01:12Z | `pool/best_value` -0.3813 after step 1 (seed pool best -0.48722; the tree's 16 seed states improved from -0.5016 mean). 128 samples: correctness 0.39, mean reward 0.886 (max 2.59), raw score mean 0.449 (max 0.577), ~16.4k action tokens per turn (525k total), env_step (sandbox execution) mean 927 s and max 2,340 s per sample, sampling 5,375 s, train 1,019 s, total 6,805 s. Checkpoint 000001 and sampler weights uploaded to `gs://sk7524-tinker-tpu-us-east5/ray-training/gptoss120b-ray-v5p-32-ttd-001/checkpoints/model_8b4c841f/`; step-1 adapter swapped onto both engines at 01:11Z and step-1 sampling started. |
| 578 | v5p worker 365 (2nd placement, head 136.83.35.143) | STEP 1 DONE 01:06Z | First completed gpt-oss 120B optimizer step, on TTD. Timeline on this placement: RUNNING 23:05Z, engines 23:10Z, create_model 23:14Z (2.2 s, HBM 31.93 GB/chip), `save_weights_for_sampler` 268 s, adapters on both engines 23:18Z, step-0 sampling 23:20Z to 00:49Z (`sampling took 5375 s`; 128 seqs/engine at 1,640 tok/s falling to ~1,100 once KV hit 92%; reasoning high, phase-1 cap 13,824), reward sandbox ~00:21Z-00:30Z, 16 streamed fb batches (n=1 51 s, n=14 313 s, n=40 866 s; ~22 s per sequence group plus a fresh XLA compile per new length bucket), `optim_step` 9.9 s, step-1 checkpoint saving 01:06Z. Full step ~1 h 50 min wall. 577 (GRPO) and 579 (PWC) were reclaimed at 00:03Z ~80 min into their first rollouts (577 was already in its fb pass); 577 re-placed 00:11Z and is sampling again, 579 waiting. Spot VM lifetime on v5p tonight (60-80 min) is shorter than a step, so only VMs that survive ~2 h produce a step; shortening the rollout (fewer groups, lower phase-1 cap) or mid-rollout resume is the lever if this repeats. |
| 577 / 578 / 579 | v5p workers 326 / 342 / 353 | launched 22:37Z | gpt-oss moved back to v5p per the user (20/48 READY there; east5b in a reclaim wave: 575/576 lost their VMs twice within minutes and were cancelled 22:36Z before v17's adapter fix could be exercised). GRPO / TTD (entropic adaptive beta) / piecewise centered (`piecewise_valid_entropic_centered`, alpha 1.0; profile written by a peer session, aligned here to bf16 experts + 128 seqs, a427df0d). Bundle v17 carries every fix so far; on single-host v5p engines the relevant ones are the block fallback, bf16 experts, 128 seqs, and the readiness count. |
| 572 (cont.) | east worker 394 | FAILED 21:39Z | Client died on its first adapter load. Two faults in the expert-LoRA install RPC on the pair engine: stage 0 raised `Expert LoRA factors target layers not present in the loaded model: [18..35]` (every stage receives the whole adapter; the installer assumed one worker holds all 36 layers), and stage 1 hit an HBM OOM while scattering its half (32.65 of 33.55 GB in use at 0.90 utilization; a 48 MB buffer failed on 49 MB of fragmented free space). Fixes: fork 484a35f1 (filter factors to the local stage's layers when a PP group exists), bundle v17 (sha 19f45284...), v6e `memory_utilization` 0.85 (user's pick; 0.80 is the fallback), compile prefix mem85. Relaunched as 575 (GRPO, worker 394) / 576 (TTD, worker 411) at 21:46Z. Per the user (21:45Z) matched comparisons come first on v5p and gpt-oss last, so the v5p hedges 569/570 (and 571) were cancelled; gpt-oss effort is v6e east5b only. |
| 572 | east worker 394 (head 34.144.188.105) | RUNNING 21:33Z | First successful decode through a tp4/pp2 pair engine on v6e. Engines loaded in 5 s, `services_ready` 21:33:01Z, `create_model` accepted; the harmony probe (75-token prompt, `Reasoning: high`, `/v1/completions`, `skip_special_tokens=False`) returned 607 tokens in 62 s with finish=stop: `<|channel|>analysis<|message|>` reasoning, then the final channel with a single ```python block (the last-code-block extractor picks the right one). Process cmdline confirms `--no-async-scheduling`. 571 (GRPO) pending on east capacity (4/48 READY, two peer qwen35 jobs hold workers 385/416). |
| 567 / 568 | east workers 411 / 394 (bundle v16) | cancelled 21:22Z / FAILED 21:20Z | First v6e runs past readiness: 567 exercised the repaired block fallback (block 1 rejected, block 2 = ranks 5,2,1,0 validated 16 s later); both reached `services_ready` with 2 replicas, `client_started`, and `create_model` (57 s, HBM 16.03 GB/chip on tp8 x fsdp2). The harmony probe against 568's engine crashed it again, one layer deeper: `persistent_batch_manager.update_states: new_token_ids[i]` IndexError on the first pipeline stage's second step. Under async scheduling only the last stage holds the previously sampled tokens, so PP>1 + async is unsupported in the runner regardless of the executor patch. Fix e23d9720: pair engines get `--no-async-scheduling` (flag verified in the serving venv). 568's client then failed its first adapter upload with connection refused because engine-0 was mid-restart from the probe (follow-up: the upload path should wait for a restarting replica). Relaunched as 571 (GRPO, worker 390) / 572 (TTD, worker 394) at 21:22Z on the same bundle v16 with the sync flag. |
| 563 (cont.) | east worker 390 | stuck at readiness 20:57Z-21:0xZ, then probed | Both pair engines registered 20:57Z but the controller never started the client: the readiness gate compared replicas (2) with `inference_hosts` (4). Fixed b5fa0ed0 (`Config.engine_count`). Used the live engines for the harmony check: one `/v1/completions` request (75-token harmony prompt) killed engine-0's EngineCore on its first step with `assert self._is_last_rank()` in the fork's `ray_distributed_executor.execute_model_ray`: the async-scheduling override only supports PP=1 (first stage returns JaxIntermediateTensors, and `AsyncResultFuture` zipped result ids against all workers). Fixed in the fork (9384733b on skyrl/v5p-bf16-experts; worktree 9349c113): forward intermediate tensors to the next stage, read results from the last stage. The engine auto-restart path worked (weights reloaded in 5 s, replica re-registered 21:04Z). Base bundle v16 building 21:05Z. The harmony response check is still pending. |
| 563 | east worker 390 (head 34.144.191.135) | RUNNING 20:48Z | Topology validated 20:48:30Z (train ranks 1,7,4,0; api_host rank 1 = 10.202.0.109; inference 2,3,5,6), cache barrier 20:49:58Z, Tinker API up 20:50:04Z; engines loading. |
| 562 / 563 | v6e east5b (workers 385 / 390) | launched 20:40Z / 20:41Z | GRPO / TTD on east5b per the user (cross-region egress from asia is the concern; nothing to migrate since no gpt-oss run has produced a checkpoint in either bucket). East profiles bumped to `max_sequences` 128 (a508e788; compile-cache prefix seq128) after 555 showed a 1.42M-token KV cache; fp8 experts kept on v6e (fp8 MXU); KV cache stays bf16 (the `fp8_e5m2 for FP8 KV cache` engine log line is the platform fp8-dtype helper, not the KV dtype). Rebuilt yamls on a41f731b. Asia 555/556 cancelled 20:41Z; v5p 560/561 (same region) stay queued as hedges. Pool had 6 READY workers at launch. |
| 555 attempt 1 | asia worker 93 (VM 6t5f, created 18:54Z) | reclaimed 19:12Z | First v6e-32 gpt-oss 120B run past engine startup. Timeline: 8 hosts in Ray 19:05:13Z, topology validated on the first full probe 19:05:49Z (train ranks 1,2,0,5 with api_host rank 1; inference ranks 3,4,6,7; no fallback needed), cache barrier 19:07:45Z, both pair engines (tp4/pp2, hosts 10+16 and 4+29) spawned EngineCore on the leader and loaded MXFP4 experts on the partner via Ray; engine B reported KV cache 1,416,532 tokens (86x at 16,384) at 19:09:21Z; both replicas registered with the controller at 19:12:16Z. GCP suspended the queued resource (SUSPENDING/SERVICE) at ~19:12Z, before the trainer reached create_model. So spawn + RAY_JOB_CONFIG_JSON_ENV_VAR + lora env + partner mkdir (bundle v11-v15) are all confirmed on real hardware. Job back to WAITING (auto-recovery). Benign noise seen: `writeback_retry_pending` on engine hosts is a gsutil no-clobber race (HTTP 412) when 4 hosts publish the same XLA compile-cache keys; retried, not fatal (follow-up: treat 412-on-no-clobber as success). |
| 555/556 (asia), 560/561 (v5p) | both pools | PENDING 18:36Z | No spot capacity, not a controller fault: every tpuswarm request in both zones is a TPU queued resource in WAITING_FOR_RESOURCES (about 35 in asia-northeast1-b since 04:27Z, 7 in us-east5-a since 17:54Z); the few that filled today were SUSPENDED by SERVICE within minutes (worker 364 twice for 561). `gcloud compute tpus tpu-vm list` shows zero tpuswarm VMs in either zone; the only v5p VMs are two PREEMPTED third-party ones. Pool "PROVISIONING/STARTING" replicas are those pending queued resources. Nothing to change code-side; jobs auto-place when a request fills. |
| 555 / 556 | asia | queued 17:41Z | GRPO / TTD, bundle v15 + fallback/api-host executor. | Per the goal (v5p empty -> east): GRPO-002 and TTD-001 on v6e (4 trainer hosts tp8 x fsdp2, pair engines tp4/pp2, fp8 experts, 32 seqs, bundle v10). New profile `gptoss120b_v6e_32_ttd.json`. The v5p copies (502/511) stay queued as a hedge; cancel whichever pair is behind once one pair is past step 1. |
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
