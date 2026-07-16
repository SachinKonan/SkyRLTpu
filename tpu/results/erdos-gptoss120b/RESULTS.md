# Erdős minimum-overlap: gpt-oss-120b run (verified c5 = 0.380887659)

> **CORRECTION (2026-07-16).** This document originally claimed the run beat the
> published record. That compared against the *reward-path logged* value (0.380932)
> discussed in [discover#19](https://github.com/test-time-training/discover/issues/19).
> The maintainers responded: the paper's real record is **c5 = 0.3808753232177187**,
> recomputed from the published construction at `results/mathematics/ttt_erdos_sequence.json`
> (n=600), which we have independently re-verified (0.3808753232177188). **Our best
> (0.380887659) is 1.23e-5 SHORT of the true record.** It beats only the buggy-logged
> figure. Claims below are retained for the record with this correction applied.

**Run**: [`erdos-gptoss120b-full` on wandb](https://wandb.ai/sk7524-princeton-university/ttt-discover-gptoss20b/runs/74zeiufa)
(project `ttt-discover-gptoss20b`, entity `sk7524-princeton-university`, run id `74zeiufa`)

**Date**: 2026-07-09 → 2026-07-11 (20 RL steps, stopped by design). Branch `agent/ttd-discover-erdos`.

## Result

| | C₅ (lower = better) |
|---|---|
| **Authors' TRUE published record** (construction in `results/mathematics`, maintainer-confirmed + independently re-verified) | **0.380875323** |
| This run, best construction (verified) | 0.380887659 |
| Authors' reward-path logged value (buggy; see discover#19) | 0.380932 / verifies to 0.380973 |

This run's best is **1.23×10⁻⁵ short of the true record** (see correction above); it
surpasses only the reward-path logged figure. Our value is recomputed from the raw
h-values with zero self-report gap, as is the true-record figure we compare against.

## The problem and the score

Find a step function `h: [0,2] → [0,1]` with `∫h = 1` (discretely: `sum(h) = n_points/2`)
minimizing the worst-shift overlap

```
C₅ = max_k  ∫ h(x)·(1 − h(x+k)) dx  =  max(np.correlate(h, 1−h, mode="full") * dx)
```

Lower C₅ = a tighter upper bound on the Erdős minimum-overlap constant. RL reward = `1/C₅`
for a valid program's construction, 0 for any failure.

## The solution

`best_construction_c5_0.380887659.json` in this directory holds the record construction:
an **n = 144** step function (found at step 15 as an n=144 with C₅ = 0.380907871, then
refined at step 18 to 0.380887659 by descendants of the same family — the run's PUCT buffer
compounded on it). Sibling file `construction_step15_c5_0.380907871.json` is the step-15
ancestor.

Re-verify in three lines:

```python
import json, numpy as np
d = json.load(open("best_construction_c5_0.380887659.json"))
h = np.array(d["construction"]); n = len(h); h *= (n/2)/h.sum()
print(np.max(np.correlate(h, 1-h, "full") * (2/n)))   # -> 0.380887659...
```

## Run setup

- **Model**: `openai/gpt-oss-120b` (the authors' intended default) on the real Thinking
  Machines Tinker prod API; LoRA rank 32, renderer `gpt_oss_high_reasoning`.
- **Code**: authors' released `test-time-training/discover` @ `6c40e82` (their latest
  release; still upstream HEAD at run time) + our infra-only patch
  (`tpu/discover-fixes.patch`: eval backends, SSL init, KL logprob alignment for prod
  Tinker, logging/checkpoint gating). **No changes to the method, reward, env, or
  hyperparameters.**
- **Config**: authors' canonical defaults — group_size 8 × groups_per_batch 64
  (512 rollouts/step), lr 4e-5, KL 0.1, temperature 1.0, phase1_max_tokens 26000,
  context 32768. No adapters persisted (`SAVE_EVERY=0`).
- **Hardware**: one 64-core neuronic node (`tpu/run_ttd_gptoss20b_neuronic.sbatch`),
  `TTD_EVAL_BACKEND=local` — grading runs 64-wide in-process; sampling hits Tinker
  directly. ~2h10m per step, ~44h total.
- **Critical fix**: `EVAL_TIMEOUT=1100`. The env prompt promises generated programs a
  1000s compute budget; an earlier 300s cap silently killed ~40% of all rollouts as
  timeouts (measured on the gpt-oss-20b run). At 1100s, timeouts fell to ~7% and the
  success rate roughly tripled.

## Trajectory (best C₅ per step, cumulative record in bold)

```
step  1  0.381103      step  8  0.380988      step 15  0.380908  <- beats the buggy-logged 0.380932 (not the true record)
step  2  0.380973      step  9  0.380984      step 16  0.380908
step  3  0.380958      step 10  0.380973      step 17  0.380908
step  4  0.380966      step 11  0.380964      step 18  0.380888  <- final record
step  5  0.380966      step 12  0.380956      step 19  0.380888
step  6  0.380966      step 13  0.380956      step 20  0.380908
step  7  0.380988      step 14  0.380956
```

Notable: step 2 reproduced the authors' verified value **exactly** (0.380972753),
suggesting both runs converge on the same underlying optimum family. The two record
leaps (steps 15, 18) came during *exploratory* phases — reward/mean dropped from ~1.9
to ~1.4 while the frontier jumped — consistent with our 20b trace analysis showing
frontier discoveries come from exploration bursts, not the converged
reproduce-the-seed behavior.

## Where the rest lives

- **wandb** ([run link](https://wandb.ai/sk7524-princeton-university/ttt-discover-gptoss20b/runs/74zeiufa)):
  reward curves, and per-step `gen&score_train_*` tables holding every rollout's full
  chain-of-thought (harmony `analysis` channel) + generated program + reward.
- **Local (not in git)**: `runs/ttd_gptoss120b_full/` — slurm log, wandb media,
  PUCT buffer snapshots `puct_sampler_step_0000{00..20}.json` (every state's
  construction + value).
- 20b baseline run + deep trace analysis: wandb run `erdos-gptoss20b-full`
  (`hei5ms3q`); best verified 0.381472 in 14 steps.
