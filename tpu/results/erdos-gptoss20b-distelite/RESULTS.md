# gpt-oss-20b distill+elite run: verified c5 = 0.380925116

> **CORRECTION (2026-07-16).** Originally titled "beats the published record" — that
> compared against the reward-path *logged* value (0.380932) from
> [discover#19](https://github.com/test-time-training/discover/issues/19). The
> maintainers clarified the paper's real record: **c5 = 0.3808753232177187**
> (published construction `results/mathematics/ttt_erdos_sequence.json`, n=600,
> independently re-verified by us). **This run is 4.98e-5 short of the true record.**
> What remains true: a 20B model surpassed the *logged* figure and came within 5e-5
> of a 120B-set record using method improvements alone.

A 20-billion-parameter model, using the authors' released method plus three
additions developed in this project (contrastive context-distillation, PUCT
elite slots, eval-budget fix), reached within 5e-5 of the record the authors
set with their 120B model — and surpassed the (buggy) reward-path logged
record value. Run: `erdos-gptoss20b-distelite15` (15 steps, July 14–16 2026).

## Result (independently verified)

```
c5 = 0.380925116     n_points = 138     found at step 12
```

Construction: `best_construction_c5_0.380925116.json` (this directory).
Re-verify:

```python
import json, numpy as np
d = json.load(open("best_construction_c5_0.380925116.json"))
h = np.array(d["construction"]); n = len(h)
h = h * ((n/2) / h.sum())
print(np.max(np.correlate(h, 1-h, mode="full") * (2/n)))   # 0.380925116...
```

## Ranking

| rank | construction | c5 (verified) | source |
|---|---|---|---|
| 1 | **authors' true published record** (their 120B; maintainer-confirmed) | **0.380875323** | `results/mathematics/ttt_erdos_sequence.json` |
| 2 | ours, gpt-oss-**120b**, 20 steps | 0.380887659 | `tpu/results/erdos-gptoss120b/` |
| 3 | **ours, gpt-oss-20b distill+elite (this result)** | 0.380925116 | this directory |
| 4 | authors' reward-path logged value (buggy¹) | 0.380932 / 0.380973 re-verified | discover#19 |
| 5 | ctrl15 (20b, RL only, timeout fix) | 0.381001033 | `runs/ttd_gptoss20b_ctrl15` |
| 6 | distill15 (20b, distill only) | 0.381039226 | `runs/ttd_gptoss20b_distill15` |
| 7 | in-context contrast experiment (offline) | 0.381182 | `runs/critic_ab/` |
| 8 | original 20b baseline (14 steps, 300s budget) | 0.381471836 | `runs/ttd_gptoss20b_full` |

¹ Per discover#19 the in-run grader logs the model's self-claimed value
(tolerance 1e-4); the maintainers confirmed the paper's real number comes from
recomputing the published construction (0.380875323), which we have
independently verified. All values in this table for OUR runs are likewise
independent recomputations from stored constructions.

## Run configuration

Authors' canonical gpt-oss config (gpt_oss_high_reasoning, group_size 8 ×
64 groups = 512 rollouts/step, lr 4e-5, KL 0.1, LoRA rank 32, 26k thinking
budget, 32k context, temperature 1.0) with:

- **`EVAL_TIMEOUT=1100`** — matches the 1000s budget the prompt promises
  generated programs (the original 300s killed ~40% of rollouts).
- **Contrastive context distillation, β=0.1** (`ttt_discover/rl/context_distill.py`):
  16 (worse, 2–3 betters) pool pairs/step → the current policy writes
  `<improve>`-block critiques with verbatim citations (grounding + leakage
  gates) → cross-entropy on the critique text accumulates into the same optim
  step as the RL loss (β·meanNLL normalization).
- **PUCT elite slots, `TTD_ELITE_SLOTS=8`**: 8 of 64 seeds/step are guaranteed
  top-value picks, at most one per lineage.
- No LoRA weights persisted (`SAVE_EVERY=0`); the discovered constructions and
  programs are the artifact.

wandb: https://wandb.ai/sk7524-princeton-university/ttt-discover-gptoss20b/runs/jhucnrgk
(siblings: ctrl15 `xkil7ddh`, distill15 `8gsb0c0d`, 120b `74zeiufa`,
original `e5vkjxyc`)

## Why it worked (mechanism, from the full behavioral analysis)

The two additions fix complementary failure modes measured in the A/B runs
(`tpu/results/erdos-distill-ab/ANALYSIS.md`):

1. **Distillation preserves width.** Pure RL collapses to seed-copying
   (improve-rate 94%→0% by step 10; the control's elite became 20 identical
   clones). The distill objective doubled the late improve-rate and kept ~7
   distinct near-frontier families alive — but alone it spread PUCT's
   attention so thin that no family got polished (best family drew 3.3% of
   rollouts vs the control's accidental 31% monoculture).
2. **Elite slots convert width to depth.** Guaranteed top-value seeding built a
   five-generation polish chain on one compact family:
   step 4 (n=69, 0.380991) → 5 → 7 → 9 (0.380979) → **step 12: the rollout
   reasoned the n=69 grid was the binding discretization floor, doubled the
   resolution to n=138, and refined to 0.380925.** The upsample-and-refine
   move is precisely the disposition the distilled critiques teach (its
   signature appears in 33% of distill improving code vs 18% in control), and
   the model's own analysis channel states the plan:
   *"the final value C5 depends slightly on resolution; the continuous optimum
   may be slightly lower than discrete … upsample to higher resolution, refine
   more and produce slightly lower bound."*
3. **The budget fix makes deep programs viable.** The winning program is a
   five-stage optimizer (guided hill-climb → upsample → annealing →
   subgradient → SLSQP) that would have been killed as a timeout ~40% of the
   time under the original 300s grading cap.

None of these levers is model scale — which is how a 20B model closed to
within 5e-5 of a 120B-set record (and past the logged figure). Training dynamics stayed healthy to the end (reward/mean peaked
at 1.58 on step 13 with no exploration collapse), and the frozen-probe metric
stayed flat (0.960±0.001), showing the distillation acts through a behavioral
disposition shift rather than content memorization at β=0.1.

## Reproduce

```bash
NUM_EPOCHS=15 TTD_DISTILL_ENABLED=1 TTD_ELITE_SLOTS=8 \
  EXPERIMENT_NAME=erdos-gptoss20b-distelite15 \
  sbatch tpu/run_ttd_gptoss20b_neuronic.sbatch
```

(discover code = upstream `6c40e82` + `tpu/discover-fixes.patch`; see
`tpu/HANDOFF-gptoss20b-neuronic.md` for environment setup.)
