# gpt-oss-20b beats the published Erdős min-overlap record (c5 = 0.380925116)

**A 20-billion-parameter model surpassed the record the ttt-discover authors set
with their 120B model** — using their exact released method plus three additions
developed in this project: a contrastive context-distillation objective, PUCT
elite slots, and an eval-budget fix. Run: `erdos-gptoss20b-distelite15`
(15 steps, July 14–16 2026).

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
| 1 | ours, gpt-oss-**120b**, 20 steps | **0.380887659** | `tpu/results/erdos-gptoss120b/` |
| **2** | **ours, gpt-oss-20b distill+elite (this result)** | **0.380925116** | this directory |
| 3 | authors' published record (their 120B run) | 0.380932 claimed / 0.380973 re-verified¹ | test-time-training/discover |
| 4 | ctrl15 (20b, RL only, timeout fix) | 0.381001033 | `runs/ttd_gptoss20b_ctrl15` |
| 5 | distill15 (20b, distill only) | 0.381039226 | `runs/ttd_gptoss20b_distill15` |
| 6 | in-context contrast experiment (offline) | 0.381182 | `runs/critic_ab/` |
| 7 | original 20b baseline (14 steps, 300s budget) | 0.381471836 | `runs/ttd_gptoss20b_full` |

¹ The published number is the model's self-reported value; per
test-time-training/discover#19 the grader returns the claim rather than the
recomputed value (tolerance 1e-4). Recomputing the published construction gives
0.380973. Our results beat it under either reading; all values in this table
for OUR runs are independent recomputations from the stored constructions.

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

None of these levers is model scale — which is why 20B overtook the published
120B record. Training dynamics stayed healthy to the end (reward/mean peaked
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
