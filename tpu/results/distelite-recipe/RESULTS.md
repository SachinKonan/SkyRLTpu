# The "distelite" recipe (context-distillation + PUCT elite slots): results across two problems

Recipe = the authors' canonical ttt-discover RL setup (gpt-oss-20b, 8×64=512
rollouts/step, lr 4e-5, KL 0.1, EVAL_TIMEOUT=1100, SAVE_EVERY=0) plus two
additions, both default-off knobs in this repo:

1. **Contrastive context distillation** (`ttt_discover/rl/context_distill.py`):
   per step, a teacher pass (current policy) sees 2–3 stronger pool programs and
   writes `<improve>`-block critiques of a weaker one (citation-grounding + leak
   gates); the final-channel critique is trained as an auxiliary cross-entropy
   loss (β=0.1, meanNLL-normalized) into the same optim step as the RL loss.
   `TTD_DISTILL_ENABLED=1`.
2. **PUCT elite slots** (`tinker_utils/sampler.py`): 8 of 64 seeds/step reserved
   for top-value states, at most one per lineage. `TTD_ELITE_SLOTS=8`.

Motivation and the component-level analysis (distill-only vs control, the
clone-chain/width-vs-depth mechanics) are in
`tpu/results/erdos-distill-ab/ANALYSIS.md`.

## Problem 1: Erdős minimum overlap (minimize c5)

15-step runs, all constructions independently re-verified by recomputation:

| arm | best c5 (verified) | family | found at |
|---|---|---|---|
| ctrl15 (RL only) | 0.381001033 | n=1000 | steps 13–14 |
| distill15 (distill only) | 0.381039226 | n=400 | steps 12–14 |
| **distelite15 (recipe)** | **0.380925116** | **n=138** | **step 12** |

References: authors' buggy-logged number 0.380932 (their true record per the
discover#19 maintainer reply is 0.380875323); our gpt-oss-120b 20-step run
0.380887659 (`tpu/results/erdos-gptoss120b/`).

- distelite beat both of its parent arms by 0.8–1.1e-4 — a larger margin than
  the gap between the parents — and beat the authors' *published/logged* value
  with a 20b model (their maintainer-confirmed true record and our own 120b run
  remain lower).
- Trajectory: distill-width surfaced a compact n=69 family by step 4 → elite
  slots polished it 0.380991 → 0.380979 → at step 12 a rollout doubled it into
  the n=138 champion at 0.380925. Width found it, depth ground it, a structural
  move crossed the line.
- wandb: `erdos-gptoss20b-distelite15` (id jhucnrgk), run dir
  `runs/ttd_gptoss20b_distelite15`.

## Problem 2: circle packing n=26 in the unit square (maximize sum of radii)

15-step runs (r2; the first pair died at steps 6/5 to a transient Tinker infra
error — `RequestFailedError: unknown infra error`, non-retryable in the client):

| arm | best sum of radii | reached at |
|---|---|---|
| ctrl15 r2 | 2.635983 | step 13 |
| **distelite15 r2** | 2.635983 | **step 7** |

Literature: pre-AlphaEvolve 2.634 → AlphaEvolve 2.63586276 → best known
(ShinkaEvolve) 2.635983283.

- Both arms independently reached the **current best-known packing family**;
  the recipe's effect here is **time-to-SOTA halved** (step 7 vs 13) on a task
  whose ceiling both arms hit.
- Champion program re-executed independently on a compute node (100s): packing
  valid, **verified sum of radii 2.635983085** — 2.0e-7 short of the exact
  best-known value (same configuration, marginally less numerical polish).
- wandb: `circle26-gptoss20b-{ctrl15r2,distelite15r2}`, run dirs
  `runs/ttd_circle26_{ctrl15,distelite15}` (crashed r1 archived as
  `*_crashed1`).

## Secondary findings

- **Frozen-probe NLL** (`distill/probe_nll`, fixed critique set scored by the
  current policy each step) was flat to ±0.2% across both distelite runs:
  at β=0.1 the CE objective produces **no measurable content internalization**.
  The behavioral effects (2× improve-rate in the erdos distill-only A/B;
  earlier frontier finds here) are a disposition shift, consistent with the
  CE gradient acting in RL-quiet parameter directions under Adam's
  per-parameter scaling.
- Env generality: circle packing evolves code-only (`result_construction=[]`),
  exercising the distill pipeline without construction context — no changes
  needed. The generic entrypoint (`tpu/run_ttd_env_gptoss.py`) also registers
  `ac_inequalities` and the Frontier-CS `frontier_erdos_ud` env (built +
  sandbox-verified, unlaunched).

## Reproduce

```bash
# distelite arm on a problem (control = drop the last two exports)
sbatch --export=ALL,NUM_EPOCHS=15,\
TTD_ENTRYPOINT=$REPO/tpu/run_ttd_env_gptoss.py,TTD_ENV=circle_packing,TTD_PROBLEM_TYPE=26,\
EXPERIMENT_NAME=...,TTD_RUN_DIR=...,\
TTD_DISTILL_ENABLED=1,TTD_ELITE_SLOTS=8 \
  tpu/run_ttd_gptoss20b_neuronic.sbatch
```

Commits: distill objective `5f1dda3`/`fe73431`, elite slots + teacher + probe
`0ca82be`, generic entrypoint `ecc3245f`, frontier env `ca0d37f` (discover
clone; mirrored in `tpu/discover-fixes.patch`).
