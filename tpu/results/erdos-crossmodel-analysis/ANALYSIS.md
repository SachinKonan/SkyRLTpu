# Erdős minimum overlap: what every model found, and the gen-0 context-mixing round

Written 2026-09-09 (fork of the v5p cells session). Scope: every Erdős run we have
run on TPUs (98 GCS run directories under `gs://sk7524-tinker-tpu-us-east5/skyrl-runs/`)
plus the Thinking Machines Tinker-API runs from July (gpt-oss 20b/120b, 64×8).

**Bottom line.** Every tree, whatever the model or objective, ends in the same basin
(C₅ ≈ 0.380857–0.380867). Eleven trees beat the published record 0.380875323 (n=600).
The project best is qwen's centered-piecewise cell at **0.380857586** (n=500),
1.77e-5 below the record and 5.8e-5 above the 0.38080 target the prompt asks for.
Weight mixing (learnable LoRA mix) is staged but unlaunched; data mixing (tree carry
across models) moved trees by at most 6e-7. This document sets up the third channel:
**context mixing at gen-0**, where each model starts a fresh PUCT tree but its prompt
carries the other models' best solutions and its sandbox carries their constructions.

## 1. Method

- For each run directory, take the latest `puct_sampler_step_*.json`, pick the state
  with the highest value, and **recompute C₅ from the stored construction**
  (`max(np.correlate(h, 1-h, "full")) * 2/n` after renormalising the mass to n/2).
  For every clean tree the recomputed value agrees with the stored value to float
  round-trip. Values in this document are tree values, not metric-row values (the
  two can differ by up to 5e-7; e.g. stageB2-g-ttd-n banks 0.380900374 in the metric
  log, the tree holds 0.380899888).
- Excluded from the leaderboard: `-j` cells (JSSP, a different problem), the `-a`
  cells (a different reward scale, stored values ≈ 1.50, and constructions outside
  [0,1], so not comparable), smoke/sweep/dbtest runs.
- Programs were read from the best state's `code`. For gpt-oss-120b the code is not
  in `tpu/results` (only the construction was saved); it was recovered from the wandb
  `gen&score_train` tables of run `74zeiufa` (6438 scored rollouts). The rollout that
  made the record leap (initial 0.380907871 → 0.380887659, table step 17) is now at
  `tpu/results/erdos-gptoss120b/record_leap_program_step17_0.380907871_to_0.380887659.py`.

## 2. Best solution per tree

### Gen 0, fresh trees, one model each (the clean per-model results)

| Model | Cell | Objective | Best C₅ | n | Found at step | Lineage depth |
|---|---|---|---|---|---|---|
| **qwen** | **stageC-pwc-n** | centered piecewise, lr 1.5e-4 | **0.380857586** | 500 | 11 (12/15 banked) | 12 |
| qwen | stageC-lr-n | GRPO, lr 1.5e-4 | 0.380859355 | 620 | 14 | 15 |
| qwen | stageA2-grpo-n | GRPO, lr 4e-5 | 0.380861631 | 1000 | 14 | 15 |
| qwen | stageC-pw-n | piecewise LOO | 0.380862309 | 300 | 13 | 14 |
| qwen | stageC-tlr-n | TTD, lr 1.5e-4 | 0.380864756 | 240 | 13 | 14 |
| qwen | stageC-v32-ttd-n | TTD, matched validity | 0.380866576 | 320 | 5 (6/15) | 6 |
| qwen | stageC-v32-grpo-n | GRPO whitened, matched validity | 0.380870379 | 400 | 5 (6/15) | 6 |
| qwen | stageA2-ttd-n | TTD, lr 4e-5 | 0.380867900 | 512 | 14 | 15 |
| **gemma** | **stageB-g-ttd-n** | TTD, 8×32, elite 0 | **0.380863196** | 536 | 13 | 14 |
| gemma | stageB2-g-pwc-n | centered piecewise | 0.380863427 | 241 | 12 (13/15 banked) | 13 |
| gemma | stageB2-g-pw-n | piecewise LOO | 0.380865312 | 500 | 14 | 16 |
| gemma | stageB2-g-v32-grpo-n | GRPO whitened, matched validity | 0.380870881 | 400 | 10 (11/15) | 11 |
| gemma | stageB2-g-ttd-n | TTD, 16×32 | 0.380899888 | 512 | 14 | 15 |
| gemma | stageB2-g-grpo-n | GRPO | 0.380911409 | 113 | 14 | 15 |
| **muse** | **stageB-m-pw-n** | piecewise LOO | **0.380860445** | 512 | 8 (11/15 banked) | 9 |
| muse | stageB-m-ttd-n | TTD (cancelled, degenerate) | 0.380865601 | 256 | 12 | 13 |
| muse | stageB-m-grpo-n | GRPO | 0.380866934 | 320 | 14 | 15 |
| muse | stageB-m-pwc-n | centered piecewise | 0.380876406 | 192 | 11 (12/15) | 12 |
| muse | stageD-m-c-n | (stage D) | 0.380890017 | 112 | 5 | 6 |
| **gpt-oss-120b** | Tinker `erdos-gptoss120b-full` | entropic TTD, 64×8, lr 4e-5, KL 0.1 | **0.380887659** | 144 | 18/20 | step-15 ancestor 0.380907871 |
| gpt-oss-20b | Tinker `ttd_20b_a8x64_delite` | entropic, 8×64, distill+elite | 0.380913928 | – | – | – |
| gpt-oss-20b | Tinker `ttd_gptoss20b_distelite15` | entropic, 64×8, distill+elite 8 | 0.380925116 | 138 | 12/15 | – |
| gpt-oss-120b | Tinker `ttd_gptoss120b_distelite` | entropic, distill+elite | 0.380946823 | – | – | – |
| gpt-oss-120b | Ray v2 on v5p / v6e (502, 511, 516, 517) | GRPO / TTD | no banked result at fork time | – | – | – |

gemma note: `stageC-g-tsw-n` holds 0.380861585 (n=1000) but is a **tree-swap arm**
(fresh gemma weights on qwen's `grpo-n` tree at snapshot 4; commit 3cf2095b), so it is
gemma-on-a-qwen-tree, not a clean gemma gen-0 result. Its qwen control `stageC-tsw-n`
was later found contaminated (0.380889524 in the final tree). `stageB-g-grpo-n`
(0.380999297) ran on the lenient grader and its tree holds fictional states.

### Ensembles (gemma + qwen in one tree, ctrlrerun-L, 20 steps)

| Cell | Best C₅ | n | Step | Winning member |
|---|---|---|---|---|
| ctrlrerun-L-ctrl-n-b | 0.380862724 | 512 | 19 | qwen |
| ctrlrerun-L-ctrl-r | 0.380864568 | 600 | 19 | gemma |
| ctrlrerun-L-ctrl-x | 0.380870087 | 512 | 11 | qwen |
| ctrlrerun-L-ours-n | 0.380898026 | 512 | 18 | qwen |

### Gen 1 (meta-wt16: tree seeded with the top-16 of the qwen GRPO tree)

| Cell | Weights | Best C₅ | n | Found at step |
|---|---|---|---|---|
| meta-wt16-carry-g0-muse | carried muse GRPO | 0.380858715 | 740 | 14 |
| meta-wt16-carry-g0-gemma-ttd | carried gemma TTD | 0.380858785 | 700 | 11 |
| meta-wt16-fresh-g0-qwen | fresh | 0.380858844 | 578 | 8 |
| meta-wt16-fresh-g0-gemma-ttd | fresh | 0.380858858 | 544 | 14 |
| meta-wt16-fresh-g0-muse-lr4e5 | fresh | 0.380858919 | 840 | 12 |
| meta-wt16-fresh-g0-gemma | fresh | 0.380859041 | 620 | 0 (= seed) |
| meta-wt16-fresh-g0-muse | fresh | 0.380859045 | 584 | 12 |
| meta-wt16-fresh-g0-gemma-lr4e5 | fresh | 0.380859355 | 620 | 14 (= the stageC-lr-n seed, no gain) |

The whole gen-1 basin spans 6.4e-7. Seeding from the qwen tree makes every model land
within 1.4e-6 of qwen's own best; nobody escapes the seed's neighbourhood.

### Cross-model tree carry (gtree48 = gemma LOO tree, mtree48 = muse GRPO tree, top-48)

| Seed tree (seed value) | qwen member | gemma member | muse member |
|---|---|---|---|
| gemma stageB2-g-pw-n (0.380865312) | 0.380864695 @3 (+6.2e-7) | cancelled | 0.380865312 @2 (= seed) |
| muse stageB-m-grpo-n (0.380866934) | 0.380866881 @2 (+5.3e-8) | 0.380866800 @0 (= seed, 1.3e-7) | 0.380866923 @0 (= seed) |

## 3. What each model actually did

All four families converge on the same mathematical move: replace the max over shifts
by a log-sum-exp smooth-max, anneal its temperature, and optimise under the exact mass
constraint. They differ in optimiser, grid size, and what they wrap around it.

**qwen (Qwen3.5-27B).** SLSQP on the smooth-max with β annealed 1e3 → 1e6, always at a
fixed large grid (n = 500–620, chosen up front by interpolating the incoming
construction), a handful of Gaussian-noise restarts, and a greedy per-coordinate
polish on the exact C₅ at the end. Its lineages are long chains of ~1e-6 gains
(0.380891602 → 0.380857586 in 12 steps). The GRPO best (n=620) is the same recipe
with a candidate-n sweep and three β values. Qwen never used numba, FFT or a
multi-grid scan.

**gemma (Gemma-4-31B).** Two distinct ideas. The TTD best (n=536) spends half of the
budget on **stochastic hill-climbing in the binary space** (binarise, swap a random
1 with a random 0, keep if exact C₅ drops) and half on SLSQP with a β ladder
10 → 1e7 from that binary seed. The centered-piecewise (n=241) and LOO (n=500) bests
use the **epigraph formulation**: minimise a slack y subject to
`y - corr_k·dx ≥ 0` for all k as a vector inequality constraint, with Gaussian /
block-swap / subset-randomisation mutations between SLSQP polishes. Gemma is the only
model that reformulated the max as constraints instead of smoothing it.

**muse (Muse-Glimmer-30B).** The LOO best (n=512) compiles the soft-max objective and
its gradient with **numba** (explicit O(n²) loops), starts from a 50/50 blend of the
incoming construction and a raised cosine, and sweeps SLSQP over τ 0.2 → 5e-6 with
box-simplex projection between stages. The GRPO best (n=320) is scipy-free: a numba
C₅ kernel driven by a list of hand-written configs. Muse prefers smaller grids
(192–512) and analytic gradients.

**gpt-oss-120b (Tinker API, 64×8).** The record program is a **multi-grid pipeline**:
scan n ∈ {144, 192, …, 480} with a cheap anneal + Adam pass, keep the best grid (144),
then loop simulated annealing → Adam on the smooth-max (α 5 → 3.3e5) → sub-gradient
descent on the true max → batch worst-shift improvement → SLSQP polish, restarting
from mass-transfer perturbations of the incumbent, with FFT correlation. It is the
only small-n specialist among the bests; at n=144 it sits 3e-5 above the large-grid
trees, which is consistent with the gap being grid resolution rather than optimiser
quality.

**Gen-1 winners** (n = 700–840) are qwen-style SLSQP with β 5e4–8e5 and a random
choice among candidate grids, i.e. the same recipe pushed to larger n.

Observations that motivate context mixing:

1. The grid size explains more of the spread than the objective does: bests at
   n ≥ 500 are all within 1e-5 of each other; n ≤ 250 trees stop 5e-6 to 3e-5 short.
2. No model has combined its neighbours' tricks: nobody runs the gpt-oss multi-grid
   scan at qwen's grid sizes; nobody outside muse uses compiled kernels; nobody outside
   gemma uses the epigraph constraints. These are exactly the things a prompt can carry.
3. The old prompt told every model "Current record: C₅ ≤ 0.38092", which is *worse*
   than the true record (0.380875323) and worse than what the trees had already
   reached, so the "target" in context was stale from step 1 of every run.

## 4. The context-mixing channel (implemented in `third_party/discover`)

Files: `examples/erdos_min_overlap/env.py`,
`examples/erdos_min_overlap/exemplars/erdos_gen0_exemplars.json`,
`tests/test_erdos_exemplars.py` (6 tests, pass with numpy only).

- `TTD_EXEMPLARS_PATH` (absolute, or relative to the discover repo root) points at a
  JSON list of `{label, model, c5, n_points, approach, code, construction, source}`.
  Off when unset: the prompt is unchanged except for the honest record line.
- `TTD_EXEMPLARS_EXCLUDE=<model>` drops the model's own entry (each model sees the
  other three); `TTD_EXEMPLARS_MAX` caps the count (best by C₅ first).
- `TTD_EXEMPLARS_MODE=summary` (default) injects, per exemplar, one header line
  (label, model, C₅, n) and the approach paragraph: ~680 tokens for three exemplars,
  so it fits gemma's 10240-token context. `code` mode adds a head/tail excerpt of the
  program (`TTD_EXEMPLARS_MAX_CODE_CHARS`, default 2500): ~2560 tokens, for gpt-oss
  (32k context) or qwen (18k).
- The sandbox prelude defines `reference_constructions[label]` as feasible numpy
  arrays next to `initial_h_values`, and the rules bullet says so; programs can start
  from, blend, or ignore them.
- The record line becomes: "Published record: C₅ ≤ 0.380875323. Best construction
  known to this project: C₅ = <min over exemplars and record>. Our goal is 0.38080."

Library contents (one clean gen-0 best per model):

| label | model | C₅ | n | source |
|---|---|---|---|---|
| qwen-centered-n500 | qwen | 0.380857586 | 500 | stageC-pwc-n step 11 |
| muse-slsqp-numba-n512 | muse | 0.380860445 | 512 | stageB-m-pw-n step 8 |
| gemma-binary-climb-slsqp-n536 | gemma | 0.380863196 | 536 | stageB-g-ttd-n step 13 |
| gptoss120b-multigrid-n144 | gptoss | 0.380887659 | 144 | Tinker 74zeiufa, step 18 |

## 5. Stage F: gen-0 context-mixed refinement (staged, NOT launched)

Fresh PUCT tree per model (no carried states, no carried weights); each model keeps
the objective that won its own gen-0 round; everything else matches its gen-0 cell.

| Cell yaml | Model | Objective / lr | Compare against | Exemplars seen |
|---|---|---|---|---|
| `stageF-q-ctx-n.yaml` | qwen | centered piecewise, 1.5e-4 | stageC-pwc-n (0.380857586 @11) | gemma, muse, gpt-oss |
| `stageF-g-ctx-n.yaml` | gemma | centered piecewise, 4e-5 | stageB2-g-pwc-n (0.380863427 @12), stageB-g-ttd-n | qwen, muse, gpt-oss |
| `stageF-m-ctx-n.yaml` | muse | piecewise LOO (nv ≥ 3 fix), 4e-5 | stageB-m-pw-n (0.380860445 @8) | qwen, gemma, gpt-oss |
| `ray_train/profiles/gptoss120b_v5p_32_pwc_ctx.json` | gpt-oss-120b | centered piecewise (Ray v2) | 502/511 GRPO/TTD gen-0 | qwen, muse, gemma, code mode |

All three yamls pin bundle **v24** (this worktree: env.py + library) and pass the
exemplar variables through `EXTRA_TTD_ENV` (a reused tmux server keeps its creator's
env, so plain `envs:` entries are not enough). The gpt-oss profile carries the same
variables in `client_env` but still points at the gpt-oss branch's bundle v11, which
predates env.py's exemplar support; that session must rebuild its bundle from a
checkout containing discover commit `ctxmix` before launching, otherwise the
variables are silently ignored.

What to measure (all from the trees and gen tables, no extra instrumentation):

1. **Descent at matched step** against the gen-0 counterpart: does the context-mixed
   tree reach the counterpart's step-k value earlier, and does its final beat
   0.380857586?
2. **Adoption**: fraction of valid programs per step that reference
   `reference_constructions`, and that contain the other models' fingerprints
   (`numba`, `rfft`/FFT, a multi-grid scan list, the epigraph `ineq` constraint).
   If adoption is high and descent is unchanged, the ideas are not the bottleneck.
3. **Grid choice**: n of the best state over steps. The hypothesis from section 3 is
   that gpt-oss moves to n ≥ 500 when it sees the others' grids, and that the
   others try the multi-grid scan.

Not launched: the pool's standing priority is gpt-oss GRPO/TTD, and the v5p pool was
fully preempted when this fork started. Launch command when the user decides:

```
sky jobs launch --pool tpuswarm-v5p32-east5a-erdos tpu/swarm/examples/v5p32-cells/stageF-q-ctx-n.yaml -d -y
```

(with the controller environment from EXPERIMENTS.md), one per cell.
