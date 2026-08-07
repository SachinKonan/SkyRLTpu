# Sweep1 Erdős leg — results and findings (2026-08-04 → 08-06)

**STATUS: both variants COMPLETE (grpo 08-05 11:58 UTC, ttd 08-06 02:52 UTC).**

The erdos_min_overlap leg of the 6-variant sweep: {entropic_adaptive_beta ("ttd"),
mean_baseline ("grpo")} on Qwen3.5-27B (thinking-on), self-hosted SkyRL tinker on
v5p-32 spot slices. 15 steps per variant, 32 groups × 16, elite 2, KL 0, LoRA r32,
lr 4e-5, ctx 18432 (phase1 13824 / tail 4608), TTD_REJECT_TRUNCATED=1, masked
answer-cue prefill on thinking exhaustion. wandb: tpu-tinker-exps
sweep1-{ttd,grpo}-erdos.

## Headline: both arms broke every prior sub-120b record

| Source | best c5 (lower = better) |
|---|---|
| Authors' true record (discover#19) | 0.380875 |
| Our gpt-oss-120b run | 0.380888 |
| **sweep1 ttd-erdos FINAL (step 14)** | **0.380953139** |
| **sweep1 grpo-erdos (step 10)** | **0.380984785** |
| In-context contrast arm best | 0.381182 |
| 21-step qwen dbtest champion | 0.381294 |
| gpt-oss-20b 15-step champion | 0.381472 |

Gap to the authors' record: 7.81e-5 (ttd). Final ttd champion: 192-pt step fn found step 14 (pool 780); the 0.380953891 (step 11, n=57) intermediate is also archived. Both champions verified three ways:
exact recomputation of the overlap correlation from the stored construction
(bit-identical to 9 dp), reward cross-check (env reward = 1/c5 matches logged
reward/max exactly), and the own-grid-exactness argument — for piecewise-constant
h the overlap integral is piecewise-LINEAR in the shift, so its continuous max
lands on a lattice shift and the discrete n-point computation is exact at any n
(empirically: ×1000 refinement moves c5 by nothing at 9 dp).

Champion constructions archived in `sweep1-champions/` (full pool states incl.
code, GCS: skyrl-runs/sweep1-*-erdos + skyrl-run-archives/0804-2245-*).

## Findings

**1. Algorithm parity on champion quality; ttd wins on process.** Final champions
differ by 3.1e-5 — noise at this depth. The two arms leapfrogged each other all
run (ttd 0.381044 → grpo 0.381127 → grpo 0.380985 → ttd 0.380968 → ttd 0.380954),
each mining an independent pool. But ttd held better correctness (up to 29.9% vs
grpo's ~22-26%) and its healthy-format phases produced champions, while grpo's
best finds came from a format-collapsed policy (see 2). At erdos, entropic vs
mean-baseline is a wash on outcome; entropic looks steadier on the way there.

**2. Format decay tracks discovery on BOTH arms — it's the reward structure, not
the advantage estimator.** grpo: 80.6% → 1.1% format over 13 steps, with its two
champions found at format 3.2% and 4.5% (~16 valid rollouts/batch). ttd pre-reset:
80.4% → 50%. The policy learns to spend ever more budget thinking, blowing the
output contract; the survivors are qualitatively better (both arms' champions
arrived DURING the slide). reject_truncated keeps the failures out of the
gradient, so this is an exploration tax, not divergence — but at ~1-3% format the
effective batch is a sliver. Next-round candidate fix: KL=0.1 anchor (gpt-oss held
98% format with it) vs. the exploration cost it imposes.

**3. Pool-persistence + policy-reset beat continuous training (the accidental
discovery).** Because weight checkpoints were disabled, every spot preemption
restarted the LoRA fresh while (after the pool-resume fix) keeping the PUCT pool.
Every post-reset incarnation immediately out-discovered the decayed policy it
replaced: 0.381044 found 1 step after a cold start on a warm pool; 0.380968 and
0.380955 found at format 71-75% / correctness ~30% (the run's healthiest metrics)
by a fresh policy reading an elite pool. Mechanism: the pool carries the search
knowledge; the reset discards the format-collapse attractor while elite seeds
re-teach competence in ~1 step. Worth promoting from accident to technique:
periodic deliberate policy resets (or a reset-on-format-collapse trigger) on a
persistent pool.

**4. The search explores parameterization resolution, not just values.** Champion
n_points sequence: 47 → 120 → 400 → 192 → 57. Coarse step functions (47-57 pts)
won first and last — fewer knobs suit the annealing/block-optimization strategies
the programs run, and own-grid c5 is exact at any resolution (no fidelity
penalty). The 400-pt grpo champion shows refinement also pays. Elite redundancy
across resolutions is real diversity, not clones.

**5. Ops (details in memory ttd-qwen35-sweep1):** pool-only resume (572aee8) +
checkpoint-404 fallback (c43e7a2) made ~hourly spot preemption a non-event —
the erdos pair completed through ~8 preemptions across 2 slices with zero manual
resumes and zero lost champions after the fixes landed. Engine model slots are
never freed (driver restarts engine per variant); circle/acineq need
problem_type (f2b522d1).

## Reproduction pointers

- Runs: wandb tpu-tinker-exps (sweep1-ttd-erdos, sweep1-grpo-erdos + crashed
  fragments from pre-fix incarnations).
- Pool snapshots: gs://sk7524-tinker-tpu-us-east5/skyrl-runs/sweep1-{ttd,grpo}-erdos/
  tinker_log/*/puct_sampler_step_*.json (per-step, incl. full programs).
- Config: sweep_driver.sh CFG line (memory ttd-qwen35-sweep1); client commits
  through c43e7a2 on discover branch ttd-dbtest-diagnostics.
