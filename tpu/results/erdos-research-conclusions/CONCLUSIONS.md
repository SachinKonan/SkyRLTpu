# ttt-discover on Erdős min-overlap: experimental conclusions

*Synthesis for maintainers of the `third_party/discover` reimplementation in SkyRLTpu. Every c5 figure below was re-derived from raw sampler buffers (`puct_sampler_step_*.json`), wandb per-rollout tables, and `train.log` config blocks by an adversarial verification pass. Verdicts are tagged **[CONFIRMED]** (re-derived + clean single-knob attribution) or **[SUGGESTIVE]** (numbers exact, attribution carries a named confound). The known true record is **c5 = 0.3808753232177187** (discover issue #19 maintainer reply); the paper's rounded figure 0.380876 is not the bar.*

---

## 1. TL;DR

1. **[SUGGESTIVE]** The narrow **8×64** batch shape reaches deep c5 frontiers **2–3× faster in rollouts** than 64×8 (steps to c5≤0.38095: 8×64 = {4,7,9} vs 64×8 = {13,15,19}); the speed is real but the two shape groups also differ in an *active* elite-slot setting (see C1). *Evidence: cumulative-best over all `puct_sampler_step_*.json` in the four delite runs + the two ctrl runs; threshold table CONFIRMED exact.*
2. **[SUGGESTIVE]** 8×64's speed comes with a **lineage-lock-in / stall failure mode**: `ttd_120b_a8x64_delite` froze at c5=0.380921331 for 15 consecutive steps while its 64×8 twin kept improving; the collapse only manifests at the `{120b, 8×64, distill, elite2}` intersection, not from narrow shape alone (C1).
3. **[CONFIRMED]** **PUCT never restarts and grows depth at exactly +1/step in every shape**; shape changes breadth/concentration, not depth velocity. The pool grows at a hard cap of 2×seeds/step (`topk_children=2`), so 64×8 saturates the top-1000 prune cap by step 8–9 while 8×64 ends with a 3–4× smaller, never-pruned ~250–330-state pool (C2).
4. **[CONFIRMED]** **Distillation alone is frontier-neutral-to-negative** (distill15 0.381039226 *lost* to ctrl15 0.381001033 by +3.8e-5) — it ~doubles improve-rate and delays exploration collapse ~2 steps but caps improvement *magnitude*; **distill + elite8 flips the sign** (distelite15 0.380925116, −7.6e-5 vs ctrl) by restoring frontier seed allocation that distillation's flattened value landscape had destroyed (C3).
5. **[CONFIRMED]** The distill+elite win **replicates on the independent circle-packing problem (n=26)** as a *speed/robustness* win, not an endpoint win: `c26_delite` led at all 12 intermediate steps and hit the joint attractor 5–6 steps early, but both runs converge to the identical value 2.63598308491… (C3).
6. **[CONFIRMED]** In the clean **16×32 GRPO-vs-entropic A/B**, GRPO (mean_baseline) currently leads entropic by **1.58e-4** (0.380927971 vs 0.381085635 at step 14), because entropic's scale-invariant winner-take-most gradient **collapses the pool onto one root lineage ~2 steps earlier** and locks onto a worse family — both runs in-flight at step 14/30 (C4).
7. **[SUGGESTIVE]** Widening entropic's seed base **delays its collapse** (ctrl15 at 64 seeds keeps ≥8 root lineages to step 9 and finishes 8.5e-5 deeper than entropic@16), consistent with a **"entropic is seeds-hungry"** interaction hypothesis — but heavily confounded (seeds vs group-size vs elite all move together) (C4).
8. **[CONFIRMED]** Our design choices have **direct published precedent**: 8×64 is the ttt-discover paper's own default; entropic's KL-constrained scale-invariance mirrors NES/CMA-ES rank-based fitness shaping; and GRPO-arm exploration die-off matches documented RLVR entropy-collapse literature (C5).

**Deepest frontier across all runs: `ttd_gptoss120b_full` at c5 = 0.380887659** (64×8, elite0, no distill, 120b, 20 steps) — still 1.23e-5 short of the true record.

---

## 2. Experiment matrix

All Erdős runs use **512 rollouts/step** (shape = GroupsPerBatch × GroupSize). Best c5 = −max(state value) over the run's final sampler buffer.

| Run dir | Model | Shape G×B | Elite | Distill | Objective | Steps | Best c5 | Status |
|---|---|---|---|---|---|---|---|---|
| `runs/ttd_gptoss20b_full` | 20b | 64×8 | 0 | no | entropic | 15 | 0.381471836 | done (300s eval baseline) |
| `runs/ttd_gptoss20b_ctrl15` | 20b | 64×8 | 0 | no | entropic | 15 | **0.381001033** | done |
| `runs/ttd_gptoss20b_distill15` | 20b | 64×8 | 0 | b=0.1 | entropic | 15 | 0.381039226 | done |
| `runs/ttd_gptoss20b_distelite15` | 20b | 64×8 | 8 | b=0.1 | entropic | 15 | **0.380925116** | done (best 20b @64×8) |
| `runs/ttd_20b_a8x64_delite` | 20b | 8×64 | 2† | b=0.1 | entropic | 15 | **0.380913928** | done (best 20b overall) |
| `runs/ttd_gptoss120b_full` | 120b | 64×8 | 0 | no | entropic | 20 | **0.380887659** | done (deepest overall) |
| `runs/ttd_gptoss120b_distelite` | 120b | 64×8 | 8 | b=0.1 | entropic | 20 | 0.380946823 | done |
| `runs/ttd_120b_a8x64_ctrl` | 120b | 8×64 | 2 | no | entropic | 20 | 0.380939058 | done |
| `runs/ttd_120b_a8x64_delite` | 120b | 8×64 | 2 | b=0.1 | entropic | 20 | 0.380921331 | done (stalled @step5) |
| `runs/ttd_obj_ttt_16x32` | 20b | 16×32 | 2 | no | entropic | 30 | 0.381085635 | **in-flight ~14/30** ‡ |
| `runs/ttd_obj_grpo_16x32` | 20b | 16×32 | 2 | no | **mean_baseline** | 30 | 0.380927971 | **in-flight ~14/30** ‡ |
| `runs/ttd_circle26_ctrl15` | 20b | 64×8 | 0 | no | entropic | 15 | 2.635983084918¶ | done (circle packing) |
| `runs/ttd_circle26_distelite15` | 20b | 64×8 | 8 | b=0.1 | entropic | 15 | 2.635983084918¶ | done (circle packing) |

† **`TTD_ELITE_SLOTS` is env-only and never logged.** For the 8×64 runs it was recovered by simulating `PUCTSampler.sample_states` pick order and matching it against recorded wandb `sampler_states_*` tables: `ttd_20b_a8x64_delite` is consistent with elite∈{0,1,2,4} but the elite knob **changed no picks** (behaviorally inert); `ttd_120b_a8x64_*` uniquely fit elite=2; both 64×8 delite runs uniquely fit elite=8; `120b_full` was launched before the elite feature existed (elite=0, confirmed by commit date 2026-07-14 vs launch 2026-07-09).
‡ 16×32 runs are the objective A/B, still running (final ranking may change). Best-c5 figures are at step 14, the latest dump present for both.
¶ Circle26 is a **different problem** (n=26 circle packing, maximize Σradii; value positive ~2.63). Both runs converge to the same attractor; final signed diff is noise-level (~1e-13).

---

## 3. Conclusions by theme

### C1 — Batch shape (64×8 vs 8×64 vs 16×32) and its interaction with the sampler

**Claim.** At matched compute (512 rollouts/step everywhere), the **narrow 8×64 shape converges to deep frontiers 2–3× faster than 64×8**, but the win carries a proven elite-slot confound and a stall failure mode; the **middle 16×32 shape is worst under the entropic objective but only because of the objective, not the shape.**

**Mechanism.** With `topk_children=2`, the pool grows at a hard cap of 2×seeds/step. 8 seeds/step forces greedy PUCT to concentrate all 512 rollouts on the top ~2-value families, which **collapses to a single root lineage by step 6** (verified: distinct roots among new children go 8→5→3→…→1, top-lineage share 1.00 from step 6 in *both* 8×64 runs, despite one being 20b+distill and the other 120b+no-distill). This is pure fast hill-climbing: the best-state ancestor chain length equals the step count in every 8×64 run (15/15, 20/20). 64 seeds/step instead maintain a diverse ~1000-state pruned frontier (2–4 live roots at step 15), which climbs slower but retains somewhere to pivot.

**Evidence.**
- **Speed [CONFIRMED]:** steps to first reach c5≤0.38095 = **{4, 7, 9}** for {120b_8x64_delite, 20b_8x64_delite, 120b_8x64_ctrl} vs **{13, 15, 19}** for {20b_64x8_delite15, 120b_64x8_full, 120b_64x8_delite}; 16×32 entropic *never* crossed it in 14 steps (GRPO variant crossed at step 6). Attribution survives because distill and model are crossed within each shape group and the one uncontrolled knob (elite) points *inconsistently* — the zero-elite `120b_full` (step 15) is still slower than every elite-2 8×64 run, and adding elite within 64×8 *slowed* it (full=15 → delite=19). Files: `runs/ttd_120b_a8x64_delite/tinker_log/erdos-120b-a8x64-distelite/puct_sampler_step_000005.json` and siblings; script gist: per run iterate sorted buffers, `best_c5 = -max(state.value)`, cumulative-min, first step ≤ threshold.
- **Deeper-at-matched-budget [SUGGESTIVE]:** with distill+elite held on, 8×64 leads 64×8 at *every* step ≥1: 20b pair 0.380913928 vs 0.380925116 at step 15 (gap 1.12e-5); 120b pair 0.380921331 vs 0.380946823 at step 20 (gap 2.55e-5). **Confound:** elite slots differ *with* shape and are behaviorally active — both 64×8 arms ran elite=8 (unique fit 14/14 and 19/19 steps; elite altered picks in 9/14 steps of the 20b run vs an elite-0 sim), while the 8×64 arms ran elite≤2 (and the 20b 8×64 arm's elite was inert). The comparison is effectively *(8×64, plain-PUCT-equivalent)* vs *(64×8, elite-8-active)*. The 120b gap is also shrinking monotonically after step 6 (8×64 flat from step 5), so the horizon endpoint matters.
- **Stall risk [SUGGESTIVE]:** `120b_8x64_delite` froze at 0.380921331 from step 5 through step 20 (improve-rate 0.708→0.173, five steps with 0/8 seeds improving) while its 64×8 twin sustained ~0.79 late improve-rate and improved to step 20. But the collapse needs the full `{120b, 8×64, distill, elite2}` intersection — `20b_8x64_delite` (same shape) had 0.969 late improve-rate and did not stall. Note the honest framing caveat: the "stalled" run's frozen 0.380921331 is *still better* than the healthy 64×8 twin's 0.380946823, so this is a diagnostic-flatness failure mode, not an absolute-outcome failure.
- **16×32 is objective-dependent [CONFIRMED]:** 16×32 entropic = 0.381085635 @14 (8.5e-5 worse than ctrl15's 0.381001034; never crossed 0.3810), but the *identical* 16×32 shape under GRPO reached 0.380927971 by step 8 — proving the shallowness is not shape-intrinsic. The within-16×32 A/B is genuinely single-knob (only `adv_estimator` differs; jobs launched/restarted as a pair on identical code).

**What would falsify/strengthen.** A **64×8-elite2 run** and an **8×64-elite8 run** would factorize shape from elite (currently they move together). Multiple seeds per arm would turn the 1–2.5e-5 endpoint gaps (comparable to a single step's improvement, n=1) into a statistical claim. Running the 16×32 pair to step 30 would settle whether GRPO's lead there is durable.

---

### C2 — PUCT branch expansion, elite slots, and the no-restart property

**Claim.** The sampler expands **only the current frontier** (depth +1/step, no restarts), keeps an **at-most-binary retained tree** (`topk_children=2`), and elite slots reshape the tree — but elite is **largely redundant once the pool has collapsed to one lineage**, so its measured benefit is real only in diverse (64×8) pools.

**Mechanism & evidence.**
- **Depth velocity = +1/step, shape-invariant [CONFIRMED].** Diffing consecutive buffers by state id, every new surviving child sits at depth exactly `t` in 15/15 steps for ctrl15, 20b-8×64, and 120b-8×64 (20/20). Max final depth = number of steps in all runs (15/15/15/15/14/20). Adding a matched pair the original omitted: 120b 64×8 (`ttd_gptoss120b_full`) has max-depth trajectory 0…20, identical to 120b 8×64. Read this as an empirical regularity (depth never grows *slower*; the +1 upper bound is mechanical), not a hard PUCT invariant — `ttd_obj_ttt_16x32` shows genuine depth-(t−1) children at steps 6–9. Code: `third_party/discover/ttt_discover/tinker_utils/sampler.py` (`_set_parent_info`, parents chain = [immediate parent … root]).
- **Pool mechanics [CONFIRMED].** Growth is capped at 2×seeds/step: 64×8 grows ~+100–128/step and saturates the top-1000 prune cap by step 8–9; 16×32 ~+31/step (443–446 states @14); 8×64 grows +16/step to only 248 (15 steps) / 310–328 (20 steps) — a 3–4× smaller, never-pruned pool. Amendment: growth is a *cap* realized 65–100% (the 300s-eval-timeout `20b_full` only grew 35–78/step, pool 933@14, never saturated). Mechanism in `sampler.py`: `max_buffer_size=1000` (line 91), `topk_children=2` (line 95), `_filter_topk_per_parent` + `flush()` keep top-2 children/parent, `_finalize_and_save` prunes to top-1000 by value with initial states pinned.
- **topk-2 saturation [CONFIRMED].** Final-pool children/parent: 8×64 `{2:120}` / `{2:160}` (every parent at cap); 16×32 `{1:7, 2:210}` (97%); ctrl15 64×8 `{1:230, 2:353}` (60.6%). Per-step survival is exactly 16/512 (3.1%) for 8×64 vs 81–116/512 (16–23%) for 64×8. (64×8's `{1:…}` counts are partly prune artifacts, not just failure to make 2 good children.)
- **Lineage collapse by shape [CONFIRMED].** First step with top-lineage-share = 1.00: **t=6 (both 8×64), t=9 (16×32), never (both 64×8)**. The elite gradient (2/2/0 across 8/16/64 seeds) points *against* the claim's direction yet 64×8 (elite0, least protection) never collapses — so collapse timing tracks seeds/step cleanly.
- **Elite reshapes the 64×8 tree [CONFIRMED for tree shape].** At 64×8, elite=8 replaces strict consecutive-step polish chains with **gap-tolerant reseeding of interior nodes**: distelite15's best (c5 0.380925116) is a depth-9 chain with 2–3-step gaps at each of which the resumption is an elite-seeded link (8/9), vs ctrl15's unbroken depth-15 consecutive chain (0.381001033). The isolation is clean because `distill15` (distill-on, elite-0) behaves like ctrl15 — so gaps track elite, not distill — and a PUCT counterfactual shows the three stale gap-resumption parents rank 118–242 under pure PUCT-64 (never sampled) yet occupy elite-8 slots. The *value* improvement remains a joint distill+elite effect (n=1).
- **Elite is redundant once collapsed [CONFIRMED for the redundancy mechanism].** Disabling elite in a counterfactual changes the seed set in **0/35** 8×64 batches (15/15 and 20/20 identical) but **9/15** distelite15 (64×8) batches. Elite-child-vs-PUCT-child mean-c5 advantage shrinks from **0.0054 (64×8-e8)** to **~0.0003 (both 8×64-e2)**; top-decile enrichment 5.6× → 1.8–2.9×. distelite15's run-best (0.380925116) came *uniquely* from an elite seed (PUCT-best only reached 0.380991), whereas both 8×64 bests are tied between elite/PUCT (within 5e-8). **Caveat on the enrichment-share sub-claim:** the "elite produces 45–73% of top-decile children" figures are largely a *relabeling* of PUCT's own picks in the two 8×64 runs (counterfactually inert), so that specific sub-claim was downgraded to PLAUSIBLE — but the redundancy conclusion (elite adds nothing once the pool is one family) is clean.

**What would falsify/strengthen.** A 64×8-elite2 vs 64×8-elite8 pair would isolate elite *count* from the mechanism. Logging `TTD_ELITE_SLOTS` into `train.log` (it is currently env-only) would end the reconstruction guesswork. A run that deliberately re-seeds a *different* root mid-training would test whether 64×8's late branch-switching (120b_full's step-15/18 jumps off a 13-ancestor side lineage) is exploitable on demand.

---

### C3 — Distillation: what it changes, why it is frontier-neutral alone, why distill+elite works

**Claim.** Contrastive context distillation (β=0.1) **raises improvement *frequency* ~2× and delays exploration collapse ~2 steps, but does not raise improvement *magnitude* and slightly *loses* the frontier race alone**; it becomes a decisive win only when paired with elite slots, which restore the frontier seed allocation that distillation's flattened value landscape had collapsed. This **replicates on circle26** as a speed/robustness win.

**Mechanism & evidence.**
- **Frequency up, collapse delayed [CONFIRMED].** Pooled steps 7–14 improve-rate **1.81% (32/1769) for distill15 vs 0.79% (15/1890) for ctrl15** (2.28×; two-proportion z=2.73, p≈0.006); distill crosses below 10% improve-rate at step 6 vs ctrl at step 4 (~2-step delay), same ~0% asymptote. Config diff isolates the knob: only `distill_enabled False→True` differs. Files: `runs/ttd_gptoss20b_{ctrl15,distill15}/tinker_log/*/wandb/latest-run/files/media/table/`; matches `tpu/results/erdos-distill-ab/ANALYSIS.md` digit-for-digit.
- **Magnitude capped [CONFIRMED].** Pooled steps 3–14, improvements bin as micro(<1e-3)/mid/big: ctrl 83/8/0 (max gain 3.55e-3), distill 165/19/0 (max 2.46e-3) — **zero gains >1e-2 in either arm, and distill's max gain is smaller than ctrl's.** Distillation teaches *more frequent* incremental edits, not bigger jumps.
- **Diversity is value-space, not lineage [CONFIRMED].** distill15's top-20 value spread is **~4,470× wider** than ctrl15's (5.50e-5 vs 1.23e-8) and spans 3 construction-size families vs ctrl's single n=1000×20 — but *all* top-20 in both arms descend from **one timestep-0 founder**, so the extra "diversity" is geometric sub-branching within one lineage, not independent lineages.
- **Distill alone loses; distill+elite flips the sign [CONFIRMED].** ctrl15 0.381001033; distill15 0.381039226 (**+3.82e-5, ctrl better**); distelite15 0.380925116 (**−7.59e-5 vs ctrl, −1.14e-4 vs distill15**); 20b_a8x64_delite 0.380913928 (−8.71e-5 vs ctrl). The distelite15-vs-distill15 comparison isolates elite8 cleanly (single default-off commit between them; elite activation confirmed behaviorally). Margins are single-run and small vs the ~1.3e-4 inter-run spread, so "flips the sign" is *directional* (clean pairs + trajectory shapes: ctrl frozen from step 5 vs distelite still descending to step 13), not statistically powered.
- **Elite mechanism [CONFIRMED].** In steps 9–14 the elite band (seed c5<0.38105) got **30.7% of ctrl15 seeds** (clone-chain over-exploitation), only **3.4% of distill15 seeds** (distillation flattened the value landscape → PUCT stops re-selecting the tip), and **11.2% of distelite15 seeds** — of which the 8 forced elite rows supplied 34/43 band picks (79%). The result was a genuinely new champion family (69-point construction, 18/20 of final top-20) polished to 0.380925 through a 9-link chain, while retaining distill-level exploration.
- **circle26 replication [CONFIRMED].** `c26_delite` (distill+elite8) led `c26_ctrl` at all 12 intermediate steps (max +0.009226 at step 1) and reached the joint-final value **2.635983084918** at step 8, 5–6 steps before ctrl (step 13–14). Final signed diff is noise-level (~1e-13) — both converge to the identical attractor (likely the known n=26 optimum), so this is a **speed/robustness win, not an endpoint win**. Combined treatment (distill AND elite together) — cannot separate the two knobs on this problem.

**What would falsify/strengthen.** The distill-alone-loses margin (+3.8e-5) is within plausible single-run seed noise (1.3e-4 spread across the four runs) — **replicate seeds** to confirm the sign. A **distill+elite8 vs no-distill+elite8** pair (currently missing) would isolate distillation's contribution *inside* the elite regime. Measure policy entropy directly to confirm the "flattened value landscape" mechanism rather than inferring it from seed-band allocation.

---

### C4 — GRPO (mean_baseline) vs entropic_adaptive_beta

**Claim.** From the advantage math, **entropic is scale-invariant winner-take-most and GRPO's gradient in a mixed group is almost purely a validity signal**; empirically in the current in-flight 16×32 A/B, **GRPO leads by 1.58e-4** because entropic collapses the pool onto one lineage ~2 steps earlier and locks onto a worse family. A **seeds-hunger interaction hypothesis** (entropic needs many seeds to avoid premature commitment) is directionally supported but heavily confounded. **Both 16×32 runs are in-flight (~14/30) — this ranking may change.**

**Mechanism (from the advantage math) [CONFIRMED].** Reimplementing and executing the actual `compute_advantages` (`third_party/discover/ttt_discover/rl/train.py:118–195`):
- **entropic** solves β per group so KL(softmax(βr)‖uniform)=log 2, giving leave-one-out advantages that depend only on *relative* rewards. On a near-tie group `[2.6240 + 1e-4·i]`, β*=6720 and the best rollout gets advantage **+5.77** while the rest are ~−1 (zero weight); shrinking spacing to 1e-6 rescales β* to 1.65e5 and the best still gets **+4.59** — a full-strength "copy the best sibling" gradient no matter how microscopic the spread. (Boundary: a β_max=1e6 cap means below ~6.7e-7 spacing (k=8) entropic advantages *also* vanish; irrelevant at Erdős scale, |r|≈0.381.)
- **mean_baseline** on the same group gives ±3.5e-4 (best gets +3.5e-4, ~16,500× smaller) and shrinks linearly with the spread to zero.
- **In the real mixed-reward regime** (`frac_mixed=1.00` every step; valid reward=1/c5≈2.62, failure=0), GRPO's gradient is 99.6% a validity indicator — every valid rollout gets ~+1.7 whether it is the group best or 1e-4 worse — while entropic still discriminates hard within the valid set (logged adv_max up to 18.6, though that ceiling is a sparse-valid *concentration* ladder, with genuine within-valid grading amplifying best-vs-2nd tie gaps 76–88× in majority-valid groups).

**Current A/B state [CONFIRMED, in-flight].** At dump 14 (latest for both), **GRPO 0.380927971 vs entropic 0.381085635, gap 1.58e-4**; GRPO leads at every dump 4–14 (entropic led only at dumps 1 and 3). Clean knob isolation: full config.json diff = `adv_estimator` only. **Post-collapse the objectives enter opposite regimes:** entropic keeps a 73–100% improve-rate but gains only 2.4e-6 c5 over dumps 8→14 (micro-grinding), while GRPO's improve-rate drops to 31–40% and its frontier is *exactly flat* (Δ=3.1e-13) — GRPO's entire lead was built *before* collapse via PUCT selection over diverse children. Files: `runs/ttd_obj_{grpo,ttt}_16x32/tinker_log/*/puct_sampler_step_000014.json`.

**Lineage-collapse cause [CONFIRMED].** Entropic's top-20 pool is single-root from dump 7; GRPO keeps ≥2 root lineages among new children through dump 9 (collapse ~2 steps later). But *which* family each locks onto is a single stochastic draw (GRPO's winning family was already better at dump 4, before the concentration trajectories diverged) — so "entropic locked onto a *worse* family" is one draw, n=1.

**Seeds-hunger interaction [SUGGESTIVE].** ctrl15 (entropic at 64 seeds) retains 42 distinct root lineages among new children at dump 2, still 8 at dump 9, and collapses its top-20 to one root only at dump 10 (vs dump 7 for entropic@16); it reaches 0.3810010342 — **8.5e-5 better than entropic@16** — yet still 7.3e-5 worse than GRPO@16. **Heavily confounded:** ctrl15 differs from the 16×32 pair in seeds (64 vs 16), group size (8 vs 32 — itself changes both estimators' per-group statistics), and elite (0 vs 2). Treat as directional corroboration only.

**What would falsify/strengthen.** **Run the 16×32 pair to step 30** — GRPO has been flat since step 8 while entropic is still creeping (2.4e-6 over 6 dumps); closing 1.58e-4 at that rate needs far more than 16 steps, but the horizon is not yet reached. The clean seeds-hunger test is **entropic 64×8 vs GRPO 64×8** (or a single-estimator 64×8-vs-16×32 sweep with elite matched). Multiple seeds per arm would separate "entropic locks onto a worse family" (mechanism) from bad luck (one draw).

---

### C5 — External grounding in the PUCT / GRPO / ES literature

**Claim.** Every major design element we studied has published precedent, and our shape/objective findings align with (or extend) it.

- **The ttt-discover paper [CONFIRMED].** "Learning to Discover at Test Time" (arXiv:2601.16175; code `github.com/test-time-training/discover`) specifies exactly the machinery we reimplement: the entropic objective *J_β = E_s[log E_a[e^{β(s)R}]]* with β set per initial state by constraining the induced policy's KL divergence (App A.1, incl. "too large β early … instabilities, too small later … advantages vanish"); PUCT reuse *Q(s)+c·P(s)·√(1+T)/(1+n(s))* with **Q = max (not mean) child reward** and P a rank prior. **The paper's default batch shape is 8 groups × 64 rollouts (512/step, 50 steps)** — so our 8×64 runs match the authors' default, and **64×8 / 16×32 are our deviations.** Paper Erdős results: Haugland 0.380927, AlphaEvolve 0.380924, TTT-Discover 0.380876 (16× larger improvement than AlphaEvolve's). Caveat: the paper trains only 120b at 50 steps; our sampler's PUCT score carries an extra value-range `scale` multiplier not in the paper's printed formula.
- **PUCT prior/visit + progressive widening [CONFIRMED].** AlphaZero-lineage PUCT (*U = c·P·√N_total/(1+N)*, Lc0 primer) concentrates on high-prior, low-visit branches; the standard fix for wide action spaces is progressive widening, whose documented pathology (Volume-MCTS, arXiv:2407.05511) is "reward-blind … very rapid branching … many short branches that explore the starting region much more than other regions." Our `topk_children=2` + top-1000 value pruning are **reward-aware analogues of capping branching** (interpretation, not a published result about this system).
- **GRPO definition [CONFIRMED] + group-size theory [PLAUSIBLE].** GRPO (our mean_baseline arm, `r − group_mean`, minus DeepSeekMath's std normalization) is DeepSeekMath (arXiv:2402.03300). A U-statistic analysis (arXiv:2603.01162) derives a "universal scaling law … for selecting the optimal group size." **Correction to the original absence claim:** a fixed-budget prompts×completions ablation *does* exist in the literature (IsoCompute Playbook, arXiv:2603.12151, sweeps rollouts×problems at fixed budget), so our 64×8/16×32/8×64 experiment is not without precedent — though our exact single-run 3-arm shape A/B at matched step counts may still be novel.
- **GRPO entropy collapse [CONFIRMED].** Exploration die-off under GRPO-style RLVR is documented with a token-level mechanism (arXiv:2605.11491: "entropy-decreasing tokens consistently outweigh entropy-increasing ones"; corroborated by SCOPE-RL, arXiv:2510.08141: positive/negative samples have opposite entropy effects). So a GRPO-arm die-off in `ttd_obj_grpo_16x32` would have independent support — though these papers study binary-reward math benchmarks, not PUCT-pooled construction search, so the transfer is inferential.
- **Max-reward objectives + ES rank shaping [PLAUSIBLE].** Optimizing max-of-N is an established family (max@k, arXiv:2510.23393; BoN-aware fine-tuning, Chow et al. arXiv:2412.15287), supporting entropic winner-take-most over vanilla GRPO for record-hunting — *correction:* only 2510.23393 (not both papers) states mean-reward RL erodes BoN-relevant diversity. Entropic's KL-constrained scale-invariance and PUCT's rank prior P mirror **NES/CMA-ES rank-based fitness shaping** (Wierstra et al. arXiv:1106.4487: "invariant under monotonically increasing … transformations … the gradient is disproportionately distorted by extreme fitness values"). Nuance: NES ranks give *full* monotone-invariance; entropic's softmax(βr) is only *affine-scale*-invariant — so only PUCT's P is literally rank-based.

**What would strengthen.** Measure policy entropy in both 16×32 arms to test whether the GRPO die-off matches the published token-level entropy-flow mechanism directly. Compare against a literal rank-shaped (NES-style hard-rank) advantage variant to test whether full monotone-invariance beats entropic's affine-only invariance.

---

## 4. Open questions / next experiments

1. **Factorize shape from elite.** Run **64×8-elite2** and **8×64-elite8** so the C1 speed/depth attribution stops being confounded by elite slots that currently move *with* shape.
2. **Finish the objective A/B.** Carry both 16×32 runs to step 30. GRPO leads by 1.58e-4 now but is flat since step 8 while entropic still micro-grinds; the horizon is not reached.
3. **Clean seeds-hunger test.** Entropic **64×8 vs GRPO 64×8** (or GRPO 8×64), to test the hypothesis that entropic needs many seeds to avoid premature lineage lock-in, without confounding on group size and elite.
4. **Isolate distillation inside the elite regime.** A **no-distill + elite8** run (currently missing at 64×8) vs distelite15 would show whether distillation adds anything once elite slots are present, or whether elite alone carries the −7.6e-5 win.
5. **Replicate the best runs with fresh seeds.** All headline margins are n=1; the sign of "distill alone loses" (+3.8e-5) and the 1–2.5e-5 shape endpoint gaps are within single-run noise.
6. **Log `TTD_ELITE_SLOTS`** into `train.log` and the sampler buffer so elite settings stop requiring behavioral reconstruction from wandb pick-order tables.
7. **Exploit 64×8's late-jump capability.** `120b_full` made frontier jumps at steps 15/18 off short side lineages — deliberately reseeding diverse roots late may push below the current deepest 0.380887659 (still 1.23e-5 above the true record).

---

## 5. Appendix

### 5.1 Claims that did not survive verification

- **"Elite slots produce 45–73% of top-decile children *because of the elite mechanism*"** — downgraded to PLAUSIBLE: in both 8×64 runs a pure-PUCT counterfactual picks the *identical* seed set every step (15/15 and 20/20), so the elite share there is a relabeling of PUCT's own picks, not a mechanism effect. The share *numbers* are correct; only the causal attribution fails. (Genuine elite divergence appears only in distelite15/64×8, 9/15 batches.)
- **"Group max is almost always a unique singleton in *all three* 8×64 runs"** — the third run (`120b_a8x64_delite`) has median tie-count 12 (21.5 late) under distill+elite convergence; the uniqueness property is config-dependent, holding for `20b_a8x64_delite` and `120b_a8x64_ctrl` only.
- **"Missing an early singleton delays the whole winning lineage"** (group-variance F3) — PLAUSIBLE: an ancestry check shows 4 of 5 claimed per-run singletons are *not* ancestors of the final record; the winning lineage advanced through different (also-rare) rollouts. The weaker "8-rollout groups would frequently miss the early progress carriers" survives.
- **"No published fixed-budget prompts×completions shape ablation exists"** (W3) — refuted; IsoCompute Playbook (arXiv:2603.12151) is exactly such a sweep.
- **"*Both* max@k papers state mean-reward RL erodes BoN diversity"** (W5) — only arXiv:2510.23393 says this; Chow et al. (2412.15287) does not.
- **"8×64 grows *exactly* +16/step, cap always saturated"** and **"64×8 grows +100–128/step"** — overstated: `120b_a8x64_delite` dips to +12/+14 on 6 steps (ends 310 not 328), and the 300s-timeout `20b_full` grew only +35–78/step (pool 933, never capped). Growth is a *cap* of 2×seeds, realized 65–100% depending on eval-timeout/validity survival. Headline 3–4× pool disparity unaffected.
- **Minor numeric corrections carried into the text:** entropic led at dumps 1 *and* 3 (not "only dump 3"); circle26 final signed diff is step-13's +1e-12 (true final step is a −1.6e-13 noise-level tie); "premature convergence to *suboptimal* behaviors" is a slight embellishment of SCOPE-RL's "converge prematurely."

### 5.2 Methodology note

- **Frontier c5** = −max(state `value`) over `puct_sampler_step_NNNNNN.json`; cumulative-min across steps for trajectories. Sign convention verified every time (step-0 pool values ~−0.48…−0.57, i.e. c5 scale, not the early 1/c5 era).
- **Improve-rate / lineage** computed by id-diffing consecutive buffers (new = ids absent from prior step incl. `initial_states`); root = `parents[-1].id`; depth = `len(parents)`.
- **Group-variance findings** parsed per-rollout `gen&score_train_*` wandb tables (512 rows/step = 8 contiguous 64-row prompt blocks; c5 from "C5 bound:" in Message; validity = Correctness==1). Expected-best-of-g via exact without-replacement order statistics, conditioned on ≥1 scored rollout in the subsample.
- **Advantage math** verified by AST-extracting and *executing* the actual `compute_advantages` (train.py:118–195) under torch float32, cross-checked against an independent float64 numpy re-derivation with own KL-bisection.
- **Elite settings** (env-only, unlogged) recovered by reimplementing `sample_states` and matching predicted pick order against recorded wandb `sampler_states_*` tables — unique fits reported; behaviorally-inert cases flagged.
- **Config isolation** checked by diffing full `train.log` Config blocks + `config.json` + slurm `[smoke]`/`[ttd]` echoes for every compared pair; commit drift diffed and checked for behavioral inertness (KL-guard never fired: no `kl_error` in any `metrics.jsonl`).
- **Literature** verified by fetching arXiv HTML/PDF and matching quoted fragments verbatim across two independent fetches where possible.
- **Every reported number was independently re-derived** by an adversarial verifier before inclusion; verdicts (CONFIRMED / PLAUSIBLE) reflect whether the *attribution* — not just the arithmetic — survived a confound audit. All comparisons here are **n=1 run per arm** unless noted; treat sub-3e-5 margins as directional.
