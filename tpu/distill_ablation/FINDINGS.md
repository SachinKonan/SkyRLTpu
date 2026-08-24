# Erdős min-overlap discovery: distillation & cross-model investigation — consolidated findings

Comprehensive log of the July 2026 investigation into (1) the RL objective, (2) contrastive
context-distillation, (3) cross-model (foreign) distillation, and (4) the off-policy /
population formulation. All c5 values are independently recomputed from stored constructions
(`np.max(np.correlate(h, 1-h, 'full') * dx)` after normalizing `sum(h)=n/2`); lower c5 = better.

Task: find a step function h:[0,2]→[0,1], ∫h=1, minimizing C5 = max_k ∫ h(x)(1−h(x+k))dx.
Method skeleton (ttt-discover): PUCT state pool + GRPO-style RL on Tinker; each step PUCT
picks seeds from the pool, the policy rolls out G times per seed, rewards = 1/(1e-8+c5),
discovered states pushed back to the pool.

---

## 0. Leaderboard / reference points

| construction | c5 (verified) | notes |
|---|---|---|
| authors' true record (their 120B, n=600) | **0.3808753232177187** | published, maintainer-confirmed (discover#19) |
| our gpt-oss-**120b**, 20 steps | 0.380887659 | best we have |
| AlphaEvolve (arXiv 2601.16175) | 0.380924 | program-search SOTA |
| **20b-a-dst** (distill+elite, 8×64) | **0.380913928** | best 20B run |
| distelite15 (distill+elite8, 64×8) | 0.380925116 | |
| **GRPO 16×32** (no distill) | 0.380927971 | objective-swap alone ≈ distill tier |
| ctrl15 (20b RL-only, 64×8) | 0.381001033 | the own-pool baseline |
| gpt-oss+nemotron-nano ensemble (15 steps) | 0.381013 | ~tied with ctrl15 |
| distill15 (distill only, 64×8) | 0.381039226 | |
| **foreign best** (Qwen / Nemotron programs) | ~0.381001 | NOT above the 20B's own frontier |

**Load-bearing fact:** the foreign models' *best* (~0.381) is worse than gpt-oss's *best*
(~0.38091). This single fact explains most of the negative results below.

---

## 1. Objective A/B — GRPO vs ttt-discover's entropic estimator (20b, 16×32, elite=2, no distill)

Only variable: advantage estimator. Both frozen since step 8 (converged monoculture).

| arm | best c5 | gap to record |
|---|---|---|
| entropic (ttt-discover, `entropic_adaptive_beta`) | 0.381086 | +2.1e-4 |
| **GRPO** (`mean_baseline`) | **0.380928** | +5.3e-5 |

**GRPO beat entropic by ~1.6e-4** on the authors' own task. Mechanism (from PUCT-buffer
analysis): PUCT behavior was *identical* in both arms (96% seed-share to one family, ~12-gen
polish chain) — the sampler didn't decide it. GRPO won because mean-baseline updates keep the
policy's early mutations larger for ~2 extra steps in the discovery window (steps 2–4), landing
a better family before both collapse. Improve-event data: at step 3 GRPO's median improving
rollout took ~1.4e-4 steps vs entropic's 4.2e-5 — entropic mode-sharpens ~2 steps earlier.

**Shape × objective interaction:** entropic is *seed-hungry*. ctrl15 (64 seeds, entropic) hit
0.381001; the 16-seed entropic arm stalled at 0.381086. Entropic's winner-take-most collapse
needs many seeds' diversity to find a good family before locking in; GRPO thrives on few seeds.
So 16×32 handicaps entropic, suits GRPO — the headline 1.6e-4 overstates the pure-objective gap.

**Takeaway:** the estimator is *not* the main lever; a textbook GRPO swap reaches distill-tier
(0.380928) for free. Shape (fewer seeds → forced depth: 8×64 > 64×8) matters more.

**Infra note (Tinker prod OOV regression):** since ~2026-07-21 prod sampling emits reserved
o200k-harmony token ids > 200018 (the trainable-vocab ceiling) in ~4% of sequences; training on
them **poisons the TrainingClient** (fatal). Both arms crashed at step 13 within 45 min. Fixed
with a guard in `train.py` (`_max_train_token_id`, drops such datums from RL + distill batches,
metric `train/dropped_oov`, ~20/step). See memory `tinker-prod-oov-regression`.

---

## 2. Self-distillation dose ablation (freeze-RL, CE-finetune base gpt-oss-20b, held-out eval)

**Design (the clean instrument).** The frontier metric is a noisy PUCT lottery, so we *froze
RL*: fine-tune base gpt-oss-20b on a distillation corpus (cross-entropy only, no rollouts, no
PUCT), then measure on a **fixed held-out battery of 18 suboptimal seeds** (6 hard / 6 mid /
6 near-frontier), K=8 rollouts/seed, graded locally. Only the corpus differs across arms.
Corpus = pool-wide (worse, betters) pairs → the policy writes `<improve>`-block critiques.

**Dose sweep (self-origin betters, 20b teacher):**

| arm | dose | valid/144 | improve% | best-of-8 (seeds) | probe NLL |
|---|---|---|---|---|---|
| A0 | base | 64 (44%) | 17.2% | 6/18 | 0.857 |
| A1 | 8 | 82 (57%) | 18.3% | 6/18 | 0.813 |
| A2 | 32 | 75 (52%) | 21.3% | 5/18 | 0.809 |
| A64 | 64 | 81 (56%) | 13.6% | 6/18 | 0.803 |
| A3 | 128 | 76 (53%) | 18.4% | 7/18 | 0.803 |

### Finding 2a — improve-rate has NO dose response (it's noise)
17→18→21→14→18 with A64 (the mid gap) *lowest* kills the apparent "peak at 32." Every arm is
|z| < 0.62 vs A0 on a two-proportion test. Best-of-8 is flat too (5–7/18). At N=144 the
improve-rate channel is **below resolution** — resolving a 4-pt difference at p<0.05 needs
~1450 valid rollouts/arm (~20× our data). **The earlier "S*=32, +4.1 pts" claim was retracted.**

### Finding 2b — the ROBUST effect is CODE VALIDITY (not improvement quality)
Base writes runnable code only **44%** of the time; every distill arm is **52–57%** (A1 vs A0:
z ≈ 2.1, p ≈ 0.034 — the one comparison that clears significance). **Validity is dose-
insensitive** — 8 datums already achieve it, 128 adds nothing. So standalone-CE distillation's
one reliable contribution is teaching the policy to write *executable* programs, not (measurably)
better-improving ones.

### Finding 2c — internalization ≠ competence (the parrot signature)
Probe NLL (frozen critique set) falls monotonically with dose (0.857→0.803, saturates ~64) —
the policy internalizes the critique *task* more with more data. But that does NOT convert to
better improving (A3 internalized most, improved no more). First appearance of the theme that
dominates everything below: **the policy can learn to reproduce the critique disposition without
gaining the competence.**

### Finding 2d — the taught moves are already in the base model's repertoire
Categorizing the corpus critiques' `<what>` directions vs the improving-rollout code: the taught
moves (grid-refine, more-iters, projection, annealing, FFT, SLSQP) appear in **base's** improving
rollouts at the *same* rates (grid-refine 10/11, more-iters 11/11 in base A0). Distillation isn't
introducing novel capability — it's nudging the frequency of moves the model already deploys. This
mechanistically explains the null on quality: you can't teach a jump with advice the model already
follows. (Matches the prior ANALYSIS.md finding: "grounded critique raises frequency, not magnitude.")

### Why the pairs are weak (pair construction)
`select_distill_pairs`: worse = random from the top-half of the pool by value (excl top-8);
betters = the global champion (always) + 2 of the next-7-best. Because the source pool (ctrl15
step-15) is *converged*, this gives **tiny value gaps (~1.5e-4)** and **champion-dominated
betters** (one polished family in triplicate). The teacher is asked to explain a razor-thin gap
between two near-frontier programs → inherently marginal, polish-level advice.

---

## 3. The measurement instrument & recurring gotchas

- **Held-out eval, not frontier.** Best-of-K per seed (does the policy find *any* improvement in
  K tries — the deployment-relevant, higher-signal metric) + per-rollout improve-rate + probe NLL.
- **EVAL_TIMEOUT is the recurring project curse.** The prompt promises `budget_s=1000`; programs
  that use the full budget get guillotined if eval_timeout < ~1100. Bit us THREE times: original
  300s (killed ~40% of rollouts), the foreign probe at 900s (killed Nemotron's *good* programs →
  falsely "17% valid"), and generally. **Always eval_timeout ≥ 1100.**
- **N is small everywhere** (18 seeds, 144 rollouts/arm). Best-of-K over 18 seeds and improve-rate
  over ~70 valid are both low-power; treat small differences as suggestive. Robust signals require
  a *large effect* (validity's 12-pt gap) or a *sign flip* (Qwen's negative delta).

---

## 4. Filtering deep-dive (grounding + leak gates)

Two gates on the teacher's **final-channel** critique only (private analysis channel excluded):
1. **Grounding** (`_grounding_ok`): every `<cite>` must be a verbatim substring of the *worse*
   code. Real work — anchors the critique to the actual program. Drops ~30%.
2. **Leak** (`_leaks`): distinctive identifiers unique to the *better* program (snake_case len≥4,
   or len≥10) that appear in the critique → drop. Designed to prevent verbatim memorization.

**Finding: the leak filter is counterproductive for cross-model and buggy in general.** On a
stored 80-pair Qwen batch: leak-on = 37 datums, leak-off = **57** (+54%). Reading the dropped
critiques (via offline re-filter, free): they carried the *exact* techniques we want (smooth-max
"softmax of convolution values (log-sum-exp)", water-filling projection), dropped on either (a)
foreign *function-name* mentions ("as in the `project_to_box_and_sum` routine" — incidental) or
(b) **false positives on generic variable names** (`c5_new`, `h_sym`, `start_time`). Decision:
**drop the leak filter for cross-model; keep grounding.** General technique names (`logsumexp`
9-char, `L-BFGS`→`BFGS` 4-char) already pass, so principle-transfer survives regardless.

**Architecture fix (important, prompted by "we regenerate the teacher every time?"):** decoupled
generation from filtering. `gen_teacher_outputs.py` samples + **stores every raw critique once**
(paid); `build_corpus_offline.py` applies gates offline (free, repeatable, toggleable). All the
leak analysis above was done offline with zero regeneration.

---

## 5. Foreign-model viability (Qwen3.6-35B-A3B, Nemotron-3-Super-120B) on Tinker

Available on Tinker: `Qwen/Qwen3.6-35B-A3B` ($1.335/M sample), `nvidia/NVIDIA-Nemotron-3-Super-
120B-A12B-BF16` ($1.44/M). Gemma NOT hosted; Llama retired.

### The 5 harness fixes that made foreign generation work (the models were capable all along)
1. Render prompts with `apply_chat_template(tokenize=False)` then `encode` (`tokenize=True`
   returns junk Encoding objects).
2. Grade only the final answer (after last `</think>`) — reasoning models write DRAFT code inside
   `<think>` and `_extract_code` grabs the FIRST ```python block.
3. **Two-phase forcing** (`QwenTwoPhaseTokenCompleter`, works for any `<think>` ChatML model) —
   single-phase lets reasoning models burn the whole budget thinking and never emit a program
   (Qwen used all 26k on thinking). Force `</think>` then sample the answer, like gpt-oss.
4. Grade concurrently (thread pool) — sequential grading serializes 1100s-timeouts into hours.
5. **eval_timeout=1100 not 900** — flipped Nemotron 17%→61% valid.

### Fair yields (two-phase, 1100s, 18 samples/model)
- **Nemotron 11/18 valid (61%)**, best 0.381001, 2 beat seed — *exceeds* base gpt-oss's 44%.
- **Qwen 6/18 (33%)**, best 0.381111.
Generated **26 foreign betters** (Nemotron 12, Qwen 14, all c5 < 0.3835), origin-tagged.

### Technique diversity (% of programs using; foreign vs gpt-oss pool of 1480)
| technique | gpt-oss | Nemotron | Qwen |
|---|---|---|---|
| annealing | 68% | 58% | 29% |
| **L-BFGS-B** | **2%** | 8% | **43%** |
| **smooth-max (log-sum-exp)** | **17%** | 42% | **71%** |
| SLSQP | 20% | 0% | 0% |
| penalty / waterfill | ~19% | 33% | ~25% |

gpt-oss's converged pool is **annealing-dominant**. Qwen swaps to **smooth-max + L-BFGS gradient**
optimization; Nemotron stays annealing-based but adds **constraint rigor** (penalty/projection,
analytical gradient). **Distinct per-model signatures — do not blur them.** Both foreign models
lean away from gpt-oss's annealing default.

### Score by technique (MEAN c5 among users; MAX is useless — every technique's best = 0.381001)
gpt-oss pool: **symmetry (0.38152)** and **smooth-max (0.38172)** associate with the best scores;
penalty (0.38224) and SLSQP (0.38219) with the worst. **Smooth-max is BOTH the foreign signature
AND high-scoring** — the strongest a-priori case that there was useful, under-adopted info to transfer.
(Correlation, not causation.)

---

## 6. Fair cross-model distillation experiment (the main test) — NEGATIVE

**Fair design (all controls held):** a fixed **shared worse set** of 40 gpt-oss programs (c5
0.3814–0.3818), critiqued in *every* arm; each worse paired with its source's top-3 betters
(matched gaps: own/qwen 6.4e-4, nemo 5.7e-4); gpt-oss teacher everywhere; grounding-on/leak-off;
count-matched at 27 datums. **Only the better-source (technique) differs.**

| arm | better source | best-of-8 (seeds) | improve% | mean best-of-8 Δ | probe NLL |
|---|---|---|---|---|---|
| base (A0) | — | 6/18 | 17.2% | +2.6e-4 | 0.857 |
| **fair_own** (null) | gpt-oss champion (annealing) | **8/18** | 20.0% | +3.0e-4 | 0.806 |
| fair_qwen | Qwen (smooth-max/L-BFGS) | **4/18** | 15.1% | **−1.4e-3** | **0.785** |
| fair_nemo | Nemotron (constraint-rigor) | 7/18 | 17.6% | +2.5e-4 | 0.859 |

**Hypothesis NOT supported.** The own-pool arm (gpt-oss's own technique) was best; Nemotron ≈
base; **Qwen actively HURT** — its best-of-K rollouts came out *worse* than the seeds (negative Δ).

**Mechanism — parrot, not assimilate (crisp here):** fair_qwen had the **lowest probe NLL (0.785,
most internalized)** yet the **worst eval**. The 20B learned to *reproduce* the smooth-max/L-BFGS
critique disposition without the *competence* to execute it, so it wrote worse code than its own
annealing. Distilling a technique the model can't execute degraded it. Nemotron (constraint-rigor,
closer to gpt-oss's repertoire) didn't hurt — consistent with "assimilable technique is safe,
un-executable technique backfires."

*Caveats:* N small; improve-rate diffs within noise. The two robust signals are Qwen's **negative**
best-of-K Δ and the **NLL/perf inversion** — both point the same way.

---

## 7. The gpt-oss + nemotron ensemble (shared pool = the "seeding" approach) — ~TIED

Prior work (worktree `SkyRLTpu-ensemble`, branch `agent/ttd-ensemble`, Nemotron-3-**Nano**-30B,
15 steps, shared PUCT pool). Trajectory: nemotron *held the frontier* at steps 4–5; both models
**cross-pollinated (cross_frac 0.5) and CONVERGED** to the same 0.381013 family by step 15.

**Final best c5 = 0.381013 ≈ solo ctrl15 (0.381001) — tied, marginally worse.** No frontier gain.
The OOD exploration was real and free early (nemotron contributed states), but the shared pool let
the models **copy each other**, annealing the diversity away — the same exploration-collapse
pathology as a single model, just with two sources feeding one converging pool.

---

## 7b. In-context foreign guidance — does gpt-oss improve better *with foreign examples in context*? NO

Cleanest test yet (avoids parrot/competence at training time): 16 gpt-oss base programs
(c5 0.3814–0.3818); gpt-oss generates **5 improvement attempts each** with better programs
**in context** (foreign Qwen+Nemo / own gpt-oss champions / none), best-of-5, keep if it beats
the base. gpt-oss writes its OWN executable program (foreign = inspiration only → no cross-
tokenizer or training-time issue), outcome-filtered on measured c5. Same base set across arms.

| arm (in-context betters) | improved (best-of-5) | mean best-of-5 Δc5 |
|---|---|---|
| vanilla (none) | 1/16 | −7.3e-4 |
| **own** (gpt-oss champions) | **5/16** | **+6.3e-5** |
| foreign (Qwen + Nemo) | **0/16** | −8.4e-5 |

**Own-context wins; foreign-context is the worst arm** (0/16, never crossed the threshold). In-
context guidance *works* (own: 5/16, +Δ — echoes the early 0.381182 in-context result) but only
with examples the model can *execute*. **THIRD independent confirmation** (distill + ensemble +
in-context all agree): foreign technique does not help through any channel. (Regret: the main run
saved only c5s — the 240 rollouts' think+programs were lost; fixed, see memory
`cache-raw-model-generations`.)

## 7c. Comprehension vs execution — it's EXECUTION (the mechanism, verified)

Re-ran foreign-context bases SAVING the full think+program; tagged techniques in the *plan*
(think) vs the *code* (n=6 attempts, bases 0–1):

| technique | gpt-oss default | foreign-ctx **PLAN** | foreign-ctx **CODE** |
|---|---|---|---|
| annealing | 68% | 66% | 33% |
| **smooth-max** | 17% | **100%** | 66% |
| **L-BFGS** | 2% | **83%** | **0%** |
| projection | 19% | 83% | 33% |

**Comprehension is excellent** — the plan adopts the foreign techniques (smooth-max 100%, L-BFGS
83%, projection 83%): a huge shift from gpt-oss's defaults, so it *understood* the examples.
**Execution fails** — the code drops them, most starkly **L-BFGS: 83% of plans, 0% of programs**
(intends it every time, never writes it). Of 6 attempts (all planning foreign techniques):
**2 valid, 0 improved**; failures were a syntax error, a timeout (non-termination), a missing
`run()`, a NameError, and the 2 valid ones were no-gain or c5=0.4327 (catastrophic). **The 20B
knows the technique to reach for and cannot write it. The barrier is skill, not knowledge.**

## 7d. Why gpt-oss-120b clears the 20B frontier — execution competence on smooth-max (the capstone)

The 120B (0.380888) is the only source *above* the 20B's frontier. Same-axis analysis of its run:

**Technique distribution (20B ctrl15 vs 120B full):**

| technique | 20B | 120B |
|---|---|---|
| annealing | 67% | 61% |
| **smooth-max** | **16%** | **64%** |
| **projection** | 18% | 43% |
| L-BFGS | 1% | 2% |

The one huge gap: **smooth-max 16%→64%** (4×) — the *exact* technique the 20B plans-at-100%
but codes-at-0%-successfully. L-BFGS is low in *both* (so the earlier L-BFGS focus was a red
herring; smooth-max is the prize).

**Family concentration:** the 120B *also* collapsed to a narrow family (top-20 = **2/20** distinct
families, best-chain depth 13) — essentially the same monoculture as the 20B (1/20, depth 14).
So the 120B did **NOT** out-explore. Its advantage is not width.

**Champions:** 20B (0.381001) = diff-evo + projection (*no* smooth-max); 120B (0.380888) =
anneal + SLSQP + **smooth-max**.

**Verdict — validity vs idea axis? Neither: it's execution competence on one high-value technique.**
Not novel ideas (both share the idea space; smooth-max is in the 20B's pool at 16% and its plans
at 100%). Not more-valid-code-in-general (the 120B is also a monoculture polisher). **The 20B–120B
frontier gap IS the ability to *implement* smooth-max** — high-scoring, foreign-favored, and the
one the 20B comprehends but botches. This closes the loop on every negative result: distilling /
seeding / in-context-showing smooth-max all fail because the 20B *cannot execute it however you
hand it over*; only a model that already can (the 120B) moves the frontier. The lever is teaching
the **skill** of implementing smooth-max, or using a model that has it — not conveying the **idea**.

## 8. Synthesis — why nothing beat solo gpt-oss, and the reframe

Four mechanisms tried to inject foreign technique, none beat solo:
- **Contrastive distillation** (critiques, §6) → parrot / neutral-to-harmful.
- **Shared-pool ensemble** (seeding, §7) → converges, ~tied.
- **In-context guidance** (§7b) → worst arm (0/16).
- **On-policy RL alone** → collapses to monoculture (documented everywhere).

Three intertwined root causes (the third, §7c–7d, is the deepest):

1. **The foreign source isn't *better at the task*** (best ~0.381 vs gpt-oss 0.38091). Its
   "betters" were only *locally* better (vs the coarse worse we paired them against), never
   globally. There's no above-frontier information to inject as a *proven win*. LUFFY-style
   off-policy guidance works because its source (R1) is *stronger*; ours isn't.

2. **Every mechanism collapsed the diversity instead of preserving it.** The value of a second
   model isn't being better — it's **out-of-distribution exploration for free** (the thing single
   models can't manufacture; every solo run collapses explore). Distillation *absorbs* the other
   → convergence. Shared-pool *copies* → convergence. On-policy RL is a *closed loop over self-
   generated info* — the model only ever gets reinforced by its own successes, so it can only get
   better at what it already does. That closed loop is the deepest reason single models collapse.

**The reframe (user's, and it's right):** the goal is a **diversity-preserving** way to reinforce
good/promising information *regardless of origin* — an evolving/population view, not a single-best
view. Reinforce by *outcome and promise*, not by *who generated it*.

---

## 9. The off-policy / population formulation (where we are now)

The user's proposal — run GRPO/ttt-discover on the whole pool, reinforce gpt-oss's own rollouts
on-policy *and* foreign rollouts off-policy weighted by advantage — is a named, published method.

### Literature
- **LUFFY: "Learning to Reason under Off-Policy Guidance"** (Yan et al., NeurIPS 2025,
  arXiv 2504.14945) — the direct match. **Mixed-Policy GRPO**: combine on-policy rollouts with
  off-policy (external-model) traces *in the same group* during advantage estimation; group-
  relative advantage over the union naturally up-weights high-reward off-policy traces;
  **regularized importance sampling ("policy shaping")** amplifies low-prob-but-crucial actions
  and prevents entropy collapse / rigid imitation. Results +7.0 math, +6.2 OOD.
- **"Group-Relative REINFORCE Is Secretly Off-Policy"** (arXiv 2509.24203); **Revisiting GRPO
  on/off-policy** (2505.22257); **LLMs can learn to reason via off-policy RL** (2602.19362);
  RePO (replay-buffer GRPO). Classical: **AWAC / AWR** (advantage-weighted regression over any-
  source data; arXiv 2006.09359); RL-from-demonstrations (DQfD, POfD, DAPG); off-policy PG
  (IMPALA/V-trace, Retrace, ACER).

### The formulation (concrete)
Per seed s, group = {G self rollouts from π} ∪ {K foreign rollouts on the SAME seed}. Advantage
`A(τ)` computed group-relative over the **union**. Loss (one optim step):
```
L = Σ_{self}    IS-loss(τ, A(τ))          # on-policy GRPO (ratio π/π_sample ≈ 1)
  + Σ_{foreign} A(τ) · CE(τ)              # advantage-weighted imitation (AWR form)
```
- **Literal cross-tokenizer IS is ill-defined** (foreign wrote its program in its own tokenizer,
  no clean per-token μ). So foreign traces enter as **advantage-weighted cross-entropy** on the
  re-tokenized foreign code — this IS the off-policy reinforcement in practice, and it reuses
  ttt-discover's existing `extra_batches=[(datums,"cross_entropy")]` mechanism (the distillation path).
- **Why same-seed:** GRPO's advantage subtracts a per-*prompt* baseline (the group mean) to cancel
  problem difficulty. Cross-seed groups make the baseline invalid (a mediocre attempt at an easy
  seed outscores a great attempt at a hard seed). Same-seed = "on THIS problem, whose attempt was
  better." The ensemble already produces same-seed multi-model rollouts (shared deterministic PUCT
  frontier) — that infra is reusable.

### Open design fork — the reward/advantage (explore vs exploit)
- **ttt-discover max (entropic, raw c5)** = *exploit*: reinforces the group winner. But foreign is
  globally worse → rarely wins → barely reinforced, and when it loses it gets *negative* advantage
  → imitation pushes *away* from foreign. Pure-max ⇒ mixed-GRPO ≈ solo (reproduces the null).
- **Improvement reward (Δc5)** = *explore*: reinforces foreign for its technique's *progress* even
  when not globally best — but optimizes progress-not-frontier and has a headroom confound
  (coarse seeds are easier to improve a lot; Δ isn't seed-invariant).
- **Recommended:** keep the max-chasing entropic backbone (it's the discovery engine), add a
  **novelty/promise bonus** to the reward (`A = entropic-max over group of [r + β·novelty]`) so
  promising-different foreign directions get a nudge into positive advantage without abandoning
  frontier focus. This is the user's "reinforce potentially-promising," and it is exactly LUFFY's
  policy-shaping. NOT improvement-reward instead of max — max PLUS an exploration term.

**Status of the current confusion:** whether the reward should be improvement-based, max-based, or
max+novelty is the open question this document exists to resolve. The negative results (§6, §7)
say: don't just re-add foreign data under the *exploit* objective (it self-suppresses foreign or
parrots it); the value is *exploration*, so the objective must explicitly reinforce *promise*, and
the mechanism must *preserve* diversity (per-technique elites, regularized IS) rather than collapse it.

---

## 9b. Decoupled scientist/executor — Stage 1 (vanilla): decoupling DOUBLES VALIDITY, does NOT move the score

Premise (from §7c/§7d): execution is the binding constraint. Test the zero-training ablation —
gpt-oss is a **scientist** emitting an improvement PLAN (not code); a high-fidelity **Codex executor**
(gpt-5.x @ reasoning=high) implements it under a FAITHFUL contract (translate, don't optimize C5).
16 shared bases (c5 0.3814–0.3818), 3 plans/base (48), each plan implemented by each executor →
graded locally at the real budget_s=1000. Code: `gen_plans.py` (scientist, Tinker), `run_executor.py`
(Codex CLI), `regrade_from_disk.py` (authoritative — grades the cached `solution.py`), `analyze_decoupled.py`.

**Result (all @ 1000s, best-of-3 per base):**
| arm | valid | best c5 | bases-improved |
|-----|-------|---------|----------------|
| DIRECT (gpt-oss writes code) | 21/48 = **44%** | 0.381197 | **3/16** |
| exec gpt-5.4-mini | 14/18 = 78%* | 0.381446 | 1/11 |
| exec gpt-5.6-luna | 41/48 = **85%** | 0.381394 | 1/16 |
| exec gpt-5.6-terra | 42/48 = **88%** | **0.381165** | 2/16 |
(*mini only wrote 18/48 files — see harness note below; 78% is of what it produced.)

- **VALIDITY: decoupling works, decisively — 44% → 85–88% (+44 pts).** Execution WAS the validity
  bottleneck. terra one-shots correct code: 48/48 parse, 48/48 clip to [0,1], 45/48 project the
  Σh=n/2 constraint; programs are genuine (CMA-ES + multigrid 80→1024 + gradient refine + symmetry +
  binary search, `project_box_sum` re-applied before return). Running the program bought ~nothing for
  validity → justifies a **read-only no-run executor** (also removes the timeout class + gives a hard
  credit-assignment guarantee: executor physically cannot tune against C5).
- **SCORE: ~null.** Best anywhere is terra 0.381165, beating direct 0.381197 by **3.2e-5 (noise)**;
  luna/mini are WORSE than direct. **bases-improved is LOWER for every executor** (terra 2, luna 1)
  than direct's 3/16.
- **Mechanism — same signature as distillation (§6): robust on validity, null on improvement.** Two
  causes: (a) plan/ideation quality is only ~as good as gpt-oss's own ideas (no above-frontier info);
  (b) the FAITHFUL contract suppresses the aggressive C5-tuning gpt-oss does when it writes to win —
  high validity, low ambition. **The improvement bottleneck is ideation, not execution fidelity.**
- **Executor ranking: terra > luna > mini** on BOTH validity and best c5 → terra is the executor.

**Harness lessons (baked into the code):** (1) `run_executor` discarded `solution.py` on codex
TIMEOUT — the 300s per-call cap fired because Codex ran the program's full 1000s optimizer to "test"
it; fixed to salvage-on-timeout + tell Codex to smoke-test with `run(budget_s=15)`. All workdirs +
programs are cached, so `regrade_from_disk.py` recovers everything (memory: cache-raw-model-generations).
(2) Grading is THE wall-clock cost — the harness calls the entrypoint as `func()` with NO args
(`sandbox_reward_evaluator.py:205`), so every program self-limits to budget_s=1000. For controlled
ablations, grade at a smaller UNIFORM budget (patch the `budget_s=1000` default; all programs honor
it via `cutoff=0.9*budget_s` deadlines) → ~10× faster, ranking preserved. (3) Fairshare: 50-core jobs
pend; split into per-tag 24-core jobs that backfill in parallel.

**Stage 2 (planned): plan cross-pollination.** Does the foreign-technique transfer that WEIGHT-distill
failed at (§6, parrot-not-assimilate) work when injected at the PLAN level and executed by terra?
2×2 {direct, decoupled-terra} × {vanilla, foreign-refs-in-context}; foreign plans already generated
(`corpora/plans_foreign.json`). Keep the faithful contract (so any gain is attributable to the plan's
technique). See memory [[ttd-decoupled-executor]].

## 10. Infrastructure map

- Code: `tpu/distill_ablation/` — `common.py`, `extract_heldout.py`, `build_corpus.py`,
  `build_corpus_offline.py`, `gen_teacher_outputs.py` (stores raw, supports `--worse-set`+`--source`
  for fair pairing), `select_worse_set.py`, `gen_foreign_betters.py`, `probe_foreign.py`,
  `finetune_and_eval.py`, `analyze.py`; jobs in `jobs/`; corpora + results in `corpora/` and
  `runs/distill_ablation/`.
- Runs on neuronic via `run_arm.sbatch` (sets `TTD_EVAL_BACKEND=local`, grabs a compute node —
  NEVER the login node). Full 64-core node fragmented → resubmit at `--cpus-per-task=32`.
- Discover lib changes captured in `tpu/discover-fixes.patch`; submodule remote is the authors'
  upstream (no push access).
- Memory: `ttd-cross-model-distillation`, `tinker-prod-oov-regression`, `ttd-ensemble-discovery`,
  `ttd-context-distillation`, `ttd-gptoss20b-neuronic-run`.
