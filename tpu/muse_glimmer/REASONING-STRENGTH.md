# Muse-Glimmer-30B: `high` vs `xhigh` reasoning strength, on recorded Qwen3.5 PUCT states

**Verdict up front: `xhigh` does not earn its tokens on solution quality, and it
does earn them on solution *validity*.** Across all three problems and both
regimes — 480 rollouts per arm, all 960 generated and all 960 graded — the
improvement rate is **48.33 % (`high`) vs 49.79 % (`xhigh`), z = −0.45**, a
difference of −1.46 points with a 95 % CI of **[−7.8, +4.9] points**. That is a
null with teeth: effects larger than ~8 points are ruled out. The one thing
`xhigh` reliably does is produce a *runnable, verified* program more often —
**75.4 % vs 82.9 % pooled, z = −2.86**, driven almost entirely by JSSP's C++17
task (**78.9 % → 95.3 %, z = −3.92**). It costs **1.3–3.5 % more total tokens**
for that.

The cost side of `xhigh` is still only partly measurable, and the reason is
structural: **`xhigh` hit the phase-1 cap on 480 of 480 rollouts. `high` hit it
on 473 of 480.** Neither arm was ever allowed to finish reasoning on a parent
state, so the reasoning-length distributions are censored, not measured — but
the censoring itself is now asymmetric and that asymmetry is a result (§5).

Status of the underlying port is in `E2E.md` / `SPEC.md`. This file covers only
the reasoning-strength experiment. Anything not measured here is called out as
unmeasured; nothing below is extrapolated.

**Output path (fixed, and this is where everything lives):**
`/n/fs/vision-mix/sk7524/muse-rs2/` — `manifests/`, `gen/` (every raw
generation), `grades/`, `report_{erdos,jssp,ac1}.json`, `tables_all.md`,
`state_ids.md`, `pools/`, `bad/` (the voided first JSSP pass, §8.2).
Driver logs: `runs/muse_glimmer/rs2-3692376.log`, `rs2-3692972.log`,
`rs2.progress`, `rs2.slice_seconds`.

---

## 1. The diagnosis this round exists to fix, and the fix

Round 1 died seven times with the same line:

```
16/1060 ok=16 err=0 tok=174025 172 tok/s 1012s elapsed
```

172 tok/s is four concurrent streams. At ~11 400 tokens an item, 960 items at
that rate is **~19 hours**; no spot slice could survive it, and the preemptions
that got the blame were a symptom. Worse, nobody could *see* it: the only
progress line printed every 16 **completions**, and nothing completes in under
four minutes, so the first number arrived 1012 s into a paid slice.

Three things changed, and all three are verified rather than asserted.

| | round 1 | this round |
|---|---|---|
| client concurrency (proven on CPU, before any slice) | unknown | `CONC-SELFTEST requested=64 peak_inflight=64 wall=0.79s (serial would be 32s) -> PASS` |
| **achieved concurrency, client gauge** | 4 (inferred) | **96 / 96, `peak=96`, every heartbeat** |
| **achieved concurrency, engine scheduler** | not looked at | **`Running: 96 reqs, Waiting: 0 reqs`** |
| **measured throughput** | **172 tok/s** | **1227 tok/s probe · 2188 tok/s sustained** |
| KV page size at `max_model_len` 16384 | 16 (tpu-inference default) | **256** (`--block-size 256`) |
| time to first throughput number | 1012 s | **~200 s** (Gate 1, before generation) |

**Gate 1** is the mechanism. Before any rollout is generated the driver fires
`--probe-conc 96` real prompts with `ignore_eos`, twice — the first burst is
discarded because with `SKIP_JAX_PRECOMPILE=1` the batch shape compiles inside
it — and refuses to proceed below `MIN_TPS=600`. The two bursts differ by
**2.5×** (491.3 → 1227.2 tok/s), which is almost certainly the same artefact
behind `E2E.md` §7.2's anomalous "217 tok/s at 64 concurrent": a single
unwarmed measurement paying XLA compilation. Gate 1 reproduced to within
0.1 tok/s on **three independent slices** (1227.3 / 1227.2 / 1227.2).

**Page size was the other half.** `tpu_inference/layers/vllm/backends/flash_attn.py`
hardcodes `page_size = 16` for any `max_model_len > 8192` — "a temporary fix
for vmem OOM" whose own comment warns that small pages spill scalar registers
and perform badly. Passing `--block-size 256` sets `user_specified_block_size`
and bypasses it; the engine booted, allocated the same **3 025 664** KV tokens
(184.67× concurrency at 16384) and served. **Caveat, stated plainly: block 16
was never measured this round.** Gate 1 short-circuits on the first
configuration that clears `MIN_TPS`, deliberately, to save slice time. Round 1's
172 tok/s was at block 16 *and* concurrency 4, so the 12.7× improvement is
**not** attributed between the two changes here.

Concurrency 96 against a KV pool that holds 184 sequences at 16384 leaves ~1.9×
of headroom; `--max-num-seqs` was 128.

---

## 2. Design

3 problems × (1 start state + 4 parent states) × 32 completions × 2 arms =
**960 rollouts**. Every cell is exactly full: 128 parent + 32 start per arm per
problem, **0 generation errors across all 960**.

**Prompts come from the env machinery, not from hand.** `rs_build_manifest.py`
builds the real `DatasetConfig`, the real env class and the real renderer by
registry name, then calls the real `ProblemEnv.initial_observation()` →
`renderer.build_generation_prompt([{"role": "user", "content": env.get_question()}])`.
Token ids leave as `ModelInput.to_ints()` and are sent to vLLM as an explicit
`list[int]`, so nothing round-trips through text. The builder asserts, per
state, that

```
prompt_high.replace("strength: high", "strength: xhigh") == prompt_xhigh
```

`ARM-DIFF-IS-ONE-LINE OK` on all three manifests — **the arms are byte-identical
apart from that one line.**

**Early-step parents, deliberately.** Qwen at step 14 has had 14 RL steps and
Muse-Glimmer has had zero, so a late-step comparison is rigged. Parents come
from the earliest pool step containing states with non-empty code: timestep 1
for erdos (`puct_sampler_step_000002.json`), timestep 0 for JSSP and ac1
(`..._000001.json`). Two parents from the `ttd` pool and two from `grpo` per
problem, so the sample is not an artefact of one advantage estimator. Selection
is `random.Random(20260813).sample` over id-sorted candidates.

**Framing, and it matters.**

* `high` vs `xhigh` is **controlled**: same model, same weights, same states,
  same prompts, one differing line.
* Muse-Glimmer vs Qwen3.5 is **observational**: different model, different
  training history, different sampler, and the pool kept only the top-2
  children per parent. Reported separately (§7), never as if it were the first.

### 2.1 Budget and grading

| knob | value | note |
|---|---|---|
| `max_model_len` | **16384** | KV holds 184 concurrent here; round 1 items averaged 11 376 output tokens |
| KV `--block-size` | **256** | §1 |
| phase-1 cap (prompt + reasoning) | **13312** | leaves 3006 for the phase-2 answer; round 1's longest answer was 2604 |
| `--max-num-seqs` / client concurrency | 128 / **96** | |
| temperature | 1.0 | sweep default |
| **`EVAL_TIMEOUT`** | **1100** | the value the Qwen3.5 sweep used |
| grading shards | **64** | round 1 used 8–16, which is why grading crawled |
| serving | 1 × spot v5p-8, TP=4, vLLM 0.23.0 + tpu-inference, `MODEL_IMPL_TYPE=flax_nnx` | `E2E.md` §4 |

Phase 2 mirrors `QwenTwoPhaseTokenCompleter`: when phase 1 exhausts its budget
still inside `to=self`, a masked close + answer cue is injected and the answer
is sampled from the remaining context. Without it a budget-exhausted rollout is
just an unparseable format error, which reads as "xhigh is worse" when the
truth is "the budget was too small". **The fence language of that cue now comes
from the environment** — see §8.2, where getting it wrong voided 320 rollouts.

### 2.2 The exact states sampled

Reproduce with `rs_build_manifest.py --seed 20260813 --current-date 2026-08-13
--n-parents 4 --n-seeds 1 --rollouts 32 --start-rollouts 32`.

#### erdos (`erdos_min_overlap`, c5, **lower better**, `maximize=False`, timestep 1)
| kind | state id | pool | recorded value | prompt tok | Qwen children |
|---|---|---|---|---|---|
| start | `f190f7ef-09a8-4b00-8dc0-93c192f59048` | ttd | −0.508264 | 631 | 2 |
| parent | `24ae7a5c-faa5-4081-afeb-294bfe2cc589` | grpo | −0.437948 | 2219 | 0 |
| parent | `315bb14c-33b8-4ff6-8eb5-d882b42a5129` | grpo | −0.396557 | 2917 | 2 |
| parent | `30f426a0-4534-4079-9b2f-f936f2f08d7b` | ttd | −0.444164 | 2730 | 0 |
| parent | `4b4928c6-e89f-48f5-913a-a3f8bdabf8b5` | ttd | −0.382679 | 3846 | 2 |

#### JSSP (`frontier_algo` problem 46, **higher better**, timestep 0, **C++17**)
| kind | state id | pool | recorded value | prompt tok | Qwen children |
|---|---|---|---|---|---|
| start | `e9a9af72-b6ca-47fb-b935-ccfcd5176b23` | ttd | 0.000000 | 1641 | 2 |
| parent | `246f775e-a87c-4610-9152-c24325ccc271` | grpo | 0.000000 | 2732 | 2 |
| parent | `65097fe4-d29a-4206-84da-4792cb37d2a1` | grpo | 0.000000 | 3264 | 2 |
| parent | `47d4cfd4-d8ae-47e3-b367-26d359442f36` | ttd | 0.000000 | 3761 | 2 |
| parent | `5a57efd4-9012-423e-9c6b-f116326913f8` | ttd | 0.000000 | 4342 | 2 |

#### ac1 (`ac_inequalities`, **higher better**, timestep 0)
| kind | state id | pool | recorded value | prompt tok | Qwen children |
|---|---|---|---|---|---|
| start | `c8731cae-899e-469d-be48-7fd5ebb1220f` | ttd | −2.000000 | 1927 | 2 |
| parent | `1f038424-f27b-4354-ad22-063b5fdaa678` | grpo | −2.004360 | 3409 | 0 |
| parent | `31df977e-c7f9-46cd-95dc-54e7309180e3` | grpo | −1.570848 | 3793 | 2 |
| parent | `4adfcdaf-28ce-457a-9390-0485bb9e812f` | ttd | −1.974088 | 3779 | 2 |
| parent | `7795f0f2-9a46-47df-bfcf-4e303e3da927` | ttd | −1.999987 | 3994 | 2 |

**Read the "recorded value" column before reading any improvement rate.** All
four JSSP parents are exactly 0.0 and so is the JSSP start state, so "strictly
beats the parent" degenerates to "scored anything at all". All four ac1 parents
sit at ≈ −2.0, and every ac1 rollout that ran at all beat its parent (§4.3), so
ac1's improvement rate is numerically *identical* to its validity rate. **Only
erdos supplies a genuine graded improvement signal** — it has a real tie rate
(1.6–3.1 %) and a real regression rate (22.7–23.4 %). This is a property of the
early pools, not of either model, and it bounds what the other two problems can
say.

---

## 3. Gates

**Gate 0 (i) — renderer unit tests: PASSED, 57/57** (unchanged from round 1;
`MuseGlimmerRenderer` is byte-exact against the checkpoint's own
`chat_template.jinja`).

**Gate 0 (ii) — live channel smoke: PASSED on all three slices.** Both arms
produce a `to=self` reasoning channel *and* a `to=user` answer channel and stop
cleanly, with `stop_token_ids=[200001, 200008]`:

```
--- high   gen_tokens  79  finish stop   to=self … <|eom|><|start|>assistant to=user<|message|>2<|eot|>
--- xhigh  gen_tokens 125  finish stop   to=self … <|eom|><|start|>assistant to=user<|message|>2<|eot|>
```

Identical token counts to round 1, on three further slices. **79 vs 125 is
1.58×**, and it is the only *uncensored* look at the knob anywhere in this study
(§5).

**Gate 1 — throughput: PASSED**, `GATE1-BEST block-size=256 throughput=1227.2 tok/s (min 600)`.

---

## 4. Results

Full tables in `/n/fs/vision-mix/sk7524/muse-rs2/tables_all.md`; the headline
rows are here. Percent change is `(v_child − v_parent) / |v_parent| × 100` on
the pool's signed value, so positive is always better.

### 4.1 The controlled comparison, all six cells

| problem | regime | metric | `high` | `xhigh` | z |
|---|---|---|---|---|---|
| **erdos** | parent | **improvement rate** | **46.9 %** | **46.9 %** | **+0.000** |
| erdos | parent | validity | 71.1 % | 73.4 % | −0.419 |
| **erdos** | start | **improvement rate** | **75.0 %** | **78.1 %** | −0.295 |
| erdos | start | validity | 75.0 % | 84.4 % | −0.932 |
| **JSSP** | parent | **improvement rate** | **17.2 %** | **18.0 %** | −0.164 |
| **JSSP** | parent | **validity** | **78.9 %** | **95.3 %** | **−3.917** |
| **JSSP** | start | **improvement rate** | **15.6 %** | **15.6 %** | **+0.000** |
| JSSP | start | validity | 78.1 % | 90.6 % | −1.377 |
| **ac1** | parent | **improvement rate** | **77.3 %** | **81.2 %** | −0.771 |
| ac1 | parent | validity | 77.3 % | 81.2 % | −0.771 |
| **ac1** | start | **improvement rate** | **68.8 %** | **68.8 %** | **+0.000** |
| ac1 | start | validity | 68.8 % | 68.8 % | +0.000 |

**Pooled over all three problems, 480 rollouts per arm:**

| metric | `high` | `xhigh` | z | difference (95 % CI) |
|---|---|---|---|---|
| **improvement rate** | 232/480 = **48.33 %** | 239/480 = **49.79 %** | **−0.452** | **−1.46 pp [−7.78, +4.87]** |
| **validity rate** | 362/480 = **75.42 %** | 398/480 = **82.92 %** | **−2.861** | +7.50 pp to `xhigh` |

Three of the six improvement-rate comparisons are *exactly* zero and none
exceeds |z| = 0.8. The CI is the useful part: at this sample size an `xhigh`
advantage bigger than about 5 points would have shown up, and it did not.

The validity result is the opposite: pooled z = −2.86 (p ≈ 0.004), and it is
not smeared across cells — it is JSSP's parent cell at z = −3.92 (p ≈ 9 × 10⁻⁵),
which survives a Bonferroni correction over all six validity comparisons
(α = 0.0083) with room to spare. JSSP is the one task with a *formal* output
contract: a single C++17 translation unit that must compile with
`g++ -O2 -std=gnu++17` and then produce a feasible schedule. **`xhigh` fails
that contract a third as often** (4.7 % vs 21.1 %).

### 4.2 erdos — the only genuine graded improvement signal

| metric | `high` (n=128) | `xhigh` (n=128) |
|---|---|---|
| format rate (code block present) | 100.0 % | 100.0 % |
| reasoning `to=self` / answer `to=user` present | 100 % / 100 % | 100 % / 100 % |
| validity rate | 71.1 % | 73.4 % |
| **improvement rate (all rollouts)** | **46.9 %** | **46.9 %** |
| improvement rate (of valid rollouts) | 65.9 % | 63.8 % |
| tie rate | 1.6 % | 3.1 % |
| regression rate | 22.7 % | 23.4 % |
| mean % change (valid) | +2.90 | +3.14 |
| **median % change (valid)** | **+0.37** | **+0.80** |
| mean % change over ALL rollouts (invalid = 0) | +2.06 | +2.31 |
| best single % change | +14.18 | +14.17 |
| grading-timeout rate | 0.0 % | 1.6 % |

Start state (from scratch, n=32/arm): improvement 75.0 % vs 78.1 %, validity
75.0 % vs 84.4 %, median % change +21.82 vs +23.54, and **100 % vs 92.6 %** of
valid rollouts beat the start baseline.

**Start-state vs improvement is the sharpest contrast in the data, and it is
not about the arms.** Expanding a parent that Qwen had already optimised
improves it 46.9 % of the time by a median of +0.4 %; writing from scratch
against the raw start state improves it ~76 % of the time by a median of
+22 %. That is headroom, not skill — the start state sits at c5 0.508 while the
parents sit at 0.383–0.444 — and it is the same shape the sweep's
"improvement-ability decays with headroom" result predicts. Both arms show it
identically.

**Best absolute construction found: c5 = 0.381037663**
(`erdos|start|f190f7ef|high|r28`). For scale, the sweep's Qwen3.5 champion was
0.380953 and the authors' record 0.380875, so an untrained Muse-Glimmer gets
into the 0.3810 neighbourhood in a single expansion but does not approach the
record. The next four best are 0.381135 (`xhigh`), 0.381171, 0.381177 and
0.381194 — the arms interleave.

### 4.3 JSSP and ac1 — read with §2.2's warning

**JSSP.** Every parent value is 0.0, so improvement = "scored above zero".
`high` 17.2 % vs `xhigh` 18.0 % (parent), 15.6 % vs 15.6 % (start). The tie rate
is the interesting column — 61.7 % vs 77.3 % — meaning `xhigh` far more often
produced a program that *compiled and ran and scored exactly the parent's 0.0*.
Mean absolute delta 0.0070 vs 0.0112. Grading timeouts: 0 % in both arms; JSSP
grades in ~7 s (p90), so no timeout confound exists here at all.

**ac1.** `improvement_rate_all` equals `valid_rate` exactly in both arms and
`improvement_rate_valid` is 100.0 % — every ac1 program that ran beat its ≈ −2.0
parent. So ac1 measures validity, not improvement. Its percent-change column is
degenerate for a second reason: the mean is **+231.6 % (`high`) vs +14902.5 %
(`xhigh`)**, an artefact of a single rollout at +1 527 937 % against a
near-zero denominator. **The median is the honest statistic and it is
+199.99 % vs +200.00 % — a difference in the fifth significant figure.** The
mean is reported only so the reader can see why it must be ignored.

---

## 5. The token-cost curve, and truncation

**This is where the study is still partly blind, and the blindness is now
one-sided.**

| problem | regime | arm | natural finish (`stop`) | truncation | reasoning med | total med | `xhigh:high` total |
|---|---|---|---|---|---|---|---|
| erdos | parent | high | **4 / 128 (3.1 %)** | 96.9 % | 10 394 | 11 798 | — |
| erdos | parent | xhigh | **0 / 128 (0.0 %)** | **100.0 %** | 10 486 | 12 238 | **1.037** |
| erdos | start | high | **2 / 32 (6.2 %)** | 93.8 % | 12 680 | 13 826 | — |
| erdos | start | xhigh | **0 / 32 (0.0 %)** | **100.0 %** | 12 679 | 14 099 | **1.020** |
| JSSP | parent | high | **1 / 128 (0.8 %)** | 99.2 % | 9 550 | 11 330 | — |
| JSSP | parent | xhigh | **0 / 128 (0.0 %)** | **100.0 %** | 9 797 | 11 724 | **1.035** |
| JSSP | start | high | 0 / 32 | 100.0 % | 11 670 | 13 580 | — |
| JSSP | start | xhigh | 0 / 32 | 100.0 % | 11 669 | 13 694 | **1.008** |
| ac1 | parent | high | 0 / 128 | 100.0 % | 9 518 | 10 792 | — |
| ac1 | parent | xhigh | 0 / 128 | 100.0 % | 9 524 | 10 943 | **1.014** |
| ac1 | start | high | 0 / 32 | 100.0 % | 11 384 | 12 790 | — |
| ac1 | start | xhigh | 0 / 32 | 100.0 % | 11 383 | 12 898 | **1.008** |

**Reasoning-token cost ratio `xhigh`:`high`, on the mean:** erdos **1.0085×**,
JSSP **1.0130×**, ac1 **1.0006×**. **Total-token ratio:** erdos **1.033×**,
JSSP **1.035×**, ac1 **1.013×**.

Two things follow, and the second is new relative to round 1.

1. **The reasoning distributions are censored, not measured.** The medians agree
   to within ~1 % because both are pinned to `phase1_max_tokens − prompt_len`.
   *No* statement of the form "xhigh reasons N× longer on a hard problem" can be
   made from this data. The ~1 % ratios above are the residue of a few short
   `high` rollouts, not a measurement of natural reasoning length.
2. **The censoring is asymmetric, and that is a measurement.** `xhigh` used its
   entire phase-1 budget on **480 of 480** rollouts — `phase1_headroom` was
   exactly 0, min = median = max, in every one of the six cells. `high` finished
   naturally **7 times out of 480** and left up to 1195 tokens of headroom. So
   the knob is doing *something* to the reasoning length; the cap simply sits
   below the point where the two distributions separate, for both arms.

**The only uncensored evidence remains the Gate 0 smoke**: on "How much is 1+1?
Reply with just the number", `high` spends **79** tokens and `xhigh` **125** —
**1.58×** for the same one-character answer, reproduced identically on six
slices across two rounds. That is one trivial prompt and says nothing about
hard problems, but it does show the knob is not inert.

**The named risk did not fire.** The brief's concern was that `xhigh` runs past
the cap, dies mid-reasoning, gets scored invalid, and reads as "no effect". It
did not happen: format rate is 100 % in eleven of twelve cells (98.4 % in one),
the phase-2 rescue fired on 82–100 % of rollouts, and `xhigh`'s validity is
*higher*, not lower. The two-phase completer absorbed the truncation exactly as
designed. The subtler version did occur — the cap flattened the very quantity
the arms were supposed to differ on.

### 5.1 Grading timeouts — checked, and not the confound

| problem | regime | `high` | `xhigh` |
|---|---|---|---|
| erdos | parent | 0.0 % | 1.6 % |
| erdos | start | 0.0 % | 0.0 % |
| JSSP | parent / start | 0.0 % | 0.0 % |
| ac1 | parent | 8.6 % | 9.4 % |
| ac1 | start | 12.5 % | 6.2 % |

ac1 is the only problem where grading timeouts are common, and they fall
*against* `high` in the start cell and marginally against `xhigh` in the parent
cell. There is no systematic asymmetry large enough to explain anything; in
particular it cannot explain the JSSP validity gap, where both arms time out
0 % of the time and p90 grading takes 6.5–7.3 s.

---

## 6. The numba correction — round 1's biggest number was an artefact

Round 1 reported that **67 % of Muse-Glimmer rollouts failed with
`No module named 'numba'`**, called it "a genuine behavioural difference"
against Qwen, and let it cut measured validity from 77.8 % to 28.0 %.

**`numba>=0.60` is a declared dependency of the discover package**
(`third_party/discover/pyproject.toml:54`). It is simply not installed in
`.venv-ttd-discover`, the venv round 1 graded in. So that 67 % was a property of
the grading environment, not of the model.

This round supplies numba from an isolated `--target` directory
(`/n/fs/vision-mix/sk7524/muse-rs2/pylibs`, on `PYTHONPATH`) rather than
installing into the shared venv, which live `rq2` jobs also use. Validity on
erdos parents goes **28.0 % → 71.1 %**, which is most of why this round has
usable statistical power at all.

The strict reading of the prompt — which lists "scipy, numpy, cvxpy[…], math"
and *not* numba — is preserved as a separate column rather than baked into the
environment, so both answers are available from one grading pass:

| problem | regime | programs importing numba, `high` | `xhigh` | validity STRICT, `high` | `xhigh` |
|---|---|---|---|---|---|
| erdos | parent | 75.0 % | 73.4 % | 14.1 % | 20.3 % |
| erdos | start | 84.4 % | 78.1 % | 12.5 % | 18.8 % |
| ac1 | parent | 15.6 % | 5.5 % | 67.2 % | 77.3 % |
| ac1 | start | 28.1 % | 3.1 % | 53.1 % | 68.8 % |
| JSSP | both | 0.0 % | 0.0 % | = permissive | = permissive |

Muse-Glimmer reaches for numba on **three quarters** of erdos programs. Under
the strict contract the arms stay null on improvement (7.8 % vs 7.8 % on erdos
parents) — so **the headline conclusion does not depend on which reading you
take**, which is the point of reporting both.

---

## 7. Muse-Glimmer vs the recorded Qwen3.5 rollouts — observational

Two structural facts limit this, and neither is a property of either model: the
PUCT pool keeps only the **top 2 children per parent**, so Qwen's recorded
children are elite-filtered; and `puct_n` counts rollouts over a state *and all
its descendants*, so **Qwen's improvement rate is not computable from the pool
at all** and none is quoted. The only honest shape is best-of-k per state.

| problem | source | states with data | mean best % change | median | states beating parent |
|---|---|---|---|---|---|
| erdos | Qwen3.5-27B (top-2 of unknown N, 2 RL steps) | 2 | +0.75 | +0.75 | 2 / 2 |
| erdos | Muse `high` (top-1 of ≤32) | 4 | **+7.63** | +8.04 | 4 / 4 |
| erdos | Muse `xhigh` (top-1 of ≤32) | 4 | **+7.82** | +8.41 | 4 / 4 |
| ac1 | Qwen3.5-27B | 3 | +5.05 | 0.00 | 2 / 3 |
| ac1 | Muse `high` | 4 | +998.1 | +998.5 | 4 / 4 |
| ac1 | Muse `xhigh` | 4 | +382230.5 | +387.3 | 4 / 4 |

**Do not read a model ranking off this.** Muse gets best-of-32 against Qwen's
best-of-2-out-of-unknown, only 2–4 states carry data on the Qwen side, and the
ac1 row is dominated by the same near-zero-denominator outlier as §4.3. JSSP has
no row because every value on both sides is 0.0. What the table supports is a
weak directional statement: **an untrained Muse-Glimmer expanding these states
is not obviously worse than a lightly-trained Qwen3.5 was**, and that is all.

---

## 8. Cost, incidents, and cleanup

### 8.1 Slice cost

**Cumulative live slice: 13 414 s = 3.73 h on 4 chips = 14.90 chip-hours**,
inside the 6 h (21 600 s) cap. Four QR lifecycles across two slurm jobs; the
cumulative ledger (`runs/muse_glimmer/rs2.slice_seconds`) is what enforces the
cap across retries, so a preemption cannot reset the budget.

| # | slurm | landed | slice alive | outcome |
|---|---|---|---|---|
| 1 | 3692376 | 22:21:46Z, 21m36s hunt | **3 477 s** | **preempted** at 288/320 erdos; all 288 banked |
| 2 | 3692376 | 23:36:22Z, 13m29s hunt | **5 184 s** | erdos 320/320 ✅, JSSP 320/320 (voided, §8.2), ac1 320/320 ✅ |
| 3 | 3692972 | 01:13:51Z, 5m56s hunt | **2 331 s** | **preempted** at 85/320 JSSP; all 85 banked |
| 4 | 3692972 | 02:11:38Z, 15m50s hunt | **2 422 s** | JSSP 320/320 ✅ |

Both preemptions were detected by the same mechanism and cost minutes, not
hours: after 600 s of consecutive ssh failures the driver queries the QR state,
saw `SUSPENDED` both times, recorded its slice seconds and exited 12; the
wrapper re-hunted and `rs_generate.py` resumed from the JSONL already on the
shared filesystem. **Resume is why 960/960 exist**: attempts 1 and 3 contributed
373 rollouts that were never regenerated. The brief's "unreachable host" rule is
implemented as the same check with a different verdict — QR `ACTIVE` but
unreachable exits 13, also retryable.

Grading: 64-shard arrays, `EVAL_TIMEOUT=1100`. The first submission ran only
6 shards concurrently despite 1774 idle CPUs — a 5-hour walltime request was
unbackfillable under fairshare. Dropping to `--time=01:00:00 --cpus-per-task=2`
took it to **62 concurrent shards immediately**. All 960 rollouts graded.

### 8.2 The JSSP C++ incident — 320 rollouts voided and regenerated

The phase-2 answer cue hardcoded ` ```python `, and `rs_grade.py` hardcoded
`codeblock_seps=["python"]`. **`frontier_algo` is a C++17 problem**
(`_get_code_languages() -> ["cpp","c++"]`, `_should_keep_code_separators() ->
False`). The first JSSP pass therefore cued the model into writing Python for a
C++ grader and scored **0/320**: 184 "No C++ program with main() found in
response", 124 "no code block".

Both sides now read the language from the manifest, which records the env's own
contract, and `rs_build_manifest.py` writes it. After the fix, **99.4 % of JSSP
answers carry a ```cpp fence and `#include`** (0 % Python). The voided pass is
kept verbatim at `/n/fs/vision-mix/sk7524/muse-rs2/bad/jssp.pythoncue.jsonl`.

Cost of the incident: one extra slurm job and ~4 753 s of the slice budget.
It is a harness bug of exactly the class this project keeps paying for — an
environment-specific contract silently defaulted to the common case.

### 8.3 Cleanup

**Verified zero `muse` queued-resources and zero `muse` TPU VMs in
us-east5-a, us-east5-b and us-east5-c** at the end of both slurm jobs, by three
independent paths (script cleanup loop, wrapper backstop, wrapper audit). Every
QR delete was issued `setsid nohup`, re-issued, and re-verified until the
describe call returned empty. One QR still reported `ACTIVE` for ~90 s after the
first delete, which is exactly why the loop re-issues rather than trusting one
call.

---

## 9. Verdict

**Does `xhigh` earn its tokens on this workload?**

* **On solution quality: no, and this is now a well-powered null.** Pooled
  improvement rate 48.33 % vs 49.79 % over 480 rollouts per arm, z = −0.45,
  difference −1.5 points with a 95 % CI of [−7.8, +4.9]. Six separate cells,
  three of them exactly tied, none above |z| = 0.8. Under the strict-contract
  reading it is null too. **A null result, reported as one.**
* **On solution validity: yes, modestly, and it is significant.** Pooled
  75.4 % → 82.9 %, z = −2.86; concentrated in JSSP's C++17 task at
  78.9 % → 95.3 %, z = −3.92, which survives Bonferroni over all six validity
  comparisons. If your task has a formal output contract that is easy to
  violate, `xhigh` violates it less.
* **On cost: 1.3–3.5 % more total tokens**, and the *reasoning* cost is still
  not measurable — both arms are censored by the phase-1 cap. What is now
  measured is that `xhigh` hit the cap **480 / 480** times while `high` finished
  naturally 7 times, so the knob does lengthen reasoning; the cap sits below
  where the distributions separate.

**Practical recommendation.** On erdos-like search problems where the answer is
a Python program and correctness is soft, `xhigh` buys nothing measurable and is
not worth switching to. On JSSP-like problems with a hard format contract, the
7.5-point validity gain is real and costs ~3.5 % more tokens — that trade is
worth taking, and it is a *format-compliance* effect, not a reasoning-quality
effect. Do not generalise it into "xhigh thinks better".

**The next experiment is still a budget experiment, not a strength experiment.**
`E2E.md` §6.1 has 32768 serving with ~92× concurrency of headroom. Running this
same design at `CTX=32768 / PHASE1=26000` is the only way to see the reasoning
distributions uncensored and to find out whether the 1.58× from the Gate 0 smoke
survives on a hard problem. Two secondary items: JSSP and ac1 need parent states
with **non-degenerate** recorded values before their improvement rates mean
anything, and block-size 16 vs 256 deserves a clean A/B now that the 12.7×
throughput win is confounded between page size and concurrency.

---

## Reproducing

```bash
# 1. renderer + rosetta_stone unit tests (CPU)
PYTHONPATH=third_party/discover third_party/discover/.venv/bin/python \
  -m pytest third_party/discover/tests/test_muse_glimmer_renderer.py -q

# 2. client-concurrency self-test + all three manifests (CPU, ~10 min)
sbatch tpu/muse_glimmer/rs_manifests.sbatch

# 3. ONE spot v5p-8: Gate 0 smoke, Gate 1 throughput probe, then generation.
#    Deletes its QR on every exit path; retries on preemption (exit 12) and on
#    a landed-but-unreachable host (exit 13); 6 h CUMULATIVE slice cap.
sbatch tpu/muse_glimmer/rs_tpu.sbatch                       # erdos jssp ac1
sbatch --export=ALL,PROBLEMS=jssp tpu/muse_glimmer/rs_tpu.sbatch   # one problem

# 4. grade on CPU, 64 shards, eval_timeout 1100, numba on PYTHONPATH.
#    Keep --time short: a 5 h request does not backfill.
for p in erdos jssp ac1; do
  sbatch --export=ALL,PROB=$p,NSH=64,EVAL_TIMEOUT=1100 \
         --array=0-63%64 tpu/muse_glimmer/rs_grade.sbatch
done

# 5. metrics + markdown tables
W=/n/fs/vision-mix/sk7524/muse-rs2
for p in erdos jssp ac1; do
  cat $W/grades/$p.shard*.jsonl > $W/grades/$p.all.jsonl
  python3 tpu/muse_glimmer/rs_analyze.py --manifest $W/manifests/$p.json \
    --gen $W/gen/$p.jsonl --grades $W/grades/$p.all.jsonl \
    --out $W/report_$p.json --phase1-max-tokens 13312
done
python3 tpu/muse_glimmer/rs_tables.py \
  erdos=$W/report_erdos.json jssp=$W/report_jssp.json ac1=$W/report_ac1.json
```
