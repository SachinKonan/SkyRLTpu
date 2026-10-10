# Feature-factory grader

`grade.py` scores one candidate feature family by how much it improves the current stack on held-out
cosmologies. It implements the evaluator from `../MOMO_HILLCLIMB_ENV.md` section 3 on top of the existing
consistency harness (`agents/consistency/`).

## Usage

```bash
cd research/fair_v2
python factory/grade.py factory/examples/dog.py --out <dir> --mode screen   # quick pre-filter
python factory/grade.py factory/examples/dog.py --out <dir> --mode full     # the real reward
sbatch factory/run_grade.sbatch factory/examples/dog.py:screen factory/examples/dog.py:full   # gpu-test A100
```

Outputs:
- `<dir>/result.json` holds everything.
- `<dir>/proposer_view.json` is the same without the hidden-shift field, keeping only `hidden_ok`. Show this
  file to a proposer (MOMO or an agent).

Reruns resume from where they stopped:
- Per-set features are cached in `feats_<mode>/`, and per-fold fits in `fold<k>_<mode>.pkl`.
- Editing the candidate (a new SHA-256 of its source) clears the cache.

## Candidate contract

```python
def features(maps, mask):   # maps: torch.float32 (B, 1424, 176) on the GPU, zero outside the footprint
                            # mask: torch.bool (1424, 176), same device
    return feats            # (B, d) torch tensor or numpy array; 1 <= d <= 4096
```

Hard failures (`ok: false`, `contract.fail` says why):

| Check | How it is tested |
|---|---|
| Per-map | 8 audit maps scored together vs one at a time vs shuffled. More than 1% of entries off by more than 1e-3 of the column scale fails. |
| Deterministic | Two identical calls. |
| Finite | On the contract batch and on every extracted set. |
| Shape | Output is `(B, d)` with 1 ≤ d ≤ 4096. |
| Runtime | 512 maps (including host-to-GPU copy), extrapolated to 10k maps, must be ≤ `--budget` (default 60 s). The final Codabench run has 1200 s for the whole stack. |
| Import | Import errors fail. |

The candidate runs in-process with normal file access. A MOMO deployment must sandbox it and must not let it
read the public test file or the shift generators (`data.py`). The grader itself never reads the test file.

## What is computed

**Inputs.** The frozen noisy maps in `fair-data/agents/diffusion/noisy`. These are the same realizations every
stack member was scored on, so per-map scores line up.

**LCO folds.** Folds k = 0, 1, 2 come from `holdout3.masks`. Each fold holds out 20 cosmologies.
- Normal (InD): held-out calib3 (2 views) plus held-out audit3 maps.
- OoD: the 19 shifts of the held-out audit3 maps.
- The candidate is refit on fit3 minus the fold's cosmologies.

**Candidate parts per fold:**
- `ridge`: a ridge θ-probe, features to 5 normalized parameters, with alpha chosen on the fold's calib3 maps
  (`probes.ridge_fit`).
- `prof` (full only): the conditional likelihood (`JLikelihood`, std variant: width 256, 3000 steps, jitter 0.1),
  profiled per map. It gives θ̂ as a vote and the standardized copula energy as a misfit member `X`
  (`holdout3`/`components` conventions).

**Stack.** The v3-12 recipe without Oh:

```
Sd .5, C 1, Dr 2.5, xSL 1, consP8 .5, consRplus 3, N2 .5, PsBh .5, SS 1
```

`consRplus` is the Mahalanobis disagreement over the views `b0–b3, cnn, cnn1, pred0, pred1, dog, den_r, den_sd`
plus the `den_b` probe. Oh is excluded because its LCO gain comes from overlap between its training generators and
the proxy shifts. The stack's per-fold members are cached once in `fair-data/factory/cache/fold{k}.pkl` (built
from `lco3/set{k}_std`, about 5 s on CPU).

The base stack's LCO: all-19 mean 0.4350 (folds 0.469 / 0.458 / 0.378), visible-13 0.4772, hidden-6 0.3435.
Without SS it is 0.4319, matching the recorded STACKDp3_denb value of 0.4316.

**Configurations.** A configuration is a misfit weight w (screen: 0; full: 0, .5, 1, 2) crossed with a vote
(none, ridge, prof), minus (0, none).
- The vote is added as an extra view inside `consRplus`.
- One configuration is chosen for all folds, by the reward below.
- Every configuration's reward is reported, so a proposer can see whether the candidate helps as a misfit or as a
  vote.

**Reward.**

```
gain_k   = mean over VISIBLE shifts of phase2_score(stack+cand) - phase2_score(stack)   on fold k
penalty  = 0.01 * (max(0, Δexc@1%) + max(0, Δexc@0.5%))     # percentage points; exc = held-out normal maps
                                                              # above in-sample normal 99% / 99.5% quantiles
reward   = mean_k gain_k + min_k gain_k - penalty
hidden_ok = mean_k (same gain on HIDDEN shifts) > 0
```

Proxy noise is about ±0.004 per fold. A sensible acceptance rule:
- reward > about 0.006,
- every visible fold gain > 0,
- `hidden_ok`,
- and, after acceptance, append the member to `STACK` and rebuild the cache, so the next candidate must beat
  the grown stack.

**Screen mode** is a cheap pre-filter: ridge vote only, with no likelihood fit.
- It uses one fit3 view on every 5th fit map (4,143 maps).
- calib3 and audit3 are used in full.
- Each shift uses every 3rd map that is held out in some fold (492 per shift).
- The base stack is evaluated on the same maps, so gains stay paired.

## Hidden shifts (one per physics family; the visible siblings stay)

| Family | Visible | Hidden |
|---|---|---|
| Baryons / power-spectrum shape | bary6.5, bary9.0, supp0.15, boost0.15 | **baryDMO** |
| Shear calibration | mult+0.05 | **mult-0.05** |
| Noise | noise+0.05, noise-0.05, noisegrad0.15, poisson120 | **noise_t5** |
| PSF / smoothing | smooth0.5, psfpat | **aniso0.7** |
| Astrophysical | srcclust0.15 | **ia0.2** |
| Large-scale / non-linear systematics | gradient | **peakcut0.3** |

The hidden shifts appear only in `result.json["hidden"]`. Of the organizers' planned test systematics, IA is
hidden entirely, so a proposer gets no IA signal and `hidden_ok` checks whether its gains transfer.

## Validation (2026-10-10)

The examples in `examples/` were graded on della-gh (GH200) and gpu-test A100s. Runs are in
`fair-data/factory/runs/` (`gh_*` = GH200).

| Candidate | Mode | ok | Reward | hidden_ok | Notes |
|---|---|---|---|---|---|
| `batch_violator.py` (log power minus batch mean) | screen | **fail** | n/a | n/a | per-map check: 100% of entries off; shuffled order passes, as it should for a batch mean |
| `random_feats.py` (16 Gaussians seeded by a map hash) | screen | ok | +0.0008 | no | ridge θ-probe calib MSE 1.00 (no information) |
| `random_feats.py` | full | ok | +0.0023 | no | best of 11 configurations: this is the selection inflation of a useless family |
| `dog.py` (stats.py DoG, 56 dims) | screen | ok | −0.0052 | no | features match `v3/stats` columns to 5e-5 relative |
| `dog.py` | full | ok | −0.0018 (GH200) / −0.0043 (A100) | no | standalone misfit visible-13 0.2937 vs cached Sd 0.2928; re-adding a stack member does not help, as expected |
| `psband.py` (stats.py power spectrum, 28 dims) | screen | ok | −0.0058 | no | features match to 7e-7 relative |
| `psband.py` | full | ok | −0.0019 | no | already covered by PsBh and consP8 |
| `dirpower.py` (new: power anisotropy, 36 dims) | full | ok | **+0.0086** | **yes** | all 3 folds positive (+.007/+.0035/+.005); psfpat +0.054 visible, aniso0.7 +0.042 hidden; all-19 LCO 0.4350 → 0.4405 |

- **Reproducibility.** Identical inputs give identical rewards up to likelihood-refit noise. dog full gave −0.0018
  on GH200 and −0.0043 on A100, so treat about ±0.003 as the refit noise of a full reward.
- **Acceptance threshold.** A threshold of about 0.006 with all folds positive and `hidden_ok` clears both the
  noise and the +0.002 selection inflation.
- **dirpower.** It is the first candidate found by this grader that clears that bar. It is a PSF / shear-systematics
  probe, and its gain transfers from the visible psfpat shift to the hidden anisotropic-PSF shift.

Wall times (one candidate; dumps partly in the page cache):

| Mode | A100 (gpu-test) | GH200 (della-gh) |
|---|---|---|
| screen (21k maps, ridge only) | 7–16 s | 3–60 s (Python-loop candidates are slow on Grace) |
| full (77k maps, 3 likelihood refits) | 101–140 s: about 13–43 s features, about 28 s per fold fit | 109–207 s |

The likelihood refits dominate full mode. On one A100 that is about 25–35 full grades per hour, or several
hundred screens per hour. Keep the dumps (25 GB) in the page cache or `/dev/shm` for a long-running service.

## Proposer loop (`loop.py`)

`loop.py` drives a proposer model, such as Qwen3.5 on a farm, against this grader. The prompt templates are in
`prompt/`: `problem.md` holds the problem, contract and worked example; `task_improve.md` and
`task_explore.md` hold the two tasks.

**One round:**
1. Plan N prompts:
   - `improve` (75% by default): the parent is picked PUCT-style from the graded pool. The score is
     rank(value) + c·sqrt(1 + total children)/(1 + parent's children). The value is the full reward.
     Screen-only candidates count as screen reward − 0.005.
   - `explore` (`--explore 0.25`): a new idea, given the leaderboard of graded ideas.
2. Generate. Extract the last complete ```python block after `</think>` (as discover's `answer_only_code`).
   An unterminated block is accepted only if it parses.
3. Run static checks (see below). Duplicate ASTs (docstrings ignored) are not graded.
4. Screen-grade every passing candidate.
5. Full-grade the best `--full-top` screens whose reward is ≥ `--screen-floor` (−0.003).

Generation of round r+1 starts once round r's screens are in, so it overlaps round r's full grades.

**Prompt contents.**
- Each prompt shows the parent's code and its proposer-view diagnostics: reward, per-fold gain, best
  configuration, best and worst visible shifts, standalone metrics, θ-probe MSE, tail and runtime.
- It also shows up to 8 other graded ideas and the 3 most recent contract failures.
- Hidden-shift numbers, `hidden_ok` and the acceptance flag never reach the prompt.
- Prompts are 3.9–4.9k Qwen tokens, which leaves an 11.2–12.5k thinking budget.

**Screen versus full.** The screen is a weak ranker. It has no misfit member and only ~500 maps per shift.
- dirpower: screen −0.0004 vs full +0.0086.
- random features: screen +0.0008 vs full +0.0023.

So the screen works as a contract and gross-failure filter. Full grades are given liberally, to up to 8 per
round. Re-expressed families (dog −0.0052, psband −0.0058) fall below the floor.

**Static checks.** `loop.static_check` is the sandbox, because the grader runs candidates in-process. It
rejects:
- imports outside numpy, torch, math, functools, itertools, collections and typing;
- the names `open`, `eval`, `exec`, `compile`, `__import__`, `getattr`, `globals`, `os`, `sys`, `subprocess`,
  `socket`, `pathlib`, `importlib`, `pickle` and `shutil`;
- the attributes `.load`, `.save`, `.fromfile`, `.memmap` and `.from_file`, and any dunder attribute;
- identifiers starting with `test` or containing `_test`;
- non-docstring string literals containing `test`, `data`, `/`, `.np`, `.pt` or `.pkl`;
- `data.py`.

The check is a deterrent against accidental data access, not a security boundary.

**Acceptance (recorded, never applied).** A candidate is marked accepted when its full reward is > 0.006, every
visible fold gain is > 0, and `hidden_ok` is true. The loop records this in `history.jsonl` and in
`leaderboard.{json,md}`, which include `hidden_ok` and the all-19 LCO for us. Adding a candidate to `STACK`
remains a manual step (see caveats).

**State and resume.**
- Everything lives in `fair-data/factory/loops/<run>/`:
  - `history.jsonl` (append-only events: cand, round_generated, submit, result, error, promote, accept);
  - `cands/<id>.py`, `prompts/<id>.txt`, `raw/<id>.txt` (the full model output);
  - `grades/<id>_<mode>/`, `leaderboard.md`, `loop.log`.
- Restart with the same `--run` to resume:
  - Finished grades are read from their `result.json`.
  - Lost grades are resubmitted, at most 2 attempts.
  - A round interrupted mid-generation is regenerated.

**Grading backends.**
- `--slurm-jobs` concurrent gpu-test jobs (QOS limit 3 running per user), each running `run_grade.sbatch` on
  up to `--batch-screen 6` screens or `--batch-full 3` full grades.
- Optional `--gh-workers` della-gh screen processes over `ssh della-gh`, with the runtime budget relaxed to
  150 s per 10k, because the Grace CPU is slow on Python loops. The A100 full grade enforces 60 s.

**Generation settings (`CompletionsGenerator`).** These are the RQ4 Qwen3.5 no-train settings.
- **Sources:**
  - profile `rq4-notrain-local-qwen35-*-20261007.json`;
  - `DirectBaseSamplingClient` and `QwenTwoPhaseTokenCompleter._native_group` in SkyRL-Main-infer
    `third_party/discover`;
  - tinker `SamplingParams` defaults.
- **Prompt:** discover's `Qwen3Renderer` with no system message:
  `<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n<think>\n`, sent as token ids with
  `add_special_tokens=False`. The tokenizer is the Qwen3.5-27B snapshot under `rq4-local/hf`.
- **Request:** `POST <url>/v1/completions` with:

  | Field | Value |
  |---|---|
  | `model` | `Qwen/Qwen3.5-27B` |
  | `n` | `--samples` |
  | `max_tokens` | 22528 − prompt − 50 |
  | `thinking_token_budget` | 16384 − prompt |
  | `temperature` | 1.0 |
  | `top_p` | 1.0 |
  | `top_k` | −1 |
  | `logprobs` | true |
  | `return_token_ids` | true |
  | `stop_token_ids` | `[<|im_end|>]` |

- **Retries:** 409, 429 and 5xx are retried with backoff, as in the RQ4 client.
- **Validation:** every choice must pass `_validate_native_choice` (the native thinking-budget audit). The text
  is `<think>\n` + the decoded `token_ids`.
- **Offline test:** checked against a local mock endpoint, including a 429 retry and a forced thinking stop.

**Usage.**

```bash
cd research/fair_v2
PY=/scratch/gpfs/ZHUANGL/sk7524/rq4-local/envs/client/bin/python    # has transformers + httpx; the loop
                                                                     # itself is stdlib, grades use fair-env
$PY factory/loop.py --run qwen35-a --check-prompts --n 8             # render prompts, count tokens, no network
$PY factory/loop.py --run test --generator replay --rounds 2 --n 5   # offline end-to-end test
nohup $PY factory/loop.py --run qwen35-a --generator completions --url http://127.0.0.1:<port> \
    --rounds 30 --n 32 --full-top 8 --slurm-jobs 3 > loop-qwen35-a.out 2>&1 &
```

`--url` must serve the `/v1/completions` contract above with native thinking budgets. That can be the RQ4
local ingress, which forwards a lease to the farm, or a tunnel to a farm ingress. The loop opens no tunnels and
takes no leases.

**Throughput.** Grading on gpu-test is free, so the farm and the 3-job gpu-test limit are the binding costs.
- **Generation:** a round's latency is set by per-sequence decode. On the RQ4 Qwen3.5 v5p-32 farms, 2–4.5k
  tokens/s are spread over 145–256 running sequences, about 17–30 tokens/s each. An 8–14k-token answer
  therefore takes about 5–12 minutes. N = 32 per round uses a small part of one farm, so `--samples 2–4` or a
  larger N costs little extra wall time.
- **Grading:** grading on gpu-test is free in money but slow in wall time. On the evening of 2026-10-10 a
  gpu-test job waited about 45–50 min in the queue (`Priority`). della-gh starts immediately:
  - a screen is 10–20 s wall, including ssh and startup;
  - a full grade is 130–170 s, with 2 running at once on the one GH200.

  So `--gh-workers 2 --gh-modes screen,full` gives about 30–40 full grades or a few hundred screens per hour.
  Add `--slurm-jobs 3` for more when gpu-test is not congested. With N = 32 and `--full-top 8`, one round
  needs about 32 screens + 8 full grades, about 10–12 GH200-minutes. That matches one round's generation
  latency.

**End-to-end test (2026-10-10, `--generator replay`, run `test-replay`).**
- **Setup:** 2 rounds × 5 candidates, plus the 3 seeds.
- **Backends:**
  - The first submission went to gpu-test. It was cancelled after a 50-minute queue estimate, which exercised
    the error, resubmit and resume path.
  - The run then resumed with della-gh for screens and full grades.

Outcomes by candidate:

| Candidate | Status | Screen | Full | Notes |
|---|---|---|---|---|
| dirpower, 8 bands | ok | −0.0010 | +0.0062 | accepted |
| dirpower, 22.5° wedges | ok | +0.0014 | +0.0081 | accepted |
| random features | ok | +0.0008 | +0.0023 | `hidden_ok` false, not accepted |
| psband, 14 bands | ok | −0.0073 | not run | below the screen floor |
| batch-mean violator | grader contract failure | n/a | n/a | "not a per-map function" |
| `import os` | static_fail | n/a | n/a | |
| no code block | no_code | n/a | n/a | |
| unclosed `<think>` | no_code | n/a | n/a | |
| copy of a seed or earlier candidate (×2) | duplicate | n/a | n/a | |

- Both accepted candidates are the dirpower family again.
- **Before a live run, add dirpower to `STACK`** and rebuild the cache. Otherwise the proposer's easiest win is
  dirpower variants, and the acceptance flag would keep firing for the same family.
- `CompletionsGenerator` was tested against a local mock `/v1/completions` that returns the native audit fields,
  including a 429 retry and a forced thinking stop. All 4 samples were extracted and passed the static checks.
  It has not been run against a real farm.

## Not done / caveats

- **Sandbox only via static checks** (`loop.static_check`). `grade.py` itself still runs candidates in-process
  with normal file access.
- **Learned featurizers** (networks trained on fit3) would need per-fold retraining for an honest LCO. This grader
  assumes fixed functions.
- **The stack is fixed** at v3-12 without Oh. To accept a candidate, add it to `STACK` and to the per-fold caches,
  then delete `fair-data/factory/cache/`.
- **The proxy is validated on only 3 LB points.** LB probes remain the final judge.
