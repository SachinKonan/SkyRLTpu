# FAIR TTT-Discover: JAX program-search environment (Phase 2)

## V2 research upgrade — recommended starting point

**Recommended candidate:** `candidates/diverse_ensemble_jax.py`.
This is a **SOTA-oriented starter**, **not a claimed SOTA solution**. It runs
three complementary in-distribution-only branches: JAX supervised latent
features and physics/statistics, multiscale conditional density, and a separate
spectral baseline. Its prediction is an InD-validation-calibrated scalar
anomaly score. Every model is trained and calibrated without proxy anomaly
labels. This ensemble incurs extra training/inference cost versus one model.

For an **open-ended one-file coding-agent target** that can freely change
architecture and training losses, start from the self-contained
`candidates/hybrid_physics_jax.py` and retain the ensemble as a comparison.
The ensemble wrapper imports sibling model files, so changing its imported
architectures requires inlining/replacing them or explicitly allowing a
multifile search. Do not tell a one-file-only agent to edit those siblings.

**New coding-agent prompt:** `prompts/SYSTEM_V2.md` and `prompts/TASK_V2.md`.
The prompt is deliberately short, scoped to one editable file, and asks the
agent for one testable hypothesis at a time. See `prompts/ITERATE_V2.md` and
`RESEARCH_PLAYBOOK_V2.md` for iterative evolution and ablation guidelines.

```bash
# From repository root after: pip install -e '.[test]'
python scripts/seed_candidate.py --source hybrid --out generated_candidates/candidate.py
python scripts/create_agent_prompt.py > prompts_for_agent.txt
python scripts/create_agent_prompt.py \
    --previous results/candidate_001/reward.json > next_agent_prompt.txt

# Use real staged data for actual research; toy is for plumbing checks only.
python -m fairdiscover.trial \
    --candidate candidates/diverse_ensemble_jax.py \
    --stage data/staged_full \
    --run-dir results/diverse_001 --steps 250 --batch-size 16

# Local-only ablations across identical synthetic proxy sets:
PYTHONPATH=. python scripts/compare_hybrid.py \
    --stage data/staged_full --out results/ablations \
    --only autoencoder,spectral,multiscale,hybrid,diverse
```

To create an **actual Phase-2 Codabench submission**, first obtain the official
noisy test arrays, then use the `fairdiscover.submission` command below with
`--candidate candidates/diverse_ensemble_jax.py` and that candidate's own
`run-dir`. The ZIP contains `result.json` and `ood_scores`, not model weights.

**Measured local proxy evidence, NOT a FAIR score:** the development ablation
was run at 160 steps × batch 32, with 1,024 normal reward maps across four
synthetic anomaly families. The single hybrid reached .17691, the spectral
baseline .11635, old multiscale .11924, AE .04048. The diverse ensemble's
physics-first mixture reached .17115 in its development run. On a newly
seeded synthetic holdout containing four original and three unseen shift
families, the frozen ensemble reached **.15829 mean**, compared to the old
multiscale detector **.10161**. Extra three family means were **.12901 vs
.10542**. Those synthetic results neither reproduce the FAIR baseline nor
establish state-of-the-art on the hidden challenge. Full selection and numbers
are included in `evidence/`. The heldout data must not be reused for tuning.

**Important:** today's public FAIR leaderboard is much higher (~.7803 in its
October 8 snapshot); published top entrants' implementations have not been
verified or recovered here. An official real-data test is necessary before
making a competitive-performance claim.


**Deliverable:** runnable research scaffold for optimizing **programs**, not merely
hyperparameters. A coding LLM edits one Python candidate; the environment
trains it on FAIR public InD weak-lensing maps, evaluates on a held-out local
proxy anomaly set, and writes a scalar reward.

**This is not the original TTT-Discover RL trainer**, nor an official competition
scoring service. It is a simple reward environment for an external coding agent
or RL system. **No official FAIR data, learned weights, or secret anomaly
labels are bundled.** It includes an executable JAX autoencoder starter, a
reproducible split/IO pipeline, synthetic OoD shifts, exact official score
*formula*, and a validated submission ZIP exporter.

## Quickest runnable smoke test (CPU, no download)

```bash
python -m pip install -e '.[test]'
python scripts/make_toy_data.py --out data/toy
python -m fairdiscover.data --raw-dir data/toy --mode sample --out data/staged
python -m fairdiscover.trial \
  --candidate candidates/baseline_jax.py --stage data/staged \
  --run-dir results/toy --steps 5 --batch-size 8 --max-reward-ind 16
cat results/toy/reward.json
```

**Warning:** synthetic toy maps contain no real cosmology; the tiny low-FPR
score is only a plumbing test, NOT a benchmark result.

## Official training data

Use the **May 2026 independent-realization release** at
https://zenodo.org/records/20056065 rather than the 2025 Phase-1 copy.
It contains **101 cosmologies × 256 realizations = 25,856** noiseless maps.
The source `.npy` maps are shape `(101,256,132019)` float16, containing
observed footprint pixels only; the full image is `(1424,176)`, built by the
common boolean mask. Labels `(101,256,5)` are
`[Omega_m, S8, T_AGN, f0, Delta_z]`. The full maps file is about 6.8 GB;
loading all reconstructed maps as float32 would be unnecessarily expensive.

Download **small sample (~24 MB)**:

```bash
python scripts/download_data.py --sample --out data/raw_sample
python -m fairdiscover.data --raw-dir data/raw_sample --mode sample --out data/staged_sample
```

For real experiments, download **full data (~6.8 GB)**:

```bash
python scripts/download_data.py --full --out data/raw_full
python -m fairdiscover.data --raw-dir data/raw_full --mode full --out data/staged_full
```

The downloader uses the public Zenodo URLs and verifies MD5s published with
the dataset. Data must be downloaded on a networked machine. Stage creates a
small manifest and index arrays that reference the source data in place (so
**don't move the raw source files after staging**). Map reconstruction and
shape-noise sampling happen lazily in batches.

### Split rules

Shared realization indices are split **75% / 12.5% / 12.5%** into train /
InD validation / proxy-reward InD, with fixed seed. We split across the
*realization axis* in the same way for each cosmology, following the official
starter's guidance against correlated realization leakage. Synthetic shifts
are made ONLY from the reward InD split. The reward samples and proxy labels
are never passed as training input to candidate.fit.

### Noise rules (important)

Public train maps are **noiseless**. Each training/validation/reward map gets
on-the-fly independent Gaussian shape noise *within observed pixels only*:

`noise_sigma = 0.4 / sqrt(2 * 30 * 2^2) ~= 0.02582`.

The public challenge test maps are **already noisy**. The final submission
exporter never adds noise again. Do not augment noise in `predict` either.

## One real candidate evaluation

```bash
python -m fairdiscover.trial \
  --candidate candidates/baseline_jax.py \
  --stage data/staged_full \
  --run-dir results/candidate_001 \
  --steps 250 --batch-size 16 \
  --max-reward-ind 2048 \
  --train-timeout 3600 --reward-timeout 1800
```

Outputs:

- `results/candidate_001/artifacts/model.npz` (candidate-defined weights)
- `results/candidate_001/reward.txt` (**single floating-point number**)
- `results/candidate_001/reward.json` (including per-proxy-family scores)
- `results/candidate_001/logs/{train,reward}.{stdout,stderr}.txt`

Valid reward range is [0,1]; **failed/time-out candidates receive -1**.
The code SHA256 is pinned at train time; editing code between train and reward
invalidates the trial. Every trial uses the same fixed training and scoring
configuration; standardize the training budget for comparisons.

For several coding-agent-generated Python files:

```bash
python scripts/batch_trials.py --candidates generated_candidates/ \
  --stage data/staged_full --runs results/search_01 --steps 250
```

## Agent contract

Your coding agent must **write one Python file** with exactly these functions:

```python
def fit(train, val, config, out_dir): ...
def load_model(out_dir, config): ...
def predict(model, maps, mask, config) -> np.ndarray: ...
```

`train`/`val` yield dictionaries with `maps` (float32 B×H×W), `labels`
(float32 B×5), and `mask` (bool H×W). `predict` receives **already noisy**
float32 maps and returns one finite score for each map, **higher = more OoD**.
The model is free to use neural AEs, flow/diffusion methods, spectral statistics,
physics-supervised objectives, ensembles, or anything else implemented in JAX.
The coding agent should modify only candidate code. See **`prompts/SYSTEM.md`**
and **`prompts/TASK.md`**, designed for a short coding-agent context window.

The source code and reward loop have no hard-coded neural architecture or loss
for generated solutions. `candidates/baseline_jax.py` is only an example.

## Local reward (proxy, not organizer ground truth)

The fixed judge produces four kinds of *synthetic field-level* anomalies from
held-out clean InD maps: amplitude, spectral tilt, nonlinear peak coupling, and
smoothing. It applies the **same observation-noise process** to both classes.
Candidates train only on InD; reward has labels for proxy OOD only.

The reward per family follows the **exact official Phase 2 formula**:

`mean(interp(TPR at 100 log-spaced FPR values from 0.001 to 0.05))`.

The external scalar is the **mean of these family-level scores**; a pooled
score is reported separately. Unlike the competition, **our OOD labels are
synthetic**, so this number is NOT directly comparable to Codabench.
At least 1,000 (ideally 2,000+) held-out InD maps are recommended to resolve
0.1% FPR. For stronger research, freeze a separate final held-out benchmark of
*unseen* anomaly types and seeds before evolving candidates.

The scoring formula was checked against the publicly released
`scoring_program/score.py` from the organizer's GitHub repository. Note that
its current public `competition.yaml` in the repository may refer to the
older Phase 1 setup, so don't use it as the Phase 2 submission specification.

## Competition submission: predictions, not a GPU/TPU binary

Download **current Phase 2 public test data** from the **Codabench Phase-2
competition** (link below). Use the test file supplied by Codabench, often
named similarly to `WIDE12H_bin2_2arcmin_kappa_test_phase2_new_v2.npy`.
Do not use Zenodo's separate labelled cosmology benchmark holdout (14k)
for this Phase-2 submission. The exporter supports `[N,132019]` masked
pixel arrays OR `[N,1424,176]` full images; preserves sample order.

```bash
python -m fairdiscover.submission \
  --candidate candidates/baseline_jax.py \
  --run-dir results/candidate_001 \
  --stage data/staged_full \
  --test-file /path/from/Codabench/WIDE12H_bin2_2arcmin_kappa_test_phase2_new_v2.npy \
  --out-zip results/fair_phase2_submission.zip
```

This writes a ZIP whose ROOT contains exactly:

```text
result.json
```

with payload `{"ood_scores": [score_0, score_1, ...]}` matching the number
and order of test maps. This matches the organizer's published scoring
program, which loads `result.json` and requires `ood_scores`. The exporter
also leaves a plain sibling JSON for inspection. The official starter shows
how to upload to **Codabench → Submissions → Upload**:
https://www.codabench.org/competitions/10902/

**Important:** verify the current Codabench phase's expected upload ZIP
hierarchy before submission. We follow the public prediction JSON contract,
not a model-execution container. The competition closes **2026-10-11 23:59 UTC**.

## TPU / GPU portability

- Pure JAX uses XLA and can execute on **TPU, GPU, or CPU** given the correctly
  installed backend build. Our starter does not require PyTorch or custom CUDA.
- Check backend: `python -c 'import jax; print(jax.devices())'`.
- On TPU, install an appropriate TPU-compatible JAX/libtpu environment per
  https://docs.jax.dev/en/latest/installation.html .
- For GPU, install the GPU-enabled JAX build on that machine. **The competition
  receives only JSON prediction scores, so TPU-generated model outputs are
  admissible; there is no need to ship TPU runtime to Codabench.**
- Checkpoints use portable NumPy `.npz`, not hardware-specific serialization.

## Safety / validity

- Agent-generated Python is arbitrary executable code. `trial.py` enforces a
  wall-clock timeout, **not a security sandbox**. Execute model programs in an
  isolated ephemeral container/VM with network disabled and read-only public
  input files; keep private scoring files outside the training container.
- The example Python workers share a filesystem for convenience, and are **not
  an adversarially sealed hidden-test system**. For an RL evaluation service,
  place judge data and reward logic in a separate inaccessible process/container.
- Never let candidate programs modify split IDs, evaluator code, or reward
  files. The test set must remain untouched until final export.
- Proxy reward optimization may overfit synthetic shifts. High local score is
  NOT proof of strong real physical anomaly detection.

## Project map

- `fairdiscover/data.py` — memory-mapped staging, survey mask, noise
- `fairdiscover/trial.py` — timeout-aware external trial and scalar feedback
- `fairdiscover/worker.py` — isolated train / private reward phases
- `fairdiscover/shifts.py` — synthetic proxy shifts, on noiseless maps
- `fairdiscover/metric.py` — organizer low-FPR score formula
- `fairdiscover/submission.py` — official JSON prediction ZIP
- `candidates/baseline_jax.py` — runnable physics-aware AE example
- `prompts/` — minimal system/task messages for coding LLMs
- `STRATEGIES.md` — organizer baselines and research ideas
- `scripts/` — public downloads, toy data, batch search trials
- `tests/` — integration/unit tests

## Authoritative references

1. White paper (data, no labeled OoD, scoring, baselines):
   https://arxiv.org/html/2604.14451v1
2. Public train release (new realizations and checksum/manifests):
   https://zenodo.org/records/20056065
3. Official GitHub starter and scoring program:
   https://github.com/FAIR-Universe/Cosmology_Challenge
4. Phase 2 Codabench:
   https://www.codabench.org/competitions/10902/
5. Phase 2 standings: 
   https://fair-universe.lbl.gov/WeakLensing-Uncertainty-Challenge-Phase2.html

### Optional second starting point

`candidates/spectral_jax.py` provides a **JAX FFT feature extractor with
shrinkage-Mahalanobis anomaly scores**. It's a fast, interpretable comparison,
but **not** the official conditional power-spectrum/MCMC implementation.
It obeys the same `fit / load_model / predict` contract and submission exporter.

Check the upload artifact before sending it:

```bash
python scripts/check_submission.py --zip results/fair_phase2_submission.zip
```
