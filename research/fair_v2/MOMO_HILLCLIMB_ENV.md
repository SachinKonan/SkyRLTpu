# FAIR Universe Phase 2: a hill-climbing environment for MOMO (design, not implemented)

Status: design note, 2026-10-10. Nothing here is implemented as an environment yet. Every component it relies on
already exists as scripts under `research/fair_v2/`, and the agents have been running this loop by hand to take
the leaderboard (LB) from 0.527 to 0.646.

## 1. What we learned about what to optimize

- **Synthetic shifts on in-sample normal maps are the wrong reward.** We built fake anomalies on audit3 maps, whose
  cosmologies and baryon models are all in training. That reward rated v3-02 and v3-04 as equal, while the LB put
  v3-04 +0.08 higher. It undervalued parameter-consistency features by about 2×.
- **Leave-cosmology-out (LCO) is the first reward that tracks the LB.** We hold out 20 of the 101 cosmologies in each
  of 3 disjoint folds, refit, and score held-out normal maps against shifted copies of the same maps. The LCO metric
  ranked all three scored submissions correctly:

  | Submission | LCO | LB |
  |---|---|---|
  | v3-01 | 0.376 | 0.527 |
  | v3-02 | 0.393 | 0.567 |
  | v3-04 | 0.413 | 0.646 |

  That is about 3 LB points per LCO point.
- **What makes a member valuable:**
  - **Off-manifold signal.** Fitting the 5 parameters per map absorbs most physics shifts. Information survives
    either as misfit or as *disagreement* between feature families.
  - **A tight normal tail on unseen parameters.** About 60% of the metric's weight is at FPR < 0.01.
  - **Marginal gain on the existing stack.** Standalone strength does not matter.

## 2. The scoring pipeline (what the environment wraps)

**Splits.** All 25,856 training maps carry labels. The split is map-level with a fixed seed, so all 101 cosmologies
and all 256 baryon models appear in every part:
- **fit3** (80%): learn "normal", and train any networks.
- **calib3** (10%): calibrate the scores.
- **audit3** (10%): evaluation. Its maps are the normal maps, and synthetic shifts of them are the stand-in OoD.
- **Public test** (10,000 unlabeled maps): scored only at the end, one map at a time, with frozen models.

A **family** is a named group of feature columns. Each family has its own normal model, misfit and parameter vote.

```python
# --- per family, fit once on fit3 ---
X = features(noisy(fit3 maps))                         # (N, d)
net: theta -> (mean[d], std[d])                        # small MLP, Gaussian log-likelihood
Z = (X - mean(theta_true)) / std(theta_true)            # residuals at the true params
R = shrunk_correlation(copula(Z))                      # one pooled matrix: how residuals co-vary

# --- scoring one map with one family ---
def fit_map(x):
    f = features(x)
    d2(theta) = z(theta)^T R^-1 z(theta) + log det + prior   # z = (f - mean(theta)) / std(theta)
    theta_hat = argmin d2 (gradient descent, training range)
    return misfit = d2(theta_hat), vote = theta_hat

# --- calibration on calib3 (normal maps never used for fitting) ---
misfit_z = (misfit - m(theta_hat)) / s(theta_hat)      # m, s: smooth fits of normal misfit vs theta
disagreement_z = std(log Mahalanobis(vote differences across families))  # covariance from calib3

# --- final per-map score ---
score = sum_k w_k * z_k / sum_k w_k                    # misfit members + disagreement members
```

## 3. The environment

### What MOMO edits

A single file, `candidate.py`:

```python
def features(maps, mask):   # maps: (B, 1424, 176) float32 noisy maps; mask: (1424, 176) bool
    ...                     # pure per-map function; GPU allowed; deterministic
    return feats            # (B, d) float32, finite
```

Later variants:
- **Learned featurizer:** MOMO trains a small network on fit3 maps only.
- **Re-grouping:** MOMO proposes a new split of existing columns into families, which can give better votes.

MOMO does **not** tune blend weights. The environment optimizes them internally.

### What the environment computes per candidate

```python
def evaluate(candidate):
    # 0. contract checks (hard fail)
    assert per_map(candidate)            # singleton vs batch identical; shuffled order identical
    assert runtime(candidate) < budget   # it must fit the final 1200 s / 10k-map code-submission limit
    assert finite_outputs(candidate)

    # 1. features on FROZEN inputs: the exact noisy maps every existing member used
    F = {s: candidate.features(DUMPS[s]) for s in
         ['fit', 'calib', 'audit', *VISIBLE_SHIFTS, *HIDDEN_SHIFTS]}

    # 2-5. the pipeline above, for the new family
    fam = fit_family(F['fit'])                     # normal model
    fam.calibrate(F['calib'])                      # misfit calibration + its vote

    # 6. marginal gain over the CURRENT BEST stack, leave-cosmology-out
    gains = []
    for fold in LCO_FOLDS:                         # 3 x 20 held-out cosmologies (cached for the stack)
        f_k = refit_family(F, without=fold)        # only the new family is refit; stack fits are cached
        base = lco_metric(STACK, fold)
        best = max(lco_metric(STACK + f_k.as_misfit_member(w) + f_k.as_extra_vote(), fold)
                   for w in WEIGHT_GRID)
        gains.append(best - base)

    reward = mean(gains) + min(gains)              # help on average AND never hurt a fold
    reward -= TAIL_PENALTY * excess_false_alarms   # held-out normal maps above nominal 1/0.5/0.1%
    hidden_ok = hidden_shift_gain(...) > 0         # never shown to MOMO
    return reward, hidden_ok, diagnostics          # per-shift / per-fold / tail stats as feedback
```

### Outer loop

```python
stack = current best recipe          # e.g. v3-12 members
while budget:
    cand = MOMO.propose(history)     # history = past candidates + diagnostics
    r, hidden_ok, diag = evaluate(cand)
    history.append((cand, r, diag))
    if r > THRESHOLD and hidden_ok:
        stack = stack + cand         # accepted; every later candidate must beat the grown stack
        refresh stack caches (fold fits, blend weights)
```

This is greedy forward selection over feature families, which is hill climbing on "stack quality on unseen
cosmologies".

### Guardrails (all required)

1. **Hidden shift families against reward hacking.** The outlier-exposure member gained +0.03–0.05 LCO because its
   training generators resembled the proxy's shifts. Its held-out-family version gained only about +0.005.
   - MOMO sees scores only on visible shift families.
   - Acceptance also requires gain on hidden families.
   - No shift generator code is ever exposed to the candidate.
2. **Learned featurizers retrain per fold.** CNN and denoiser members were trained on all cosmologies, and LCO only
   refit their likelihoods, so their LCO is optimistic. For candidates that train networks, the environment retrains
   them per fold (more expensive) or caps their reward.
3. **Proxy noise.** LCO noise is about ±0.004.
   - Require the gain to be positive on every fold.
   - Use a minimum threshold.
   - Re-validate accepted stacks with occasional LB probes; the proxy is validated on only 3 LB points.
4. **Competition rules, enforced mechanically.**
   - Per-map independence.
   - Fixed-shape batches.
   - No access to the public test file during fitting or evaluation.
   - Test scoring happens only in the final export, outside the loop.
5. **Runtime.** Every accepted member must also run inside the final code submission (Codabench `model.py`, about
   1200 s for 10k maps on unknown hardware). Charge candidates for inference cost.

## 4. Cost and infrastructure

- **Per candidate:** feature extraction on about 107k frozen maps plus 1 full and 3 fold likelihood fits plus a weight
  grid. That is about 10–15 GPU-minutes with the stack's fold fits cached, so roughly 4–6 candidates per GPU-hour.
- **Hardware:** the existing harness is torch on della GPUs (GH200 or A100). MOMO's grading farms are JAX on TPU, so
  it needs either a GPU grading backend or a JAX port of the likelihood and LCO code. The feature functions themselves
  can be JAX or torch.
- **Frozen inputs:** the noisy maps were dumped once, bit-identical to all existing feature sources:
  - fit3 ×2 views
  - calib3 ×2
  - audit3
  - 19 shifts
  - (the test file is scored only at export)

## 5. Existing pieces to reuse

| Step | Code (under `research/fair_v2/`) | Data |
|---|---|---|
| Frozen noisy maps | `agents/diffusion/dump_noisy.py` | `fair-data/agents/diffusion/noisy/` |
| Family definitions | `agents/consistency/lib.py` (`FAMILIES`) | `fair-data/v3/<source>/` |
| Normal model, fit, copula, calibration | `model.py`, `agents/consistency/jlik.py`, `fit_families.py` | `fair-data/agents/consistency/fam/` |
| LCO refits | `agents/consistency/holdout3.py` | `fair-data/agents/consistency/lco3/set{0,1,2}_std/` |
| Disagreement, components, metric | `agents/consistency/lco3_analyze.py` (`cons`, `components`, `evaluate`) | n/a |
| Blend search | `agents/consistency/stack_search*.py` | n/a |
| Full-data export, merge, zip | `agents/consistency/export3.py`, `merge.py`, `export.py` | `fair-data/merged/`, `fair-data/submissions/` |
| Code-submission packaging | `package.py`, `pack/` | `fair-data/packages/` |

## 6. Recommendation

- **Before the 2026-10-11 deadline:** keep running the loop by hand. Agents propose members and the LCO harness scores
  them. Building the guardrails takes longer than the remaining time.
- **After the deadline, or as a low-priority parallel track:** package the environment from Section 3 with the frozen
  dumps, the LCO-stack reward and the guardrails. This answers the open item from the previous handoff: "decide the
  search reward".
