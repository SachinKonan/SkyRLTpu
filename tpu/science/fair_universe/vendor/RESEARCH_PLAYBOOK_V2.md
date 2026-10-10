# FAIR Universe v2: Kaggle-strength research design

## What we know vs. what we do NOT know

- 2026 Phase 2 competition is **unlabeled OOD detection**, not supervised anomaly
  classification. Hidden OOD generative physics is undisclosed.
- Public maps are noiseless; official test maps are noisy; mask is shared; 5
  labels mark the cosmology and known nuisance variation. Train-time labels can
  supervise encoders; predict does not get labels.
- Organizer metric: mean ROC TPR at 100 log-spaced FPR from .001 to .05.
- Public leaderboard top entry was ~.7803 on October 8, 2026, but public
  leaderboard does **not** identify or disclose winning implementation.
- None of the methods in this repo has been scored on the official hidden set.

## Principles adopted from strong Kaggle/Jane Street work

1. **Evaluate the true downstream objective.** AE reconstruction MSE is not
   enough; the paper's power-spectrum baseline beat its AE.
2. **Trust validation design.** Separate training, normal-only calibration,
   development proxy reward, locked holdout with unseen shifts, and official
   leaderboard. Tune only on development. Repeated testing on an allegedly
   untouched holdout destroys its independence.
3. **Diversity, not just model count.** Physics spectrum, wavelet/peak features,
   supervised neural representations and local/conditional density fail on
   different perturbations. An individually weaker branch can still help the
   ensemble. Ablate every branch before accepting it.
4. **Make numerical nuisances explicit.** Fixed survey mask, shape-noise
   treatment, preprocessing fit on training data, stable score tails, log
   transforms, shrinkage covariances and deterministic random seeds.
5. **Optimize reliability and cost.** Record peak RAM/accelerator memory,
   training time, feature extraction time, checkpoint size, 0.1% FPR detection,
   worst-family score and multi-seed variance. The ensemble costs more than a
   single model; never compare under a falsely equal compute claim.
6. **Require causality / no leakage.** No synthetic OOD train labels, no test
   transduction using hidden labels, no test-time noise injection, no tuning
   score thresholds to Codabench; test only at final export.

## V2 initial candidate

`candidates/hybrid_physics_jax.py`:
JAX FFT radial + angular spectra, Gaussian and bandpass statistics, high-order
peaks, directional gradients, cross-scale responses, location heterogeneity,
plus a JAX masked denoising/physics-head MLP on summary features. Score with
shrinkage global/conditional distance, spectral Mahalanobis, kNN, neural
feature reconstruction; rank-calibrate from InD validation.

`candidates/diverse_ensemble_jax.py` (recommended default): fits the above
alongside independent multiscale conditional and classical power-spectrum
branches, calibrates their scalar outputs on normal validation maps, and fuses
them. This costs roughly three fitted branches; train and export through the
usual `fit/load_model/predict` interface. It deliberately favors physically
complementary statistics over a single trained network.

## Better discoveries to try next (not implemented)

- Conditional *normalizing flow* on learned summaries, marginalizing over
  cosmological/nuisance parameters instead of global Gaussian distance.
- Fourier/wavelet scattering or equivariant CNN with mask-aware operations and
  supervised + contrastive training, across 3-5 seeds.
- Log-likelihood ratios against alternate noise / perturbation models, with
  branch-specific calibration and multiple-testing controls.
- Simulation-conditioned diffusion or score matching on patches (much more
  costly; may encode irrelevant pixel detail).
- Leave-one-cosmology-out *in-distribution* stress tests in addition to normal
  realization holdouts; high score on new cosmologies can be a false positive.
- Robust soft-min / soft-max fusion, chosen on development only, with calibrated
  low-FPR tails. Avoid blindly maximizing a noisy component.
- Bootstrapped seed ensembles and learned weights, BUT weights must not train on
  the final locked holdout or official leaderboard.

## Measured synthetic-proxy evidence (not FAIR SOTA)

Development 16 synthetic cosmologies, 1,024 normal reward maps, four synthetic
shift families, fixed noise: v2 hybrid robust .17691, spectral .11635,
autoencoder .04048, multiscale conditional .11924. These values were used to
select a robust design; they are not directly comparable to official scores.

Fresh holdout with different random seed, 1,024 InD maps, four original +
three novel perturbation families, *physics-first diverse ensemble frozen in
advance*: mean across 7 families .15829 vs multiscale .10161; extra three
family mean .12901 vs .10542. Synthetic shifts are imperfect surrogates for
hidden physically different universes; no SOTA or ranking claim follows.

Full reproduction logs live in `evidence/` (summaries only, not real data).

## Sources

- Competition white paper: https://arxiv.org/html/2604.14451v1
- Official repository: https://github.com/FAIR-Universe/Cosmology_Challenge
- Updated dataset (independent realizations): https://zenodo.org/records/20056065
- Official leaderboard: https://fair-universe.lbl.gov/WeakLensing-Uncertainty-Challenge-Phase2.html
- Jane Street reference: https://github.com/evgeniavolkova/kagglejanestreet/blob/master/solution.md
- Jane Street leakage-safe reconstruction: https://github.com/cweill/jane-street-causal-forecasting
