# What to search for (ideas, not claims of participant implementations)

## Verified organizer baselines (Dai et al., arXiv:2604.14451v1)

- Power spectrum -> fitted cosmology -> chi-squared/p-value outlier statistic:
  reported historical Phase 2 score **0.2143**.
- CNN feature statistic + fitted cosmology -> chi-squared/p-value:
  **0.1053**.
- Low-dimensional convolutional AE -> reconstruction-error tail:
  **0.1307**.
- Random score **0.0128**.

These were published on an earlier public test setup; DO NOT compare these
numbers directly to a new local synthetic reward or claim they are the current
leaderboard cutoffs. The 2026 public leaderboard lists method names/scores but
not enough algorithm details to reconstruct top private solutions. We make no
claims about unreleased participant methods. Check official website for live
current standings.

## High-priority TTT-Discover candidate families

1. **Power spectrum + multi-scale statistics**: radial/two-dimensional FFT
   bands, cross-band moments, peak counts, wavelet energy and tail statistics.
   Important: model both mean and covariance across cosmological parameters.
2. **Conditional goodness-of-fit**: regress the expected statistic as a
   function of (Omega_m, S8) or of an inferred parameter estimate, then test
   the *residual* rather than raw distance from a single global mean. This
   avoids classifying rare but valid cosmologies as anomalies.
3. **Neural AE with physics supervision**: combine pixel/masked-pixel and
   power-spectrum reconstruction objectives with labelled physical prediction
   to preserve meaningful latent variables. Score both reconstruction and
   latent-density deviation.
4. **Masked/denoising prediction**: train on independent shape-noise draws and
   masked regions; test for structural inconsistency on seen spatial scales.
5. **Likelihood ratio / energy**: normalizing flows, diffusion residuals,
   autoregressive image-patch models, score matching. Raw likelihood alone can
   be unreliable for OoD; compare against a background/reference model.
6. **Calibrated ensembles**: combine tail-calibrated non-neural and neural
   outlier statistics to protect against unseen anomaly families.
7. **Distribution shift discipline**: keep survey mask and noise matched to
   official data, use known valid nuisance variations as invariances, and
   reserve unseen proxy-shift TYPES to measure generalization.

## Suggested search curriculum

- Stage 0: reproduce starter AE and FFT power spectrum on InD data.
- Stage 1: use cheap short runs/small subset to reject invalid candidates.
- Stage 2: run 2k+ held-out InD samples to resolve FPR=0.001; compare per-shift
  scores, not merely pooled ROC-AUC.
- Stage 3: full data & training budget; rank robust improvements; inspect
  reconstruction and score distributions.
- Stage 4: evaluate once on a frozen final local holdout with unseen shift
  families / independent seeds; submit only genuinely promising candidates.

## Key research risk

There are NO organizer-labeled OoD training/validation maps. Artificial image
perturbations make a useful program-optimization surrogate, but optimizing
against them can produce shortcuts that do not transfer to hidden physics.
The true leaderboard can only be established by an official submission.

## Reading

- Official white paper: https://arxiv.org/html/2604.14451v1
- Official starting kits: https://github.com/FAIR-Universe/Cosmology_Challenge
- Public independent-realization training data: https://zenodo.org/records/20056065
- Phase 2 challenge and leaderboard: https://fair-universe.lbl.gov/WeakLensing-Uncertainty-Challenge-Phase2.html
