# Task: invent a new feature family for out-of-distribution detection on weak-lensing maps

You are helping a team in a scientific machine-learning competition (FAIR Universe, out-of-distribution
detection). Write ONE Python function, `features(maps, mask)`, that turns each weak-lensing map into a short
vector of summary statistics. An automatic grader adds your features to an existing detector and measures how
much the detector improves. Only the improvement counts, not how good your features are on their own.

## The data

- Each map is a simulated weak-lensing convergence map (kappa): a 1424 x 176 pixel image with 2-arcmin pixels.
  132,019 pixels lie inside the survey footprint (`mask` is True there). Pixels outside are exactly 0. The
  footprint is a fixed irregular shape, not the full rectangle, so FFTs and filters see its edges. The
  footprint is the same for every map, so its own effects are constant and get absorbed.
- The clean signal has a pixel rms of about 0.010-0.013. Every map also carries white Gaussian shape noise
  with sigma = 0.4/sqrt(240) = 0.0258 per pixel, independent from pixel to pixel. The maps are stored as
  float16. Per pixel the noise is about twice the signal, so the information lives in correlations over many
  pixels and in larger scales.
- The normal (training) maps come from 101 cosmologies x 256 realizations. Five parameters vary:
  - Omega_m: matter density.
  - S8: amplitude of matter clustering.
  - T_AGN and f0: two baryonic-feedback parameters. They suppress small-scale power.
  - Delta_z: a photometric-redshift shift of the source galaxies. It changes the lensing amplitude.
- Test maps can contain physics or systematics that never appear in training: other baryonic physics, PSF or
  instrument effects, shear-calibration errors, wrong noise properties, astrophysical effects, and others.
  - The detector must score such maps higher than normal maps.
  - Normal maps, including normal maps at cosmologies never seen in training, must not raise false alarms.
  - The metric averages the true-positive rate over false-positive rates from 0.1% to 5% (log-spaced), so the
    extreme upper tail of normal-map scores matters a lot.

## The existing detector: what is already measured

Every feature family has its own model of "normal". A neural net predicts the mean and spread of the family's
features given the 5 parameters. Each map is then fit by searching for the 5 parameters that make its features
most likely. Families already in the detector:
- power spectrum in 28 bands, and bispectrum;
- one-point PDF and histogram moments;
- peak and minimum counts;
- Minkowski functionals;
- difference-of-Gaussians scale-space statistics, and wavelet and scattering-transform coefficients;
- two CNNs trained to predict the 5 parameters (their embeddings and predictions);
- a denoising U-Net: statistics of the residual (map minus denoised map), compared with what pure noise gives;
- a diffusion-model small-scale score;
- four band-limited views of the map (large to small scales), each giving its own parameter estimate.

The detector's score combines two signals:
- **Misfit:** how badly the best-fitting parameters still explain each family.
- **Disagreement:** every family gives its own estimate of the 5 parameters. On normal maps the estimates
  agree. Under unseen physics they often disagree. A scale-consistency term also compares large-scale and
  small-scale estimates.

## Why most new features do NOT help

Every family is fit over the 5 parameters. So any distortion that looks like a change of cosmology, baryons or
redshift is absorbed: the fit moves to other parameters and the map looks normal again. A new family helps
only if it does one of these:
1. **Measures something no combination of the 5 parameters can mimic** (off-manifold). Examples:
   - statistical anisotropy;
   - non-Gaussian structure that is inconsistent with the power spectrum;
   - spatial non-stationarity across the footprint;
   - noise properties (the true noise is white, Gaussian, stationary, and has a known sigma);
   - relations between scales that physics fixes.
2. **Gives a parameter estimate that reacts to distortions differently from the existing families**, so the
   disagreement grows.

Re-measuring the power spectrum, peaks, PDF and so on in a slightly different way adds nothing. Re-expressed
existing families score at or below 0. Features must also be stable on normal maps at unseen cosmologies,
meaning smooth in the parameters and low-variance. Noisy or heavy-tailed features create false alarms, and
the grader penalizes them.

## How you are scored

- **What the grader computes:**
  - Your features are computed on about 77,000 frozen maps, normal and distorted.
  - The grader refits on the training cosmologies with 3 different sets of 20 cosmologies held out, then
    evaluates on the held-out ones: normal maps against distorted copies of the same maps.
  - Your family enters the detector as an extra misfit term and/or as an extra parameter estimate in the
    disagreement term. The grader picks the best entry from a small grid.
- **The reward** = the mean gain of the detector's metric over the 3 held-out folds + the worst fold's gain −
  a penalty if the normal maps' tail gets heavier.
  - Gains are small numbers. About +0.003 is noise level. A gain of +0.006 or more with all folds positive is
    a real improvement.
  - Re-expressed existing families score about -0.005 to 0.
- **Visible shifts are only part of the test.** The per-shift numbers you see cover the "visible shifts".
  Acceptance also requires improvement on a separate set of other systematics that is never shown. Aim for
  general sensitivity to "this map is not a normal map", not for one listed shift.

Visible shifts (the names in the per-shift numbers):
- `bary6.5`, `bary9.0`: baryonic feedback much weaker / much stronger than anything in training.
- `supp0.15`, `boost0.15`: extra suppression / enhancement of small-scale power.
- `mult+0.05`: shear calibration error (the signal is scaled by 1.05).
- `noise+0.05`, `noise-0.05`: shape noise 5% above / below the assumed sigma.
- `noisegrad0.15`: noise level drifting by ±15% along the long axis of the map.
- `poisson120`: noise variance varying from pixel to pixel with Poisson galaxy counts.
- `smooth0.5`: extra isotropic smoothing (PSF / resolution).
- `psfpat`: an additive coherent anisotropic pattern (PSF residual).
- `srcclust0.15`: noise level anti-correlated with the local smoothed signal (source clustering).
- `gradient`: an additive large-scale gradient along the long axis.

## The contract (hard requirements: violating any of them gives no reward)

```python
def features(maps, mask):
    # maps: torch.float32 tensor (B, 1424, 176) on a CUDA GPU, zero outside the footprint
    # mask: torch.bool tensor (1424, 176) on the same device (True = inside the footprint)
    # return: (B, d) torch tensor or numpy array, 1 <= d <= 4096 (8-64 is a good size), all finite
```
- **Pure per-map function.** Row i of the output depends only on `maps[i]`.
  - No batch statistics, for example no normalizing by the batch mean.
  - No state carried across calls. Caching constants computed from `mask` or from fixed numbers is fine.
- **Deterministic:** the same input gives the same output. Do not use random numbers. If you need random
  projections, build them once from a fixed `torch.Generator` seed inside your module.
- **Fast:** at most 60 s per 10,000 maps on one GPU, called in batches of 64. Vectorize over the batch. No
  Python loops over maps or pixels.
- **Imports:** only numpy, torch, math, functools, itertools, collections and typing.
  - No file, network, OS or subprocess access. Do not try to read any data.
  - The two arguments are all you get.
- **Well-conditioned outputs:**
  - Use logs of powers and variances.
  - Avoid dividing by quantities that can be near 0.
  - Clamp where needed.
  - Do not return constant columns.

## Worked example (accepted)

This family gained +0.0086 with all three folds positive, and it also helped on the hidden systematics.
It is not yet part of the detector, so its gain is what any new idea is compared with. A near-copy of it
teaches nothing new. Variants are useful only when they capture clearly more, or different, information.

```python
{EXAMPLE_CODE}
```

Its grader diagnostics:
{EXAMPLE_DIAG}

How to read diagnostics:
- **reward** is what is optimized.
- **best config** says how the family entered the detector: misfit weight 0 means it was used only as a
  parameter vote; vote `none` means it was used only as misfit.
- **per-fold gain** has one number per held-out fold.
- **per-shift gains** show which visible distortions the family helped or hurt.
- **misfit alone** is the metric of the family's misfit used as a detector by itself.
- **theta-probe MSE** measures how well a linear map from the features predicts each normalized parameter
  (1.0 = no information).
- **tail** is the fraction of held-out normal maps above the training-normal 1% / 0.5% / 0.1% thresholds.
  Lower is better, and increases are penalized.
- **screen** results come from a cheap first pass: features on a subsample, vote only, no misfit. Only the
  best screen candidates get the full evaluation.
