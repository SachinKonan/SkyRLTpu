# Review of the original fair_ttt_discover_strong_v2 package

Inspected directly on 2026-10-09; historical claims below are from source, not official leaderboard evidence.

## What the package actually does

1. `spectral_jax.py`: 20 log-power FFT bands plus map mean, RMS, skewness and kurtosis (24 features); standardization and Ledoit-Wolf covariance fitted to noisy normal training maps; Mahalanobis anomaly score. It explicitly is NOT an exact organizer MCMC baseline.
2. `multiscale_jax.py`: 12 spectrum bands, original plus four Gaussian-smoothed fields, moments/tails, and cross-scale interactions. Normal training labels create a center for each cosmology; shared within-cosmology covariance regularizes distances. Nearest valid cosmology and nearest normal references are alternative scores.
3. `hybrid_physics_jax.py`: radial/angular spectra, spatial gradients, peaks, bandpasses, cross-scale and spatial-stripe features. Robust feature scaling, PCA to 32 dimensions, and global/conditional/neighbor/spectral detectors. An additional MLP reconstructs masked SUMMARY FEATURES and predicts all FIVE physical labels; it is not a full-image denoising autoencoder.
4. `diverse_ensemble_jax.py`: trains all three branches, ranks branch outputs against normal-only validation samples, and applies fixed ensemble weights.

## Important hidden detail: the default score does not use its trained neural branch

Hybrid default `hybrid_robust` is .60 spectral + .25 conditional + .15 neighbor. Neural reconstruction has zero weight. The ensemble's default `physics_first` combines hybrid_robust, hybrid_spectral, multiscale conditional, and standalone spectral. None uses neural latent or neural reconstruction. The physics MLP training is therefore unnecessary work for the default prediction function. The neural latent is not the representation fed into the PCA/conditional/kNN detectors either: those operate on handcrafted features.

## Useful ideas to retain

- Account for known valid cosmological variation: distance from a nearest normal cosmology is more appropriate than assuming every normal map lies near one global mean.
- Fit covariance regularization and all calibration on normal training/validation only.
- Preserve power spectrum and multiscale features as an independent physical comparator. Low pixel MSE alone need not preserve the statistics relevant to unseen physics.
- Keep the mask and shape-noise distribution matched at fitting and inference.
- Record distinct raw component scores rather than only an opaque ensemble scalar.

## Limitations and bugs

- The archived hybrid spatial gradient rolls axis 0 of B,H,W, mixing different samples. The separate `candidates/hybrid_physics_axes_fixed.py` corrects this to spatial axis 2; the archive remains unchanged. Do not reuse the original module unchanged.
- Per-feature clipping at +/-8 can suppress information in extreme tails. Repeated empirical rank calibration saturates beyond the largest calibration sample, introducing ties precisely in the low-FPR tail of interest.
- Mixture/ensemble weights and preferred modes were motivated by synthetic-proxy experiments, not official OOD labels.
- The toy-data evidence shipped in the ZIP is distinct from our later real-training-data/synthetic-anomaly TPU matrix. Neither is an official competition evaluation.
- A nearest cosmology score can still mistake valid continuous parameter variation for an anomaly. Calibrate false-positive rates across cosmology and nuisance bins and report them.
- The archive's baseline autoencoder reconstructs a resized NOISY input and uses hard-coded parameter scaling. Our new model learns from noisy inputs to CLEAN full-resolution targets, uses train-fitted scales, and makes the neural bottleneck central to scoring.

## Consequences for current research

Proceed with the requested supervised denoising autoencoder, but retain the physics-statistics branch as a meaningful comparator. Do not blindly reuse the recommended ensemble. Compare latent density scoring with known-cosmology conditional density and check normal false positives by cosmology. Keep raw continuous anomaly scores rather than rank-normalizing public test predictions. Final submission variants must be frozen without using test-set statistics or labels.
