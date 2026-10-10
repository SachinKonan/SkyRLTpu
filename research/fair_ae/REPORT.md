# FAIR Universe research results — October 9, 2026

Nine learned configurations completed on the assigned v5p-32. Three valid prediction ZIPs are ready. We have normal-data validation and synthetic-shift evidence, but **no official competition score, hidden anomaly labels, or demonstrated SOTA result**. The files contain predicted anomaly scores, not labeled ground truth.

The strongest result is that good reconstruction or parameter prediction does not automatically yield the best anomaly detector. Fixed physical summaries remain competitive in this study. Minimum standardized distance to their normal clusters improved several synthetic sensitivities compared with mixture likelihood. A neural ensemble provides a different hypothesis to test, but did not win our local comparison.

## Upload files

Directory: `/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-fair-universe/.fair/deliverables/fair-2026-10-09/`.

| Suggested order | File | What it tests |
|---|---|---|
| 1 | `01-physical-cluster-distance.zip` | Distance from the closest normal physical-summary cluster, normalized by its coordinate variances |
| 2 | `02-physical-likelihood.zip` | The same physical representation with Gaussian-mixture negative log likelihood |
| 3 | `03-complementary-ensemble.zip` | Equal combination of physical likelihood, joint-summary neural likelihood, and spatial reconstruction-latent likelihood |

Each archive contains exactly `result.json`, with one `ood_scores` list of 10,000 finite numbers in the original Phase-2-v2 test order. Higher means more anomalous. Singleton and permutation checks passed for each component and for the ensemble. `manifest.json` and the adjacent receipts contain SHA-256 hashes and frozen-model provenance. No upload was made automatically.

The joint-summary and spatial-only predictions are also preserved as internal ensemble components in the reproduction bundle. They are not additional recommended uploads for this three-file round. Three manual uploads use three submission slots; this work has consumed none of the user's account quota.

## Data and isolation

We used the official independent-realization training set: 101 cosmologies × 256 realizations, stored as 132,019 valid mask pixels per map. The mask reconstructs the 1424 × 176 image. Training labels contain two cosmological and three nuisance parameters, not binary anomaly labels.

All three training files were checked against published checksums. The current `WIDE12H_bin2_2arcmin_kappa_test_phase2_new_v2.npy` was extracted from the official public-data ZIP and hashed. The superseded test version was not used. `official-download.json` preserves archive/member details.

Per cosmology, seed 2026 partitions realizations into 160 fitting, 32 normal validation, 32 normal calibration, and 32 normal audit examples: 16,160 / 3,232 / 3,232 / 3,232 total. Splits are disjoint. They cover the same 101 cosmologies; this is not leave-cosmology-out validation. Independent Gaussian shape noise with sigma 0.025819889 is added to clean training maps. Public maps already contain noise, so inference adds none.

Training, latent normalization, clustering, covariance fitting, calibration, and ensemble scaling use training-derived data only. Checkpoints and cluster count use normal validation. Development perturbations inform the final shortlist. The untouched perturbation families were examined only after the shortlist was fixed. An earlier control export was made while other models trained; its predictions were not used to select or modify them.

Public inputs are evaluated independently: per-map features and GroupNorm, fixed train-fitted normalization, no test clustering or test ranking. This follows the organizer's [per-sample inference rules](https://github.com/FAIR-Universe/Cosmology_Challenge/blob/master/pages/Phase2/overview.md). The test labels remain unavailable.

## Learned models and observed normal-data results

All learned models use seed 0. The image models use batch 16, AdamW, gradient clipping, and a warmup/cosine learning-rate schedule. Fresh noise is sampled during training. Best checkpoints minimize each run's specified normal-validation objective; fixed validation noise makes checkpoints comparable within a run. Parameter errors below are direct head RMSE on the fixed 1,024-map image validation subset. Untrained heads are omitted.

| Image architecture/objective | Updates completed | Selected update | Clean image NMSE | Omega_m RMSE | S8 RMSE |
|---|---:|---:|---:|---:|---:|
| Global latent 32, reconstruction + 0.10 × parameter loss | 20,000 | 9,000 | 0.9615 | 0.0402 | 0.0328 |
| Global latent 64, reconstruction + 0.25 × parameter loss | 20,000 | 12,000 | 0.9465 | 0.0375 | 0.0323 |
| Global latent 128, reconstruction + 0.50 × parameter loss | 20,000 | 14,000 | 0.9315 | 0.0366 | 0.0324 |
| Spatial, reconstruction only | 12,000 | 12,000 | **0.5882** | — | — |
| Spatial, parameter prediction only | 12,000 | 11,000 | — | **0.0344** | **0.0295** |
| Spatial, reconstruction + 0.25 × parameter loss | 12,000 | 12,000 | 0.6189 | 0.0356 | 0.0303 |
| Coarser spatial, same joint objective | 12,000 | 12,000 | 0.6874 | 0.0370 | 0.0303 |

Image NMSE is masked clean-target MSE divided by the fitting-set signal variance. Global models use a dense spatial bottleneck; spatial models retain a 16-channel latent map and use its channel means, log standard deviations, skewness, and kurtosis as a 64-dimensional anomaly representation. Decoders have no encoder bypasses. Global latent size and supervision weight changed together, so these three rows do not isolate the causal effect of latent dimension.

The two summary MLPs take 72 noisy physical statistics and learn a 32-dimensional code. They reconstruct clean statistics, with optional supervision of all five physical labels. Both completed 10,000 updates. Reconstruction-only selected update 9,500 with standardized feature MSE 0.0630; joint supervision selected update 6,000 with feature MSE 0.0643 and Omega_m/S8 RMSE 0.0300/0.0259 on all 3,232 normal validation examples. These feature losses and validation sets differ from the image table and must not be compared as identical reconstruction tasks.

A common ridge probe gives a fairer comparison of latent parameter content: audit normalized MSE for Omega_m/S8 was 0.1195/0.1945 for the fixed physical summaries, 0.1044/0.1918 for the reconstruction-summary code, and 0.0945/0.1847 for the joint-summary code. Joint supervision improved this probe; it did not establish better unknown-OOD detection.

The spatial reconstruction code had strong noise repeatability (Spearman 0.843), compared with 0.764 for physical likelihood and 0.721 for joint-summary likelihood. The cluster-distance score was less stable at 0.562. Correlation effective ranks and all parameter probes are in `reports/normal-and-training.csv`.

## Clusters and scores

The physical vector contains 20 radial Fourier-power bands, 40 multiscale moment/tail/gradient statistics, and 12 cross-scale moments. It is a fixed descriptor, not a trained neural representation or a full wavelet-scattering implementation.

For each representation, fitting-set standardization and regularized whitening precede K-means. Candidate counts are 1, 4, 16, and 64, plus centers grouped by the 101 supplied cosmologies. Normal validation likelihood selects the density. Variances shrink toward unit scale, avoiding tiny clusters with singular covariance. These are K-means-initialized diagonal Gaussian mixtures, not EM-fitted mixtures.

The physical model selected 16 clusters; summary models and most spatial models selected 64. The likelihood score includes cluster weights and covariance-volume terms. The distance variant uses the **minimum squared standardized distance** to those same physical clusters, without either term. It is a cluster-based goodness-of-fit proxy, not exact continuous cosmological profiling or proof of nuisance invariance.

The ensemble keeps physical likelihood, joint-summary likelihood, and spatial reconstruction-latent likelihood. Each score is centered at its normal-calibration median and divided by its calibration q90-minus-median. Their fixed arithmetic mean is the final score. No test-set ranks or statistics enter it. We also examined max and log-sum-exp combinations on development probes; neither gave a general advantage. The final equal mean was frozen before novel-family results were inspected.

## Pseudo-test results

All entries below use the competition's low-FPR ROC functional, applied to **synthetic positive examples**, not the official unlabeled test. The metric averages TPR over 100 logarithmically spaced FPRs from 0.001 to 0.05. [Official metric](https://github.com/FAIR-Universe/Cosmology_Challenge/blob/master/pages/Phase2/evaluation.md)

Perturbations act on clean held-out maps before adding the correct observation noise. Each model receives the same map IDs and noise. Each family uses 512 perturbed examples against 3,232 independent normal calibration examples. Valid cosmological/nuisance variation is not labeled anomalous.

| Synthetic family and strength | Physical likelihood | Physical cluster distance | Joint-summary latent | Complementary ensemble |
|---|---:|---:|---:|---:|
| Development: spectral suppression 0.2 | 0.0312 | **0.1099** | 0.0322 | 0.0174 |
| Development: spectral suppression 0.5 | 0.6451 | **0.8101** | 0.5168 | 0.4207 |
| Development: spectral enhancement 0.2 | 0.1145 | **0.1786** | 0.0911 | 0.0713 |
| Development: spectral enhancement 0.5 | **0.8543** | 0.8322 | 0.7528 | 0.7479 |
| Development: nonlinear peaks 0.2 | **0.3924** | 0.3600 | 0.3494 | 0.3734 |
| Development: nonlinear peaks 0.5 | **0.9594** | 0.9467 | 0.8897 | 0.9257 |
| Development: same-cosmology blending 0.5 | 0.0098 | 0.0394 | 0.0197 | 0.0061 |
| Held out: anisotropic filtering 0.2 | 0.0995 | **0.2096** | 0.0548 | 0.0426 |
| Held out: anisotropic filtering 0.5 | 0.9660 | **0.9804** | 0.8387 | 0.8732 |
| Held out: spatial modulation 0.2 | 0.0174 | 0.0190 | 0.0168 | 0.0171 |
| Held out: spatial modulation 0.5 | 0.0262 | 0.0386 | 0.0289 | 0.0257 |

The complete 13-method table, both strengths of every development family, and one-/two-sided diagnostics are in `reports/synthetic-diagnostics.csv`. Global bottleneck latents were especially weak on these probes. Adding three reconstruction-residual features to the joint-summary latent changed performance only modestly; it did not overturn the physical comparator.

The cluster-distance minus likelihood improvement on mild anisotropy was 0.1101, with an exploratory paired cosmology-block bootstrap interval of [0.0796, 0.1476]. For mild spectral suppression, the difference was 0.0788 [0.0527, 0.1149]. These 300-resample intervals describe this synthetic experiment; they are not training-seed variation or uncertainty on official performance. `reports/paired-bootstrap.json` preserves all comparisons.

At the threshold fitted to the normal calibration set's 99th percentile, normal-audit exceedances were:

| Detector | Exceedances / 3,232 | Observed fraction |
|---|---:|---:|
| Physical likelihood | 40 | 1.24% |
| Physical cluster distance | 32 | 0.99% |
| Joint-summary latent | 38 | 1.18% |
| Spatial reconstruction latent | 44 | 1.36% |
| Complementary ensemble | 36 | 1.11% |

Aggregate calibration conceals uneven performance across cosmologies: the distance detector's worst audit group had 8 exceedances among 32 maps at this threshold. It has not solved conditional false positives. At FPR 0.001, only about three calibration examples define the tail; estimates there are inherently fragile. Mild spatial modulation and same-cosmology blending remain difficult. Some synthetic changes may overlap valid physics, which further limits what these scores mean.

## What the literature changes about the plan

The [organizer paper](https://arxiv.org/abs/2604.14451) compares cosmology-conditioned summary goodness of fit with neural summaries and an autoencoder. It motivates keeping physical-statistics controls and separating parameter inference from anomaly detection. Our cluster-distance comparison follows that distinction, but is not a reproduction of their baseline.

[Optimal neural summarisation for weak lensing](https://arxiv.org/abs/2407.10877) warns against treating parameter-MSE compression as sufficient information. [DAGMM](https://bzong.github.io/doc/iclr18-dagmm.pdf) motivates combining latent and reconstruction information; our separate-stage residual experiment is not its joint training algorithm. [Weak-lensing scattering research](https://arxiv.org/abs/2006.08561) motivates richer multiscale nonlinear summaries rather than simply widening the global AE. None of those papers proves our current detector is competitive on this competition.

The original supplied ZIP was reviewed separately in `PACKAGE_REVIEW.md`. Its default final blends largely used physical statistics even though it included a neural branch. A batch-axis gradient issue and empirical-rank tail saturation also mattered. We preserved the vendor source and did not assume that “contains a neural model” meant “uses neural predictions.”

My next experimental priority would be masked scattering or another richer spatial summary with normal-parameter-conditioned density/goodness-of-fit modeling. A second direction is independent-noise-view representation learning with an explicit anti-collapse objective, followed by the same audit. Neither is implemented or validated here. Another large global AE sweep is poorly supported by the present results.

## Reproduction, compute, and remaining limits

Worktree: `/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-fair-universe`, branch `agent/fair-universe-momo`, base commit `d5088ad21561ce9afd0bb1ac073639960e7f3927`. Research additions are uncommitted; the delivered source snapshot provides the exact code. Existing unrelated farm-integration edits were preserved.

The delivery contains `reproduce/` with selected checkpoints, normal density/calibration arrays, exact runtime versions, receipts, and `predict.sh`. `research/fair_ae/RUN.md` documents training. `all-experiment-artifacts.tar.gz` preserves the broader comparison, including initial-pilot records and intermediate spatial snapshots. Raw data are excluded; download the official public ZIP and use its `public_data` directory. The public test is about 2.64 GB uncompressed.

The tested environment was Python 3.12, JAX/JAXlib 0.10.1, libtpu 0.0.41, Flax 0.12.4, Optax 0.2.6, NumPy 2.3.5, and SciPy 1.18.1. A clean reinstall and a different accelerator have not been validated. Exports enforce original source/checkpoint/density hashes; use each preserved runtime rather than mixing current and older scripts.

Experiments ran as independent single-chip systemd services on the granted four-host `erdos-3242` v5p-32, not as a production-farm rollout or automated MOMO population search. Global image training's recorded update/validation loop took about 37.9 minutes per configuration; spatial training about 22.8–22.9 minutes. These loops ran concurrently. Large data stayed in `/dev/shm`; no GCS production reads/writes or unrelated jobs were changed.

The first pilot needed `LimitMEMLOCK=infinity`; its corrected rerun succeeded. Artifact transfers intermittently failed SSH authentication checks; bounded retries completed, and all four final archives were extracted and verified on GPFS. At 19:13 UTC, all four hosts had no running `fair-*` services and no `/dev/vfio` owners. Root disks retained 67–70 GiB free. Holding job 2306 remains for the coordinator to release; this work did not cancel it. The first assigned machine, erdos-3191, was not used or re-polled in this research phase.

Best model weights are preserved, but optimizer states were not saved; exact mid-training continuation is not provided. There was only one training seed per learned configuration. Synthetic benchmark design and final candidate selection introduce researcher choices. Three official submissions are the next external check, not a guaranteed route to the current leaderboard leader's score.
