# Research decisions and primary sources

Reviewed 2026-10-09. These papers motivate experiments; none establishes that our implementation is competitive on the hidden FAIR test.

| Source | Relevant finding | Experiment or constraint here |
|---|---|---|
| [FAIR organizer paper, Sections 5.1–5.2](https://arxiv.org/html/2604.14451v1) | Uses cosmology-conditioned summary-statistic goodness of fit and an 8-dimensional convolutional AE baseline; neural parameter estimation and OOD detection have different outcomes. | Compare normal-cosmology centers against K-means; retain a physical-statistics detector; compare reconstruction and latent scoring. Published baseline scores are historical, not our achieved results. |
| [Lanzieri et al., optimal neural summarisation for weak lensing](https://arxiv.org/abs/2407.10877) | Parameter MSE need not produce sufficient summaries; information-theoretic objectives can retain information lost by point estimation. | Do not use low parameter RMSE as sole evidence of a good OOD representation. Preserve morphological information through reconstruction and compare different objectives. |
| [Alain and Bengio, regularized autoencoders](https://jmlr.org/papers/v15/alain14a.html) | Relates regularized/denoising reconstruction to data-distribution structure under particular assumptions, including small-noise limits. | Denoising is motivated, but our observational noise is substantial; no claim that reconstruction residual is an exact density or score estimator. |
| [Ruff et al., Deep SVDD](https://proceedings.mlr.press/v80/ruff18a.html) | Learns representations with a one-class compactness objective, with architectural restrictions needed to avoid degenerate solutions. | Do not blindly shrink all 101 valid cosmologies to one point. Retain reconstruction/physics tasks and inspect variance/effective rank. This study does not implement or reproduce Deep SVDD. |
| [Ren et al., likelihood ratios for OOD](https://arxiv.org/abs/1906.02845) | Raw likelihood can be dominated by background statistics; reference-model likelihood ratios can reduce that problem. | Keep observational noise identical in controls and synthetic stress tests; compare meaningful physical deviations rather than trivial noise-level changes. No claimed reproduction of their likelihood-ratio algorithm. |
| [Nalisnick et al., typicality](https://arxiv.org/abs/1906.02994) | High density need not imply typicality in high dimensions. | Diagnose both tails of distance/likelihood, using a fixed normal calibration set. Do not use a test-batch typicality test: the competition requires independent per-map inference. |
| [Bardes et al., VICReg](https://arxiv.org/abs/2105.04906) | Separates view agreement, variance preservation, and covariance regularization to prevent collapsed self-supervised embeddings. | Relevant optional next experiment: independent observational-noise views of the same clean map. Noise stability is measured now; a faithful VICReg training experiment is not yet implemented. |

## Additional primary sources

- [Zong et al., DAGMM](https://bzong.github.io/doc/iclr18-dagmm.pdf) combines learned representations with reconstruction information and jointly trains a density model. Our latent-plus-residual ablation is inspired by this combination, but uses separate representation training and density fitting; it is not a reproduction of DAGMM.
- [Cheng et al., scattering for observational cosmology](https://arxiv.org/abs/2006.08561) motivates fixed multiscale nonlinear summaries that retain non-Gaussian structure. Their weak-lensing parameter-inference results do not establish FAIR Phase-2 OOD performance.
- [Cheng et al., HSC Y1 scattering constraints](https://arxiv.org/abs/2404.16085) provides a real-survey application and reinforces the need to handle photometric-redshift systematics. A faithful masked scattering implementation is a next experiment, not something implemented in our 72-feature extractor.

Public competition-related repositories were also inspected. The [Denario repository](https://github.com/ParallelScience/denario-3-weak-lensing-phase2) suggests conditional scattering flows, but its README mixes a different submission schema, outdated test naming, and unverified claims about hidden OOD construction. Its reported number is not used as a verified current leaderboard comparison or as evidence of the hidden test mechanism. Organizer documentation remains authoritative.

## Final empirical update

Three global bottleneck CNNs completed 20,000 updates; four spatial CNN variants completed 12,000; two summary MLPs completed 10,000. Spatial reconstruction NMSE reached 0.588 versus 0.931–0.962 for global models. That improvement did not make the spatial latents the best synthetic anomaly detectors. The fixed physical features remained stronger on the tested spectral/peak perturbations.

Joint supervision improved the summary representation's normal parameter probe relative to reconstruction-only. Adding residual features changed synthetic detection only modestly. Minimum standardized cluster distance improved mild spectral/anisotropic sensitivity relative to mixture likelihood, but reduced noise repeatability and did not eliminate per-cosmology false-positive disparities. See REPORT.md and the artifact CSVs for all measured comparisons.

## Pseudo-testing protocol

- Clean maps receive the official shape-noise distribution. All transformations act on the clean signal before noise.
- Development probes: mild suppression/enhancement of small-scale structure, a variance-matched nonlinear peak change, and blending same-cosmology independent realizations with variance matching. Two strengths are fixed in advance.
- Held-out probe families: anisotropic filtering and spatial modulation. Evaluate once after candidate selection. These are synthetic diagnostics, not organizer OOD physics, even when motivated by physical statistics.
- Changing cosmological or nuisance parameters within their valid ranges is NOT labeled anomalous. Normal audit false positives are reported by cosmology.
- All candidates see identical held-out normal map IDs and deterministic noise seeds. Report noise repeatability, normal tail counts, and separate per-family diagnostics, not only a pooled score.
- A normal-only validation split selects checkpoint/density complexity. A separate calibration split fits score scales; a separate audit split measures resulting normal false positives. Any final synthetic-family evaluation uses the audit only after freezing variants.
- The public test dataset is never used for fitting, clustering, normalization, choice of architectures/weights, or threshold setting. Only final independent inference is permitted.

## What cannot be inferred locally

Normal-only metrics and synthetic probes do not establish true-positive rates for the organizer's unknown alternative physics. Improvements here can justify a candidate, not a SOTA claim. Official submissions contain continuous anomaly predictions; we do not possess hidden ground-truth anomaly labels.
