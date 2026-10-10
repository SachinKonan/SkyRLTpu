# FAIR Universe: representation pipeline proposal and evidence review

Prepared 2026-10-10, approximately 18:34 UTC, for discussion and handoff.

Worktree: `/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-fair-universe-improved`

Branch: `agent/fair-universe-improved`; inspected HEAD: `dfe0a673`.

This records the user's proposal, the completed experiments inspected in this conversation, and recommended follow-up work. The worktree is being developed concurrently, so statements about missing implementations describe the inspected snapshot. This document does not report new training or new competition submissions.

## 1. The user's proposed experiment

The user proposed separating representation learning from physical-parameter prediction, then using the resulting predictions to assess whether a map is unusual.

1. **Learn representations through reconstruction.** Compare multiple autoencoders, including standard, denoising, masked, and variational variants. Explore training augmentation and a Vision Transformer encoder. Encode each map into a vector, or a distribution over latent vectors, and reconstruct the map.
2. **Freeze the encoders and predict the five physical parameters.** Start with linear predictors to measure how much physical information each representation contains. Train a model that combines the representations or their predictions into an ensemble.
3. **Score new maps.** Pass each map through the frozen encoders and parameter ensemble. Assess how unusual its inferred physical parameters are relative to the normal training distribution, accounting for prediction uncertainty.

These choices are separate experimental axes: a ViT is an encoder architecture; denoising and masking are training objectives/corruptions; a VAE introduces a probabilistic latent model. They can be combined. We have not completed this controlled comparison.

## 2. A necessary correction to the generative picture

The five labels are physical generating parameters, not five summary statistics:

| Parameter | Meaning |
|---|---|
| Omega_m | Matter abundance |
| S8 | Clustering amplitude |
| T_AGN | Parameter controlling baryonic feedback |
| f0 | Another parameter controlling baryonic effects |
| Delta_z | Photometric-redshift shift |

A useful schematic is `map = simulator(theta, random_realization) + observation_noise`, with the survey mask applied. The five-dimensional vector `theta` specifies a family of map distributions. Random initial conditions and the realized matter structures also matter. The five labels do not uniquely determine an image or prove that its reconstructive latent space is five-dimensional.

This supports learning a compact representation, while allowing it to retain both physical information and information about the particular spatial realization. A VAE's latent posterior is a distribution over learned codes; its coordinates and uncertainty are not automatically the five physical parameters or calibrated uncertainty about them.

Sources: [competition simulation description](https://arxiv.org/html/2604.14451v1#S2), [VAE formulation](https://arxiv.org/html/1312.6114v11#S2).

## 3. Data and evaluation contract

- Normal training data: 25,856 noiseless maps, from 101 cosmologies with 256 realizations each. Each map has five physical labels.
- Image geometry: 1424 x 176 pixels, with 132,019 observed pixels in the supplied mask.
- Observation noise is added to training maps. The public test maps already contain noise.
- The public test has 10,000 maps without physical labels or normal/anomalous labels. No organizer-provided labeled anomaly training set is available.
- All model fitting, clustering, normalization, calibration, and ensemble training must use training-derived data. Public test inputs are used for frozen per-map inference.
- The competition score averages true-positive rates at 100 logarithmically spaced false-positive rates between 0.001 and 0.05. It is not ordinary classification accuracy.

Each realization index carries the same T_AGN/f0 pair across all 101 cosmologies: there are 256 such pairs. This parameter reuse does not mean the new training maps share simulation random seeds. Splitting by realization holds out whole baryon parameter pairs. The newer map-level split puts all 256 pairs in fitting data. Both split types answer useful, different questions; the latter does not establish generalization to unseen parameter combinations.

## 4. What exists today

| Proposed component | Observed implementation/evidence | Remaining work |
|---|---|---|
| Reconstruction representations | Global image AEs, spatial image AEs, and summary-feature AEs; reconstruction-only and joint-supervision variants | Controlled comparison across AE objectives and encoder architectures |
| Denoising | Fresh observational noise for original image training; newer U-Net denoiser with completed training and residual features | Compare objectives and training budgets fairly |
| Masked training | Summary-feature models drop 5% of inputs and add small noise | Random image-patch masking/MAE experiment |
| VAE and ViT | No completed experiments found in the inspected code/results | Implement, train, and evaluate |
| Geometric augmentation | No random image rotation, flip, or crop augmentation in the checked training paths | Controlled augmentation experiment preserving mask and physical scale |
| Frozen latent to physical parameters | Separate linear probes fitted to each representation for all five labels | Trained ensemble of reconstruction representations; nonlinear controls |
| Parameter uncertainty | Point-prediction error measured | Calibrated intervals/joint regions for the proposed ensemble |
| Parameter rarity | Prototype fits a density to inferred parameters on normal calibration maps | Establish usefulness and robustness beyond synthetic diagnostics |
| Map compatibility | Conditional feature likelihoods and profiled goodness-of-fit scores | Extend to proposed representations and check misspecification |
| Parameter agreement | Cross-scale and cross-feature consistency tests | Evaluate additional representations without hiding disagreement through averaging |

The established ensembles mostly combine **anomaly scores**. They are not the trained ensemble of frozen reconstruction latents for five-parameter prediction proposed by the user.

## 5. What the original model names mean

- **Global image latent:** the encoder turns the entire image into one vector of 32, 64, or 128 numbers. The decoder reconstructs the image from that vector.
- **Spatial image latent:** the encoder retains a smaller spatial grid with 16 feature channels. Reconstruction uses that grid. For anomaly scoring and the linear probes, channel means, log standard deviations, skewness, and kurtosis yield a separate 64-number summary.
- **Joint:** reconstruction and physical-parameter prediction are trained together with a weighted loss.

The spatial model has a much larger internal representation than the global vector model. Their reconstruction results are not a comparison at equal compression.

The original study completed nine learned configurations:

| Family | Configurations |
|---|---|
| Global image models | Latent 32, 64, 128; joint losses with parameter weights 0.10, 0.25, 0.50 respectively |
| Spatial image models | Reconstruction only; parameter prediction only; joint; more compressed joint |
| Summary-feature models | 72 physical features to a 32-number latent; reconstruction only or joint prediction of all five parameters |

The parameter-only model is a control, not a trained reconstruction model. Global latent dimension and supervision weight changed together, preventing a clean attribution to latent dimension. Original image-model heads supervised only Omega_m and S8; the joint summary model supervised all five labels.

The supplied starter separately included a model reconstructing its own downsampled noisy input. It was not a matched standard-versus-denoising control for the later full-resolution study. Its historical proxy evidence should not be treated as an official competition score.

## 6. Training and convergence evidence

The original study used 16,160 fitting maps and separate sets of 3,232 maps each for validation, calibration, and audit. Image training used batches of 16, fresh Gaussian observation noise, and clean reconstruction targets. Input/label scales were fitted on training data. AdamW used weight decay 1e-4, gradient clipping, and a warmup/cosine learning-rate schedule.

Image validation ran every 1,000 updates on a fixed 1,024-map subset with fixed noise. The best checkpoint minimized the model's specified validation objective. Training stopped at fixed budgets, not an automatic convergence criterion. Global models completed 20,000 updates and spatial models 12,000. Summary models completed 10,000 updates with batches of 128 cached feature vectors, fresh feature corruption, and validation every 500 updates. All nine used seed 0.

| Model | Saved evidence | Assessment |
|---|---|---|
| Global models | Best joint objectives at updates 9k, 12k, 14k; no later best by 20k | Broad flattening under the chosen schedules, not proof of optimality |
| Spatial reconstruction | Reconstruction NMSE 0.5930 at 8k -> 0.5882 at 12k | Still improving, with diminishing gains |
| Spatial joint | Joint objective 0.6662 at 8k -> 0.6594 at 12k | Still improving |
| Summary joint | Best objective 0.17120 at 6k; 0.17153 at 10k | Broad flattening |

Compare losses within a row; different objectives/scales prevent comparing these values as one common loss. We have not established that longer training or better augmentation cannot improve the spatial AEs.

Sources: [original report](../../../fair_ae/REPORT.md), [image training](../../../fair_ae/train.py), [summary training](../../../fair_ae/summary_train.py). Saved curves are under `/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-fair-universe/.fair/ae-research-evidence/snapshot-05/`.

## 7. Do the latents predict the five physical parameters?

Yes, especially Omega_m and S8. We froze each encoder, fitted a separate ridge linear predictor on fitting representations, and evaluated on the normal audit split. These are normalized mean-squared errors: zero is perfect; around one means little improvement over predicting the average, with exact interpretation depending on the audit distribution and training normalization.

| Representation | Omega_m | S8 | T_AGN | f0 | Delta_z |
|---|---:|---:|---:|---:|---:|
| Handcrafted physical features | 0.1195 | 0.1945 | 0.7401 | 0.4159 | 0.9053 |
| Global joint image latent, 128 dimensions | 0.1609 | 0.2586 | 1.1581 | 1.1184 | 0.9570 |
| Spatial image, reconstruction only | 0.2271 | 0.2511 | 0.9176 | 0.9489 | 0.9142 |
| Spatial image, joint | 0.1346 | 0.2129 | 0.9093 | 0.9471 | 0.9122 |
| Spatial image, parameter-only control | 0.1304 | 0.2074 | 0.9779 | 0.9771 | 0.9152 |
| Summary latent, reconstruction only | 0.1044 | 0.1918 | 0.8202 | 0.4457 | 0.9083 |
| Summary latent, joint | 0.0945 | 0.1847 | 0.7505 | 0.3592 | 0.9254 |

Reconstruction-only representations retained useful physical information. Explicit supervision improved several errors. Summary representations performed better than the image representations on these probes. This does not establish better unknown-anomaly detection, and weak linear predictability does not prove that information is absent from a latent.

The probes fitted all five labels but selected ridge strength using only the mean validation error for Omega_m/S8. A new all-five comparison should make that choice explicit and evaluate appropriate alternatives. These audit maps used the same 101 cosmologies as fitting, so this is not leave-cosmology-out evidence.

Evidence: [probe implementation](../../../fair_ae/probe.py). Full five-parameter arrays are in the per-model `linear_probe.json` files under the snapshot-05 directory above; the condensed report is [normal-and-training.csv](/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-fair-universe/.fair/deliverables/fair-2026-10-09/reports/normal-and-training.csv).

## 8. Latest recorded competition progress

The following are reported leaderboard results from [PROGRESS.md](../../PROGRESS.md), dated 2026-10-10 18:30 UTC. This review did not authenticate to Codabench to verify the scores independently.

| Submission | Recipe | Reported score |
|---|---|---:|
| Original candidate 09 | Summary flows + field flows + regional detector | 0.525304, originally supplied by the user |
| v3-01-wavelet-cnn-mean.zip | (Sd + C) / 2 | 0.5272 |
| v3-02-consistency.zip | (Sd + C + 0.5 xSL + 0.5 consP8) / 3 | 0.5669 |
| v3-04-den-cons1.zip | (Sd + C + Dr + xSL + consP8) / 5 | 0.6461 |

Definitions: `Sd` is the wavelet-statistics compatibility score; `C` is the CNN-feature compatibility score; `Dr` measures denoiser residual structure; `xSL` tests large/small-scale parameter compatibility; `consP8` measures disagreement among eight parameter estimates from feature families/models. Components are standardized before combination.

The v3 gain is +0.1189. The last transition changes both the denoiser component and consistency weights, so it does not isolate the denoiser's causal contribution. The report lists v3-03 as an unsubmitted intermediate attribution comparison. It also lists v3-05 and v3-06 as unsubmitted at that snapshot.

The v3-01, v3-02, and v3-04 archives were inspected: each contains only `result.json`, exactly 10,000 finite `ood_scores`, and matches its recorded SHA-256 prefix (`c841bb9f9b971e46`, `292fcde652720d3c`, `1771471c3f57b2c5`). They live in `/scratch/gpfs/ZHUANGL/sk7524/fair-data/submissions/`. This verifies artifact identity/format, not the reported leaderboard score.

The newer U-Net denoiser report records 30 epochs, selection of epoch 26, and reconstruction R-squared about 0.47. Its residual features helped synthetic noise/systematic shifts; its reconstructed-image statistics alone were weaker. The current evidence favors retaining what reconstruction fails to explain, alongside the latent. See [denoiser report](../reports/denoiser.md).

## 9. Refining stage 3: preserve three different signals

The user's parameter-rarity idea is useful, but a predictor trained only on normal maps can map an anomalous input to an ordinary physical-parameter vector. A valid map near the edge of the normal parameter distribution can also produce an unusual estimate.

We should evaluate these separately before combining them:

1. **Parameter rarity:** are the inferred parameters unusual relative to predictions on independent normal calibration maps? Comparing estimated parameters with estimated parameters accounts for prediction bias and noise better than directly comparing point predictions to true labels alone.
2. **Map compatibility:** after allowing the physical parameters to vary, is the observed representation or reconstruction residual still unusual? This retains information beyond five point predictions.
3. **Internal consistency:** do independent representations or scale ranges infer compatible physical parameters, after calibrating their ordinary correlated errors? Keep individual predictions and disagreement even when forming an ensemble mean.

The newer consistency code already implements parts of this. `agents/consistency/final.py` includes a parameter-rarity prototype: a 12-component Gaussian mixture fitted to normal-calibration profiled parameter estimates. It also implements cross-family disagreement and cross-scale compatibility. These are not a completed calibrated posterior for the proposed reconstruction-latent ensemble.

Synthetic oracle/absorption diagnostics report that some anomalies become less detectable when fitted physical parameters adjust to mimic them. The observed leaderboard improvements are consistent with the usefulness of residuals and consistency tests; they do not reveal the hidden anomaly categories. The broader approach is related to [posterior predictive consistency testing](https://arxiv.org/html/2505.00632v2#S2.SS1).

## 10. How to validate uncertainty

For each map, the proposed parameter model should return a point estimate plus intervals or a joint region. On untouched labeled normal maps, a claimed 90% interval should contain the corresponding true parameter approximately 90% of the time. Repeat at multiple nominal levels.

- Separate representation/predictor fitting, uncertainty calibration, and final audit data. Keep repeated noise views of the same clean map together.
- Check each parameter's coverage and interval width. Very wide intervals can achieve coverage without being informative.
- Check simultaneous coverage for the full five-parameter vector; five individual 90% intervals are not a 90% joint region.
- Check coverage across cosmological and nuisance-parameter groups and on parameter-held-out stress splits. Good overall coverage can conceal poorly calibrated subgroups.
- Compare ensemble spread with actual errors. Agreement among models is not sufficient: they can share a bias.
- If using a VAE, validate uncertainty after mapping latent samples to physical parameters, including predictive errors the latent sampling does not capture.

Conformal calibration is one possible interval-calibration method; its standard coverage guarantee requires exchangeability with calibration data and is marginal rather than a guarantee for every parameter regime. Our finite simulation design and group splits need explicit empirical checks. See [conformal uncertainty reference](https://arxiv.org/html/2107.07511v6#S2.SS3).

The old probes measured point-prediction errors, not calibrated intervals. Normal-data calibration also does not prove reliable uncertainty on OoD inputs. A normal-tail percentile is not a posterior probability that an input is anomalous.

## 11. Recommended experiment sequence

This is a proposal for subsequent work, not an execution record.

1. **Preserve the 0.6461 reference and make it reproducible.** Freeze its exact components, weights, checkpoints, normal calibrators, source versions, and prediction archive. Complete the code-submission packaging described below.
2. **Establish a controlled reconstruction baseline.** Compare reconstruction-only spatial CNN models with matched data, budgets, validation, and multiple seeds. Extend training enough to assess the still-improving curves. Keep a matched noisy-input reconstruction control and clean-target denoising model.
3. **Add representation variants in stages.** Test image masking, VAE objectives, and a ViT encoder without changing every factor simultaneously. Geometric augmentations should transform map, target, and mask consistently and preserve physical pixel scale; cropping needs an explicit treatment of lost context and map-level aggregation.
4. **Freeze encoders and fit the parameter ensemble.** Compare per-encoder ridge predictors, a predictor on concatenated latents, and a learned combination of per-encoder predictions. Use held-out or out-of-fold predictions for training any stacking layer. Compare nonlinear predictors and report all five errors.
5. **Calibrate uncertainty and test the three stage-3 signals.** Measure parameter rarity, map/residual compatibility, and prediction disagreement separately. Compare additions against the existing winning recipe using normal audit data and diverse synthetic probes before spending submission slots.

Select encoder checkpoints using normal validation objectives. Improved reconstruction or parameter prediction is not, by itself, evidence of improved OoD detection. Report proxy performance separately from competition feedback. Changing the split or training budget must be documented so improvements are interpretable.

## 12. Packaging gap and interpretation corrections

At the inspected snapshot, `package.py` and `pack/model_template.py` support statistics/CNN feature extraction and a mean of fitted detector scores. They do not include the U-Net residual extractor or the cross-family/cross-scale consistency calculations required by v3-04. The existence of its prediction ZIP does not establish a runnable code package for new private-holdout maps. The earlier package parity/runtime claims should not be generalized to the full winning recipe without checking that recipe end to end.

Before delivery, the complete recipe needs packaging of all required weights/calibrators and score agreement checks against the frozen pipeline, including per-map behavior and runtime/memory measurements.

Other claims to keep qualified:

- A score around 0.525 does not establish that a fixed 52% of anomalies were detected, or that two methods detected the same examples. That interpretation assumes a special mixture model not established by the score.
- The association between lower base-detector weight and higher leaderboard scores is not proof about the hidden test's cosmologies. New components and weights changed together.
- Improved behavior with a smaller likelihood network is not proof that unseen-parameter generalization is solved; retain parameter-held-out checks.
- A supervised linear classifier's proxy result is a useful diagnostic, not a mathematical upper bound on all possible anomaly detectors.
- Synthetic anomaly families are diagnostic hypotheses. They are not verified descriptions of the competition's hidden OoD generation process.

The evidence supports testing the user's reconstruction-and-ensemble direction while preserving the map information that five predicted parameters can leave unexplained.
