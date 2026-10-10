# FAIR parameter-supervised denoising autoencoder research

Authorized machines: erdos-3242 and erdos-3191 only. Preserve holding jobs 2306 and 2317. Large remote data stay under /dev/shm/fair-ae-research. Do not use other pools/services or production cache paths.

Goal: three frozen-model result.json prediction ZIPs for user upload, with reproducible report. No automatic Codabench submissions.

## Protocol

- Official independent-realization training maps, mask, and five parameter labels; published MD5 verification.
- Reuse immutable seed-2026 160/32/32/32 realization split per cosmology: fit / model-selection / density-calibration / final normal audit.
- Image models train on newly noised maps and compare reconstruction, standardized Omega_m/S8 prediction, and joint losses. Summary models reconstruct clean physical features and optionally predict all five supplied parameters. Checkpoint and density complexity selection use normal validation only; final candidate selection also considers explicitly synthetic development diagnostics.
- Fixed training-derived global signal/input scales and parameter mean/std; no per-map standardization that removes cosmological amplitude.
- Convolutional encoder, compact global bottleneck, decoder without bypass connections, parameter head; per-example normalization only.
- Fixed validation noise seed for comparable checkpoint selection. Report clean-target masked MSE and each parameter RMSE separately as well as weighted objective. Compare zero-map and mean-parameter baselines.
- Fit latent normalization and candidate cluster/density models on training latents. Choose complexity using held-out normal density likelihood and stability; explicitly no claim that this optimizes hidden-OOD accuracy.
- Calibrate anomaly scores against a separate normal calibration set. No test-set clustering, normalization, threshold fitting, ranking, or model selection.
- Freeze each candidate's artifacts and scoring rule before its public-test inference. A preliminary control was exported while other training continued; its test predictions were not used for fitting or selection. Verify singleton/batch/permutation independence and finite 10,000 scores per archive.
- Three submissions must represent documented, predeclared model/scoring variants; preserve exact configuration, source hashes, checkpoints, and normal validation evidence.

Nine learned configurations completed. See REPORT.md for measured results, three upload files, limitations, and reproduction instructions. No official OOD score or SOTA result has been established.
