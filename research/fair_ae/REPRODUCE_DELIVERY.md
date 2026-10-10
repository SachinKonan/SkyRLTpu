# Reproduce the delivered predictions

The standalone delivery has three upload ZIPs, reports, trained models, frozen runtimes, and the complete research source. Nothing here submits to Codabench or modifies a SkyPilot job.

1. Download the official [public-data ZIP](https://www.codabench.org/datasets/download/09b7a27a-bfb3-4031-a5b0-9b3c533f45e6/) and extract `public_data/`. The directory must contain the mask and current `WIDE12H_bin2_2arcmin_kappa_test_phase2_new_v2.npy`. Keep large files in RAM-backed storage on a TPU lab VM.
2. Use the package versions in `requirements-observed.txt`. These versions were tested in the supplied Python 3.12 serving environment; a clean install has not been tested.
3. On an allocated TPU chip you own, set the single-chip bounds below, then run `predict.sh`. If launched through systemd, set `LimitMEMLOCK=infinity`, as the default limit was insufficient.

```bash
export TPU_VISIBLE_CHIPS=0
export TPU_CHIPS_PER_HOST_BOUNDS=1,1,1
export TPU_HOST_BOUNDS=1,1,1
export FAIR_PYTHON=/path/to/the/tested/python
bash predict.sh /path/to/public_data /path/to/new-output
```

The script invokes the correct frozen runtime for each model, validates source/model hashes and per-map independence, produces the three named predictions, and writes receipts. Two `component-*.zip` files are intermediate ensemble inputs, not additional recommended submissions. ZIP timestamps may differ on reproduction; compare parsed scores and receipts' model hashes rather than expecting an identical archive-byte hash across export times. Numerical equality on a different JAX/hardware stack has not been established.

The inference recipe requires no fitting and uses no test-derived statistics. Models' stored original training paths are provenance; export does not require those original paths. For retraining, use `RUN.md`, the archived configuration/source files, and the original training data. Best weights were saved, but optimizer resume state was not.

The `models/` directory includes normal calibration/density arrays. Runtime `v2` scores physical likelihood, `v4` scores the joint-summary model, `v5` scores spatial reconstruction latents, and `v6` adds the nearest-cluster-distance rule. Do not replace one runtime with another without refitting or explicitly versioning the change: hashes intentionally reject silent source drift.

To inspect every trained configuration, extract `../all-experiment-artifacts.tar.gz`. Its `host0`–`host3` directories preserve all nine full training runs, four intermediate spatial evaluations, derived score variants, source snapshots, normal latent caches, and reports. Rebuild aggregate CSVs from that extracted directory with:

```bash
python research-source/report_data.py --snapshot /path/to/extracted-artifacts --out /path/to/new-reports
```

The primary research report is `../REPORT.md`. `../reports/paired-bootstrap.json` contains synthetic-only uncertainty estimates. There are no hidden true anomaly labels and no official competition score in this delivery.
