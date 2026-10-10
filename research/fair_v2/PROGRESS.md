# FAIR Universe Phase 2 (OoD detection): v3 progress

As of 2026-10-10 18:30 UTC. Branch `agent/fair-universe-improved`, worktree
`/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-fair-universe-improved`, branched from `agent/fair-universe-momo` (6833f76e).
The competition deadline is **2026-10-11 23:59 UTC**. Limits: 5 submissions per day, 100 total.

## Leaderboard history

| Submission | What it is | LB score |
|---|---|---|
| previous team, candidate 09 | summary flows + field flows + regional detector | 0.5253 |
| `v3-01-wavelet-cnn-mean.zip` | mean of z: wavelet-stats likelihood (Sd) + CNN-embedding likelihood (C) | 0.5272 |
| `v3-02-consistency.zip` | (Sd + C + 0.5 xSL + 0.5 consP8) / 3 | 0.5669 |
| **`v3-04-den-cons1.zip`** | **(Sd + C + Dr + 1 xSL + 1 consP8) / 5** | **0.6461** |
| `v3-03-stats-cnn-den-cons.zip` | (Sd + C + Dr + 0.5 xSL + 0.5 consP8) / 4 (attribution test) | not submitted |
| `v3-05-den-consRbands.zip` | (Sd + C + Dr + 1 xSL + 1 consRbands) / 5 | not submitted |
| `v3-06-den-noise-consR.zip` | v3-05 + N2 (noise battery) + K (peak clustering) + PsBh (spectrum + bispectrum) | not submitted |
| `v3-07-BIG_RE2.zip` | Dr 1 + N2 1 + PsBh 0.5 + xSL 1 + consRbands 2 (Sd and C dropped); chosen by leave-cosmology-out metric 0.4292 | pending |

The leader is at 0.7835 (others 0.7823, 0.7729, ...). User dkn20, at 0.7358, is very likely the first author of the
organizers' flow-matching OoD paper.

Submission zips are in `/scratch/gpfs/ZHUANGL/sk7524/fair-data/submissions/`. Each contains a single `result.json`
with 10000 finite scores, written by `export.py`. sha256 prefixes:
- v3-01 c841bb9f9b971e46
- v3-02 292fcde652720d3c
- v3-03 5fadaee9f6f322a4
- v3-04 1771471c3f57b2c5
- v3-05 51a0c861eaecb365
- v3-06 a370c2f52f204a7f

Score reading: if a fraction q of OoD maps is caught cleanly and the rest look normal, score ≈ q + 0.0125 (1 - q).
Both 0.525 pipelines caught about 52% of OoD maps.

## Key findings

1. **Each realization index has a fixed baryon model.** (T_AGN, f0) is identical across all 101 cosmologies for a
   given index; only 256 baryon models exist. Realization-level splits, including the previous team's, therefore hold
   out whole baryon models, and the previous detector was fit on 160 of the 256. v3 uses map-level splits
   (`data.index('fit3'|'calib3'|'audit3')`) with all 256 models in the fit set.
2. **The conditional likelihood network memorized baryon models.** It was overconfident on unseen ones: the 99.9th
   percentile of normal energy was about 5.9 million on unseen models vs 234 on seen ones. Width 256 trained for 3000
   steps fixes this (`baryon_cv.py`).
3. **The normal tail decides the metric.** About 60% of the weight sits at FPR < 0.01. A Gaussian copula on the
   residuals plus an eigenvalue floor tightened the tail (`model.py`).
4. **Profiling absorbs anomalies** (`oracle.py`, `absorb.py`). At the true parameters our features detect IA-like and
   baryon-swap shifts well (0.4-0.87). After profiling over (Om, S8, T_AGN, f0, Dz) they drop to 0.03-0.38: the fitted
   parameters mimic the anomaly. IA-like shifts move to lower S8; baryon swaps move T_AGN and f0.
5. **Consistency tests recover part of the absorbed signal** (consistency agent). Parameters are estimated separately
   from scale bands and statistic families and scored on their disagreement (xSL, consP8, consRbands). On the LB this
   was worth about 2x what the synthetic proxies predicted.
6. **The LB rises as the share of the base likelihoods (Sd, C) falls** (100% → 67% → 40%). The leading hypothesis
   is that the test's normal maps sit at unseen cosmologies or baryon draws, where the base detectors (especially the
   CNN, whose leave-cosmology-out tail is about 2.5x heavier) raise false alarms. Robust members dilute them.
   Leave-cosmology-out weighting is being run now.
7. **Web research.** The organizers originally planned 6 systematics for the test: baryons, IA, photo-z, shear bias,
   PSF, and source clustering. Training contains only baryons and photo-z. No public source names the OoD categories.
   Final ranking runs our packaged `model.py` on a private holdout.

## Leave-cosmology-out (LCO) proxy

The proxy holds out 20 of the 101 cosmologies, in 3 disjoint sets, and refits every member without them.
- Normal maps: maps from the held-out cosmologies.
- Anomalous maps: the synthetic shifts of those maps.

Its ordering matches the leaderboard where audit3 did not:

| Recipe | LCO metric | LB |
|---|---|---|
| BASE | 0.3757 | 0.527 |
| REC1 | 0.3929 | 0.567 |
| v3-04 | 0.4126 | 0.646 |

That is roughly 3.2 LB per LCO point. BIG_RE2 scores 0.4292 on LCO, which would be about 0.70 on the LB if the
relation stays linear. Cosmology input jitter and smaller nets did not help.
Details: `agents/consistency/` (`holdout3.py`, `lco3_*.py`, `export3.py`).

## Code map (`research/fair_v2/`)

- `data.py`: data paths, splits (v2 realization-level, v3 map-level), noise, and 19 synthetic diagnostic shifts.
  These include HMcode baryon extrapolation (`tk_grid.npz` from the public toolkit), source clustering, Poisson noise,
  an IA-like shift, and a PSF pattern.
- `stats.py`: 466 per-map statistics: power spectrum, PDF, peaks/minima, Minkowski V1/V2, moments, DoG wavelets.
- `cnn.py`, `pipeline_v3.py`: CNN parameter regressor, and aligned extraction of stats and embeddings for every set
  using identical noise seeds.
- `model.py`: conditional Gaussian likelihood, profile, copula, standardizer, and the official metric.
- `fit_v3.py`: fit, evaluate, and test-score any feature source. `merge.py` / `combine.py`: cross-run combinations.
- `oracle.py`, `absorb.py`, `baryon_cv.py`, `check_independence.py`: diagnostics.
- `package.py` + `pack/`: Codabench code submission (`model.py` with `Model.predict` on a float16 memmap).
  Verified against pipeline scores and per-map independence; about 250 s per 10k maps on a GH200.
- `agents/`: shared brief (`BRIEF.md`), each agent's code, and copies of their reports in `agents/reports/`.

## Data and artifacts (not in git)

All under `/scratch/gpfs/ZHUANGL/sk7524/fair-data/`:
- Raw data, with the public test SHA-256 verified (`1b854da9...`).
- `v3/<source>/` feature sources: `stats`, `cnn0`, `cnn1`, `den*`, `stx_*`, `dif*`, `oe*`.
- `v3/runs/` fitted runs. `agents/<name>/` agent outputs.
- `merged/` merged test z.
- `submissions/` zips.

Environments: `fair-env` (x86, A100) and `fair-env-gh` (aarch64, della-gh GH200).

## Rules kept

Every score is per map. Batches are fixed-shape and run in eval mode; scoring a map alone vs in a batch differs by
≤ 0.003. Nothing was fitted, calibrated or selected on test data. Only frozen per-map test scores were produced.
Leaderboard feedback is used for choosing among submissions.

## Next

- Outlier-exposure and diffusion agents report about 20:30 UTC.
- Consistency agent: leave-cosmology-out weighting and cosmology jitter.
- Build the "big swing" submission for today's last slot before 23:59 UTC.
- Package the final method with `package.py` for the private holdout.
