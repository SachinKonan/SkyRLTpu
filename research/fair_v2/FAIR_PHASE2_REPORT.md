# FAIR Universe Phase 2 (out-of-distribution detection): full report

Status as of **2026-10-11, 20:35 UTC**, about 3.5 hours before the 23:59 UTC deadline.

**Best leaderboard score: 0.7037** (`v3-16b`). We started this effort at 0.5253; the leader is at 0.7835. We have
closed 0.178 of the original 0.258 gap (69%). One submission is left today (`v3-16c`, prepared). The final code
package is `packages/v3-16b.zip`, or `packages/v3-16c.zip` if v3-16c scores higher.

---

## 1. The challenge

### 1.1 Task

Each test sample is a simulated weak-lensing convergence map. Some test maps come from the same physics as the
training data ("in-distribution", InD). Others contain physics or systematics never seen in training
("out-of-distribution", OoD). For every map we output one number, an OoD score: higher means more likely OoD.

### 1.2 Data

| Item | Value |
|---|---|
| Training maps | 101 cosmologies x 256 realizations = **25,856** labelled maps (clean, no noise) |
| Map size | 1424 x 176 pixels, 2 arcmin per pixel |
| Survey footprint (mask) | **132,019** of 250,624 pixels (52.7%); pixels outside are 0 |
| Labels per map | 5 parameters: Omega_m, S8 (cosmology), T_AGN, f0 (baryonic feedback), Delta_z (photometric-redshift shift) |
| Baryon models | Each realization index has a fixed (T_AGN, f0) pair, so only **256** baryon models exist |
| Shape noise | White Gaussian, sigma = 0.4 / sqrt(240) = **0.0258** per pixel. Signal rms is about 0.010-0.013, so per pixel the noise is about 2x the signal |
| Public test set | **10,000** maps, already noisy, stored as float16, shape (N, 132019) |

### 1.3 Metric

The leaderboard score is the mean true-positive rate (TPR) at **100 log-spaced false-positive rates (FPRs) between
0.001 and 0.05**:

    score = (1/100) * sum_i TPR(FPR_i),   FPR_i log-spaced in [0.001, 0.05]

What this means in practice:
- **Only the ranking matters.** The organizers set each threshold so that a fixed fraction of the normal maps lands
  above it. Any monotone transform of our score gives the same result.
- **The strictest thresholds dominate.** log(0.01/0.001) / log(0.05/0.001) = 59%, so **about 59% of the 100 points sit
  below FPR 1%**. The score is decided by the most extreme normal maps (the top 0.1% to 5% of normal scores) and by
  how many OoD maps rank above them.
- **What hurts most** is a group of normal maps that our method finds odd, for example normal maps from universes that
  differ from the training ones. They use up the allowed false alarms at every strict threshold and raise the bar.
- A rough reading from our earlier analysis: if a fraction q of OoD maps is caught cleanly and the rest look normal,
  score ≈ q + 0.0125 (1 - q).

### 1.4 Rules and logistics

| Item | Value |
|---|---|
| Per-map requirement | Every score must be a function of that map alone. No transductive methods: the public test may not be used to fit, calibrate, normalize, rank, select or adapt |
| Leaderboard submissions | A zip with one `result.json` holding 10,000 finite `ood_scores` |
| Submission limits | 5 per day, 100 in total |
| Final ranking | The organizers run our packaged `model.py` (`Model().predict` on a float16 memmap) on a **private holdout**. Execution limit about 1200 s. Scoring hardware is not published |
| Deadline | 2026-10-11, 23:59 UTC |
| Leaderboard top (Oct 10) | 0.7835, 0.7823, 0.7729, 0.7685, 0.7571; a cluster at 0.60-0.65 |

Public background (research agent): the organizers' first README planned 6 systematics for the test set: baryons,
intrinsic alignment (IA), photo-z, shear measurement bias, PSF, and source clustering. Training contains only baryons
and photo-z. No public source names the actual OoD categories.

---

## 2. Starting point (before 2026-10-10)

The previous team's best was **0.5253** (candidate 09, "cosmology-corrected spatial": summary flows + field flows +
a regional residual detector). Their synthetic probes had improved, but the leaderboard had not.

Our review found four failure points:

| # | Finding | Evidence | Consequence |
|---|---|---|---|
| 1 | **Baryon models leaked through the split.** Each realization index has a fixed baryon pair, so realization-level splits hold out whole baryon models. The previous detector was fit on 160 of 256 | `data.py`, label inspection | We moved to map-level splits with all 256 baryon models in the fit set |
| 2 | **The conditional likelihood network memorized baryon models.** 99.9th percentile of normal "energy" was about **5.9 million** on unseen baryon models vs **234** on seen ones | `baryon_cv.py` | Width 256, 3000 steps fixed it |
| 3 | **The normal-map tail decides the metric** (59% of the weight below FPR 1%) | metric analysis | Gaussian copula on residuals plus an eigenvalue floor (0.02) tightened the tail |
| 4 | **Profiling absorbs anomalies.** At the true parameters our features detect IA-like and baryon-swap shifts well (0.4-0.87). After fitting the 5 parameters per map they drop to **0.03-0.38**: the fit moves to parameters that mimic the anomaly (IA-like -> lower S8; baryon swaps -> T_AGN, f0) | `oracle.py`, `absorb.py` | Motivated the consistency (disagreement) members |

---

## 3. Our method

### 3.1 Splits

| Split | Share | Maps (≈) | Used for |
|---|---|---|---|
| `fit3` | 80% | 20,700 (x2 noise views) | Training every model ("what is normal") |
| `calib3` | 10% | 2,580 (x2 views) | Setting each member's scoring scale; frozen |
| `audit3` | 10% | 2,580 | Checking; synthetic shifts are applied to these maps |

Map-level split with a fixed seed (2027), so all 101 cosmologies and all 256 baryon models appear in every part.
Noise is added to the clean training maps with fixed seeds, quantized to float16 like the test maps, and dumped once,
bit-exactly, so every member sees identical noisy maps.

### 3.2 Pipeline in plain words

1. **Add noise** to the clean training maps (test maps are already noisy).
2. **Measure each map** with several feature families (Section 3.3).
3. **Learn "normal" per family:** a small network maps the 5 parameters to the expected value and spread of every
   feature (a Gaussian per feature, trained by maximum likelihood on `fit3`, with a little input jitter). The
   co-variation of the leftovers (actual minus predicted, in spreads) is estimated once per family: a Gaussian copula
   plus an OAS-shrunk correlation matrix with an eigenvalue floor of 0.02.
4. **Score a new map, two ways:**
   - **Misfit:** gradient descent (Adam, 3 restarts) over the 5 parameters finds the setting under which the map's
     features are most likely, with a Delta_z prior and a soft penalty keeping the search inside the training range.
     The misfit is the remaining distance between the map's features and the prediction at that best fit, accounting
     for each feature's spread and correlations.
   - **Disagreement:** each family (and each scale band) gives its own parameter estimate. Normal maps give consistent
     estimates; distorted maps often do not. Measured as a Mahalanobis distance of the estimates (covariance from
     `calib3`).
5. **Put scores on one scale:** each member's raw value becomes a z-score relative to normal `calib3` maps
   (a smooth function of the fitted parameters: a Standardizer quadratic in theta-hat).
6. **Combine:** final score = weighted mean of the members' z-scores.

### 3.3 Members

| Member | What it measures | Kind | Source / dims |
|---|---|---|---|
| **Sd** | Wavelet / difference-of-Gaussians scale-space statistics | misfit | `stats:dog` |
| **C** | Embedding of a CNN trained to predict the 5 parameters | misfit | `cnn0` |
| **Dr** | Residual of a denoising U-Net (map minus denoised map) vs what pure noise gives | misfit | `den_r` (112) |
| **N2** | Noise-structure battery (high-pass moments, tile variances, tails) | misfit | `stx_nois` (columns dropped) |
| **PsBh** | Power spectrum + binned bispectrum | misfit | `stx_bisp` + ps |
| **X** | Power-spectrum anisotropy ("dirpower"): log power in kx- vs ky-dominated wedges, 12 bands | misfit | `dirpower` (36) |
| **Oh** | Outputs of the outlier-exposure (OE) network (OoD logit, 16 family logits, 5 parameter predictions), scored as a family | misfit | `oe_head` |
| **Ol** | The OE network's own "this map is distorted" logit, robust z vs `calib3` (median −0.8047, scale 0.3417) | direct | `oe_head` column 0 |
| **SS** | Diffusion-model small-scale score | misfit | `dn_SS` |
| **xSL** | Large-scale vs small-scale parameter estimates (Dai-Seljak scale consistency) | disagreement | profiles |
| **consP8** | Parameter estimates of 8 families | disagreement | profiles |
| **consRplus** | Parameter estimates of 13 views: 4 scale bands, CNN predictions (x2), DoG, denoiser views, denoiser-bottleneck ridge probe, anisotropy vote | disagreement | profiles + probes |

**The OE network (Oh, Ol).** A CNN trained to tell normal `fit3` maps (fresh noise) from synthetic "alternative
physics" maps: clean `fit3` maps passed through one of 14 generator families before noise is added (`tell`, `nz`,
`shear`, `psf`, `noise`, `mask`, `nonlin`, `hmcode`, `halo`, `ia`, `srcclust`, `poisson`, `psfpat`, `holes`), with
log-uniform random strengths. Loss = normal-vs-distorted + which-family + parameter regression (normal maps only).

### 3.4 Weights of every submitted recipe

| Member | v3-01 | v3-02 | v3-04 | v3-12 | v3-15b | v3-15d | v3-15g | **v3-16b** | v3-16c |
|---|---|---|---|---|---|---|---|---|---|
| Sd | 1 | 1 | 1 | 0.5 | 0.5 | 0.5 | 0.5 | **0.5** | 0.5 |
| C | 1 | 1 | 1 | 1 | 1 | 0 | 0 | **0** | 0 |
| Dr | | | 1 | 2.5 | 2.5 | 2.5 | 2.5 | **2.5** | 2.5 |
| xSL | | 0.5 | 1 | 1 | 1 | 1 | 1 | **1** | 1 |
| consP8 | | 0.5 | 1 | 0.5 | 0.5 | 0.5 | 0.5 | **0.5** | 0.5 |
| consRplus | | | | 3 | 3 (13 views) | 3 | 3 | **3** | 3 |
| N2 | | | | 0.5 | 0.5 | 0.5 | 0.5 | **0.5** | 0.5 |
| PsBh | | | | 0.5 | 0.5 | 0.5 | 0.5 | **0.5** | 0.5 |
| SS | | | | 1 | 0 | 0 | 0 | **0** | 0 |
| Oh | | | | 1.5 | 1.5 | 1.5 | 0 | **1.5** | 1.5 |
| X | | | | | 2 | 2 | 2 | **2** | 2 |
| Ol | | | | | | | | **3** | 6 |
| **Share of base likelihoods (Sd + C)** | 100% | 67% | 40% | 13% | 11.5% | 4.2% | 4.8% | **3.3%** | 2.8% |
| **Leaderboard** | 0.5272 | 0.5669 | 0.6461 | crashed | 0.6667 | 0.6676 | 0.6479 | **0.7037** | pending |

---

## 4. Validation: how we judged changes before the leaderboard

### 4.1 Synthetic shifts (19)

| Shift | What it does | Factory grader |
|---|---|---|
| `mult+0.05`, `mult-0.05` | Shear calibration bias (signal x1.05 / x0.95) | visible / **hidden** |
| `smooth0.5`, `aniso0.7` | Isotropic / anisotropic smoothing (PSF, resolution) | visible / **hidden** |
| `noise+0.05`, `noise-0.05`, `noisegrad0.15`, `noise_t5` | Misestimated shape noise: level ±5%, ±15% gradient, Student-t (df 5) | visible x3 / **hidden** |
| `supp0.15`, `boost0.15` | Small-scale suppression / enhancement beyond a smooth baryon transfer | visible |
| `peakcut0.3` | Mass-dependent peak flattening (BCM-like) | **hidden** |
| `gradient` | Additive large-scale gradient | visible |
| `bary9.0`, `bary6.5`, `baryDMO` | Baryonic feedback outside the training prior (HMcode-2020 ratio) | visible x2 / **hidden** |
| `srcclust0.15` | Noise anti-correlated with local structure (source clustering) | visible |
| `poisson120` | Poisson galaxy counts per pixel | visible |
| `ia0.2` | IA-like: part of the signal replaced by a phase-randomized copy | **hidden** |
| `psfpat` | Additive coherent anisotropic PSF pattern | visible |

### 4.2 Proxies we tried

| Proxy | Definition | Verdict |
|---|---|---|
| audit3 synthetic | Normal `audit3` maps vs their synthetic shifts; cosmologies and baryon models all seen in training | **Failed:** rated v3-02 and v3-04 equal while the LB put v3-04 0.08 higher; undervalued consistency members ~2x |
| **LCO** (leave-cosmology-out) | Hold out 20 of the 101 cosmologies, in 3 disjoint sets; refit every member without them; normal = held-out maps, anomalous = their synthetic shifts; mean metric over the 19 shifts | **Used.** Ranked the first three LB points correctly |
| Factory grader (LCO, marginal) | LCO gain of one candidate family added to the current stack; reward = mean + min of the 3 visible-fold gains − tail penalty; 6 hidden shifts gate acceptance | Used for feature search |

### 4.3 Proxy vs leaderboard

| Recipe | LCO (19 shifts) | Leaderboard | ΔLB / ΔLCO vs previous row |
|---|---|---|---|
| BASE (≈ v3-01) | 0.3757 | 0.5272 | |
| REC1 (= v3-02) | 0.3929 | 0.5669 | 2.3 |
| v3-04 | 0.4126 | 0.6461 | 4.0 |
| v3-15b | 0.4624 | 0.6667 | 0.41 |
| v3-15d | 0.4720 | 0.6676 | 0.09 |
| v3-15g | 0.4421 | 0.6479 | 0.66 (vs v3-15d) |
| **v3-16b** | **0.5166** | **0.7037** | 0.81 (vs v3-15d) |

- **Early on, LB moved ~3.2x the LCO change.** That predicted v3-15b at about 0.80; it scored 0.667. The members added
  after v3-04 (Oh, anisotropy, more consistency views) were partly tuned to the kinds of shifts we generated, so the
  proxy flattered them.
- **The proxy still got the direction right on every recent pair** (b→d up/up, d→g down/down, d→16b up/up). The
  slope is now 0.1-0.8, not 3.2.
- **Oh's proxy gain is inflated:** +0.03-0.05 LCO on visible shifts vs about +0.005 when graded on generator families
  it never trained on. On the real LB, though, Oh is worth +0.020 (Section 6).

---

## 5. Timeline

Codabench times in the submission list are UTC−4; UTC is given here.

### 5.1 Leaderboard submissions

| # | Date / time (UTC) | File | Change | LB | Δ vs best so far |
|---|---|---|---|---|---|
| 0 | 10-10 14:07 | `09-cosmology-corrected-spatial.zip` | Previous team's best | 0.5253 | |
| 1 | 10-10 16:10 | `v3-01-wavelet-cnn-mean.zip` | New v3 pipeline: Sd + C, map-level splits, fixed likelihood | 0.5272 | +0.0019 |
| 2 | 10-10 17:53 | `v3-02-consistency.zip` | + xSL 0.5 + consP8 0.5 (disagreement members) | 0.5669 | +0.0397 |
| 3 | 10-10 18:01 | `v3-04-den-cons1.zip` | + Dr (denoiser residual), consistency at full weight | 0.6461 | +0.0792 |
| 4 | 10-10 ≈23:45 | `v3-12-STACKDp3+SS+Oh1.5.zip` | + consRplus, N2, PsBh, SS, Oh | crashed on Codabench (zip valid) | |
| 5 | 10-11 12:32 | `v3-15b-pkg-v313a-noSS.zip` | v3-12 − SS + anisotropy (X), scored by the code package itself | 0.6667 | +0.0206 |
| 6 | 10-11 13:00 | `v3-15d-pkg-v315b-noC.zip` | − C (CNN embedding) | 0.6676 | +0.0009 |
| 7 | 10-11 19:09 | `v3-15g-pkg-v315d-noOh.zip` | − Oh (test) | 0.6479 | −0.0197 vs v3-15d |
| 8 | 10-11 20:25 | **`v3-16b-pkg-v315d-Ol3.zip`** | + Ol (OE network's own logit) at 3 | **0.7037** | **+0.0361** |
| 9 | (last slot) | `v3-16c-pkg-v315d-Ol6.zip` | Ol at 6 | pending | |

The submission list shows v3-02 with status "Failed" but a score of 0.5669.

### 5.2 Phases and design choices

**Phase A, Oct 10 morning: review and rebuild (→ 0.5272).**
Read the previous handoff, identified the four failure points of Section 2, rebuilt the pipeline as `research/fair_v2`
(map-level splits, fixed likelihood width/steps, copula + eigenvalue floor), with two members (Sd, C). The first
submission only tied the old score: both pipelines caught about 52% of OoD maps cleanly.

**Phase B: disagreement members (→ 0.5669, +0.040).**
Profiling absorbs anomalies (finding 4), but a distortion rarely fools every family into the same wrong parameters.
xSL (scale consistency) and consP8 (8-family Mahalanobis of parameter estimates) on audit3: 15-shift mean
0.4667 vs 0.4496 (paired bootstrap +0.0171 ± 0.0025, 200/200 positive); audit q99/99.5/99.9 1.61/1.88/2.67 vs
2.09/2.47/3.74. ia0.2 is a pure parameter degeneracy that no consistency test sees.

**Phase C: denoiser (→ 0.6461, +0.079).**
U-Net denoiser (7.3M params, 30 epochs on A100, MSE 0.1035 sigma², R² 0.47). Features of the residual (map minus
denoised: spectrum, moments, SURE, tile variance, tails). On audit3: mean:Dr+Sd+C 0.4564 vs 0.4396, q99.9 2.78 vs
3.74. Gains came only on noise/systematics shifts (psfpat 0.54→0.69, srcclust/noise→1.0); physics shifts unchanged.
On the LB it was the largest single jump.

**Phase D: multi-agent sprint (members for v3-05 … v3-12).**
Six agents in parallel: denoiser, field diffusion, outlier exposure, new statistics, consistency, web research.

| Agent | Result |
|---|---|
| Statistics | 7 new feature sources (scattering, peak profiles, peak clustering, position-dependent power, bispectrum, noise battery, tails). Small gains: mean:J3+N3+K+PsBh+W+C+Sd 0.4615 vs 0.4396, from psfpat (0.54→0.66), srcclust, ia0.2 (0.08→0.115). Supervised "ceilings" show bary/peakcut/poisson information barely exists per map (bary9.0 0.33, baryDMO 0.30, peakcut 0.10) |
| Consistency (rounds 2-3) | consRbands over 11 views (best consistency member, .437 alone on audit3); built the LCO proxy; single-member LCO: consR .399, Dr .380, C1 .364, C .343 (one set .219), consP8 .339, N2 .313, xSL .294, Sd .278, PsBh .220, K .025. Cosmology input jitter and width-128 nets hurt |
| Denoiser | den_b bottleneck ridge probe as an extra consistency view (STACKDp3_denb, LCO .4316) |
| Diffusion | SS small-scale score (+0.003 LCO in the stack) |
| Outlier exposure | Oh (14 generator families); +0.03-0.05 LCO, ~+0.005 on held-out generator families |
| Web research | Organizers' planned systematics, LB, rules (Section 1) |

The LCO proxy ranked BASE .3757 < REC1 .3929 < v3-04 .4126, matching the LB. Candidates built: STACK_D (.4277),
BIG_RE2 (.4292), HEDGE_RE3 (.4291), STACKDp3_denb (.4316), v3-12 (.4593 with Oh). v3-12 was submitted and crashed
on Codabench's side.

**Phase E: TPU sprint on a free v5p-32 (negative results).**

| Experiment | Result |
|---|---|
| Throughput | JAX ports of the U-Net and the CNN, one model per chip, 16 in parallel: 89 s per U-Net epoch, 73 s per CNN epoch; converted to torch to ~1e-3 |
| 13-model denoiser ensemble (den_e) | LCO alone .379 vs .380; in STACK_D .4271 vs .4277. Residual features are limited by shape noise, not by denoiser variance |
| Wide-denoiser ensemble (den_w) | Worse: .350 vs .380 |
| 12-CNN ensemble | consR alone .399→.407, full stack +0.001-0.002 (inside ±0.004 proxy noise); not adopted |

**Phase F: feature factory and search (→ anisotropy feature).**
Built a grader (`factory/grade.py`) that scores the marginal LCO gain of one feature family on top of the stack
(stack = v3-12 without Oh; baseline LCO 0.4350). A100 timing: screen 7-16 s, full 101-140 s.

| Candidate | Full reward | hidden_ok | Note |
|---|---|---|---|
| batch_violator | fails | | per-map check |
| random features | +0.0023 | no | "any vote" artifact |
| dog (existing family) | −0.0018 / −0.0043 | | re-expressed family ≈ 0, as expected |
| psband (existing) | −0.0019 | | |
| **dirpower (new)** | **+0.0086** | **yes** | folds +.007/+.0035/+.005; psfpat +0.054 (visible), aniso0.7 +0.042 (hidden); all-19 0.4350→0.4405; 5.9 s per 10k maps |

Then two Claude searchers (33 full grades) and a stack-combination study:

| Search | Grades | Accepted | Best near-miss | Lesson |
|---|---|---|---|---|
| A: noise / non-stationarity | 13 | 0 | ramp4n +0.0144 (gradient-only), hidden fails | Narrow detectors dilute the mean-of-z stack |
| B: scale relations / non-Gaussian | 20 | 0 | linecorr +0.0052, modestat +0.0047 | Any vote adds ~+0.002 even from random features |
| Stack levers (11 configs) | | 0 | prior view + eccentricity +0.0013 (v3-13a) | Max/softplus rules move score between folds; log-sum-exp −0.028 |

**Phase G: code packaging (Oct 10 night).**
The final ranking runs our `model.py`, so every member must run inside it, per map, in fixed batches of 64.

| Package | Runtime (10k maps) | Parity with leaderboard file | Note |
|---|---|---|---|
| v3-13a (with SS) | 231 s on A100 (est. ~3,100 s on a T4) | Spearman 0.998, top-100 99/100 | SS cannot be reproduced (bf16 drift, Spearman 0.93 for SS alone) |
| v3-15b (no SS) | 169 s on A100 | Spearman 0.99999 | Leaderboard files generated by the package itself from here on |
| v3-16a/b (+ Ol) | ~104 s on GH200 | Spearman 0.999997 | Ol reuses the OE forward pass |

v3-13a runtime breakdown on A100: statistics 26 s, CNNs 23 s, denoiser 64 s, anisotropy 4 s, OE CNN 11 s,
diffusion 63 s, noise/bispectrum 7 s, 21 family fits 27 s, model load 5 s. Per-component parity: Sd 1.00000,
C 0.99956, Dr 0.99985, xSL 0.99998, consP8 0.99981, consRplus 0.99891, N2 1.00000, PsBh 1.00000, SS 0.93158,
Oh 0.99645, X 0.99992; final 0.99805.

**Phase H, Oct 11: leaderboard-guided choices.**

| Step | Hypothesis | Proxy (all-19 / hidden / tail at 0.1%) | LB | Decision |
|---|---|---|---|---|
| v3-15b | Package the v3-13a recipe without the non-reproducible SS | .4624 / .3604 / 0.88% | **0.6667** | New best; the post-v3-04 work helps, but far less than the proxy said |
| v3-15d | The CNN's held-out tail is 2.5x heavier; drop C | .4720 / .3669 / 0.63% | **0.6676** | Keep C out |
| v3-15g | Is Oh's gain real or proxy overlap? Drop Oh | .4421 / .3514 / 0.65% | 0.6479 | **Oh is real (+0.020)**; keep it |
| v3-16b | Use the OE network's own logit directly (Ol 3) | .5166 / .3843 / 0.16% | **0.7037** | New best (+0.036) |
| v3-16c | More Ol (6) | .5258 / .3866 / 0.09% | pending | Last slot |

Ol weight sweep on the proxy (v3-15d + Ol):

| Ol weight | All-19 | Visible | Hidden | Per fold | False alarms 1 / 0.5 / 0.1% |
|---|---|---|---|---|---|
| 0 (v3-15d) | 0.4720 | 0.5205 | 0.3669 | .4978 / .4746 / .4435 | 2.29 / 1.39 / 0.63 |
| 1.5 (v3-16a) | 0.5026 | 0.5595 | 0.3794 | .5209 / .5051 / .4819 | 1.70 / 1.02 / 0.31 |
| **3 (v3-16b)** | **0.5166** | 0.5776 | 0.3843 | .5279 / .5189 / .5030 | 1.32 / 0.64 / 0.16 |
| 4.5 | 0.5228 | 0.5859 | 0.3860 | .5299 / .5249 / .5135 | 1.21 / 0.59 / 0.11 |
| 6 (v3-16c) | 0.5258 | 0.5901 | 0.3866 | .5304 / .5281 / .5189 | 1.01 / 0.61 / 0.09 |
| 9 | 0.5282 | 0.5936 | 0.3866 | .5302 / .5310 / .5234 | 1.03 / 0.61 / 0.07 |
| 12 | 0.5290 | 0.5949 | 0.3863 | .5300 / .5319 / .5252 | 1.06 / 0.57 / 0.05 |
| Ol alone | 0.5289 | 0.5959 | 0.3837 | .5277 / .5339 / .5250 | 1.10 / 0.57 / 0.05 |

**Final hours (20:38-22:35 UTC): levers tested against v3-16c.** None beat it outside the ±0.003 proxy tie band.

| Lever | Result (proxy, all 19 shifts) | Verdict |
|---|---|---|
| Average of 3 OE networks (Ol3) | Single nets alone: 0.5289 (the one in v3-16b) vs 0.4801 / 0.5009 (siblings). 15d + Ol3 6 = 0.5174, worse on fold 0 | Dropped: dilutes the best net |
| OE embedding Mahalanobis (Oe3) | Alone 0.409-0.424 with heavy tails (2.3-2.9% at the 1% threshold); 15d + Ol 6 + Oe3 1.5 = 0.5242 | Dropped |
| Honest overlap check (7-family OE net) | Adds only ~+0.003 on shift kinds it never trained on | Ol's proxy gain is mostly generator overlap; its +0.036 LB gain implies the real distortions resemble the 14 OE families |
| Config neighbours of v3-16c | +Oh 3: 0.5256; −Sd: 0.5260; −Sd −consP8: 0.5265; consRplus 4.5: 0.5238; Oh 3 + Ol 9: 0.5281 | Ties |
| Lever 3: tail diagnosis | Top held-out normals come from edge / sparse low-Omega_m universes; their fits hit the T_AGN/f0 box edge 2-2.5x more often. Ol is the largest contributor for 68% of them. Dr 1.25 on top of Ol 6: 0.5271 | Tie; the boundary issue needs a model change |
| Lever 1: learned non-negative combiner (leave-fold x leave-family-out, shrunk toward v3-16b) | v3-18: 0.5275 in-sample; mostly "more Ol" (~37% of the weight) | Tie; not submitted |
| Lever 2: OE head re-fit (low-FPR ranking loss, or plain re-fit on 330k fresh normals + 222k synthetic positives) | Ranking loss worse in every setting. v3-19a (16c + re-fit head): +0.0047 overall, hidden +0.0013, only +0.0018 without the gradient shift. v3-19b: 0.5237 | Weak; adds a code-path change; not submitted |

Other weight variants checked on the proxy and not submitted:

| Variant | Change | All-19 | Per fold vs v3-15b |
|---|---|---|---|
| v3-15c | v3-15b − Oh | 0.4379 | −.028 / −.023 / −.023 |
| v3-15e | consRplus 4.5 | 0.4602 | −.004 / −.004 / +.001 |
| v3-15f | − C, consRplus 4.5 | 0.4685 | −.009 / −.010 / +.037 |
| v3-15h | v3-15d with Oh 3 | 0.4817 | (vs v3-15d) +.003 / +.013 / +.014 |

**Phase I: MOMO feature search on TPU (infrastructure built; no candidates before the deadline).**
Goal: let Qwen3.5-27B write new feature families under RL (MOMO-Train, 16 groups x 16 samples per step), graded on
TPU chips.

| Piece | Status |
|---|---|
| Problem environment `fair_features` | Prompt (JAX contract, physics, diagnostics feedback), reward = clip(1 + 100 x gain, 0.1, 3), 0 for invalid code |
| TPU grader | JAX port, validated against the A100 grader: dirpower +0.0083 (vs +0.0086), all-19 0.4405 exact; deterministic across chips; rl grade 31-37 s warm, 56-62 s for a new feature-size bucket |
| Data bundle | 20.8 GB (frozen noisy maps cut to needed rows, stack caches v1/v2, venv) in the us-east5 bucket; staging 147-150 s per host |
| Farm grading family `fair` | One TPU chip per grade on non-engine hosts; static code checks, clean child env, infrastructure vs candidate failure split, results archived with the hidden block |
| Codex adversarial review | 5 findings (bundle, isolation, failure classes, archive, runtime budget); all fixed |
| Farms | 2 FAIR farms (2 Qwen engines + 8 grading chips each), jobs 2381/2382; 4/4 reference grades ok (−0.00139, ~42 s) |
| Control replay | dirpower −0.00026, random +0.00135, dog −0.00353, violator rejected, static reject ok |
| 1-step smoke (job 2398) | **Preempted.** Farms and smoke went to RECOVERING; ~2 h of runtime in 17 h; no MOMO candidates |

---

## 6. What moved the leaderboard

| Change | LB effect | Confidence |
|---|---|---|
| Disagreement members (xSL, consP8) | +0.040 | measured |
| Denoiser residual (Dr) + consistency at full weight | +0.079 | measured |
| Everything from v3-04 to v3-15b (consRplus, N2, PsBh, Oh, anisotropy, − SS) | +0.021 | measured as a bundle |
| Drop the CNN member (C) | +0.0009 | measured |
| Oh (OE network outputs as a family) | +0.0197 | measured by removal |
| **Ol (OE network's own logit, weight 3)** | **+0.0361** | measured |

**Lessons.**
1. **Training on our own synthetic distortions works on the real test.** The OE network (Oh + Ol) is worth about
   +0.056 combined, the largest effect after the denoiser. The real OoD categories overlap its generator families.
2. **The normal-map tail decides the score.** Every reduction of the flexible base likelihoods (Sd, C) helped. The
   members that help most also lighten the false-alarm tail at 0.1% (0.63% → 0.16% from v3-15d to v3-16b).
3. **Our proxy is direction-correct but biased.** It overrates members tuned to our own shifts by a factor of
   1.5-10, and its LB slope fell from 3.2 to 0.1-0.8.

## 7. What did not help

| Idea | Result |
|---|---|
| Denoiser ensembles (13 models), wide denoisers | No gain (.379 vs .380) / worse (.350) |
| 12-CNN ensemble as a consistency view | +0.001-0.002 in the stack, not adopted |
| Cosmology input jitter, width-128 likelihood nets | Hurt the LCO metric |
| Phase / coupling statistics (scattering, peak profiles, position-dependent power, bispectrum) | Little gain; the information for bary/peakcut/poisson shifts barely exists per map |
| 33 hand-designed feature families (searchers A and B) | None passed the hidden-shift gate |
| Max / softplus / log-sum-exp combination rules | Move score between folds; never better on hidden shifts |
| Prior-centre pseudo-view and vote eccentricity | +0.0013; below the acceptance bar |
| Diffusion small-scale member (SS) | Small proxy gain (+0.0013 all-19), not reproducible in the package; dropped |

---

## 8. Where we are now

### 8.1 Submissions

| Day | Used | Left |
|---|---|---|
| 2026-10-10 | 5 (09, v3-01, v3-02, v3-04, v3-12) | 0 |
| 2026-10-11 | 4 (v3-15b, v3-15d, v3-15g, v3-16b) | **1** → `v3-16c` |

### 8.2 Final code submission

| Candidate | Leaderboard file | Code package | sha256 (package) | Runtime |
|---|---|---|---|---|
| **v3-16b (LB 0.7037)** | `submissions/v3-16b-pkg-v315d-Ol3.zip` (`84e91078…`) | `packages/v3-16b.zip` (140.6 MB, 43 files) | `b11071313e46a61e…` | ~104 s on GH200 |
| v3-16c (pending) | `submissions/v3-16c-pkg-v315d-Ol6.zip` (`3eaa1fa2…`) | `packages/v3-16c.zip` (same files, Ol weight 6) | `eefabf4ef3168209…` | same |

Decision rule: if v3-16c scores above 0.7037, the final code submission is `packages/v3-16c.zip`; otherwise
`packages/v3-16b.zip`. Upload the package with margin before 23:59 UTC.

All paths are under `/scratch/gpfs/ZHUANGL/sk7524/fair-data/`.

### 8.3 Risks

| Risk | Status |
|---|---|
| Unknown scoring GPU | The package runs in ~104 s (GH200) / ~170 s (A100, v3-15b) against a ~1200 s limit. A T4-class GPU was estimated ~13x slower than an A100 (v3-13a: ~3,100 s vs 231 s), which would not fit. Codabench's hardware is not published |
| Private holdout differs from the public test | The final recipe is chosen on the public LB (4 data points today); some LB overfitting is possible but each step was motivated by a prior hypothesis |
| Package vs leaderboard file | Verified: the leaderboard files are the packages' own outputs (Spearman ≥ 0.99999) |

---

## 9. Next steps (after the deadline)

| Priority | Idea | Why |
|---|---|---|
| 1 | Retrain / ensemble the OE network with generators closer to the organizers' plan (baryons, IA, photo-z, shear bias, PSF, source clustering) | Largest measured lever (+0.056) |
| 2 | Train the OE head with a low-FPR ranking loss (partial AUC / top-k) | The metric only counts the top 0.1-5% of normal scores |
| 3 | Calibrate each member on held-out universes (reweight by tail growth on LCO normal maps) | Targets false alarms on unfamiliar universes |
| 4 | Leave-baryon-model-out and leave-distortion-type-out validation | Our proxy's bias is the main obstacle to choosing changes |
| 5 | MOMO run on the OE-centric stack (infrastructure ready; needs non-preemptible capacity) | Automated feature search at scale |
| 6 | Detection-vs-strength curves per distortion type | Know how weak a distortion can be and still be caught |
| 7 | Learned likelihood ratio (classifier on matched vs shuffled (features, parameters) pairs) | Replace the Gaussian misfit model; check its tail on unseen universes |

---

## 10. Where things live

| What | Location |
|---|---|
| Research code | `SkyRLTpu-fair-universe-improved`, branch `agent/fair-universe-improved` (`research/fair_v2/`) |
| Progress log | `research/fair_v2/PROGRESS.md` |
| This report | `research/fair_v2/FAIR_PHASE2_REPORT.md`, branch `agent/fair-universe-report` |
| Feature factory + TPU grader | `research/fair_v2/factory/` (`grade.py`, `tpu_grader/`, `momo_watch.py`) |
| MOMO environment + farm grading | `SkyRL-Main-infer-fair`, branch `agent/fair-momo` (`tpu/fair/`, `third_party/discover/examples/fair_features/env.py`) |
| Data, features, runs | `/scratch/gpfs/ZHUANGL/sk7524/fair-data/` (`v3/<source>`, `agents/`, `factory/`) |
| Leaderboard files | `fair-data/submissions/` |
| Code packages | `fair-data/packages/` (`_verify/` holds the parity runs) |
| MOMO grading archive | `gs://sk7524-tinker-tpu-us-east5/fair-factory/results/` |
