# Statistics agent report: new per-map statistic families (FAIR Universe Phase 2)

Agent: statistics. Code: `research/fair_v2/agents/statistics/` (worktree SkyRLTpu-fair-universe-improved; not committed).
Outputs: `/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/statistics/`. Feature sources: `/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3/stx_*`.

## 1. Bottom line

- **Seven new feature sources are complete.** Each has fit (2 views), calib (2), audit (1), all 19 `data.SHIFTS` (including srcclust0.15, poisson120, ia0.2, psfpat) and test (10000x1). They are checked against `stats` index order and are finite. The noise is bit-identical to `stats`/`cnn0`: the first batch of every 40-batch chunk was re-scored with `stats.py`, with max relative error at most 1e-6, except for noise_t5 (see caveats).
- **Our hypothesis that the missing information is phase or coupling information is NOT supported by the synthetic proxies.** Scattering S1/S2, phase harmonics, stacked peak/void profiles, peak clustering, position-dependent power, the binned bispectrum and high-S/N tails add almost nothing to the conditional-likelihood detector on the HMcode baryon extrapolations (bary9.0/6.5/DMO), peakcut0.3 or poisson120. A supervised linear "ceiling" test explains why: for these shifts the information is barely present in ANY per-map summary (stats+cnn0+all new families). Supervised and cross-validated, the best is bary9.0 0.33, baryDMO 0.30, peakcut 0.10, poisson 0.20, ia0.2 0.29. Adding the new families raises none of these ceilings.
- **For mass-conserving halo-profile (BCM-like) shifts I built (spread/concen), the bispectrum and peak profiles DO add information.** The supervised ceiling for spread0.6_t2 goes from 0.44 to 0.54 with bisp and to 0.51 with prof; for concen0.6_t3 it goes from 0.25 to 0.33. But no unsupervised detector recovers it: all detectors score 0.03-0.08 here, because the profile over theta (T_AGN, S8, Delta_z) absorbs the change. Targeted residual-direction tests trained on synthetic negatives recover only a little (spread0.6_t2 0.13, concen0.6_t3 0.10 in-family) and hurt generic shifts.
- **Small but consistent unsupervised gains (19 synthetic shifts, mean phase-2 metric):**
  - current best `mean:Sd+C` scores **0.4396**;
  - `mean:J3+C+Sd` scores **0.4553**, where J3 is a joint likelihood over stats:all + cnn0 + bisp + prof + nois, 1068 dims;
  - `mean:J3+N3+K+PsBh+C+Sd` scores **0.4607**;
  - `mean:J3+N3+K+PsBh+W+C+Sd` scores **0.4615**.

  The gains come from psfpat (0.54 to 0.66), srcclust (0.97 to 1.00), ia0.2 (0.08 to 0.115), aniso, smooth and noisegrad. The noise-structure battery (R1) and peak clustering/Hessian shape drive them, not the baryon shifts.
- **Recommendation.** Use the stx_* sources mainly as **independent theta-views for the consistency test**. The coordinator reports that consistency scores gave LB 0.5272 to 0.5669. Families that profile well on their own are in section 4. As an ensemble member, add the joint model and the noise battery to the mean combo. The tail-safer choice is `mean:J+N2+K+PsBh+C+Sd`: 0.4600, with audit q99.9 2.65 vs 3.10 for the J3/N3 variant (see section 10). Do not expect the new families to fix the baryon-extrapolation or peakcut blind spot. On the synthetic evidence those shifts are nearly undetectable per map with any summary.

## 2. Families implemented (GPU torch, per map, frozen constants from fit3 InD maps)

All work on the masked map zero-padded to 1536x240, so the FFTs do not wrap around. Interiors are scale-dependent (mask-normalised smoothing, eroded). Frozen constants are the extremum height-bin edges (`edges_fam.npz`, quantiles 0.5/0.8/0.93/0.98 of peak/minimum heights) and the noise-battery scales (`edges_nois.npz`). Both were fitted on 768 fit3 maps with their own noise seed (999).

| source | dims | content (column layout) | GH200 cost / 64 maps |
|---|---|---|---|
| `stx_scat` | 66 | Morlet scattering J=5 (sigma=0.8*2^j px), L=4. [0:20] log S1(j,l) (j-major); [20:25] log mean or orientation-averaged S1 with q=2; [25:55] log reduced S2/S1, orientation-averaged, per (j1<j2, \|dl\|=0,1,2); [55:59] phase harmonics Re<z_j conj([z_{j+1}]^2)>/norm; [59:62] the same with [z_{j+2}]^4; [62:66] corr(\|z_j\|,\|z_{j+1}\|). S2 uses Fourier cropping by 2^(j2-1). | ~0.9 s |
| `stx_prof` | 200 | Stacked radial kappa profiles from FFT cross-correlation with exact mask normalisation. Detection is on G_1.5 and G_3 px maps; profiles are measured on the raw noisy map; 5 frozen height bins; radii edges 0,0.5,1.5,2.5,3.5,5,7,10,14,20 px. Blocks of 50 in the order (1.5 px peaks, 1.5 px minima, 3 px peaks, 3 px minima); each block is 5 bins x 9 radii (bin-major) + 5 mean heights. | ~0.3 s (with pkcl) |
| `stx_pkcl` | 82 | Per detection scale (41 each): peak ellipticity (5 bins), curvature/height (5 bins), the same two for minima (10), then the 2-point excess xi(r) for top-7% peaks x all peaks, top peaks x all minima, and deep minima x all minima at 7 radii (1.5-28 px). | (with prof) |
| `stx_ipow` | 44 | Position-dependent power. For L = 8, 16 px and DoG bands (0-1, 1-2, 2-4, 4-8 px): corr(local band power, delta_L), corr(local band power, delta_L^2), log relative variance of local power, corr(local band cube, delta_L) (16), then 6 band-band local-power correlations. | ~0.15 s |
| `stx_bisp` | 92 | Binned reduced bispectrum Q(b1,b2,b3) = <I1 I2 I3>/(rms1 rms2 rms3) over 9 bands (edges 0.006-0.5 cycles/px, ell ~65-5400), all closed triangles (`famfeats.TRI`). [62:92] are triangles with all bands >= 0.065 (high ell). | ~0.1 s |
| `stx_nois` | 54 | R1 noise-structure battery for high-pass h = x - G_s x, s = 1 and 2 px (20 each). Per scale: log var, skew, kurtosis, tail counts beyond +-3/4/5 sigma (frozen sigma), and 16x16-px tile variances v_t vs tile mean m_t of G_4: log mean, CV, Pearson/Spearman(v,m), Pearson(v,\|m\|), trend along each axis, log max/mean, log min/mean, lag-1 autocorrelation, block-variance fraction. Then [40:52] 2D power wedge ratios (horizontal/vertical and diagonal) in 6 ell bands, [52] log1p of the exactly-zero observed-pixel count, [53] 2x2 all-zero blocks (constant in InD). **Drop the discrete rare columns 6, 8, 26, 28 and the constant 53 (`stx_nois:x6.8.26.28.53`).** | ~0.3 s (with tail) |
| `stx_tail` | 45 | R3 high-S/N tails at G_1, 2, 3 px: log1p pixel counts above +3/4/5 and below -3/-4 frozen sigma, peaks above 3/4/5 sigma, minima below -3/-4 sigma, the top-1/5/20 pixel SNR and the bottom-1/5. | (with nois) |

Scripts:
- `famfeats.py` holds the families.
- `extract_fam.py` does resumable extraction with the pipeline_v3 noise loop and per-chunk RNG checkpoints (`--kind main` or `--kind nois`).
- `fit_eval.py` is fit_v3 plus column keep/drop syntax `src:x1.2-5` / `src:k...` and `--reuse`.
- `oracle.py` runs the supervised ceilings.
- `dshift.py` and `dshift_eval.py` produce and score the BCM-like diagnostic shifts.
- `trainshift.py` and `gdirect.py` run the targeted residual-direction tests.
- `effect.py` gives paired per-feature effect sizes; `theta_r2.py` gives held-out theta information per family.

## 3. Results on the 19 synthetic shifts (fit_v3 protocol, run dir `runs/r2`)

These are the phase-2 metric on audit3 normal vs shifted maps; "normal q" means audit z quantiles. For mean combos the z-scale is compressed, so compare combos by their metric, not their q99.9. Rows are single models; `combo:` rows average standardized z.
- C = cnn0:raw; Sd = stats:dog; SC = stats:all|cnn0:raw.
- J3 = stats:all|cnn0:raw|stx_bisp:raw|stx_prof:raw|stx_nois:x6.8.26.28.53.
- W = stx_scat; P = stx_prof; K = stx_pkcl; I = stx_ipow; B = stx_bisp; T = stx_tail; N3 = stx_nois:x6.8.26.28.53.
- PsB = stats:ps|stx_bisp:raw; PsBh = stats:ps|stx_bisp:k62-91.

My refit of `mean:Sd+C` reproduces `v3/runs/r1/test_combo_mean_Sd+C.npy` bit-identically.

| model | dims | normal q99 / q99.5 / q99.9 | shift_mean (19) | bary9.0 | bary6.5 | baryDMO | peakcut0.3 | ia0.2 | poisson120 | psfpat | srcclust0.15 | supp0.15 | boost0.15 | aniso0.7 | smooth0.5 | noisegrad0.15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C | 261 | 3.39 / 4.40 / 6.06 | 0.3972 | 0.135 | 0.342 | 0.202 | 0.023 | 0.069 | 0.020 | 0.546 | 0.740 | 0.095 | 0.166 | 0.789 | 0.874 | 0.973 |
| Sd | 56 | 2.90 / 3.27 / 3.87 | 0.2955 | 0.036 | 0.099 | 0.039 | 0.016 | 0.034 | 0.018 | 0.047 | 0.936 | 0.024 | 0.020 | 0.519 | 0.748 | 0.092 |
| SC | 727 | 3.46 / 4.45 / 7.32 | 0.4040 | 0.112 | 0.368 | 0.222 | 0.017 | 0.079 | 0.014 | 0.443 | 0.954 | 0.069 | 0.168 | 0.703 | 0.815 | 0.958 |
| J3 | 1068 | 3.35 / 4.12 / 6.63 | 0.4372 | 0.133 | 0.372 | 0.219 | 0.017 | 0.097 | 0.011 | 0.562 | 1.000 | 0.089 | 0.164 | 0.834 | 0.926 | 0.984 |
| W | 66 | 2.70 / 3.00 / 3.69 | 0.2885 | 0.033 | 0.123 | 0.045 | 0.019 | 0.026 | 0.014 | 0.218 | 0.814 | 0.028 | 0.021 | 0.723 | 0.530 | 0.076 |
| P | 200 | 2.62 / 3.56 / 5.35 | 0.0712 | 0.026 | 0.028 | 0.022 | 0.024 | 0.042 | 0.016 | 0.117 | 0.276 | 0.015 | 0.020 | 0.022 | 0.023 | 0.017 |
| K | 82 | 2.57 / 2.94 / 3.93 | 0.0280 | 0.022 | 0.020 | 0.020 | 0.022 | 0.088 | 0.016 | 0.027 | 0.092 | 0.019 | 0.019 | 0.018 | 0.016 | 0.013 |
| I | 44 | 2.62 / 2.94 / 4.13 | 0.1243 | 0.021 | 0.021 | 0.019 | 0.022 | 0.029 | 0.016 | 0.025 | 1.000 | 0.015 | 0.019 | 0.020 | 0.023 | 0.853 |
| B | 92 | 2.35 / 2.63 / 3.42 | 0.0739 | 0.011 | 0.025 | 0.019 | 0.022 | 0.016 | 0.011 | 0.034 | 0.998 | 0.014 | 0.022 | 0.025 | 0.037 | 0.018 |
| N3 | 49 | 2.61 / 3.14 / 4.61 | 0.3301 | 0.014 | 0.061 | 0.026 | 0.012 | 0.019 | 0.018 | 0.345 | 1.000 | 0.013 | 0.020 | 0.709 | 0.186 | 1.000 |
| T | 45 | 3.47 / 4.04 / 4.86 | 0.0390 | 0.013 | 0.017 | 0.018 | 0.015 | 0.019 | 0.010 | 0.034 | 0.147 | 0.011 | 0.018 | 0.015 | 0.019 | 0.010 |
| PsB | 120 | 2.42 / 2.57 / 2.97 | 0.2417 | 0.041 | 0.114 | 0.045 | 0.024 | 0.026 | 0.015 | 0.205 | 1.000 | 0.045 | 0.027 | 0.348 | 0.585 | 0.035 |
| PsBh | 58 | 2.61 / 2.89 / 3.60 | 0.2250 | 0.038 | 0.125 | 0.045 | 0.013 | 0.018 | 0.014 | 0.109 | 0.864 | 0.029 | 0.023 | 0.331 | 0.531 | 0.030 |
| combo:mean:Sd+C |  | 2.09 / 2.47 / 3.74 | 0.4396 | 0.139 | 0.345 | 0.199 | 0.024 | 0.080 | 0.021 | 0.537 | 0.970 | 0.096 | 0.155 | 0.828 | 0.916 | 0.968 |
| combo:mean:SC+C+Sd |  | 2.29 / 2.73 / 3.96 | 0.4465 | 0.140 | 0.376 | 0.223 | 0.021 | 0.086 | 0.018 | 0.558 | 0.986 | 0.093 | 0.177 | 0.827 | 0.919 | 0.988 |
| combo:mean:W+C+Sd |  | 1.59 / 1.92 / 3.39 | 0.4456 | 0.145 | 0.348 | 0.198 | 0.025 | 0.081 | 0.022 | 0.587 | 0.989 | 0.097 | 0.146 | 0.878 | 0.920 | 0.947 |
| combo:mean:P+C+Sd |  | 1.70 / 2.25 / 2.76 | 0.4319 | 0.126 | 0.328 | 0.188 | 0.026 | 0.088 | 0.020 | 0.544 | 0.976 | 0.084 | 0.138 | 0.804 | 0.898 | 0.934 |
| combo:mean:K+C+Sd |  | 1.56 / 1.83 / 2.68 | 0.4361 | 0.131 | 0.337 | 0.196 | 0.028 | 0.120 | 0.023 | 0.520 | 0.967 | 0.090 | 0.151 | 0.820 | 0.910 | 0.950 |
| combo:mean:I+C+Sd |  | 1.67 / 1.86 / 2.67 | 0.4321 | 0.121 | 0.332 | 0.186 | 0.024 | 0.079 | 0.019 | 0.498 | 1.000 | 0.082 | 0.133 | 0.809 | 0.902 | 0.997 |
| combo:mean:B+C+Sd |  | 1.52 / 1.74 / 2.45 | 0.4417 | 0.134 | 0.342 | 0.202 | 0.030 | 0.085 | 0.023 | 0.550 | 1.000 | 0.097 | 0.156 | 0.827 | 0.915 | 0.966 |
| combo:mean:N3+C+Sd |  | 1.69 / 1.99 / 3.73 | 0.4447 | 0.120 | 0.328 | 0.184 | 0.021 | 0.071 | 0.022 | 0.616 | 1.000 | 0.081 | 0.133 | 0.879 | 0.906 | 1.000 |
| combo:mean:PsBh+C+Sd |  | 1.64 / 1.95 / 2.64 | 0.4472 | 0.151 | 0.346 | 0.202 | 0.026 | 0.080 | 0.024 | 0.558 | 0.998 | 0.105 | 0.153 | 0.835 | 0.920 | 0.956 |
| combo:mean:J3+C+Sd |  | 2.22 / 2.74 / 4.15 | 0.4553 | 0.152 | 0.379 | 0.224 | 0.022 | 0.098 | 0.018 | 0.609 | 0.999 | 0.106 | 0.179 | 0.856 | 0.937 | 0.992 |
| combo:mean:J3+N3+C+Sd |  | 1.87 / 2.35 / 4.51 | 0.4551 | 0.139 | 0.364 | 0.212 | 0.021 | 0.089 | 0.019 | 0.646 | 1.000 | 0.095 | 0.161 | 0.884 | 0.930 | 1.000 |
| combo:mean:N2+K+PsBh+C+Sd |  | 1.18 / 1.43 / 2.24 | 0.4506 | 0.132 | 0.328 | 0.187 | 0.026 | 0.109 | 0.024 | 0.623 | 1.000 | 0.086 | 0.136 | 0.881 | 0.914 | 1.000 |
| combo:mean:J3+N3+K+PsBh+C+Sd |  | 1.42 / 1.60 / 3.10 | 0.4607 | 0.146 | 0.364 | 0.215 | 0.025 | 0.115 | 0.020 | 0.655 | 1.000 | 0.099 | 0.167 | 0.887 | 0.935 | 1.000 |
| combo:mean:J3+N3+K+PsBh+W+C+Sd |  | 1.38 / 1.48 / 2.78 | 0.4615 | 0.149 | 0.363 | 0.213 | 0.025 | 0.115 | 0.019 | 0.673 | 1.000 | 0.099 | 0.160 | 0.901 | 0.935 | 0.999 |

Findings from the table:
- No new family moves bary9.0, baryDMO or peakcut0.3 beyond noise level, either alone or in combos. bary6.5 improves only through the joint J3 model (0.345 to 0.37-0.38).
- ia0.2 (20% phase-randomised signal, the phase-information test case) gains only via peak clustering/Hessian shape (K: 0.080 to 0.120 in combo).
- poisson120 (per-pixel Poisson noise modulation) stays at chance for everything. Its excess noise kurtosis (~0.025) is smaller than the signal-driven scatter of the high-pass kurtosis (~0.06).
- The noise battery N2/N3 is the strongest new standalone family: shift_mean 0.33, srcclust 1.00, noisegrad 1.00, aniso 0.71, psfpat 0.35. It is complementary to C/Sd on psfpat and aniso.

## 4. Which stx sources profile well on their own (for the consistency / multi-view test)

Standalone fit_v3 models (copula, 3000 steps) are below; normal = audit3 z. Theta information is held-out linear ridge R^2 on calib3 (`theta_r2.txt`). The Likelihood's own "regressor mse" is a training loss and is misleading here: P reaches 0.02 by memorising the fit maps.

| source (spec) | dims | normal q99 / q99.9 / max | R^2 Om | S8 | T_AGN | f0 | dz | verdict |
|---|---|---|---|---|---|---|---|---|
| stats (reference) | 466 | - | 0.910 | 0.841 | 0.398 | 0.681 | 0.113 | |
| cnn0 (reference) | 261 | 3.39 / 6.06 / 14.3 | 0.935 | 0.864 | 0.450 | 0.764 | 0.114 | |
| stx_scat:raw | 66 | 2.70 / 3.69 / 5.0 | 0.822 | 0.694 | 0.278 | 0.478 | 0.088 | **good**: sane tail, strong Om/S8 view |
| stx_prof:raw | 200 | 2.62 / 5.35 / 7.4 | 0.783 | 0.758 | 0.273 | 0.121 | 0.105 | usable; heavier tail. The high-peak-only subset (Ph, 40 dims) has a bad tail (q99.9 6.3, max 14.9): avoid it |
| stx_pkcl:raw | 82 | 2.57 / 3.93 / 6.2 | 0.542 | 0.437 | 0.186 | 0.072 | 0.066 | **good** tail, weak theta view |
| stx_ipow:raw | 44 | 2.62 / 4.13 / 6.0 | 0.498 | 0.482 | 0.098 | 0.063 | 0.037 | ok tail, weak theta view |
| stx_bisp:raw | 92 | 2.35 / 3.42 / 3.7 | 0.506 | 0.612 | 0.152 | 0.192 | 0.049 | **best tail**; moderate S8 view |
| stats:ps\|stx_bisp:raw (PsB) | 120 | 2.42 / 2.97 / 4.1 | | | | | | **best tail of all** (P and B consistency in one model) |
| stx_nois:x6.8.26.28.53 (N3) | 49 | 2.61 / 4.61 / 16.3 | 0.786 | 0.608 | 0.131 | 0.406 | 0.067 | ok; two InD extremes come from massive-cluster maps (tile-variance max, high-pass kurtosis). `x7.8.25.26.53` (N2) has max 8.0 |
| stx_tail:raw | 45 | 3.47 / 4.86 / 8.8 | 0.700 | 0.710 | 0.172 | 0.042 | 0.086 | heavy-ish tail, little standalone value |
| J3 (stats:all+cnn0+bisp+prof+N3) | 1068 | 3.35 / 6.64 / 9.9 | | | | | | best single model (0.437) but heavy tail |

For per-scale theta views:
- scat j-subsets: columns j*4..j*4+3 and 20+j.
- prof radii blocks.
- bisp band subsets: low-ell triangles are columns with max band <= 3; high-ell ones are [62:92].
- ipow L=8 [0:22] vs L=16 [22:44].
- nois s=1 [0:20] vs s=2 [20:40].

## 5. Supervised ceilings: is the information in the statistics at all? (`oracle1.txt`, `oracle2.txt`)

Method: shrinkage LDA trained to separate shift_X from audit3, with 5-fold CV over maps (paired maps kept together); the best of 3 shrinkages is reported, which is optimistic. This is a diagnostic only: synthetic shifts of training maps, never test. An unsupervised detector cannot beat it on that shift.

| feature set | bary9.0 | bary6.5 | baryDMO | peakcut0.3 | ia0.2 | poisson120 | psfpat | supp0.15 |
|---|---|---|---|---|---|---|---|---|
| stats | 0.294 | 0.398 | 0.274 | 0.093 | 0.300 | 0.197 | 0.998 | 0.197 |
| cnn0 | 0.336 | 0.416 | 0.312 | 0.038 | 0.249 | 0.043 | 0.894 | 0.253 |
| stats+cnn0 (SC) | 0.332 | 0.419 | 0.304 | 0.095 | 0.289 | 0.198 | 0.995 | 0.248 |
| SC + scat / prof / pkcl / bisp / ipow / nois / tail | 0.33 | 0.41-0.43 | 0.27-0.31 | 0.09-0.11 | 0.28-0.29 | 0.17-0.20 | ~1.0 | 0.24-0.25 |
| unsupervised best (`mean:J3+N3+K+PsBh+W+C+Sd`) | 0.149 | 0.363 | 0.213 | 0.025 | 0.115 | 0.019 | 0.673 | 0.099 |

Takeaways:
- Even with supervision, the HMcode baryon extrapolations, peakcut and poisson are weak per map at FPR 0.1-5%.
- The new families add no new linear information for these shifts.
- The gap between supervised and unsupervised (e.g. ia0.2 0.29 vs 0.115, bary9.0 0.33 vs 0.15) is the price of not knowing the direction plus the theta-profile absorbing the change. It is not caused by missing statistics.

## 6. BCM-like (mass-conserving halo-profile) diagnostic shifts (`dshift.py`, `oracle_dshift.txt`, `dshift_eval.txt`)

I built these shifts for the halo-profile hypothesis. Excess E = relu(G1*kappa - t*sd) around high peaks, and kappa -> kappa -/+ a(G1 E - G3 E):
- spread: core mass moves to the outskirts with total mass conserved, like gas expulsion in BCM;
- concen: the opposite.

They are applied to audit3 clean maps with new noise. stats, cnn0 (the cnn_b checkpoint, which reproduces cnn0 to bf16 precision) and all stx families were computed on the same noisy maps.

| | spread0.3_t2 | spread0.6_t2 | spread0.6_t3 | concen0.3_t2 | concen0.6_t3 |
|---|---|---|---|---|---|
| supervised stats | 0.110 | 0.454 | 0.212 | 0.138 | 0.237 |
| supervised stats+cnn0 | 0.104 | 0.437 | 0.205 | 0.134 | 0.248 |
| supervised SC + prof | 0.116 | 0.511 | 0.221 | 0.123 | 0.247 |
| supervised SC + bisp | 0.120 | **0.543** | **0.277** | 0.149 | **0.328** |
| unsupervised `mean:Sd+C` | 0.029 | 0.076 | 0.037 | 0.031 | 0.045 |
| unsupervised `mean:B+C+Sd` | 0.033 | 0.083 | 0.039 | 0.036 | 0.053 |
| unsupervised `mean:J+N2+K+PsBh+C+Sd` | 0.029 | 0.074 | 0.035 | 0.028 | 0.044 |

This is the one place the "statistics" hypothesis holds: the bispectrum and peak profiles carry extra, profile-specific information (+0.1 supervised). The omnibus conditional likelihood cannot use it, because the change is mostly absorbable by T_AGN/S8/Delta_z.

## 7. Targeted tests trained on synthetic negatives (`trainshift.py`, `gdirect.py`, run `runs/g1`)

Method: the joint J model is used to profile theta and form residual normal scores g. For each of 12 synthetic shifts applied to fit3 maps (tshift/, training maps only), delta_k is the mean paired change of g; then:
- s_k = delta_k^T Rc^-1 g, standardized on calib3;
- T = z^T C^-1 z over the K directions, mapped to a normal score, with a theta-conditional Standardizer.

Evaluation is leave-family-out (lofo): when scoring family F, its own directions are removed. In-family numbers are optimistic.

Result: **not useful as a general detector.**
- Mean over the 19 shifts: lofo 0.175. Combined with Sd+C it gives 0.438 vs 0.4396 without it.
- In-family it helps only the profile shifts: spread0.6_t2 0.13 vs 0.05 omnibus, concen0.6_t3 0.10 vs 0.035.
- It loses most generic sensitivity: smooth 0.02, aniso 0.02.
- An earlier raw-feature LDA variant (`direct.py`) had strongly skewed, theta-dependent projections and was unstable. Not recommended.

## 7b. Quick cross-family theta-consistency check (`consist.py`, `consist_C.txt`)

Method: theta_hat comes from each frozen r2 model's profile and is normalized by the training theta std. Score = Mahalanobis distance of [theta_view - theta_C] under calib3, mapped to a normal score.
- With all views (Sd, W, P, K, B, N2, I, PsB), the score alone gives mean19 = 0.320 with audit q99.9 3.98.
- Added to Sd+C it gives 0.4385, vs 0.4396 for Sd+C alone. Single views give 0.430-0.442; the best is C-vs-B at 0.4421 and C-vs-I at 0.4411.

So on the synthetic proxies, this naive consistency adds nothing. The coordinator's LB gain (0.5272 to 0.5669) says the proxies underpredict this kind of score, so let the LB decide.

For the consistency agent, per-map normalized theta_hat for every set, including test, are cached as `runs/r2/theta_<tag>.npz` for tags C, Sd, W, P, K, B, N2, I, PsB (keys calib, audit, shift_*, test). Frozen test scores: `runs/r2/test_consist_C_vs_Sd+W+P+K+B+N2+I+PsB.npy` and `..._plus_Sd_C.npy`.

## 8. Frozen per-map test z-scores

All are under `/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/statistics/runs/r2/`. Models are fit on fit3, standardized on calib3, and scored per map on test with fixed batch shape; no test statistics were used. Each array is (10000,), finite, in test row order.
- Per model: `z_<tag>.npz['test']` for tags C, Sd, SC, J, J3, W, P, K, I, B, N2, N3, T, PsB, PsBh, Bh, Ph, PhBh, SdW, SdP. Pickled models are in `model_<tag>.pkl` and specs in `spec_<tag>.txt`.
- Combos: `test_combo_mean_<...>.npy`. Recommended:
  - `test_combo_mean_J3+N3+K+PsBh+C+Sd.npy` (0.4607 synthetic);
  - `test_combo_mean_J3+N3+K+PsBh+W+C+Sd.npy` (0.4615);
  - simpler: `test_combo_mean_J3+C+Sd.npy` (0.4553).

  The baseline is `test_combo_mean_Sd+C.npy` (0.4396), which is identical to the LB-0.5272 submission's scores.
- Tail-safer recommended combo: `test_combo_mean_J+N2+K+PsBh+C+Sd.npy` (0.4600; J = stats:all|cnn0:raw|stx_bisp:raw|stx_prof:raw|stx_nois:x7.8.25.26.53, N2 = stx_nois:x7.8.25.26.53).
- The targeted detector is in `runs/g1/z_D.npz` (`test`, `omni_test`). Not recommended.

## 9. Caveats

- **noise_t5 is not bit-identical to stats/cnn0.** Its Chi2 draws use the *global* CUDA RNG, whose state is not reproducible: the relative error differed between fresh processes. Only the Gaussian part matches. Every detector flags noise_t5 at about 1.0 anyway. All other sets are verified.
- **Synthetic shifts are proxies.** The +0.02 gain of the best combos is concentrated in psfpat, srcclust, ia, aniso and noisegrad. Whether the real OoD resembles them is unknown, and LB feedback should decide.
- **The noise battery is contaminated by signal from massive clusters.** Tile-variance max and high-pass kurtosis produce a few extreme InD maps (N3 max 16.3). Keep columns that are discrete rare-event counts dropped. Column 52 (exact-zero pixel count) is a hole detector: it is Poisson(~0.12) in InD and would explode on masked or zeroed regions.
- **Compute.** All families together cost about 1.7 s per 64 maps on a shared GH200 (scattering is about half of that). This matters for the 1200 s final-run limit if the final model must recompute them. scat could be dropped (W adds little).
- Combo normal quantiles are on a compressed scale; compare combos by the phase-2 metric.
- J/J3 (1068 dims) have heavier normal tails (q99.9 about 6.6-7.1) than the small models. They help in mean combos but are risky alone.

## 10. Recommendation

1. Use the complete `stx_*` sources as extra theta-views in the consistency test. They are most useful there because they are nearly independent statistics of the same map. Use scat (Om/S8, tail sane), bisp/PsB (best tails), pkcl and prof (prof at full 200 dims, not the high-peak subset), and N3. Drop the stx_nois rare columns (`x6.8.26.28.53`).
2. For the omnibus ensemble, add the joint model and the noise battery (plus K, PsBh) to `mean:Sd+C`. This gives +0.02 on synthetic shifts. The tail-safer variant is **`mean:J+N2+K+PsBh+C+Sd`**: metric 0.4600, audit q99.5/q99.9 = 1.62/2.65, ratio 1.64. Compare `mean:J3+N3+K+PsBh+C+Sd`: metric 0.4607, q 1.60/3.10, ratio 1.94, with its tail driven by one massive-cluster InD map in N3. With ~60% of the metric weight at FPR < 1%, I would submit the N2/J variant.
3. Do not invest more in hand-built non-Gaussian statistics for the baryon/peakcut blind spot. The supervised ceilings show that information is mostly absent per map or degenerate with nuisance parameters. The remaining handle is cross-view consistency, which LB feedback already supports. For profile-type physics (BCM), the bispectrum is the statistic that carries extra information, so feed it to the consistency test as its own view.
