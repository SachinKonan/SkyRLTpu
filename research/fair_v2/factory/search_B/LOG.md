# Searcher B log: scale relations and non-Gaussian structure

Grader: research/fair_v2/factory/grade.py (stack = v3-12 without Oh; dirpower not yet in the stack).
Acceptance: full reward > 0.006, all three visible fold gains > 0, hidden_ok.

## Base stack per visible shift (LCO metric, folds 0/1/2, mean)
Saturated (~1.0): noise+-0.05, noisegrad0.15, srcclust0.15. Mid: smooth0.5 0.886, psfpat 0.616, bary6.5 0.333.
Weak: gradient 0.117, boost0.15 0.096, bary9.0 0.094, supp0.15 0.072, poisson120 0.021, mult+0.05 0.014.
So gains must come from the baryon/power-shape shifts, gradient, psfpat, smoothing and poisson.

## Design notes
- The footprint is narrow: no pixel has 16-px-smoothed coverage > 0.98 (s=8: 13k px). Interiors use
  mask-normalized smoothing with coverage > 0.85-0.9.
- Amplitude-free statistics (band-normalized, phase-only, rank-Gaussianized, profile shapes) give parameter
  votes that do not follow power changes, so isotropic Fourier-filter shifts (supp/boost/bary-like) should open a
  disagreement with the power-based families.

## Wave 1 (della-gh, full)
| cand | idea | dims | s/10k (GH) |
|---|---|---|---|
| xcoup | cross-band energy coupling + cross third moments of unit-rms DoG bands (0..16 px) | 21 | 1.2 |
| modul | position-dependent small-scale energy vs large-scale field (s=8,16) | 16 | 3.1 |
| peakprof | stacked ring profiles around 2-px-smoothed peaks/troughs, divided by centre | 32 | 3.1 |
| hier | noise-debiased skewness/kurtosis of Gaussian-smoothed fields (1..16 px) | 10 | 1.6 |
Wave 1b (gpu-test job 15340127): phasew (phase-only band fields, 17), gaussz (Gaussianization power ratio, 10).

## Results
| cand | reward | folds | config | hidden_ok | biggest visible gains | theta-probe MSE (Om,S8) | tail pen |
|---|---|---|---|---|---|---|---|
| xcoup | -0.0015 | +.0006/+.0005/+.0002 | mis .5 | yes | smooth +.0025 | .48/.42 | .0022 |
| modul | **+0.0022** | +.0008/+.0044/+.0043 | ridge vote | no | psfpat +.022, bary9 +.009, gradient +.007, supp +.006 | .51/.49 | .0018 |
| peakprof | -0.0016 | -.0003/+.0001/-.0006 | ridge | no | bary9 +.004, supp +.004, psfpat -.005 | .35/.35 | .0007 |
| hier | +0.0010 | +.0011/+.0026/+.0010 | ridge | no | bary9 +.010, supp +.007, psfpat +.006 | .33/.89 | .0016 |
| phasew | -0.0003 | +.0030/+.0016/-.0004 | ridge | no | psfpat +.009, bary9 +.008, supp +.006 | .64/.61 | .0014 |
| gaussz | +0.0014 | +.0010/+.0026/+.0008 | ridge | no | bary9 +.010, supp +.006, psfpat +.006 | .55/.68 | .0009 |

Reading: every amplitude-free family helps only as a ridge parameter vote in consRplus, always on the same
visible shifts (bary9.0, supp0.15, psfpat): power-shape shifts move the power-based votes but not amplitude-free
ones. Peak profiles carry the most theta information but move with the power (no gain). Each vote is noisy, so
the held-out normal tail at 0.5% gets heavier (penalty 0.001-0.002). Next: one family with all amplitude-free
statistics (sharper vote): combo_mp (modul+phasew), combo_mpx (+xcoup), combo_af (+hier+gaussz).
| combo_mp (modul+phasew) | -0.0008 | +.0003/+.0018/-.0006 | ridge | no | psfpat +.008, bary9 +.006 | .47/.43 | .0007 |
| combo_mpx (+xcoup) | -0.0010 | +.0003/-.0009/+.0022 | mis .5 | yes | noisegrad +.007 | .35/.34 | .0007 |
| phasemf (MFs of phase-only field) | -0.0036 | all < 0 | mis .5 | no | none | .52/.52 | .0009 |
| modul2 (L=4,8,16) | +0.0014 | +.0022/+.0012/+.0026 | mis 1 + ridge | no | psfpat +.011, noisegrad +.008, bary9 +.007 | .49/.39 | .0018 |
| condmom (two-scale conditional moments) | -0.0015 | +.0027/-.0008/-.0007 | prof | yes | gradient +.003 | .41/.30 | .0011 |
| **linecorr** (phase-only line correlation) | **+0.0052** | +.0026/+.0045/+.0028 | ridge | **no** | bary9 +.012, psfpat +.011, supp +.009, noisegrad +.004, bary6.5 +.003, boost +.003 | .63/.83 | .0007 |
| **modestat** (Fourier mode-power distribution) | **+0.0047** | +.0028/+.0041/+.0021 | mis 2 + ridge | **no** | psfpat +.022, bary9 +.0105, supp +.006 | .97/.99 | .0004 |
| modulz (modul2, Fisher-z) | +0.0007 | +.0012/+.0013/+.0026 | mis .5 + ridge | no | psfpat +.012, noisegrad +.009 | .50/.40 | .0022 |

Reading (wave 2): combining amplitude-free families into one vote does not add up (the joint vote follows power
more). The two near-misses are both "Gaussian-symmetry" statistics: phase-only line correlation (as a vote) and
the per-band distribution of Fourier mode powers (as a tight misfit, like dirpower). Both clear the reward bar
region (+0.005) with all folds positive, but hidden_ok is false for both.
Wave 3: linecorr2 (scale-adaptive LCF + fixed cuts), phaseall (linecorr2 + phasew), ms_lc (modestat + linecorr),
combo_af (all amplitude-free from wave 1).
| linecorr2 (adaptive LCF + fixed cuts) | +0.0030 | +.0019/+.0053/+.0013 | ridge | no | bary9 +.012, psfpat +.010, supp +.008 | .63/.82 | .0011 |
| phaseall (linecorr2 + phasew) | +0.0011 | +.0033/+.0026/+.0002 | ridge | no | psfpat +.012, bary9 +.009 | .61/.61 | .0011 |
| combo_af (modul+phasew+xcoup+hier+gaussz) | -0.0004 | +.0016/-.0003/+.0019 | mis .5 | yes | noisegrad +.006, gradient +.006 | .22/.32 | .0011 |
| ms_lc (modestat + linecorr) | +0.0019 | +.0003/+.0057/+.0022 | ridge | no | bary9 +.011, psfpat +.008, supp +.008 | .62/.82 | .0011 |
| linecorr3 (8 dirs, 4 cuts, taper) | +0.0010 | +.0030/+.0040/+.0005 | ridge | no | psfpat +.013, bary9 +.011 | .63/.71 | .0020 |
| modestat2 (+ per-band max) | +0.0027 | +.0015/+.0055/+.0005 | mis 1 + ridge | no | bary9 +.012, psfpat +.011 | .97/.99 | .0002 |

## The null-vote effect (important for reading every vote-type gain)
`random_feats` (no information, theta-probe MSE ~1.0) gets +0.0023 with exactly the same pattern as most of the
families above: config (0, ridge), bary9.0 +0.012, supp0.15 +0.0085, psfpat +0.0077, gradient +0.003. A ridge vote
from uninformative features is ~the prior mean, so it adds a "prior-centre" view to consRplus. Shifts that push
the other views' parameter estimates to the edge of the training range (bary9.0, supp0.15, psfpat) then get a
larger disagreement. So:
- Most "amplitude-free vote" gains here are this artifact. Measured against null (+0.0023): linecorr is about
  +0.003 above null (extra on bary6.5, boost0.15, noisegrad0.15, psfpat), modestat about +0.0024 (its misfit at
  weight 2 lifts psfpat to +0.022 vs null +0.008). Variants of both landed at or below null, so even those two may
  be partly proxy luck.
- Stack-level suggestion (not a feature): a fixed prior-centre pseudo-view in consRplus, or an explicit
  "distance of the mean vote from the training prior centre" member, should capture the null-vote gain without any
  features. Worth testing directly in the stack (cheap), and it means the acceptance bar of 0.006 is the right
  scale for vote-type candidates.

## Summary
No candidate passed acceptance (reward > 0.006, all folds > 0, hidden_ok). 20 full grades.
Best near-misses, both all folds positive but hidden_ok false:
- linecorr.py: +0.0052 (+.0026/+.0045/+.0028), vote only. Phase-only three-point line correlation.
- modestat.py: +0.0047 (+.0028/+.0041/+.0021), misfit 2 + vote. Per-band distribution of Fourier mode powers.
Combining families into one vote never helped (the joint vote is less shift-invariant than its parts).
Scale-relation / non-Gaussian statistics carry real theta information (theta-probe MSE down to .22/.32 for the
combo), but their votes and misfits do not separate the visible shifts from cosmology much beyond the null vote.
modestat's misfit gain is mostly on psfpat, which dirpower already covers; expect it to shrink on stack v2.
