# Consistency agent report (saved by coordinator; 17:31Z)
- Recommended: BASE+0.5xSL+0.5consP8 = (z_Sd + z_C + 0.5 z_xSL + 0.5 z_consP8)/3.
  15-shift mean 0.4667 vs 0.4496 (paired bootstrap +0.0171 +- 0.0025, 200/200 positive); new-4 mean 0.424 vs 0.402;
  audit3 normal q99/99.5/99.9 1.61/1.88/2.67 vs 2.09/2.47/3.74.
  xSL = Dai-Seljak scale consistency (both directions); consP8 = Mahalanobis of theta estimates from 6 stat families +
  C model + CNN predictions (calib3 covariance).
- With C1: BASE3+0.5xSL+0.5consP8 0.4716 vs 0.4620, but C1 tail fragile under leave-cosmology-out (held-out q99.9 23.4).
- ia0.2 is a pure parameter degeneracy (all estimators move Om -1 sd, S8 -1.75..-3.3 sd); no consistency test sees it.
- LCO inflates 1% tail 1.5-1.9x for all detectors; LBO ~nominal. Split likelihoods worse than joint; boundary flags add
  nothing; keep plain mean of z.
- Frozen test z: /scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/consistency/export/test_z_*.npy (calib3-only fitting,
  per-map checked). Code: research/fair_v2/agents/consistency/.
