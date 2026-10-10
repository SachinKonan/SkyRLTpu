# Research agent report (saved by coordinator; full text in conversation)
Key facts (with sources in the agent's report):
- No public statement names the OoD categories, fractions or recipe; Codabench forum is login-gated.
- First challenge README (commit 4ec1b26, 2025-08-15) planned 6 systematics: baryons, intrinsic alignment (IA), photo-z,
  shear measurement bias, PSF, source clustering, some "as a post-processing step" via "a biasing script". Training has only
  baryons + photo-z. Zenodo metadata lists IA, PSF residuals, source clustering and additional survey masks as ABSENT.
- Phase-1 test: "random cosmological parameters, random nuisance parameters, random pixel-level noises"; slides mention
  "tunable pixel-level noises". A 14k-map set at 82 unseen cosmologies exists (to be released after Phase 2).
- Organizers' AE figure: InD MSE = ~210k noise + 12k-75k signal; OoD low tail ~212k (almost no small-scale signal or
  less noise), high tail ~330k. Power-spectrum chi2 baseline v2 = 0.2357.
- Diao, Dai & Seljak 2025: flow matching 0.87 AUROC vs scattering 0.73 vs CNN 0.54 for BCM baryons; GLOW mis-ranks
  smoothed/scaled/masked maps. Dai & Seljak multiscale flow: scale-consistency score dlogp AUROC 0.80, independent of logp.
- Leaderboard: 0.7835, 0.7823, 0.7729, 0.7685, 0.7571; dkn20 (likely K. Diao) 0.7358; cluster at 0.60-0.65.
- Limits: 5 submissions/day, 100 total, 1200 s execution. Final ranking runs our model.py (Model.predict on float16
  memmap (N,132019)) on a private holdout; prizes need source, write-up, AI-tool disclosure.
Recommendations: R1 noise-structure battery (tile variance of high-pass map vs local kappa, high-pass kurtosis, 2D power
anisotropy, zero-pixel count); R2 scale-consistency (theta from large vs small scales); R3 peaks/minima/PDF tails/stacked
peak profiles at theta fitted from power spectrum only; R4 field-level flow matching/diffusion on patches (avoid NFs);
R5 leave-cosmology-out / leave-baryon-model-out tail checks, min-p/Fisher combination.
