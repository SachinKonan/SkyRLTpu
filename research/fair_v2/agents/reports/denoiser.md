# Denoiser agent report (saved by coordinator)
- U-Net denoiser (7.3M params, fit3, A100 30 epochs; frozen net_ep26.pt -> unet_final.pt), MSE 0.1035 sigma^2, R^2 0.47.
- Source `den` (1090 dims): den_rr (80: residual spectrum/moments/cross-corr), den_rt (32: SURE, tile variance vs D(x),
  residual tails), den_s (466: stats on D(x)), den_b (512: bottleneck). Noise identical to stats (9e-6), per-map checked.
- Recommended mean:Dr+Sd+C: shift_mean 0.4564 vs 0.4396 (mean:Sd+C), q99.9 2.78 vs 3.74; gains only on noise/systematics
  shifts (psfpat 0.54->0.69, srcclust/noise->1.0); physics ("signal") shifts unchanged 0.257 vs 0.256.
- Denoiser pulls OoD structure toward InD-looking D(x); stats on D(x) are weaker than on noisy maps.
- Test z: agents/denoiser/runs/combo/test_mean_Dr+Sd+C.npy (and test_mean_Dr+D+Sd+C.npy, 0.4621).
