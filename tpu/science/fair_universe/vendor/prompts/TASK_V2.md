# Optimize this single FAIR Phase-2 anomaly detector program

Edit the provided `candidate.py` and nothing else. For full architecture/loss freedom, seed it from the **self-contained** `candidates/hybrid_physics_jax.py`; the stronger `candidates/diverse_ensemble_jax.py` is a multi-file ensemble reference. You can inline/reimplement any branch inside your one candidate file. Do not edit the judge or another candidate's modules.

**Your objective**: Return a higher anomaly score for an image inconsistent with the known simulated universe family, while avoiding false positives on known simulations. The only external reward is the judge's mean TPR at 100 log-spaced FPRs between .001 and .05. Hidden OoD labels are not available.

Required functions:

```python
def fit(train, val, config, out_dir) -> dict:  # saves portable state
    ...
def load_model(out_dir, config):
    ...
def predict(model, maps, mask, config) -> np.ndarray:  # finite float[B]
    ...
```

`train` and `val` yield `{'maps':float32[B,H,W], 'labels':float32[B,5], 'mask':bool[H,W]}` via `iter_batches`. Labels: `(Omega_m, S8, T_AGN, f0, Delta_z)`. Real image shape is `[1424,176]`; masked pixels are invalid. Public maps are **clean**; `DataView.iter_batches` adds the correct observation noise. Input to `predict` is **already noisy**. Return one score per map, larger means **more** anomalous. The judge never passes labels to `predict`.

**Research priorities** (pick ONE hypothesis):
- Distinguish novel physics from normal cosmology/nuisance variation: conditional residual densities, manifold distance, calibrated uncertainty.
- Capture signals invisible to Fourier power: multiscale/wavelet statistics, peaks, phase relationships, masked spatial structure.
- Learn useful representations: supervised/masked encoders, consistency or denoising objectives; combine with non-neural baselines.
- Robustly ensemble complementary scores using only InD validation calibration. Do not assume a larger model is better.

**Implementation constraints:** no PyTorch/CUDA, no network, no official test data in fit, no reward examples or proxy-shift labels in fit. Honor `config['steps']`, `config['batch_size']` and outer wall-time. Fit all normalizations and calibrators from train/normal val only. Handle shape noise, survey mask, extreme scores, NaNs and device portability.

**Your response:** if running as a coding agent, edit only the assigned Python file and give a short one-line change summary. If running as a code-generation model, output the COMPLETE Python module and no Markdown. Before declaring success, run a smoke trial and verify `reward.json`; a small toy score checks plumbing only. Record the hypothesis in a code comment. Never claim a score on the hidden FAIR test.
