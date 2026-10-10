# Coding-agent task: FAIR-Universe Phase 2 JAX anomaly detection

Modify `candidate.py` in place. The only required top-level functions are:

```python
def fit(train, val, config: dict, out_dir: pathlib.Path):
    # Train using train (and in-distribution validation val); save model state
    # in out_dir. Must respect config['steps'] and config['batch_size'].
    return {"train_loss": float(...)}  # optional JSON-compatible metadata

def load_model(out_dir: pathlib.Path, config: dict):
    # Load saved params or fit statistics to CPU/GPU/TPU-compatible state.
    return model

def predict(model, maps: np.ndarray, mask: np.ndarray, config: dict) -> np.ndarray:
    # maps is float32 [B,H,W], ALREADY NOISY, H=1424, W=176 on real data.
    # mask is bool [H,W], same for every map.
    # return a finite float array [B]. Higher = MORE anomalous.
    return scores
```

`train` and `val` are in-distribution DataView objects. They offer:

```python
len(train)
train.mask                          # bool [H,W]
train.iter_batches(batch_size=16, epochs=5, shuffle=True,
                   seed=0, add_noise=True, drop_last=False)
# yields {"maps": float32[B,H,W], "labels": float32[B,5], "mask": bool[H,W]}
```

Labels are [Omega_m, S8, T_AGN, f0, Delta_z]. TRAIN maps are noiseless on
disk, and DataView adds correctly scaled observational shape noise. The
official test maps have noise added already: do NOT add it in `predict`.

You MAY edit architecture, feature engineering, loss, learning rate schedule,
regularization, model selection, latent spaces, normalizing flows, score
calibration, and ensembles. You MAY use labels during training but predict
MUST NOT require labels. Use pure JAX (Flax/Optax allowed if installed), plus
NumPy/SciPy/sklearn when needed. Training must stay within the given compute
budget. Keep output arrays finite and exactly length B. Checkpoints must be
portable across TPU and GPU (e.g., NumPy .npz, not accelerator-specific blobs).

Do not read held-out reward data or official hidden labels. Never modify the
harness. The official test is for FINAL export only, not repeated optimization.

## Command to verify

```bash
python -m fairdiscover.trial --candidate candidate.py \
  --stage /path/to/staged --run-dir /path/to/run_001 --steps 250
```

Inspect `run_001/reward.json` and logs. The `reward` is a number in [0,1]
for valid candidates, -1 for failures. We prioritize improvement on unseen
shift families and low-FPR stability, not training reconstruction accuracy.

## Suggested first experiments

1. Spectral features + conditional Mahalanobis residual score.
2. Noise-aware autoencoder with spectral and parameter-prediction losses.
3. Hybrid outlier score combining multiscale residuals and learned latent density.
4. Self-supervised masking or noise2noise objectives, with a calibrated score.

Keep changes to one file per proposal. Make each version testable, with a
single main hypothesis.
