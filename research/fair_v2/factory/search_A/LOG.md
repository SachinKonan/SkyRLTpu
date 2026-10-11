# Feature search A: noise and non-stationarity probes (2026-10-10/11)

Graded with `factory/grade.py --mode full` on della-gh (GH200, `--budget 150`) against the v3-12-without-Oh stack.
Acceptance rule: reward > 0.006, every visible fold > 0, `hidden_ok`. Hidden per-shift numbers were not looked at;
only `hidden_ok` was read. Run outputs: `fair-data/factory/runs/A_<name>_full/`.

**No candidate was accepted.** Every candidate failed `hidden_ok`.

| Candidate | Idea | Reward | Folds | hidden_ok | Config (w, vote) | s / 10k | gradient gain | misfit alone |
|---|---|---|---|---|---|---|---|---|
| ramp4n | ramp4 + robust noise moments (level, kurtosis, L1/L2 away from peaks) | **+0.0144** | .0075 / .0107 / .0062 | no | 2, ridge | 1.3 | +0.144 | 0.191 |
| ramp4 | linear-trend coefficients / own large-scale rms, + log rms | **+0.0113** | .0115 / .0098 / .0032 | no | 2, ridge | 0.45 | +0.129 | 0.029 |
| lsexcess | ramp4 + low-k band power relative to a reference band | +0.0092 | .0028 / .0147 / .0025 | no | 2, ridge | 4.9 | +0.125 | 0.035 |
| ramp5 | ramp4 without the log rms column | +0.0068 | .0120 / .0012 / .0056 | no | 1, ridge | 0.45 | +0.087 | 0.038 |
| lsvote | ramp4 + 10 low-k log band powers | +0.0056 | .0016 / .0103 / .0013 | no | 2, ridge | 1.9 | +0.087 | 0.032 |
| ramp6 | 5 low-order mode ratios + log rms | +0.0049 | .0024 / .0047 / .0030 | no | 0, ridge | 2.3 | +0.010 | 0.023 |
| ramp | raw low-order polynomial coefficients + energies | +0.0048 | .0026 / .0047 / .0026 | no | 0.5, ridge | 0.23 | +0.021 | 0.025 |
| ramp3 | raw linear-trend coefficients (2 dims) | +0.0036 | .0065 / .0058 / −.0004 | no | 2, ridge | 0.22 | +0.053 | 0.020 |
| lsamp | log large-scale rms at 4 smoothing scales | +0.0028 | .0010 / .0031 / .0011 | no | 0.5, ridge | 5.2 | +0.010 | 0.011 |
| tilepow | tile-to-tile band-power dispersion and trends (5 DoG bands) | +0.0025 | .0014 / .0033 / .0017 | no | 0, ridge | 5.4 | +0.001 | 0.153 |
| pixkurt2 | robust per-pixel noise moments + lag-1 whiteness | −0.0038 | −.0010 / .0019 / −.0019 | no | 0, ridge | 1.3 | −0.001 | 0.196 |

Not graded:
- `pixkurt.py`: the first noise-moment version. The signal's pixel-scale peaks dominate its kurtosis (within-cosmology sd 0.21 against an expected Poisson-mixture effect of 0.02).
- `ramp2.py`: the whitened (GLS) template fit. It was less sensitive than the plain least-squares coefficient on a sanity check.

## Findings

1. **The noise shifts are already saturated in the stack.**
   - noise±0.05, noisegrad0.15 and srcclust0.15 score 0.97–1.00 base LCO.
   - The only unsaturated noise shift is poisson120 (0.021). A per-pixel scale mixture with n ≈ 120 raises the excess kurtosis by only about 0.02.
   - The pixel-scale signal leaks into every noise residual. Even after masking peaks, the best separation was about 0.7σ (L1/L2 ratio), so noise-moment families add nothing visible. Measured in `scratch/kurt_exp*.py` with self-made Gaussian vs Poisson-mixture noise on clean maps.
2. **Absorption, and how self-normalization fixes it.**
   - The raw linear-trend coefficient (ramp3) is mostly absorbed: the per-map fit moves to a cosmology with larger large-scale variance.
   - Dividing by the map's own large-scale rms (ramp4) removes that escape. Gradient gain went from +0.053 to +0.129 and reward from 0.0036 to 0.0113.
3. **Specialized detectors dilute the stack.**
   - The stack is a weighted mean of member z-scores. A gradient-only member at misfit weight 2 lifts gradient (+0.13–0.14) but costs psfpat, bary6.5, boost0.15 and smooth0.5 (−0.007 to −0.021 each).
   - That is why the hidden check fails even with all visible folds positive.
4. **Vote-only configurations give a generic +0.002–0.005** (bary9.0, supp0.15, psfpat +0.006–0.013). The pattern is the same whatever the features, so it is not specific information.

## Best near-misses

ramp4n (+0.0144) and ramp4 (+0.0113) pass reward and folds but fail `hidden_ok`. They are useful only if the real test
contains additive large-scale gradients.
