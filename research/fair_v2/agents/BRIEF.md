# FAIR Universe Phase-2 OoD sprint: shared brief for all agents (2026-10-10)

## Situation
- Competition: FAIR Universe Weak Lensing ML Uncertainty Challenge, **Phase 2 = per-map out-of-distribution (OoD) detection**
  (https://www.codabench.org/competitions/10902/). Deadline **2026-10-11 23:59 UTC**.
- Metric: mean TPR at 100 log-spaced FPRs in [0.001, 0.05] (`model.phase2_score`). ~60% of the weight is at FPR < 0.01,
  so the **normal (InD) score tail** matters as much as sensitivity.
- Leaderboard: leader **0.7835**. Our new pipeline scored **0.5272**; the previous, completely different pipeline scored **0.5253**.
  Interpretation: score ~ q + 0.0125(1-q) => both catch the SAME ~52% of OoD maps cleanly and are BLIND to the rest.
  We must find information our statistics discard. Small tuning of the same features will not help.
- Organizers' baseline figure (power-spectrum chi2 / CNN / autoencoder on the labeled test): OoD maps differ from InD in
  small-scale amplitude/structure in BOTH directions (some OoD reconstruct better than any InD map, some worse);
  a large fraction of OoD overlaps InD in power-spectrum chi2. White paper: "some test data generated from alternative
  physical models"; baselines fail "across some OoD categories". Organizers' prior OoD paper (Diao, Dai & Seljak 2025,
  arXiv:2505.00632): BCM baryons vs HMcode-transfer baryons; field-level density (diffusion / flow matching) >> feature-level.

## Hard rules (competition): violating these disqualifies us
1. Every test score must be a fixed function of ONE map. No batch statistics, clustering, kNN across test maps,
   normalization or ranking across test maps. Output must not depend on other test maps (use fixed-shape batches, eval mode).
2. The public test set must NOT be used to fit, train, calibrate, normalize, select or tune anything. Do not look at test-score
   distributions to make decisions. Only produce per-map test scores with frozen models at the very end.
3. Synthetic OoD built from TRAINING maps is allowed (diagnostics or training negatives).

## Data (della scratch, read-only)
- `/scratch/gpfs/ZHUANGL/sk7524/fair-data/`:
  `WIDE12H_bin2_2arcmin_kappa_newrealization.npy` (101 cosmologies x 256 realizations x 132019 valid pixels, float16, NOISELESS),
  `label_newrealization.npy` (101,256,5) = [Omega_m, S8, T_AGN, f0, Delta_z],
  `WIDE12H_bin2_2arcmin_mask.npy` (1424x176 bool, 132019 True),
  `WIDE12H_bin2_2arcmin_kappa_test_phase2_new_v2.npy` (10000 x 132019 float16, already NOISY), `tk_grid.npz` (HMcode T(k) grid).
- Noise: Gaussian sigma = 0.4/sqrt(2*30*4) = 0.025820 per observed pixel; test maps are float16 after noise ->
  emulate with `data.add_noise` (adds noise, casts to float16 and back).
- **Each realization index r carries a FIXED baryon model (T_AGN, f0), identical in all 101 cosmologies** (only 256 baryon
  models exist). Delta_z is continuous per map. Splitting by realization holds out whole baryon models.
- Splits (map-level, all 256 baryon models in fit): `data.index('fit3')` (20715 maps), `'calib3'` (2582), `'audit3'` (2559).
  Use fit3 for training, calib3 for standardization/calibration, audit3 for normal-tail evaluation and synthetic shifts.

## Harness (worktree `/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-fair-universe-improved`, code in `research/fair_v2/`)
- `data.py` (splits, noise, `SHIFTS` + `noisy_shift`), `stats.py` (466 summary stats), `cnn.py` (CNN regressor net),
  `pipeline_v3.py` (how features are extracted for every set), `model.py` (conditional Gaussian likelihood
  `Likelihood`, `Standardizer`, `phase2_score`), `fit_v3.py` (fit + evaluate + score test for any feature source).
- **Feature-source format** (so new features combine with existing ones on the SAME noisy maps):
  directory `/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3/<source>/` with files `fit.npz`, `calib.npz`, `audit.npz`,
  `shift_<name>.npz` (one per `data.SHIFTS` key) and `test.npz`; each npz has `feats` float32 (N, V, d),
  plus `index`, `labels` for non-test sets. Views/seeds MUST match pipeline_v3: fit3 2 views seed 31, calib3 2 views seed 32,
  audit3 1 view seed 33, shifts 1 view seed 300+i (i = position in data.SHIFTS) built with
  `data.noisy_shift(name, full, m, gen, labels)`; for view v: `gen.manual_seed(seed*1000+v)`, iterate maps in index order in
  batches of 64 drawing noise per batch exactly like `pipeline_v3.embed.run` (copy that loop) so noise is bit-identical.
  Test: batches of 64 with zero padding to fixed shape. Run on the same GPU type when reusing seeds (della-gh GH200).
- Evaluate: `fit_v3.py --root /scratch/gpfs/ZHUANGL/sk7524/fair-data/v3 --models "X=<source>:raw" --combos "mean:X+C"
  --out <dir> [--test]`. Existing sources: `stats` (466 dims; good subsets `stats:dog`, `stats:all`), `cnn0` (CNN embedding
  256 + 5 predictions). Current best = `mean:Sd+C` with Sd=stats:dog, C=cnn0:raw (LB 0.5272).
  fit_v3 reports normal quantiles of the standardized score (want 99.9% near 3-4) and per-shift metric scores.
- Synthetic shifts are imperfect proxies (multiplicative bias is undetectable by construction; noise-type shifts are easy).
  Do not over-tune to them; prefer detectors that are principled and complementary.

## Compute etiquette
- **della-gh** (GH200 96 GB, aarch64): `ssh della-gh`, python `/scratch/gpfs/ZHUANGL/sk7524/fair-env-gh/bin/python`.
  Processes are killed after roughly 10 minutes of runtime: keep each run short and resumable (checkpoint, skip finished
  files). Run inside a blocking ssh call (`timeout 1500 ssh -o BatchMode=yes -n della-gh '...'`). Shared by all agents:
  check `nvidia-smi` and keep memory modest.
- **Slurm gpu-test** (A100 80 GB, max 59 min, at most 3 running jobs for the whole user): `sbatch --qos=gpu-test -A zhuangl
  --gres=gpu:1 -c 8 --mem=64G -t 00:59:00 ...`, python `/scratch/gpfs/ZHUANGL/sk7524/fair-env/bin/python` (x86).
  At most ONE gpu-test job per agent at a time; checkpoint every epoch and resume across jobs. Example script:
  `/scratch/gpfs/ZHUANGL/sk7524/fair-data/slurm/cnn_gtest.sbatch`. You may also try `--qos=gpu-short` (up to 24 h; may queue).
  CPU jobs: `sbatch -A zhuangl -c 32 --mem=96G -t 03:00:00`.
- Never cancel or touch jobs you did not submit. Never touch TPU clusters, SkyPilot, or other della services.
  Set `UV_CACHE_DIR=/scratch/gpfs/ZHUANGL/sk7524/.cache/uv` if installing anything (home quota is tiny).

## Where to write
- Code: `research/fair_v2/agents/<your-name>/` (do not edit shared files; copy what you need). Do NOT git commit.
- Data/outputs: `/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/<your-name>/`; new feature sources under
  `/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3/<your-source-name>/`.
- Final report: `/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/<your-name>/REPORT.md` with: method, why it should see what
  current statistics miss, normal-tail quantiles, per-shift scores vs the current best, path to test z-scores
  (per-map, frozen), honest caveats, and your recommendation.
