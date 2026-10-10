# Feature-factory grader

`grade.py` scores one candidate feature family by how much it improves the current stack on held-out
cosmologies. It implements the evaluator from `../MOMO_HILLCLIMB_ENV.md` section 3 on top of the existing
consistency harness (`agents/consistency/`).

## Usage

```bash
cd research/fair_v2
python factory/grade.py factory/examples/dog.py --out <dir> --mode screen   # quick pre-filter
python factory/grade.py factory/examples/dog.py --out <dir> --mode full     # the real reward
sbatch factory/run_grade.sbatch factory/examples/dog.py:screen factory/examples/dog.py:full   # gpu-test A100
```

Outputs:
- `<dir>/result.json` holds everything.
- `<dir>/proposer_view.json` is the same without the hidden-shift field, keeping only `hidden_ok`. Show this
  file to a proposer (MOMO or an agent).

Reruns resume from where they stopped:
- Per-set features are cached in `feats_<mode>/`, and per-fold fits in `fold<k>_<mode>.pkl`.
- Editing the candidate (a new SHA-256 of its source) clears the cache.

## Candidate contract

```python
def features(maps, mask):   # maps: torch.float32 (B, 1424, 176) on the GPU, zero outside the footprint
                            # mask: torch.bool (1424, 176), same device
    return feats            # (B, d) torch tensor or numpy array; 1 <= d <= 4096
```

Hard failures (`ok: false`, `contract.fail` says why):

| Check | How it is tested |
|---|---|
| Per-map | 8 audit maps scored together vs one at a time vs shuffled. More than 1% of entries off by more than 1e-3 of the column scale fails. |
| Deterministic | Two identical calls. |
| Finite | On the contract batch and on every extracted set. |
| Shape | Output is `(B, d)` with 1 ≤ d ≤ 4096. |
| Runtime | 512 maps (including host-to-GPU copy), extrapolated to 10k maps, must be ≤ `--budget` (default 60 s). The final Codabench run has 1200 s for the whole stack. |
| Import | Import errors fail. |

The candidate runs in-process with normal file access. A MOMO deployment must sandbox it and must not let it
read the public test file or the shift generators (`data.py`). The grader itself never reads the test file.

## What is computed

**Inputs.** The frozen noisy maps in `fair-data/agents/diffusion/noisy`. These are the same realizations every
stack member was scored on, so per-map scores line up.

**LCO folds.** Folds k = 0, 1, 2 come from `holdout3.masks`. Each fold holds out 20 cosmologies.
- Normal (InD): held-out calib3 (2 views) plus held-out audit3 maps.
- OoD: the 19 shifts of the held-out audit3 maps.
- The candidate is refit on fit3 minus the fold's cosmologies.

**Candidate parts per fold:**
- `ridge`: a ridge θ-probe, features to 5 normalized parameters, with alpha chosen on the fold's calib3 maps
  (`probes.ridge_fit`).
- `prof` (full only): the conditional likelihood (`JLikelihood`, std variant: width 256, 3000 steps, jitter 0.1),
  profiled per map. It gives θ̂ as a vote and the standardized copula energy as a misfit member `X`
  (`holdout3`/`components` conventions).

**Stack.** The v3-12 recipe without Oh:

```
Sd .5, C 1, Dr 2.5, xSL 1, consP8 .5, consRplus 3, N2 .5, PsBh .5, SS 1
```

`consRplus` is the Mahalanobis disagreement over the views `b0–b3, cnn, cnn1, pred0, pred1, dog, den_r, den_sd`
plus the `den_b` probe. Oh is excluded because its LCO gain comes from overlap between its training generators and
the proxy shifts. The stack's per-fold members are cached once in `fair-data/factory/cache/fold{k}.pkl` (built
from `lco3/set{k}_std`, about 5 s on CPU).

The base stack's LCO: all-19 mean 0.4350 (folds 0.469 / 0.458 / 0.378), visible-13 0.4772, hidden-6 0.3435.
Without SS it is 0.4319, matching the recorded STACKDp3_denb value of 0.4316.

**Configurations.** A configuration is a misfit weight w (screen: 0; full: 0, .5, 1, 2) crossed with a vote
(none, ridge, prof), minus (0, none).
- The vote is added as an extra view inside `consRplus`.
- One configuration is chosen for all folds, by the reward below.
- Every configuration's reward is reported, so a proposer can see whether the candidate helps as a misfit or as a
  vote.

**Reward.**

```
gain_k   = mean over VISIBLE shifts of phase2_score(stack+cand) - phase2_score(stack)   on fold k
penalty  = 0.01 * (max(0, Δexc@1%) + max(0, Δexc@0.5%))     # percentage points; exc = held-out normal maps
                                                              # above in-sample normal 99% / 99.5% quantiles
reward   = mean_k gain_k + min_k gain_k - penalty
hidden_ok = mean_k (same gain on HIDDEN shifts) > 0
```

Proxy noise is about ±0.004 per fold. A sensible acceptance rule:
- reward > about 0.006,
- every visible fold gain > 0,
- `hidden_ok`,
- and, after acceptance, append the member to `STACK` and rebuild the cache, so the next candidate must beat
  the grown stack.

**Screen mode** is a cheap pre-filter: ridge vote only, with no likelihood fit.
- It uses one fit3 view on every 5th fit map (4,143 maps).
- calib3 and audit3 are used in full.
- Each shift uses every 3rd map that is held out in some fold (492 per shift).
- The base stack is evaluated on the same maps, so gains stay paired.

## Hidden shifts (one per physics family; the visible siblings stay)

| Family | Visible | Hidden |
|---|---|---|
| Baryons / power-spectrum shape | bary6.5, bary9.0, supp0.15, boost0.15 | **baryDMO** |
| Shear calibration | mult+0.05 | **mult-0.05** |
| Noise | noise+0.05, noise-0.05, noisegrad0.15, poisson120 | **noise_t5** |
| PSF / smoothing | smooth0.5, psfpat | **aniso0.7** |
| Astrophysical | srcclust0.15 | **ia0.2** |
| Large-scale / non-linear systematics | gradient | **peakcut0.3** |

The hidden shifts appear only in `result.json["hidden"]`. Of the organizers' planned test systematics, IA is
hidden entirely, so a proposer gets no IA signal and `hidden_ok` checks whether its gains transfer.

## Validation (2026-10-10)

The examples in `examples/` were graded on della-gh (GH200) and gpu-test A100s. Runs are in
`fair-data/factory/runs/` (`gh_*` = GH200).

| Candidate | Mode | ok | Reward | hidden_ok | Notes |
|---|---|---|---|---|---|
| `batch_violator.py` (log power minus batch mean) | screen | **fail** | n/a | n/a | per-map check: 100% of entries off; shuffled order passes, as it should for a batch mean |
| `random_feats.py` (16 Gaussians seeded by a map hash) | screen | ok | +0.0008 | no | ridge θ-probe calib MSE 1.00 (no information) |
| `random_feats.py` | full | ok | +0.0023 | no | best of 11 configurations: this is the selection inflation of a useless family |
| `dog.py` (stats.py DoG, 56 dims) | screen | ok | −0.0052 | no | features match `v3/stats` columns to 5e-5 relative |
| `dog.py` | full | ok | −0.0018 (GH200) / −0.0043 (A100) | no | standalone misfit visible-13 0.2937 vs cached Sd 0.2928; re-adding a stack member does not help, as expected |
| `psband.py` (stats.py power spectrum, 28 dims) | screen | ok | −0.0058 | no | features match to 7e-7 relative |
| `psband.py` | full | ok | −0.0019 | no | already covered by PsBh and consP8 |
| `dirpower.py` (new: power anisotropy, 36 dims) | full | ok | **+0.0086** | **yes** | all 3 folds positive (+.007/+.0035/+.005); psfpat +0.054 visible, aniso0.7 +0.042 hidden; all-19 LCO 0.4350 → 0.4405 |

- **Reproducibility.** Identical inputs give identical rewards up to likelihood-refit noise. dog full gave −0.0018
  on GH200 and −0.0043 on A100, so treat about ±0.003 as the refit noise of a full reward.
- **Acceptance threshold.** A threshold of about 0.006 with all folds positive and `hidden_ok` clears both the
  noise and the +0.002 selection inflation.
- **dirpower.** It is the first candidate found by this grader that clears that bar. It is a PSF / shear-systematics
  probe, and its gain transfers from the visible psfpat shift to the hidden anisotropic-PSF shift.

Wall times (one candidate; dumps partly in the page cache):

| Mode | A100 (gpu-test) | GH200 (della-gh) |
|---|---|---|
| screen (21k maps, ridge only) | 7–16 s | 3–60 s (Python-loop candidates are slow on Grace) |
| full (77k maps, 3 likelihood refits) | 101–140 s: about 13–43 s features, about 28 s per fold fit | 109–207 s |

The likelihood refits dominate full mode. On one A100 that is about 25–35 full grades per hour, or several
hundred screens per hour. Keep the dumps (25 GB) in the page cache or `/dev/shm` for a long-running service.

## Not done / caveats

- **No sandbox.** Candidates run in-process with normal file access.
- **Learned featurizers** (networks trained on fit3) would need per-fold retraining for an honest LCO. This grader
  assumes fixed functions.
- **The stack is fixed** at v3-12 without Oh. To accept a candidate, add it to `STACK` and to the per-fold caches,
  then delete `fair-data/factory/cache/`.
- **The proxy is validated on only 3 LB points.** LB probes remain the final judge.
