# Reproduce or resume the FAIR research

Worktree: `/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-fair-universe`, branch `agent/fair-universe-momo`. Scripts in this directory are standalone; do not launch or modify production SkyPilot jobs.

## Data

Download the public Phase-2-v2 ZIP from https://www.codabench.org/datasets/download/09b7a27a-bfb3-4031-a5b0-9b3c533f45e6/ and extract its `public_data/` folder. Use that as RAW below. The current test is `WIDE12H_bin2_2arcmin_kappa_test_phase2_new_v2.npy`; never use the superseded filename without `_v2`.

`train.py` verifies the three training-file MD5s against the published Zenodo checksums. No hidden binary OOD labels are available. All export scripts write predicted continuous scores, not ground truth.

## Runtime

The experiments use Python 3.12 and the package versions in `requirements-observed.txt`. The supplied serving environment already contains these packages. We verified this environment; we have not validated a fresh install of the requirements file.

On the assigned second machine, the Python executable is:

```
/home/gcpuser/.cache/inference-farm-rq4-qwen35-east5a-v5p32-008/envs/serving/bin/python
```

For one physical chip set `TPU_VISIBLE_CHIPS=0`, `TPU_CHIPS_PER_HOST_BOUNDS=1,1,1`, `TPU_HOST_BOUNDS=1,1,1`. Choose a free chip you own. A systemd service MUST set `LimitMEMLOCK=infinity`; the default 8 MiB limit caused TPU mmap initialization failure. Research services also use `MemoryMax=80G` and `MemorySwapMax=0`.

Keep raw data and trial files under `/dev/shm/fair-ae-research`; transfer checkpoints and final outputs to durable GPFS storage before releasing the VM. Model files alone do not preserve optimizer state; these trials save their best evaluation checkpoint, not a full optimizer resume checkpoint.

## Image models

Run from this code directory with your chosen Python interpreter, replacing RAW and RUN with real paths:

```
python train.py --raw RAW --out RUN --architecture spatial --levels 3 --latent 64 --batch 16 --steps 12000 --eval-every 1000 --val-count 1024 --reconstruction-weight 1 --physics-weight 0.25
python score.py fit --raw RAW --run RUN
python probe.py --raw RAW --run RUN
python diagnostics.py --raw RAW --run RUN --count 512
```

Reconstruction-only uses `--physics-weight 0`. Parameter-only uses `--reconstruction-weight 0 --physics-weight 1`. The global architecture uses `--architecture global`; investigated latent dimensions were 32, 64, and 128. Do not compare unused, untrained prediction-head errors for the reconstruction-only model; use the common linear probe.

Outputs are immutable by convention: use a new RUN directory for each fit. `snapshot.py SOURCE DEST` captures a coherent best checkpoint while training continues, using metadata and file timestamps to avoid a partially updated checkpoint.

## Physical comparator and learned-summary models

```
python init_physics.py --out PHYSICS_RUN
python score.py fit --raw RAW --run PHYSICS_RUN
python summary_train.py --raw RAW --features PHYSICS_RUN/density --out SUMMARY_RUN --physics-weight 0.25 --steps 10000
python score.py fit --raw RAW --run SUMMARY_RUN
python probe.py --raw RAW --run SUMMARY_RUN
python diagnostics.py --raw RAW --run SUMMARY_RUN --count 512
```

The learned-summary model uses noisy physical features as inputs and clean physical features plus five physical labels as targets. The initial physical fit supplies deterministic normal-feature caches. Keep its four `{train,val,calibration,audit}_latent.npy` caches available for efficient subsequent fits; cached arrays contain physical summaries, not official test maps.

## Export

After freezing the checkpoint, fitting density and normal calibration, and inspecting normal/proxy diagnostics:

```
python score.py export --raw RAW --run RUN --test RAW/WIDE12H_bin2_2arcmin_kappa_test_phase2_new_v2.npy --out prediction.zip
```

Export verifies frozen source/artifact hashes, a correct filename and 10,000-map count, finite predictions, and singleton/permutation independence. Use the exact runtime source directory with which that model's density was fitted: changing source afterward deliberately causes export to fail. Do not disable these checks to mix implementations. The ZIP contains only `result.json` with the `ood_scores` list.

Public test maps already include shape noise. Export adds no noise and fits nothing. It does not log test distributions or use test scores to choose models. Upload is manual; no script here submits to Codabench.
