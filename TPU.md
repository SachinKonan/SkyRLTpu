# SkyRLTpu TPU Workflow

This repository is the SkyRL checkout that should run on TPU. The helper scripts
sync this committed checkout to every TPU worker at:

```text
/home/sk7524_princeton_edu/SkyRLTpu
```

The launch scripts in `third_party/jobman` default to that path, so they do not
pull an untracked SkyRL branch on the TPU.

## Layout

- `.`: pinned SkyRL source.
- `third_party/jobman`: Jobman submodule with the v5p-64 spot configs and Tinker launch scripts.
- `tinker-cookbook`: planned future submodule. For now, the math-RL runner still installs the cookbook from its nightly git spec through `uv`.

## Setup

Initialize submodules after cloning:

```bash
git submodule update --init --recursive
```

Reserve or reuse the east v5p-64 spot TPU:

```bash
cd third_party/jobman
jobman create config/skyrl_tinker_v5p64_spot_us_east5a.yaml
```

Sync this exact SkyRLTpu commit to all TPU workers:

```bash
cd /scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu
./tpu/sync_skyrl_to_tpu.sh
```

Start the Tinker server from the synced SkyRLTpu checkout:

```bash
TP_SIZE=4 FSDP_SIZE=8 SAMPLE_MAX_NUM_SEQUENCES=256 \
  MODEL_NAME=Qwen/Qwen3.5-4B \
  ./tpu/start_skyrl_tinker.sh
```

Run the math-RL client:

```bash
MODEL_NAME=Qwen/Qwen3.5-4B RENDERER_NAME=qwen3_5 \
  GROUPS_PER_BATCH=64 MAX_STEPS=5 \
  LOG_PATH=/home/sk7524_princeton_edu/gcs/skyrl-logs/math-rl-qwen35-4b-lora-tp4 \
  ./tpu/run_tinker_math_rl.sh
```

The sync script refuses to run from a dirty checkout. Commit the SkyRL changes
you want to test, sync, then launch.
