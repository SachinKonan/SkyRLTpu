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
- `third_party/tinker-cookbook`: pinned Tinker cookbook submodule used by the math-RL client recipe.

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

Run the cookbook Qwen3.5-9B MATH target against worker 0's Tinker API:

```bash
./tpu/run_tinker_math_rl_qwen35_9b.sh
```

That starts a local `skyrl-tinker-tunnel` tmux session forwarding
`127.0.0.1:18000` to the TPU worker 0 API server, then starts a local
`skyrl-math-rl` tmux session with:

```text
env=math model_name=Qwen/Qwen3.5-9B group_size=16 groups_per_batch=64
learning_rate=2e-5 max_tokens=512 lora_rank=32 max_steps=180
```

Plot a reward/correctness curve from a completed metrics file:

```bash
uv run --with matplotlib \
  python tpu/plot_math_rl_metrics.py /path/to/metrics.jsonl \
    --monitoring benchmark_artifacts/math_rl_qwen35_9b_tpu_monitoring.json \
    --run-dir /path/to/run-dir \
    --out benchmark_artifacts/math_rl_qwen35_9b_reward_curve.png \
    --summary benchmark_artifacts/math_rl_qwen35_9b_summary.json
```

The plot also includes derived generation diagnostics when the metrics file has
the standard MathRL token and timing keys: generated tokens/sec for the full
batch, generated tokens/sec per trajectory, average prompt/generated length, and
step timing. Passing `--monitoring` and `--run-dir` adds step-aligned tensor core
and HBM utilization from Cloud Monitoring.

Fetch TPU utilization from Cloud Monitoring for the same run window:

```bash
python tpu/fetch_tpu_monitoring.py \
  --project vision-mix \
  --location us-east5-a \
  --start 2026-07-08T03:10:00Z \
  --end 2026-07-08T18:15:00Z \
  --out benchmark_artifacts/math_rl_qwen35_9b_tpu_monitoring.json
```

The sync script refuses to run from a dirty checkout. Commit the SkyRL changes
you want to test, sync, then launch.
