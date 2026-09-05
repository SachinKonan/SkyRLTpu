#!/usr/bin/env bash
# Prompt-v2 2x2 on v5p-8 in us-east5-a, running in PARALLEL with the v6e arms.
#
# A second, independent pool: v5p lives in us-east5-a and v6e only in
# us-east5-b, so these requests do not compete with the v6e ones -- whichever
# pool frees up first produces the cells.
#
# Both models already have PROVEN 32k v5p-8 XLA caches
# (vllm-xla-cache-{qwen35,gemma4}-32k-b16-v5p8), so this is a previously
# validated serving path at the same context length, not a new configuration.
#
# RUNTIME is v2-alpha-tpuv5, which is the sbatch default AND correct here --
# the mismatch that broke the v6e attempts was asking for v6e with this image.
#
# CAVEAT to record with any result: the baseline was served on v6e-8/TP=8 and
# these cells would be v5p-8/TP=4. The variable under test is the PROMPT, and
# sharding changes numerics not the sampling distribution, but the serving
# hardware differs from the baseline and every reported number must say so.
set -uo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
cd "$REPO"
S=$REPO/tpu/pallas_arena/probe

launch() {  # launch <tag> <cells> <seedfile> <obstag> [gemma]
  ( export SUP_TAG="$1" SMOKE_CELLS="$2" SEED_FILE="$3"
    export SEED_OBS="$REPO/runs/pallas_arena/seed-obs-$4.txt"
    export SEED_REWARD="$(cat "$REPO/runs/pallas_arena/seed-reward-$4.txt")"
    export LIB_IMPORTS=1
    export THINK_BUDGET=12288 QWEN_MAXLEN=32768 GROUP_SIZE=32
    export ACCEL=v5p-8 SERVE_TP=4 RUNTIME=v2-alpha-tpuv5
    export BENCH_ZONES="us-east5-a"
    export LAND_DEADLINE_S=64800 LAND_GRACE_TRIES=12
    export BUCKET=gs://sk7524-tinker-tpu-us-east5
    if [ "${5:-}" = "gemma" ]; then
      export QWEN_MODEL=google/gemma-4-31B-it
      export QWEN_HF_GCS="${BUCKET}/hf-cache-gemma4"
      export QWEN_XLA_GCS="${BUCKET}/vllm-xla-cache-gemma4-32k-b16-v5p8"
      export EXTRA_PIP="transformers==5.14.0" ENABLE_THINKING_KWARG=1
    else
      export QWEN_MODEL=Qwen/Qwen3.5-27B
      export QWEN_HF_GCS="${BUCKET}/hf-cache"
      export QWEN_XLA_GCS="${BUCKET}/vllm-xla-cache-qwen35-32k-b16-v5p8"
    fi
    nohup bash "$S/bench_supervisor.sh" > "runs/pallas_arena/bench-sup-$1.log" 2>&1 &
    echo "launched $1 (v5p-8/TP=4, us-east5-a, model=$QWEN_MODEL)" )
}

launch qwen-v5p-splash   "splash_attention:rf3s" "$S/seed_splash_flash.py"  splash
launch gemma-v5p-splash  "splash_attention:rf3s" "$S/seed_splash_flash.py"  splash gemma
launch qwen-v5p-rglru    "rg_lru:rf3s"           "$S/seed_rglru_active.py"  rglru
launch gemma-v5p-rglru   "rg_lru:rf3s"           "$S/seed_rglru_active.py"  rglru gemma
echo "all four v5p arms launched $(date +%H:%M:%S)"
