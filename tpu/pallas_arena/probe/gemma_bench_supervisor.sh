#!/usr/bin/env bash
# Keep resubmitting the gemma rf3c bench until a generations file exists.
# The evolve_smoke sbatch gives up after its bounded QR-wait (correct for a
# slurm job); through a spot drought that means several attempts. This loop
# is login-safe: it only sleeps, checks squeue, and resubmits.
set -uo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
cd "$REPO"

LOCK=/tmp/gemma-bench-supervisor.lock
exec 9>"$LOCK"
flock -n 9 || { echo "another supervisor holds $LOCK"; exit 0; }

export SMOKE_CELLS="rg_lru:rf3c,splash_attention:rf3c"
export QWEN_MODEL=google/gemma-4-31B-it QWEN_MAXLEN=16384
export QWEN_HF_GCS=gs://sk7524-tinker-tpu-us-east5/hf-cache-gemma4
export QWEN_XLA_GCS=gs://sk7524-tinker-tpu-us-east5/vllm-xla-cache-gemma4-31b-16k
export EXTRA_PIP="transformers==5.14.0" ENABLE_THINKING_KWARG=1

job=""
while true; do
  if [ -n "$job" ]; then
    gens="runs/pallas_arena/evolve-smoke-gens-${job}.jsonl"
    if [ -s "$gens" ] && ! squeue -h -j "$job" -o %T 2>/dev/null | grep -q .; then
      echo "[supervisor] gens landed: $gens $(date +%H:%M:%S)"
      echo "$job" > runs/pallas_arena/gemma-bench-final-job.txt
      exit 0
    fi
    if squeue -h -j "$job" -o %T 2>/dev/null | grep -q .; then
      sleep 300; continue
    fi
    if [ -s "$gens" ]; then
      echo "[supervisor] gens landed post-exit: $gens"
      echo "$job" > runs/pallas_arena/gemma-bench-final-job.txt
      exit 0
    fi
    echo "[supervisor] job $job died without gens $(date +%H:%M:%S); resubmitting"
  fi
  job=$(sbatch --export=ALL --job-name=bench-gemma-rf3c \
          tpu/pallas_arena/probe/evolve_smoke.sbatch 2>/dev/null | awk '{print $4}')
  [ -n "$job" ] || { echo "[supervisor] sbatch failed; retry in 600s"; sleep 600; continue; }
  echo "[supervisor] submitted $job $(date +%H:%M:%S)"
  sleep 300
done
