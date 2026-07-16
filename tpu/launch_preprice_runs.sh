#!/usr/bin/env bash
# The three pre-price-increase launches (gate: beta=0.2 partial evidence from
# erdos-gptoss20b-distelite15-b02, ~5 steps). Set BETA to the winner (0.2 or 0.1)
# and run:  bash tpu/launch_preprice_runs.sh <BETA>
# Submission order = priority order: 120b solo, then 20b+20b ensemble, then
# 20b+120b ensemble (user-specified ordering).
set -euo pipefail
BETA="${1:?usage: launch_preprice_runs.sh <beta e.g. 0.1>}"
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
cd "$REPO"

# ---- 1. gpt-oss-120b solo: distill + elite slots + chosen beta, weights saved
mkdir -p runs/ttd_gptoss120b_distelite
sbatch --parsable -t 2-00:00:00 \
  -o "$REPO/runs/ttd_gptoss120b_distelite/slurm_%j.out" \
  -e "$REPO/runs/ttd_gptoss120b_distelite/slurm_%j.out" \
  -J ttd-120b-delite \
  --export=ALL,MODEL_NAME=openai/gpt-oss-120b,NUM_EPOCHS=20,SAVE_EVERY=10,EXPERIMENT_NAME=erdos-gptoss120b-distelite,TTD_RUN_DIR="$REPO/runs/ttd_gptoss120b_distelite",TTD_DISTILL_ENABLED=1,TTD_DISTILL_WEIGHT="$BETA",TTD_ELITE_SLOTS=8 \
  tpu/run_ttd_gptoss20b_neuronic.sbatch

# ---- 2. ensemble 20b + 20b peers: symmetric cross-model distill, shared pool
mkdir -p runs/ttd_ens_20b20b
sbatch --parsable -t 2-00:00:00 \
  -o "$REPO/runs/ttd_ens_20b20b/slurm_%j.out" \
  -e "$REPO/runs/ttd_ens_20b20b/slurm_%j.out" \
  -J ttd-ens-20-20 \
  --export=ALL,NUM_EPOCHS=15,SAVE_EVERY=10,EXPERIMENT_NAME=erdos-ens-20b20b,GROUPS_PER_BATCH=64,TTD_RUN_DIR="$REPO/runs/ttd_ens_20b20b",TTD_ENSEMBLE_MODELS="openai/gpt-oss-20b:gpt_oss_high_reasoning:alpha,openai/gpt-oss-20b:gpt_oss_high_reasoning:beta",TTD_DISTILL_ENABLED=1,TTD_DISTILL_WEIGHT="$BETA",TTD_ELITE_SLOTS=8 \
  tpu/run_ttd_ensemble_neuronic.sbatch

# ---- 3. ensemble A=20b, B=120b: symmetric cross-model distill, shared pool
mkdir -p runs/ttd_ens_20b120b
sbatch --parsable -t 2-00:00:00 \
  -o "$REPO/runs/ttd_ens_20b120b/slurm_%j.out" \
  -e "$REPO/runs/ttd_ens_20b120b/slurm_%j.out" \
  -J ttd-ens-20-120 \
  --export=ALL,NUM_EPOCHS=15,SAVE_EVERY=10,EXPERIMENT_NAME=erdos-ens-20b120b,GROUPS_PER_BATCH=64,TTD_RUN_DIR="$REPO/runs/ttd_ens_20b120b",TTD_ENSEMBLE_MODELS="openai/gpt-oss-20b:gpt_oss_high_reasoning:g20,openai/gpt-oss-120b:gpt_oss_high_reasoning:g120",TTD_DISTILL_ENABLED=1,TTD_DISTILL_WEIGHT="$BETA",TTD_ELITE_SLOTS=8 \
  tpu/run_ttd_ensemble_neuronic.sbatch

squeue -u "$USER" -o "%.10i %.2t %.6C %.16j %R" | grep -E "ttd|JOBID"
