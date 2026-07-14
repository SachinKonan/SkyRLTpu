#!/usr/bin/env bash
# Full ttt-discover Erdos min-overlap run with gpt-oss-20b against the REAL
# Thinking Machines Tinker API. Uses the discover venv + third_party/discover/.env
# (TINKER_API_KEY). Hyperparameters follow the authors' canonical gpt-oss config
# (docs/api.md), adapted to gpt-oss-20b. No LoRA adapters are persisted
# (SAVE_EVERY=0 disables periodic AND final checkpoints).
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
discover_root="${repo_root}/third_party/discover"
discover_venv="${TTD_DISCOVER_VENV:-${discover_root}/.venv-ttd-discover}"
discover_python="${TTD_DISCOVER_PYTHON:-3.11}"

# --- run identity ---
export TTD_RUN_DIR="${TTD_RUN_DIR:-${repo_root}/runs/ttd_gptoss20b_full}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-erdos-gptoss20b-full}"

# --- model / renderer ---
export MODEL_NAME="${MODEL_NAME:-openai/gpt-oss-20b}"
export RENDERER_NAME="${RENDERER_NAME:-gpt_oss_high_reasoning}"

# --- discover hyperparameters (authors' canonical gpt-oss defaults) ---
export GROUP_SIZE="${GROUP_SIZE:-8}"
export GROUPS_PER_BATCH="${GROUPS_PER_BATCH:-64}"
export LEARNING_RATE="${LEARNING_RATE:-4e-5}"
export NUM_EPOCHS="${NUM_EPOCHS:-50}"
export KL_PENALTY_COEF="${KL_PENALTY_COEF:-0.1}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export LORA_RANK="${LORA_RANK:-32}"
export PHASE1_MAX_TOKENS="${PHASE1_MAX_TOKENS:-26000}"
export CONTEXT_WINDOW="${CONTEXT_WINDOW:-32768}"

# --- no LoRA adapter saving (0 disables periodic + final checkpoints) ---
export SAVE_EVERY="${SAVE_EVERY:-0}"

# --- contrastive context distillation (off by default; rl/context_distill.py) ---
export TTD_DISTILL_ENABLED="${TTD_DISTILL_ENABLED:-0}"
export TTD_DISTILL_PAIRS_PER_STEP="${TTD_DISTILL_PAIRS_PER_STEP:-16}"
export TTD_DISTILL_NUM_BETTERS="${TTD_DISTILL_NUM_BETTERS:-3}"
export TTD_DISTILL_WEIGHT="${TTD_DISTILL_WEIGHT:-0.1}"
export TTD_DISTILL_GROUNDING_FILTER="${TTD_DISTILL_GROUNDING_FILTER:-1}"
export TTD_DISTILL_LEAK_FILTER="${TTD_DISTILL_LEAK_FILTER:-1}"
export TTD_DISTILL_TEACHER_PHASE1_TOKENS="${TTD_DISTILL_TEACHER_PHASE1_TOKENS:-0}"
export TTD_DISTILL_MAX_TARGET_TOKENS="${TTD_DISTILL_MAX_TARGET_TOKENS:-8192}"
export TTD_DISTILL_MAX_CODE_CHARS="${TTD_DISTILL_MAX_CODE_CHARS:-10000}"
# Frozen stronger teacher for the critique pass (empty -> current policy).
export TTD_DISTILL_TEACHER_MODEL="${TTD_DISTILL_TEACHER_MODEL:-}"
# PUCT elitism: reserve N of 64 seeds/step for top-value states, at most one
# per lineage (0 = off).
export TTD_ELITE_SLOTS="${TTD_ELITE_SLOTS:-0}"

# --- eval backend ---
# On neuronic, compute nodes reach the internet, so the whole client runs on one
# big node with TTD_EVAL_BACKEND=local (grading runs in-process across all cores).
# Launch via tpu/run_ttd_gptoss20b_neuronic.sbatch. The TTD_SLURM_* vars below
# only matter for the submitit backend (unused on neuronic).
export TTD_EVAL_BACKEND="${TTD_EVAL_BACKEND:-local}"
export TTD_SAFE_GRADE_MAX_WORKERS="${TTD_SAFE_GRADE_MAX_WORKERS:-64}"
export TTD_SLURM_PARTITION="${TTD_SLURM_PARTITION:-all}"
export TTD_SLURM_ACCOUNT="${TTD_SLURM_ACCOUNT:-seas}"
export TTD_SLURM_MEM="${TTD_SLURM_MEM:-4G}"
# Must cover the 1000s budget_s the prompt promises the generated programs
# (300s killed ~40% of all rollouts as timeouts — the dominant failure mode).
export EVAL_TIMEOUT="${EVAL_TIMEOUT:-1100}"

# --- wandb (online; WANDB_API_KEY comes from third_party/discover/.env) ---
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-ttt-discover-gptoss20b}"

export UV_PROJECT_ENVIRONMENT="${discover_venv}"

# Drop any stale TINKER_API_KEY from the login/tmux env so .env is authoritative
# (discover's load_dotenv does not override an already-set var).
unset TINKER_API_KEY

cd "$discover_root"
if [[ "${TTD_DISCOVER_SYNC:-1}" != "0" ]]; then
  uv sync --extra math --python "${discover_python}"
fi

# TLS trust store for the Tinker HTTPS client.
export SSL_CERT_FILE="${SSL_CERT_FILE:-$("${discover_venv}/bin/python" -c 'import certifi; print(certifi.where())')}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-${SSL_CERT_FILE}}"

exec "${discover_venv}/bin/python" "${repo_root}/tpu/run_ttd_smoke_gptoss20b.py"
