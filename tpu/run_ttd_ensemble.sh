#!/usr/bin/env bash
# Ensemble ttt-discover run (multiple generators, one shared PUCT pool) against
# the real Tinker API. See tpu/run_ttd_ensemble.py for the member spec format.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
discover_root="${repo_root}/third_party/discover"
discover_venv="${TTD_DISCOVER_VENV:-${discover_root}/.venv-ttd-discover}"
discover_python="${TTD_DISCOVER_PYTHON:-3.11}"

# --- run identity ---
export TTD_RUN_DIR="${TTD_RUN_DIR:-${repo_root}/runs/ttd_ensemble15}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-erdos-ensemble15}"

# --- members (model:renderer:tag, comma-separated) ---
export TTD_ENSEMBLE_MODELS="${TTD_ENSEMBLE_MODELS:-openai/gpt-oss-20b:gpt_oss_high_reasoning:gptoss,nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16:qwen3:nemotron}"

# --- shared hyperparameters (authors' canonical defaults) ---
export GROUP_SIZE="${GROUP_SIZE:-8}"
export GROUPS_PER_BATCH="${GROUPS_PER_BATCH:-32}"   # per member
export LEARNING_RATE="${LEARNING_RATE:-4e-5}"
export NUM_EPOCHS="${NUM_EPOCHS:-15}"
export KL_PENALTY_COEF="${KL_PENALTY_COEF:-0.1}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export LORA_RANK="${LORA_RANK:-32}"
export PHASE1_MAX_TOKENS="${PHASE1_MAX_TOKENS:-26000}"
export CONTEXT_WINDOW="${CONTEXT_WINDOW:-32768}"
export SAVE_EVERY="${SAVE_EVERY:-0}"

# --- eval (local in-process grading on this node) ---
export TTD_EVAL_BACKEND="${TTD_EVAL_BACKEND:-local}"
export TTD_SAFE_GRADE_MAX_WORKERS="${TTD_SAFE_GRADE_MAX_WORKERS:-64}"
export NUM_CPUS_PER_TASK="${NUM_CPUS_PER_TASK:-1}"
export EVAL_TIMEOUT="${EVAL_TIMEOUT:-1100}"

# --- wandb ---
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-ttt-discover-gptoss20b}"

export UV_PROJECT_ENVIRONMENT="${discover_venv}"

# Stale TINKER_API_KEY in the shell shadows .env (load_dotenv doesn't override).
unset TINKER_API_KEY

cd "$discover_root"
if [[ "${TTD_DISCOVER_SYNC:-1}" != "0" ]]; then
  uv sync --extra math --python "${discover_python}"
fi

export SSL_CERT_FILE="${SSL_CERT_FILE:-$("${discover_venv}/bin/python" -c 'import certifi; print(certifi.where())')}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-${SSL_CERT_FILE}}"

exec "${discover_venv}/bin/python" "${repo_root}/tpu/run_ttd_ensemble.py"
