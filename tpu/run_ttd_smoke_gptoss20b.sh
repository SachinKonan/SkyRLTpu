#!/usr/bin/env bash
# Minimal ttt-discover smoke test vs the real Thinking Machines Tinker API.
# Uses the discover venv and the gitignored third_party/discover/.env (TINKER_API_KEY).
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
discover_root="${repo_root}/third_party/discover"
discover_venv="${TTD_DISCOVER_VENV:-${discover_root}/.venv-ttd-discover}"
discover_python="${TTD_DISCOVER_PYTHON:-3.11}"

export TTD_RUN_DIR="${TTD_RUN_DIR:-${repo_root}/runs/ttd_smoke_gptoss20b}"
# Always grade via submitit (SLURM) -- this login node has no CPU for in-process
# local grading. Dispatch eval jobs to the della cpu partition.
export TTD_EVAL_BACKEND="${TTD_EVAL_BACKEND:-submitit}"
export TTD_SAFE_GRADE_MAX_WORKERS="${TTD_SAFE_GRADE_MAX_WORKERS:-64}"
export TTD_SLURM_PARTITION="${TTD_SLURM_PARTITION:-cpu}"
export TTD_SLURM_ACCOUNT="${TTD_SLURM_ACCOUNT:-zhuangl}"
export TTD_SLURM_MEM="${TTD_SLURM_MEM:-4G}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export UV_PROJECT_ENVIRONMENT="${discover_venv}"

# The tmux/login environment may carry a stale TINKER_API_KEY (e.g. the local
# 'tml-dummy' from the TPU work). discover's load_dotenv does NOT override an
# already-set var, so that stale value would shadow third_party/discover/.env
# and cause 401s against prod Tinker. Drop it so .env is authoritative.
unset TINKER_API_KEY

cd "$discover_root"
if [[ "${TTD_DISCOVER_SYNC:-1}" != "0" ]]; then
  uv sync --extra math --python "${discover_python}"
fi

# Ensure TLS trust store is set for the Tinker HTTPS client.
export SSL_CERT_FILE="${SSL_CERT_FILE:-$("${discover_venv}/bin/python" -c 'import certifi; print(certifi.where())')}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-${SSL_CERT_FILE}}"

exec "${discover_venv}/bin/python" "${repo_root}/tpu/run_ttd_smoke_gptoss20b.py"
