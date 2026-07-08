#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
discover_root="${repo_root}/third_party/discover"
discover_venv="${TTD_DISCOVER_VENV:-${discover_root}/.venv-ttd-discover}"
discover_python="${TTD_DISCOVER_PYTHON:-3.11}"

export TINKER_API_KEY="${TINKER_API_KEY:-tml-dummy}"
export TINKER_BASE_URL="${TINKER_BASE_URL:-http://127.0.0.1:18000}"
export TTD_RUN_DIR="${TTD_RUN_DIR:-${repo_root}/runs/ttd_erdos}"
export UV_PROJECT_ENVIRONMENT="${discover_venv}"
export TTD_EVAL_BACKEND="${TTD_EVAL_BACKEND:-submitit}"
export TTD_SAFE_GRADE_MAX_WORKERS="${TTD_SAFE_GRADE_MAX_WORKERS:-64}"
export TTD_SLURM_PARTITION="${TTD_SLURM_PARTITION:-cpu}"
export TTD_SLURM_ACCOUNT="${TTD_SLURM_ACCOUNT:-zhuangl}"
export TTD_SLURM_MEM="${TTD_SLURM_MEM:-4G}"

cd "$discover_root"
if [[ "${TTD_DISCOVER_SYNC:-1}" != "0" ]]; then
  uv sync --extra math --python "${discover_python}"
fi

export SSL_CERT_FILE="${SSL_CERT_FILE:-$("${discover_venv}/bin/python" -c 'import certifi; print(certifi.where())')}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-${SSL_CERT_FILE}}"

exec "${discover_venv}/bin/python" "${repo_root}/tpu/run_ttd_erdos_qwen35_9b.py"
