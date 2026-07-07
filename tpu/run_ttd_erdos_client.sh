#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
discover_root="${repo_root}/third_party/discover"

export TINKER_API_KEY="${TINKER_API_KEY:-tml-dummy}"
export TINKER_BASE_URL="${TINKER_BASE_URL:-http://127.0.0.1:18000}"
export TTD_RUN_DIR="${TTD_RUN_DIR:-${repo_root}/runs/ttd_erdos}"

cd "$discover_root"
exec uv run --extra math python "${repo_root}/tpu/run_ttd_erdos_qwen35_9b.py"
