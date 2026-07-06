#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

export REMOTE_SKYRL_DIR="${REMOTE_SKYRL_DIR:-/home/${REMOTE_USER:-sk7524_princeton_edu}/SkyRLTpu}"
exec "${repo_root}/third_party/jobman/scripts/run_tinker_math_rl.sh" "$@"
