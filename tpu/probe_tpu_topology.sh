#!/usr/bin/env bash
# Probe the physical host order of a multi-host TPU slice.
#
# After every spot VM recreation the worker-index -> physical-z mapping can
# reshuffle, and multi-host sub-slices (e.g. a 2-host train mesh) only form
# between physically adjacent hosts. This script runs a full-slice JAX init on
# all workers, prints each worker's z position, and suggests adjacent pairs.
#
# Requires: no other process holding the TPU chips (stop vLLM/tinker first),
# and a synced SkyRLTpu checkout at REMOTE_SKYRL_DIR with uv available.
set -euo pipefail

PROJECT="${PROJECT:-vision-mix}"
ZONE="${ZONE:-us-east5-a}"
TPU_NAME="${TPU_NAME:?set TPU_NAME}"
REMOTE_USER="${REMOTE_USER:-sk7524_princeton_edu}"
SSH_KEY_FILE="${SSH_KEY_FILE:-$HOME/.ssh/jobman_tpu_ed25519}"
REMOTE_SKYRL_DIR="${REMOTE_SKYRL_DIR:-/home/${REMOTE_USER}/SkyRLTpu}"
COORD_PORT="${COORD_PORT:-7788}"
PROBE_TIMEOUT_SEC="${PROBE_TIMEOUT_SEC:-240}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
probe_py="${script_dir}/probe_topology.py"
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

mapfile -t internal_ips < <(
  gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" \
    --project="$PROJECT" --zone="$ZONE" \
    --format='value(networkEndpoints.ipAddress)' | tr ';' '\n'
)
num_workers="${#internal_ips[@]}"
if (( num_workers < 2 )); then
  echo "Expected a multi-host TPU; found ${num_workers} endpoint(s)." >&2
  exit 1
fi
coordinator="${internal_ips[0]}:${COORD_PORT}"
echo "Workers: ${num_workers}; coordinator ${coordinator}"

remote_cmd_prefix="export PATH=\"\$HOME/.local/bin:\$PATH\"; cd '${REMOTE_SKYRL_DIR}'; python3 -c \"
from pathlib import Path
lf = Path('uv.lock')
lf.write_text(lf.read_text().replace('https://download-r2.pytorch.org', 'https://download.pytorch.org'))
\""

pids=()
for ((w = 0; w < num_workers; w++)); do
  gcloud alpha compute tpus tpu-vm scp "$probe_py" "${REMOTE_USER}@${TPU_NAME}:~/probe_topology.py" \
    --project="$PROJECT" --zone="$ZONE" --worker="$w" --ssh-key-file="$SSH_KEY_FILE" --quiet
done
for ((w = 0; w < num_workers; w++)); do
  (
    gcloud alpha compute tpus tpu-vm ssh "${REMOTE_USER}@${TPU_NAME}" \
      --project="$PROJECT" --zone="$ZONE" --worker="$w" --ssh-key-file="$SSH_KEY_FILE" --quiet \
      --command "${remote_cmd_prefix}; JAX_PLATFORMS=tpu timeout '${PROBE_TIMEOUT_SEC}' uv run --extra tpu --extra tinker --extra jax python ~/probe_topology.py ${w} '${coordinator}' ${num_workers}" \
      > "${tmpdir}/probe_w${w}.log" 2>&1
  ) &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done

echo
echo "worker -> physical z position:"
declare -A z_of_worker=()
for ((w = 0; w < num_workers; w++)); do
  z="$(grep -oE 'coords=\[[0-9]+, [0-9]+, [0-9]+\]' "${tmpdir}/probe_w${w}.log" | head -1 | grep -oE '[0-9]+\]$' | tr -d ']')" || true
  if [[ -z "${z:-}" ]]; then
    echo "  worker ${w}: PROBE FAILED (see below)"
    tail -3 "${tmpdir}/probe_w${w}.log" | sed 's/^/    /'
    failed=1
    continue
  fi
  z_of_worker[$w]="$z"
  echo "  worker ${w}: z=${z}"
done
if (( failed )); then
  exit 1
fi

echo
echo "adjacent worker pairs (valid multi-host TRAIN_WORKERS choices):"
for a in "${!z_of_worker[@]}"; do
  for b in "${!z_of_worker[@]}"; do
    if (( a < b )); then
      dz=$(( z_of_worker[$a] - z_of_worker[$b] ))
      if (( dz == 1 || dz == -1 )); then
        echo "  TRAIN_WORKERS=${a},${b}"
      fi
    fi
  done
done
