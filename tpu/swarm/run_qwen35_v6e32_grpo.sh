#!/usr/bin/env bash
# Runs the proven Qwen3.5-27B GRPO cell on one mixed v6e-32 pool worker.
set -euo pipefail

: "${SKYPILOT_NODE_RANK:=0}"
: "${SKYPILOT_NODE_IPS:?SkyPilot must provide all v6e-32 TPU VM IPs}"
: "${CELL:=grpo-n}"

export JOBMAN_WORKER_ID="$SKYPILOT_NODE_RANK"
export JOBMAN_TPU_INTERNAL_IPS
JOBMAN_TPU_INTERNAL_IPS=$(printf '%s\n' "$SKYPILOT_NODE_IPS" | awk 'NF' | paste -sd, -)
node_count=$(awk -F, '{print NF}' <<<"$JOBMAN_TPU_INTERNAL_IPS")
if [ "$node_count" -ne 8 ]; then
  echo "mixed v6e-32 requires 8 TPU VMs; SkyPilot supplied $node_count" >&2
  exit 2
fi

export REMOTE_USER="${REMOTE_USER:-$(id -un)}"
export SSH_KEY_FILE="${SSH_KEY_FILE:-$HOME/ray_bootstrap_key.pem}"
if [ ! -f "$SSH_KEY_FILE" ]; then
  echo "SkyPilot cluster key is missing: $SSH_KEY_FILE" >&2
  exit 2
fi
mkdir -p "$HOME/.ssh"
ln -sfn "$SSH_KEY_FILE" "$HOME/.ssh/jobman_tpu_ed25519"

# SkyPilot may invoke a multi-node run command on every TPU VM.  The existing
# cell is deliberately head-driven and reaches the other seven ranks over the
# cluster's internal network.
if [ "$JOBMAN_WORKER_ID" != "0" ]; then
  echo "rank $JOBMAN_WORKER_ID ready; rank 0 owns the GRPO process"
  exit 0
fi

REPO="${SKYRL_REPO_DIR:-$PWD}"
bash "$REPO/tpu/jobman/cell_worker.sh"
exec bash "$REPO/tpu/jobman/cell_monitor.sh"
