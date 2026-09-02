#!/usr/bin/env bash
# Run the GPT-OSS 120B training acceptance gate on one v6e-32 pool worker.
set -euo pipefail

: "${SKYPILOT_NODE_RANK:=0}"
: "${SKYPILOT_NODE_IPS:?SkyPilot must provide all v6e-32 TPU VM IPs}"

export JOBMAN_WORKER_ID="$SKYPILOT_NODE_RANK"
export JOBMAN_TPU_INTERNAL_IPS
JOBMAN_TPU_INTERNAL_IPS=$(printf '%s\n' "$SKYPILOT_NODE_IPS" | awk 'NF' | paste -sd, -)
node_count=$(awk -F, '{print NF}' <<<"$JOBMAN_TPU_INTERNAL_IPS")
if [ "$node_count" -ne 8 ]; then
  echo "GPT-OSS 120B v6e-32 requires 8 TPU VMs; SkyPilot supplied $node_count" >&2
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
export JOBMAN_ATTEMPT_ID="${JOBMAN_ATTEMPT_ID:-${TPUSWARM_TASK_ID:-gptoss120b-v6e32-smoke}}"

# SkyPilot invokes the run command on all eight TPU VMs.  The existing smoke
# launcher is head-driven and starts the other seven JAX processes over the
# slice's internal network.
if [ "$JOBMAN_WORKER_ID" != "0" ]; then
  echo "rank $JOBMAN_WORKER_ID ready; rank 0 owns the acceptance process"
  exit 0
fi

REPO="${SKYRL_REPO_DIR:-$PWD}"
exec bash "$REPO/tpu/jobman/v6e_tunix_smoke_worker.sh"
