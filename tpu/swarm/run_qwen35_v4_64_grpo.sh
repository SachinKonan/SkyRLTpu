#!/usr/bin/env bash
# Run one Qwen3.5-27B GRPO/Erdos cell on an eight-host v4-64 pool worker.
set -euo pipefail

: "${SKYPILOT_NODE_RANK:=0}"
: "${SKYPILOT_NODE_IPS:?SkyPilot must provide all v4-64 TPU VM IPs}"
: "${CELL:=grpo-n}"

export JOBMAN_WORKER_ID="$SKYPILOT_NODE_RANK"
export JOBMAN_TPU_INTERNAL_IPS
JOBMAN_TPU_INTERNAL_IPS=$(printf '%s\n' "$SKYPILOT_NODE_IPS" | awk 'NF' | paste -sd, -)
node_count=$(awk -F, '{print NF}' <<<"$JOBMAN_TPU_INTERNAL_IPS")
if [ "$node_count" -ne 8 ]; then
  echo "mixed v4-64 requires eight TPU VMs; SkyPilot supplied $node_count" >&2
  exit 2
fi

export REMOTE_USER="${REMOTE_USER:-$(id -un)}"
export SSH_KEY_FILE="${SSH_KEY_FILE:-$HOME/ray_bootstrap_key.pem}"
mkdir -p "$HOME/.ssh"

# SkyPilot invokes the run command on every TPU VM. The existing cell launcher
# is head-driven and reaches the other seven ranks over their internal IPs.
if [ "$JOBMAN_WORKER_ID" != "0" ]; then
  echo "rank $JOBMAN_WORKER_ID ready; rank 0 owns the GRPO process"
  exit 0
fi
if [ ! -f "$SSH_KEY_FILE" ]; then
  echo "SkyPilot TPU pod key is missing: $SSH_KEY_FILE" >&2
  exit 2
fi
ln -sfn "$SSH_KEY_FILE" "$HOME/.ssh/jobman_tpu_ed25519"

REPO=$(readlink -f "${SKYRL_REPO_DIR:-$PWD}")
export SKYRL_REPO_DIR="$REPO"
export TPUSWARM_BUNDLE_ID="${TPUSWARM_BUNDLE_ID:-$(basename "$REPO")}" # Runtime generation identity.

# A SkyPilot recovery starts a new internal job on an existing pool worker.
# Stop the previous attempt's client before any backend is reconciled; otherwise
# it can continue sending requests while vLLM/trainer are being replaced.
if [[ -n "${SKYPILOT_INTERNAL_JOB_ID:-}" ]]; then
  session="${CELL_SESSION:-cell}"
  if tmux has-session -t "=$session" 2>/dev/null; then
    echo "fencing stale client before SkyPilot attempt $SKYPILOT_INTERNAL_JOB_ID"
    tmux kill-session -t "=$session" 2>/dev/null || true
  fi
  rm -f "$HOME/ENGINE-SICK"
fi
if [[ "${V4_64_AUTO_TOPOLOGY:-1}" == "1" ]]; then
  topology_cache="$HOME/.cache/tpuswarm/v4-64-topology.env"
  topology_fingerprint=$(printf '%s' "$JOBMAN_TPU_INTERNAL_IPS" | sha256sum | awk '{print $1}')
  topology_env=$(mktemp)
  if [[ -f "$topology_cache" ]]; then
    # shellcheck disable=SC1090
    source "$topology_cache"
  fi
  if [[ "${V4_64_TOPOLOGY_FINGERPRINT:-}" == "$topology_fingerprint" &&
        -n "${TRAIN_WORKERS:-}" && -n "${VLLM_WORKERS:-}" ]]; then
    echo "reusing cached v4-64 topology TRAIN_WORKERS=$TRAIN_WORKERS VLLM_WORKERS=$VLLM_WORKERS"
  else
    unset TRAIN_WORKERS VLLM_WORKERS
    OUTPUT_ENV_FILE="$topology_env" bash "$REPO/tpu/swarm/discover_v4_64_topology.sh"
    printf 'V4_64_TOPOLOGY_FINGERPRINT=%q\n' "$topology_fingerprint" >> "$topology_env"
    mkdir -p "$(dirname "$topology_cache")"
    install -m 600 "$topology_env" "$topology_cache"
  fi
  # shellcheck disable=SC1090
  source "$topology_cache"
  rm -f -- "$topology_env"
  export TRAIN_WORKERS VLLM_WORKERS
fi
bash "$REPO/tpu/swarm/reconcile_v4_64_role_caches.sh"
bash "$REPO/tpu/jobman/cell_worker.sh"
exec bash "$REPO/tpu/jobman/cell_monitor.sh"
