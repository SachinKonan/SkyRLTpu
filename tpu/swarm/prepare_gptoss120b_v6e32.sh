#!/usr/bin/env bash
# Pool setup hook: restore the immutable 120B Orbax checkpoint on all 8 hosts.
set -euo pipefail

: "${SKYPILOT_SETUP_NODE_RANK:?SkyPilot setup rank is required}"
: "${SKYPILOT_SETUP_NODE_IPS:?SkyPilot setup IPs are required}"
: "${TPUSWARM_BUNDLE_GENERATION:?bundle generation is required}"
if ! [[ "$TPUSWARM_BUNDLE_GENERATION" =~ ^[0-9]+$ ]]; then
  echo "invalid GCS bundle generation: $TPUSWARM_BUNDLE_GENERATION" >&2
  exit 2
fi

export JOBMAN_WORKER_ID="$SKYPILOT_SETUP_NODE_RANK"
export JOBMAN_TPU_INTERNAL_IPS
JOBMAN_TPU_INTERNAL_IPS=$(printf '%s\n' "$SKYPILOT_SETUP_NODE_IPS" | awk 'NF' | paste -sd, -)
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
mkdir -p "$HOME/.ssh" "$HOME/skyrl-logs"
ln -sfn "$SSH_KEY_FILE" "$HOME/.ssh/jobman_tpu_ed25519"

REPO="${SKYRL_REPO_DIR:-$HOME/SkyRLTpu-tpuswarm}"
TUNIX_MAXTEXT_CKPT_REQUIRE_MARKER=1 \
  bash "$REPO/tpu/jobman/ensure_orbax_ckpt.sh"
echo "GPT-OSS 120B pool setup rank $JOBMAN_WORKER_ID restored its checkpoint"
