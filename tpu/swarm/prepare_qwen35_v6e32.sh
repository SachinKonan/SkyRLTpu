#!/usr/bin/env bash
# Pool setup hook: restore trainer checkpoints and prewarm every engine.
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
  echo "mixed v6e-32 requires 8 TPU VMs; SkyPilot supplied $node_count" >&2
  exit 2
fi

export REMOTE_USER="${REMOTE_USER:-$(id -un)}"
export SSH_KEY_FILE="${SSH_KEY_FILE:-$HOME/ray_bootstrap_key.pem}"
mkdir -p "$HOME/.ssh" "$HOME/.cache/tpuswarm" "$HOME/skyrl-runs" \
  "$HOME/skyrl-logs"
chmod 700 "$HOME/.ssh"
if [ "$JOBMAN_WORKER_ID" = "0" ]; then
  if [ ! -f "$SSH_KEY_FILE" ]; then
    echo "SkyPilot TPU pod key is missing: $SSH_KEY_FILE" >&2
    exit 2
  fi
  chmod 600 "$SSH_KEY_FILE"
  ln -sfn "$SSH_KEY_FILE" "$HOME/.ssh/jobman_tpu_ed25519"
fi

REPO="${SKYRL_REPO_DIR:-$HOME/SkyRLTpu-tpuswarm}"
bash "$REPO/tpu/jobman/ensure_orbax_ckpt.sh"

# Setup runs concurrently on all TPU VMs.  Rank 0 must not begin the
# head-driven prewarm until every peer has installed tools, unpacked the exact
# bundle generation, and restored its trainer checkpoint (where applicable).
marker_name="qwen35-v6e32-node-ready-$TPUSWARM_BUNDLE_GENERATION"
marker="$HOME/.cache/tpuswarm/$marker_name"
mkdir -p "$(dirname "$marker")"
marker_tmp="${marker}.tmp.$$"
printf '%s\n' "$JOBMAN_WORKER_ID" >"$marker_tmp"
mv "$marker_tmp" "$marker"
if [ "$JOBMAN_WORKER_ID" != "0" ]; then
  echo "pool setup rank $JOBMAN_WORKER_ID ready"
  exit 0
fi

ssho=(-F /dev/null -i "$SSH_KEY_FILE" -o IdentitiesOnly=yes \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20)
IFS=, read -r -a node_ips <<<"$JOBMAN_TPU_INTERNAL_IPS"
declare -a ordered_ips=()
for ip in "${node_ips[@]}"; do
  ready=0
  node_rank=""
  for _attempt in $(seq 1 120); do
    node_rank=$(ssh "${ssho[@]}" "$REMOTE_USER@$ip" \
      "cat \"\$HOME/.cache/tpuswarm/$marker_name\"" 2>/dev/null || true)
    if [[ "$node_rank" =~ ^[0-7]$ ]]; then
      ready=1
      break
    fi
    sleep 5
  done
  if [ "$ready" != "1" ]; then
    echo "pool setup node $ip did not publish a valid rank" >&2
    exit 1
  fi
  if [ -n "${ordered_ips[$node_rank]+set}" ]; then
    echo "duplicate pool setup rank $node_rank from $ip" >&2
    exit 1
  fi
  ordered_ips[$node_rank]="$ip"
done
for rank in $(seq 0 7); do
  if [ -z "${ordered_ips[$rank]+set}" ]; then
    echo "pool setup rank $rank did not publish an IP" >&2
    exit 1
  fi
done
JOBMAN_TPU_INTERNAL_IPS=$(IFS=,; echo "${ordered_ips[*]}")
export JOBMAN_TPU_INTERNAL_IPS
echo "ordered TPU VMs by setup rank: $JOBMAN_TPU_INTERNAL_IPS"

echo "all eight TPU VMs prepared; prewarming TP8/FSDP2 trainer and four TP4 engines"
bash "$REPO/tpu/jobman/cell_worker.sh"
echo "mixed v6e-32 pool worker is warm"
