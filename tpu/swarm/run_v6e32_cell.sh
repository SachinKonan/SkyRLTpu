#!/usr/bin/env bash
# Runs ONE proven stage/meta Erdős cell on one eight-VM tpu-v6e-32 pool worker
# in the 2+6 layout: VMs 0-1 (one physical host, 8 chips) = trainer + RL client
# + grader Ray head on VM 0; VMs 2-7 = six TP=4 vLLM engines.
#
# The cell itself (cell_worker.sh -> start_colocated_vllm_tinker.sh ->
# cell_monitor.sh) is the unchanged jobman unit; like run_v5p32_cell.sh this
# wrapper only adapts SkyPilot's multi-node environment to it. Differences from
# the v5p-32 wrapper: 8 VMs, TRAIN_WORKERS spans two VMs, so every trainer VM
# stages the orbax checkpoint, and the IP list is NOT rotated (TRAIN_WORKERS
# indexes SKYPILOT_NODE_IPS positions and VMs 0,1 must stay physically adjacent
# for the 2,1,1 process grid); if this VM is rank 0 but not first in the list
# we exit 33 (recover_on_exit_codes) instead of guessing a topology.
#
# Layout knobs come from the task yaml: TRAIN_WORKERS=0,1 VLLM_WORKERS=2..7
# TRAIN_TP_SIZE=8 TRAIN_FSDP_SIZE=1 TUNIX_ROW_SHARD=1
# TRAIN_TPU_PROCESS_BOUNDS=2,1,1 TRAIN_TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1.
# The v6e chip has 32 GB HBM (v5p: 95 GB): TP 8 over 8 chips keeps the
# per-chip weight footprint under the v5p TP-4 value and the yaml halves the
# token budget so one fb call holds two rows, as on v5p.
set -euo pipefail
: "${SKYPILOT_NODE_RANK:=0}"
: "${SKYPILOT_NODE_IPS:?SkyPilot must provide all v6e-32 TPU VM IPs}"
: "${CELL:?CELL selects the model and cell knobs, e.g. g-v32-ttd-n}"

export JOBMAN_WORKER_ID="$SKYPILOT_NODE_RANK"
ips=$(printf '%s\n' "$SKYPILOT_NODE_IPS" | awk 'NF' | paste -sd, -)
node_count=$(awk -F, '{print NF}' <<<"$ips")
if [ "$node_count" -ne 8 ]; then
  echo "a v6e-32 cell requires 8 TPU VMs; SkyPilot supplied $node_count" >&2
  exit 2
fi
export JOBMAN_TPU_INTERNAL_IPS="$ips"
export TRAIN_WORKERS="${TRAIN_WORKERS:-0,1}"
export VLLM_WORKERS="${VLLM_WORKERS:-2,3,4,5,6,7}"

export REMOTE_USER="${REMOTE_USER:-$(id -un)}"
export SSH_KEY_FILE="${SSH_KEY_FILE:-$HOME/ray_bootstrap_key.pem}"
mkdir -p "$HOME/.ssh" "$HOME/skyrl-runs" "$HOME/skyrl-logs"
chmod 700 "$HOME/.ssh"
REPO="${SKYRL_REPO_DIR:-$PWD}"

# Pool workers are reused across jobs and models: drop the other models' caches
# and superseded bundle generations on EVERY VM first (best effort).
bash "$REPO/tpu/swarm/reconcile_v5p32_worker.sh" \
  || echo "worker reconcile failed (continuing)" >&2

# Every TRAINER VM needs the MaxText/orbax checkpoint locally before the
# multi-host trainer starts (each JAX process restores its own shards).
# ensure_orbax_ckpt.sh no-ops on VMs outside TRAIN_WORKERS.
bash "$REPO/tpu/jobman/ensure_orbax_ckpt.sh"

# SkyPilot invokes the run command on every VM. The cell is head-driven and
# reaches the other ranks over the internal network, so only rank 0 continues.
if [ "$JOBMAN_WORKER_ID" != "0" ]; then
  echo "rank $JOBMAN_WORKER_ID ready; rank 0 owns cell $CELL"
  exit 0
fi
if [ ! -f "$SSH_KEY_FILE" ]; then
  echo "SkyPilot TPU pod key is missing: $SSH_KEY_FILE" >&2
  exit 2
fi
chmod 600 "$SSH_KEY_FILE"
ln -sfn "$SSH_KEY_FILE" "$HOME/.ssh/jobman_tpu_ed25519"

# cell_worker treats the FIRST address as the trainer/client host and
# TRAIN_WORKERS as positions in that list. Rank 0 must therefore be first.
first_ip=$(cut -d, -f1 <<<"$ips")
if ! hostname -I 2>/dev/null | tr ' ' '\n' | grep -qxF "$first_ip"; then
  echo "rank 0 ($(hostname -I)) is not first in SKYPILOT_NODE_IPS ($ips); recovering onto another worker" >&2
  exit 33
fi
echo "cell $CELL on v6e-32 (2+6): trainer VMs $(cut -d, -f1,2 <<<"$ips") (rank 0 = client/API); engine VMs $(cut -d, -f3- <<<"$ips")"

# Whatever way the cell dies, leave all eight VMs clean for the next job (the
# pool fork never re-places a failed job). cleanup_v5p32_worker.sh iterates
# over JOBMAN_TPU_INTERNAL_IPS, so it covers eight hosts unchanged.
cleanup_on_failure() {
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    bash "$REPO/tpu/swarm/cleanup_v5p32_worker.sh" "$rc" 2>&1 \
      || echo "cleanup_v5p32_worker.sh itself failed (rc=$?)" >&2
  fi
  exit "$rc"
}
trap cleanup_on_failure EXIT

bash "$REPO/tpu/jobman/cell_worker.sh"
monitor_rc=0
bash "$REPO/tpu/jobman/cell_monitor.sh" || monitor_rc=$?
exit "$monitor_rc"
