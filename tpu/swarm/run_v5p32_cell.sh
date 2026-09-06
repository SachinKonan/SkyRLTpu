#!/usr/bin/env bash
# Runs ONE proven stage/meta Erdős cell on one four-host tpu-v5p-32 pool worker.
#
# Rank 0 = trainer + RL client + grader Ray head; ranks 1-3 = vLLM engines.
# The cell itself (cell_worker.sh -> launch_cell.sh -> cell_monitor.sh) is the
# unchanged jobman unit from tpu/jobman/RUNBOOK.md; this wrapper only adapts
# SkyPilot's multi-node environment to it, the way run_qwen35_v6e32_grpo.sh does
# for the mixed v6e-32 worker.
set -euo pipefail
: "${SKYPILOT_NODE_RANK:=0}"
: "${SKYPILOT_NODE_IPS:?SkyPilot must provide all v5p-32 TPU VM IPs}"
: "${CELL:?CELL selects the model and cell knobs, e.g. m-grpo-n}"

export JOBMAN_WORKER_ID="$SKYPILOT_NODE_RANK"
ips=$(printf '%s\n' "$SKYPILOT_NODE_IPS" | awk 'NF' | paste -sd, -)
node_count=$(awk -F, '{print NF}' <<<"$ips")
if [ "$node_count" -ne 4 ]; then
  echo "a v5p-32 cell requires 4 TPU VMs; SkyPilot supplied $node_count" >&2
  exit 2
fi

export REMOTE_USER="${REMOTE_USER:-$(id -un)}"
export SSH_KEY_FILE="${SSH_KEY_FILE:-$HOME/ray_bootstrap_key.pem}"
mkdir -p "$HOME/.ssh" "$HOME/skyrl-runs" "$HOME/skyrl-logs"
chmod 700 "$HOME/.ssh"

# Pool workers are reused across jobs and models. Before anything else, on
# EVERY rank, drop the other models' HF/orbax caches and superseded bundle
# generations (a gemma cell on an ex-muse head could not fit its 44 GB orbax
# restore next to 40 GB of muse orbax: jobs 200/212, 2026-09-05). Best effort:
# a failure here must not take the cell down with it.
bash "${SKYRL_REPO_DIR:-$PWD}/tpu/swarm/reconcile_v5p32_worker.sh" \
  || echo "worker reconcile failed (continuing)" >&2

# SkyPilot invokes a multi-node run command on every TPU VM.  The cell is
# deliberately head-driven and reaches the other ranks over the internal
# network, so only rank 0 does anything here.
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

# cell_worker treats the FIRST address as the trainer host (W0INT), and
# launch_cell/cell_monitor start the client LOCALLY on the host running them
# (a cross-host ray.init hangs -- meta-arm bug #3).  SkyPilot does not promise
# that rank 0's own address leads SKYPILOT_NODE_IPS, so rotate the list until
# this VM is first; the remaining three hosts become the engines in any order.
local_ips=$(hostname -I 2>/dev/null || true)
IFS=, read -r -a all_ips <<<"$ips"
self_idx=-1
for i in "${!all_ips[@]}"; do
  for l in $local_ips; do
    if [ "${all_ips[$i]}" = "$l" ]; then self_idx=$i; fi
  done
done
if [ "$self_idx" -lt 0 ]; then
  echo "this VM ($local_ips) is not among SKYPILOT_NODE_IPS ($ips)" >&2
  exit 2
fi
ordered=("${all_ips[$self_idx]}")
for i in "${!all_ips[@]}"; do
  if [ "$i" -ne "$self_idx" ]; then ordered+=("${all_ips[$i]}"); fi
done
export JOBMAN_TPU_INTERNAL_IPS
JOBMAN_TPU_INTERNAL_IPS=$(IFS=,; echo "${ordered[*]}")
export TRAIN_WORKERS="${TRAIN_WORKERS:-0}"
echo "cell $CELL on v5p-32: trainer/client host ${ordered[0]} (rank 0); engine hosts ${ordered[*]:1}"

REPO="${SKYRL_REPO_DIR:-$PWD}"
# The trainer's MaxText/orbax checkpoint must be complete BEFORE any engine
# starts (RUNBOOK: a torn restore surfaces two layers down as DATA_LOSS or
# RESOURCE_EXHAUSTED).  Model is derived from CELL; no-op off the trainer host.
bash "$REPO/tpu/jobman/ensure_orbax_ckpt.sh"
bash "$REPO/tpu/jobman/cell_worker.sh"
exec bash "$REPO/tpu/jobman/cell_monitor.sh"
