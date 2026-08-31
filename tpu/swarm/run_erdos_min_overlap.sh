#!/usr/bin/env bash
# Adapts SkyPilot's multi-node environment to the existing Erdős cell scripts.
set -euo pipefail

: "${SKYPILOT_NODE_RANK:=0}"
: "${SKYPILOT_NODE_IPS:?SkyPilot must provide the allocated node IPs}"
: "${CELL:=ttd-n}"

export JOBMAN_WORKER_ID="$SKYPILOT_NODE_RANK"
export JOBMAN_TPU_INTERNAL_IPS
JOBMAN_TPU_INTERNAL_IPS=$(echo "$SKYPILOT_NODE_IPS" | paste -sd, -)

# The current cell launcher is head-driven and reaches the remaining hosts over
# their internal addresses. A SkyPilot pool job invokes this wrapper on the head;
# non-head ranks are intentionally no-ops if invoked directly.
if [ "$JOBMAN_WORKER_ID" != "0" ]; then
  exit 0
fi

REPO="${SKYRL_REPO_DIR:-$PWD}"
bash "$REPO/tpu/jobman/cell_worker.sh"
exec bash "$REPO/tpu/jobman/cell_monitor.sh"
