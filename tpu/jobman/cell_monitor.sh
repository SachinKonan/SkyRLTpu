#!/usr/bin/env bash
# jobman monitor hook (workers: 0, timeout: 0) for a Stage-A cell.
#
# Ensures the client is running (launch_cell.sh is idempotent: restore from GCS,
# re-register checkpoints, sidecar, tmux client), then supervises:
#   exit 0  -> run complete (jobman stops looping via completion_probe)
#   exit 1  -> client/engine death (jobman loops: re-request TPU if preempted,
#              re-ensure engines, relaunch client, resume at the same step)
# Modeled on skyrl_math_rl_monitor.sh, which carried a 180-step run through
# 4 preemptions.
set -euo pipefail
: "${CELL:?}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONITOR_INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-30}"
SYNC_EVERY_SECONDS="${SYNC_EVERY_SECONDS:-300}"
TINKER_FAILURE_LIMIT="${TINKER_FAILURE_LIMIT:-4}"

if ! tmux has-session -t cell 2>/dev/null; then
  echo "client not running -- launching via launch_cell.sh"
  CELL="$CELL" bash "$HOME/ttd-client/tpu/launch_cell.sh"
  sleep 5
  tmux has-session -t cell 2>/dev/null || { echo "client failed to launch"; exit 1; }
fi

last_sync=0
tinker_failures=0
while true; do
  if bash "${SCRIPT_DIR}/cell_probe.sh"; then
    echo "run complete -- final sync"
    bash "${SCRIPT_DIR}/cell_sync.sh" || true
    exit 0
  fi

  if ! tmux has-session -t cell 2>/dev/null; then
    echo "client tmux session gone before completion" >&2
    # ENGINE-SICK marker: the client died because the trainer is wedged (fails
    # every fb while still answering health checks). Kill the tinker session so
    # the next loop's engines_healthy check fails and forces a FULL engine
    # rebuild -- otherwise the loop relaunches the client against the same
    # wedged engine forever.
    if [ -f "$HOME/ENGINE-SICK" ]; then
      echo "ENGINE-SICK marker present ($(cat "$HOME/ENGINE-SICK" 2>/dev/null | head -1)) -- killing tinker for full rebuild" >&2
      tmux kill-session -t skyrl-tinker 2>/dev/null || true
      rm -f "$HOME/ENGINE-SICK"
    fi
    bash "${SCRIPT_DIR}/cell_sync.sh" || true
    exit 1
  fi

  if curl -fsS --max-time 6 http://127.0.0.1:8000/api/v1/get_server_capabilities >/dev/null 2>&1; then
    tinker_failures=0
  else
    tinker_failures=$((tinker_failures + 1))
    echo "tinker health check failed (${tinker_failures}/${TINKER_FAILURE_LIMIT})" >&2
    if (( tinker_failures >= TINKER_FAILURE_LIMIT )); then
      bash "${SCRIPT_DIR}/cell_sync.sh" || true
      exit 1
    fi
  fi

  now="$(date +%s)"
  if (( now - last_sync >= SYNC_EVERY_SECONDS )); then
    bash "${SCRIPT_DIR}/cell_sync.sh" || true
    last_sync="$now"
  fi
  sleep "$MONITOR_INTERVAL_SECONDS"
done
