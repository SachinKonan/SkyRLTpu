#!/usr/bin/env bash
# jobman monitor hook (workers: 0, timeout: 0) for a ctrl-rerun league arm.
# Ensures the client runs (league_launch.sh is idempotent), then supervises:
#   exit 0 -> run complete (cell_probe blesses batch >= NUM_EPOCHS)
#   exit 1 -> client/engine death (jobman loops; league_worker re-ensures both
#             engine halves; client resumes at the same step)
# Health-checks BOTH tinker halves: qwen local :8000 and gemma at w4 :8000.
set -euo pipefail
: "${CELL:?}"; : "${JOBMAN_TPU_INTERNAL_IPS:?}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONITOR_INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-30}"
SYNC_EVERY_SECONDS="${SYNC_EVERY_SECONDS:-300}"
TINKER_FAILURE_LIMIT="${TINKER_FAILURE_LIMIT:-4}"
W4INT=$(echo "$JOBMAN_TPU_INTERNAL_IPS" | cut -d, -f5)

if ! tmux has-session -t cell 2>/dev/null; then
  echo "client not running -- launching via league_launch.sh"
  CELL="$CELL" GEMMA_INT="$W4INT" bash "$HOME/ttd-client/tpu/jobman/league_launch.sh"
  sleep 5
  tmux has-session -t cell 2>/dev/null || { echo "client failed to launch"; exit 1; }
fi

last_sync=0
fail_q=0; fail_g=0
while true; do
  if bash "${SCRIPT_DIR}/cell_probe.sh"; then
    echo "run complete -- final sync"
    bash "${SCRIPT_DIR}/cell_sync.sh" || true
    exit 0
  fi

  if ! tmux has-session -t cell 2>/dev/null; then
    echo "client tmux session gone before completion" >&2
    if [ -f "$HOME/ENGINE-SICK" ]; then
      echo "ENGINE-SICK marker ($(head -1 "$HOME/ENGINE-SICK" 2>/dev/null)) -- killing qwen tinker for rebuild" >&2
      tmux kill-session -t skyrl-tinker 2>/dev/null || true
      rm -f "$HOME/ENGINE-SICK"
    fi
    bash "${SCRIPT_DIR}/cell_sync.sh" || true
    exit 1
  fi

  if curl -fsS --max-time 6 http://127.0.0.1:8000/api/v1/get_server_capabilities >/dev/null 2>&1; then
    fail_q=0
  else
    fail_q=$((fail_q + 1)); echo "qwen tinker health fail ($fail_q/$TINKER_FAILURE_LIMIT)" >&2
  fi
  if curl -fsS --max-time 6 "http://$W4INT:8000/api/v1/get_server_capabilities" >/dev/null 2>&1; then
    fail_g=0
  else
    fail_g=$((fail_g + 1)); echo "gemma tinker health fail ($fail_g/$TINKER_FAILURE_LIMIT)" >&2
  fi
  if (( fail_q >= TINKER_FAILURE_LIMIT || fail_g >= TINKER_FAILURE_LIMIT )); then
    bash "${SCRIPT_DIR}/cell_sync.sh" || true
    exit 1
  fi

  now="$(date +%s)"
  if (( now - last_sync >= SYNC_EVERY_SECONDS )); then
    bash "${SCRIPT_DIR}/cell_sync.sh" || true
    last_sync="$now"
  fi
  sleep "$MONITOR_INTERVAL_SECONDS"
done
