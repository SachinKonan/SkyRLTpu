#!/usr/bin/env bash
# jobman monitor hook (workers: 0, timeout: 0) for a Stage-A cell.
#
# Ensures the client is running (launch_cell.sh is idempotent: restore from GCS,
# re-register checkpoints, sidecar, tmux client), then supervises:
#   exit 0  -> run complete (jobman stops looping via completion_probe)
#   exit 1  -> non-engine client failure (bounded user-error restart)
#   exit 33 -> recoverable setup failure (unlimited SkyPilot recovery)
#   exit 34 -> recoverable engine/runtime failure (unlimited recovery)
# Modeled on skyrl_math_rl_monitor.sh, which carried a 180-step run through
# 4 preemptions.
set -euo pipefail
: "${CELL:?}"
: "${JOBMAN_TPU_INTERNAL_IPS:?}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONITOR_INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-30}"
SYNC_EVERY_SECONDS="${SYNC_EVERY_SECONDS:-300}"
TINKER_FAILURE_LIMIT="${TINKER_FAILURE_LIMIT:-4}"
SETUP_RETRY_EXIT_CODE="${SETUP_RETRY_EXIT_CODE:-33}"
RUNTIME_RECOVERY_EXIT_CODE="${RUNTIME_RECOVERY_EXIT_CODE:-34}"
RUN="${RUN_DIR_NAME:-stageA-$CELL}"
SESSION="${CELL_SESSION:-cell}"
OWNER_FILE="$HOME/.cache/tpuswarm/${SESSION}.owner"
OWNER_TOKEN="$RUN:${TPUSWARM_BUNDLE_ID:-unversioned}"
EXPECTED_BUNDLE_ID="${TPUSWARM_BUNDLE_ID:-}"
REMOTE_USER="${REMOTE_USER:-sk7524_princeton_edu}"
SSH_KEY_FILE="${SSH_KEY_FILE:-$HOME/.ssh/jobman_tpu_ed25519}"
VLLM_ENGINES_PER_HOST="${VLLM_ENGINES_PER_HOST:-1}"
VLLM_PORT="${VLLM_PORT:-8001}"
node_count=$(awk -F, '{print NF}' <<<"$JOBMAN_TPU_INTERNAL_IPS")
VLLM_WORKERS="${VLLM_WORKERS:-$(seq -s, 1 $((node_count - 1)))}"
SSHO=(-F /dev/null -i "$SSH_KEY_FILE" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20)
mkdir -p "$(dirname "$OWNER_FILE")"

worker_ip() {
  local worker="$1"
  echo "$JOBMAN_TPU_INTERNAL_IPS" | tr ',' '\n' | sed -n "$((worker + 1))p"
}

tinker_healthy() {
  if ! curl -fsS --max-time 6 http://127.0.0.1:8000/api/v1/get_server_capabilities >/dev/null 2>&1; then
    echo "tinker health check failed at http://127.0.0.1:8000" >&2
    return 1
  fi
  if [[ -n "$EXPECTED_BUNDLE_ID" ]]; then
    local pid actual_bundle found=0
    while read -r pid; do
      [[ -n "$pid" ]] || continue
      found=1
      actual_bundle=$(tr '\0' '\n' < "/proc/$pid/environ" | sed -n 's/^TPUSWARM_BUNDLE_ID=//p' | head -1)
      [[ "$actual_bundle" == "$EXPECTED_BUNDLE_ID" ]] || {
        echo "trainer bundle mismatch: running=${actual_bundle:-unmarked} expected=$EXPECTED_BUNDLE_ID" >&2
        return 1
      }
    done < <(pgrep -u "$USER" -f '[p]ython.*-m skyrl\.tinker\.api.*--port 8000' || true)
    (( found == 1 )) || return 1
  fi
}

vllm_healthy() {
  local strict_identity="${1:-0}" worker ip engine port
  for worker in $(echo "$VLLM_WORKERS" | tr ',' ' '); do
    ip="$(worker_ip "$worker")"
    for ((engine = 0; engine < VLLM_ENGINES_PER_HOST; engine++)); do
      port=$((VLLM_PORT + engine))
      if ! curl -fsS --max-time 6 "http://$ip:$port/v1/models" >/dev/null 2>&1; then
        echo "vLLM health check failed: worker=$worker ip=$ip port=$port" >&2
        return 1
      fi
    done
    if [[ "$strict_identity" == "1" && -n "$EXPECTED_BUNDLE_ID" ]]; then
      timeout 30 ssh "${SSHO[@]}" "$REMOTE_USER@$ip" \
        "pids=\$(pgrep -u \"\$USER\" -f '[p]ython.*vllm_tpu_server\\.py'); \
         test \$(wc -w <<<\"\$pids\") -ge '$VLLM_ENGINES_PER_HOST' || exit 1; \
         for pid in \$pids; do \
           actual=\$(tr '\\0' '\\n' < /proc/\$pid/environ | sed -n 's/^TPUSWARM_BUNDLE_ID=//p' | head -1); \
           test \"\$actual\" = '$EXPECTED_BUNDLE_ID' || exit 1; \
         done" >/dev/null 2>&1 || {
          echo "vLLM bundle mismatch on $ip: expected=$EXPECTED_BUNDLE_ID" >&2
          return 1
        }
    fi
  done
}

engines_healthy() {
  local strict_identity="${1:-0}"
  tinker_healthy && vllm_healthy "$strict_identity"
}

if ! engines_healthy 1; then
  echo "full engine readiness check failed; refusing to launch client" >&2
  exit "$SETUP_RETRY_EXIT_CODE"
fi

if tmux has-session -t "=$SESSION" 2>/dev/null &&
   [[ "$(cat "$OWNER_FILE" 2>/dev/null || true)" != "$OWNER_TOKEN" ]]; then
  echo "stale client session belongs to another run -- replacing it"
  tmux kill-session -t "=$SESSION" 2>/dev/null || true
  tmux kill-session -t "=${SESSION}-backup" 2>/dev/null || true
fi

if ! tmux has-session -t "=$SESSION" 2>/dev/null; then
  echo "client not running -- launching via launch_cell.sh"
  CELL="$CELL" bash "$HOME/ttd-client/tpu/launch_cell.sh"
  sleep 5
  tmux has-session -t "=$SESSION" 2>/dev/null || {
    echo "client failed to launch" >&2
    exit "$SETUP_RETRY_EXIT_CODE"
  }
  printf '%s\n' "$OWNER_TOKEN" > "$OWNER_FILE"
fi

last_sync=0
engine_failures=0
while true; do
  if bash "${SCRIPT_DIR}/cell_probe.sh"; then
    echo "run complete -- final sync"
    bash "${SCRIPT_DIR}/cell_sync.sh" || true
    exit 0
  fi

  if ! tmux has-session -t "=$SESSION" 2>/dev/null; then
    echo "client tmux session gone before completion" >&2
    # ENGINE-SICK marker: the client died because the trainer is wedged (fails
    # every fb while still answering health checks). Kill the tinker session so
    # the next loop's engines_healthy check fails and forces a FULL engine
    # rebuild -- otherwise the loop relaunches the client against the same
    # wedged engine forever.
    if [ -f "$HOME/ENGINE-SICK" ]; then
      echo "ENGINE-SICK marker present ($(cat "$HOME/ENGINE-SICK" 2>/dev/null | head -1)) -- killing tinker for full rebuild" >&2
      tmux kill-session -t =skyrl-tinker 2>/dev/null || true
      rm -f "$HOME/ENGINE-SICK"
      bash "${SCRIPT_DIR}/cell_sync.sh" || true
      exit "$RUNTIME_RECOVERY_EXIT_CODE"
    fi
    bash "${SCRIPT_DIR}/cell_sync.sh" || true
    exit 1
  fi

  # The sidecar (run-dir sync + checkpoint writeback) is what makes progress
  # durable on pool workers; it was seen to vanish mid-cycle. Re-create it
  # from the script launch_cell.sh rendered rather than lose the next step.
  if ! tmux has-session -t "=${SESSION}-backup" 2>/dev/null && [ -s "$HOME/sidecar_${RUN}.sh" ]; then
    echo "sidecar session gone -- restarting ${SESSION}-backup" >&2
    tmux new-session -d -s "${SESSION}-backup" "bash $HOME/sidecar_${RUN}.sh"
  fi

  if engines_healthy 0; then
    engine_failures=0
  else
    engine_failures=$((engine_failures + 1))
    echo "engine health check failed (${engine_failures}/${TINKER_FAILURE_LIMIT})" >&2
    if (( engine_failures >= TINKER_FAILURE_LIMIT )); then
      bash "${SCRIPT_DIR}/cell_sync.sh" || true
      exit "$RUNTIME_RECOVERY_EXIT_CODE"
    fi
  fi

  now="$(date +%s)"
  if (( now - last_sync >= SYNC_EVERY_SECONDS )); then
    bash "${SCRIPT_DIR}/cell_sync.sh" || true
    last_sync="$now"
  fi
  sleep "$MONITOR_INTERVAL_SECONDS"
done
