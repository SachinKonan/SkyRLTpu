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
OWNER_TOKEN="$RUN:${TPUSWARM_BUNDLE_ID:-unversioned}:${SKYPILOT_INTERNAL_JOB_ID:-standalone}"
EXPECTED_BUNDLE_ID="${TPUSWARM_BUNDLE_ID:-}"
REMOTE_USER="${REMOTE_USER:-sk7524_princeton_edu}"
SSH_KEY_FILE="${SSH_KEY_FILE:-$HOME/.ssh/jobman_tpu_ed25519}"
VLLM_ENGINES_PER_HOST="${VLLM_ENGINES_PER_HOST:-1}"
VLLM_PORT="${VLLM_PORT:-8001}"
VLLM_RAY_EXECUTOR="${VLLM_RAY_EXECUTOR:-0}"
VLLM_INPLACE_RESTART_LIMIT="${VLLM_INPLACE_RESTART_LIMIT:-2}"
VLLM_RESTART_READY_ATTEMPTS="${VLLM_RESTART_READY_ATTEMPTS:-120}"
VLLM_RESTART_READY_INTERVAL_SECONDS="${VLLM_RESTART_READY_INTERVAL_SECONDS:-15}"
node_count=$(awk -F, '{print NF}' <<<"$JOBMAN_TPU_INTERNAL_IPS")
VLLM_WORKERS="${VLLM_WORKERS:-$(seq -s, 1 $((node_count - 1)))}"
SSHO=(-F /dev/null -i "$SSH_KEY_FILE" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20)
mkdir -p "$(dirname "$OWNER_FILE")"
declare -A vllm_restart_counts=()
UNHEALTHY_VLLM_WORKER=""
UNHEALTHY_VLLM_IP=""
UNHEALTHY_VLLM_PORT=""

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
        UNHEALTHY_VLLM_WORKER="$worker"
        UNHEALTHY_VLLM_IP="$ip"
        UNHEALTHY_VLLM_PORT="$port"
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

capture_vllm_diagnostics() {
  local worker="$1" ip="$2" port="$3" timestamp output
  timestamp=$(date -u +%Y%m%dT%H%M%SZ)
  output="$HOME/skyrl-logs/vllm-recovery-worker${worker}-${timestamp}.log"
  mkdir -p "$HOME/skyrl-logs"
  {
    echo "captured_at=$timestamp worker=$worker ip=$ip failed_port=$port"
    curl -sv --max-time 10 "http://$ip:$port/v1/models" -o /dev/null || true
    timeout 60 ssh "${SSHO[@]}" "$REMOTE_USER@$ip" 'bash -s' <<'REMOTE_DIAGNOSTICS'
set +e
date -u +%Y-%m-%dT%H:%M:%SZ
hostname
echo "--- uptime/memory/disk ---"
uptime
free -h
df -h "$HOME" /tmp
echo "--- tmux/processes ---"
tmux list-sessions
pgrep -a -u "$USER" -f 'vllm_tpu_server.py|vllm serve|VLLM::EngineCore|api_server'
echo "--- exit metadata ---"
tail -n 80 "$HOME"/skyrl-logs/vllm-tpu*.exits.log
echo "--- current vLLM logs ---"
for log in "$HOME"/skyrl-logs/vllm-tpu*.log; do
  test -f "$log" || continue
  echo "### $log"
  tail -n 240 "$log"
done
echo "--- kernel fatal/OOM signals ---"
dmesg --ctime | grep -Ei 'out of memory|oom-kill|killed process|gasket|tpu|segfault' | tail -n 160
REMOTE_DIAGNOSTICS
  } >> "$output" 2>&1
  echo "vLLM pre-restart diagnostics saved to $output" >&2
}

vllm_worker_ready() {
  local worker="$1" ip="$2" engine port
  for ((engine = 0; engine < VLLM_ENGINES_PER_HOST; engine++)); do
    port=$((VLLM_PORT + engine))
    curl -fsS --max-time 6 "http://$ip:$port/v1/models" >/dev/null 2>&1 || return 1
  done
  if [[ -n "$EXPECTED_BUNDLE_ID" ]]; then
    timeout 30 ssh "${SSHO[@]}" "$REMOTE_USER@$ip" \
      "pids=\$(pgrep -u \"\$USER\" -f '[p]ython.*vllm_tpu_server\\.py'); \
       test \$(wc -w <<<\"\$pids\") -ge '$VLLM_ENGINES_PER_HOST' || exit 1; \
       for pid in \$pids; do \
         actual=\$(tr '\\0' '\\n' < /proc/\$pid/environ | sed -n 's/^TPUSWARM_BUNDLE_ID=//p' | head -1); \
         test \"\$actual\" = '$EXPECTED_BUNDLE_ID' || exit 1; \
       done" >/dev/null 2>&1 || return 1
  fi
}

restart_vllm_worker() {
  local worker="$1" ip="$2" port="$3" attempt restart_count
  restart_count="${vllm_restart_counts[$worker]:-0}"
  if [[ "$VLLM_RAY_EXECUTOR" != "0" ]]; then
    echo "in-place vLLM restart is disabled for Ray executor mode" >&2
    return 1
  fi
  if (( restart_count >= VLLM_INPLACE_RESTART_LIMIT )); then
    echo "worker $worker exhausted its $VLLM_INPLACE_RESTART_LIMIT in-place vLLM restarts" >&2
    return 1
  fi
  capture_vllm_diagnostics "$worker" "$ip" "$port"
  restart_count=$((restart_count + 1))
  vllm_restart_counts[$worker]="$restart_count"
  echo "restarting only vLLM worker=$worker ip=$ip ($restart_count/$VLLM_INPLACE_RESTART_LIMIT)" >&2
  timeout 900 ssh "${SSHO[@]}" "$REMOTE_USER@$ip" \
    "test -x \"\$HOME/start_vllm_tpu_bootstrap.sh\" && \
     VLLM_RELATIVE_WORKER_ID=0 VLLM_USE_RAY_EXECUTOR=0 \
     VLLM_START_SERVER=1 VLLM_CLEANUP=1 \
     bash \"\$HOME/start_vllm_tpu_bootstrap.sh\"" || return 1
  for ((attempt = 1; attempt <= VLLM_RESTART_READY_ATTEMPTS; attempt++)); do
    if vllm_worker_ready "$worker" "$ip"; then
      echo "in-place vLLM restart recovered worker=$worker after $attempt readiness checks"
      return 0
    fi
    if (( attempt % 20 == 0 )); then
      echo "waiting for restarted vLLM worker=$worker ($attempt/$VLLM_RESTART_READY_ATTEMPTS)" >&2
    fi
    sleep "$VLLM_RESTART_READY_INTERVAL_SECONDS"
  done
  echo "in-place vLLM restart did not recover worker=$worker" >&2
  return 1
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
last_engine_failure_kind=""
vllm_outage_seen=0
while true; do
  if bash "${SCRIPT_DIR}/cell_probe.sh"; then
    echo "run complete -- final sync"
    bash "${SCRIPT_DIR}/cell_sync.sh" || true
    exit 0
  fi

  client_running=1
  tmux has-session -t "=$SESSION" 2>/dev/null || client_running=0
  if (( client_running == 0 )) && [ -f "$HOME/ENGINE-SICK" ]; then
    echo "client tmux session gone before completion" >&2
    # ENGINE-SICK marker: the client died because the trainer is wedged (fails
    # every fb while still answering health checks). Kill the tinker session so
    # the next loop's engines_healthy check fails and forces a FULL engine
    # rebuild -- otherwise the loop relaunches the client against the same
    # wedged engine forever.
    echo "ENGINE-SICK marker present ($(cat "$HOME/ENGINE-SICK" 2>/dev/null | head -1)) -- killing tinker for full rebuild" >&2
    tmux kill-session -t =skyrl-tinker 2>/dev/null || true
    rm -f "$HOME/ENGINE-SICK"
    bash "${SCRIPT_DIR}/cell_sync.sh" || true
    exit "$RUNTIME_RECOVERY_EXIT_CODE"
  fi

  # The sidecar (run-dir sync + checkpoint writeback) is what makes progress
  # durable on pool workers; it was seen to vanish mid-cycle. Re-create it
  # from the script launch_cell.sh rendered rather than lose the next step.
  if ! tmux has-session -t "=${SESSION}-backup" 2>/dev/null && [ -s "$HOME/sidecar_${RUN}.sh" ]; then
    echo "sidecar session gone -- restarting ${SESSION}-backup" >&2
    tmux new-session -d -s "${SESSION}-backup" "bash $HOME/sidecar_${RUN}.sh"
  fi

  engine_failure_kind=""
  if ! tinker_healthy; then
    engine_failure_kind="tinker"
  elif ! vllm_healthy 0; then
    engine_failure_kind="vllm"
    vllm_outage_seen=1
  fi

  if [[ -z "$engine_failure_kind" ]]; then
    engine_failures=0
    last_engine_failure_kind=""
  else
    if [[ "$engine_failure_kind" != "$last_engine_failure_kind" ]]; then
      engine_failures=0
      last_engine_failure_kind="$engine_failure_kind"
    fi
    engine_failures=$((engine_failures + 1))
    echo "$engine_failure_kind health check failed (${engine_failures}/${TINKER_FAILURE_LIMIT})" >&2
    if (( engine_failures >= TINKER_FAILURE_LIMIT )); then
      if [[ "$engine_failure_kind" == "vllm" ]] &&
         restart_vllm_worker "$UNHEALTHY_VLLM_WORKER" "$UNHEALTHY_VLLM_IP" "$UNHEALTHY_VLLM_PORT"; then
        engine_failures=0
        engine_failure_kind=""
      else
        bash "${SCRIPT_DIR}/cell_sync.sh" || true
        exit "$RUNTIME_RECOVERY_EXIT_CODE"
      fi
    fi
  fi

  if (( client_running == 0 )); then
    if [[ "$engine_failure_kind" == "vllm" ]]; then
      echo "client exited during vLLM outage; waiting for the bounded host restart" >&2
    elif (( vllm_outage_seen == 1 )); then
      echo "vLLM recovered but client exited during the outage -- relaunching client"
      CELL="$CELL" bash "$HOME/ttd-client/tpu/launch_cell.sh"
      sleep 5
      tmux has-session -t "=$SESSION" 2>/dev/null || {
        echo "client failed to relaunch after vLLM recovery" >&2
        bash "${SCRIPT_DIR}/cell_sync.sh" || true
        exit "$SETUP_RETRY_EXIT_CODE"
      }
      printf '%s\n' "$OWNER_TOKEN" > "$OWNER_FILE"
      vllm_outage_seen=0
    else
      echo "client tmux session gone before completion" >&2
      bash "${SCRIPT_DIR}/cell_sync.sh" || true
      exit 1
    fi
  elif [[ -z "$engine_failure_kind" ]]; then
    vllm_outage_seen=0
  fi

  now="$(date +%s)"
  if (( now - last_sync >= SYNC_EVERY_SECONDS )); then
    bash "${SCRIPT_DIR}/cell_sync.sh" || true
    last_sync="$now"
  fi
  sleep "$MONITOR_INTERVAL_SECONDS"
done
