#!/usr/bin/env bash
# Fire-and-forget autoresume launcher for colocated tunix math-RL runs on
# spot TPUs. One command per model:
#
#   ./tpu/autoresume_run.sh tpu/runs/qwen35-27b.env
#
# By default this detaches into a local tmux session (autoresume-$RUN_NAME)
# and returns immediately; run with AUTORESUME_FOREGROUND=1 to supervise in
# the current shell. The supervision loop:
#
#   (a) QR ensure   — SUSPENDED/FAILED queued-resource -> delete --force,
#                     wait until gone, `jobman create` from $JOBMAN_DIR.
#                     Missing QR -> jobman create.
#   (b) wait READY  — poll node state with the basename() filter (a bare
#                     name= filter NEVER matches: the API returns full
#                     resource paths), plus an ssh probe on worker 0, then
#                     allow $SETUP_GRACE_SEC for jobman's base setup.
#   (c) bring-up    — tpu/start_colocated_vllm_tinker.sh with the env file's
#                     exports (or, for BRINGUP_MANUAL=1 models, wait for a
#                     manually started Tinker API to become healthy).
#   (d) client      — tpu/run_tinker_math_rl_qwen35_9b.sh. First launch ever
#                     uses BEHAVIOR_IF_LOG_DIR_EXISTS=$FIRST_BEHAVIOR
#                     (default delete); every later launch resumes the same
#                     LOG_PATH. A state file + metrics.jsonl presence make a
#                     restarted watcher resume too, never delete.
#   (e) health loop — node READY, client tmux alive, metrics.jsonl fresh.
#                     Node lost -> kill client, back to (a). Client dead at
#                     MAX_STEPS-1 -> clean exit. Client dead early ->
#                     relaunch with resume. Metrics stale twice -> kill +
#                     relaunch with resume.
#
# Operational lessons encoded here (see the tunix-tinker-backend memory):
#   - name.basename() filter for tpu-vm list (full-path names otherwise);
#   - delete SUSPENDED queued-resources before jobman create;
#   - jobman create on an existing READY node is idempotent;
#   - kill and start remote processes in SEPARATE ssh invocations, and only
#     ever with bracketed pkill patterns ("[s]kyrl.tinker") — a plain
#     process-name string anywhere in the same remote command makes pkill
#     shoot its own shell. This script never embeds killable names remotely;
#     all remote kills live inside start_colocated_vllm_tinker.sh.
#   - client caches and logs live under /home (uv cache + LOG_ROOT): the
#     /scratch group fileset filling up killed two runs;
#   - a preempted-and-recreated TPU gets new external IPs, so the stale ssh
#     tunnel session is killed before every client (re)launch;
#   - gcloud API failures (quota/IAM blips) are NOT preemptions: an UNKNOWN
#     node state never kills a client, only a definitive non-READY does.
set -uo pipefail

usage() {
  echo "usage: $0 <run.env>   (see tpu/runs/*.env)" >&2
  exit 2
}

[[ $# -eq 1 ]] || usage
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

env_file="$1"
[[ "$env_file" = /* ]] || env_file="$(pwd)/$env_file"
[[ -f "$env_file" ]] || { echo "env file not found: $env_file" >&2; exit 2; }

# Export everything the env file sets so the bring-up and client scripts see
# it (MODEL_NAME, TUNIX_MAXTEXT_*, VLLM_EXTRA_ARGS, GROUP_SIZE, ...).
set -a
# shellcheck disable=SC1090
source "$env_file"
set +a

: "${RUN_NAME:?env file must set RUN_NAME}"
: "${TPU_NAME:?env file must set TPU_NAME}"
: "${JOBMAN_YAML:?env file must set JOBMAN_YAML}"
: "${LOG_PATH:?env file must set LOG_PATH}"
: "${CLIENT_SESSION:?env file must set CLIENT_SESSION}"

PROJECT="${PROJECT:-vision-mix}"
ZONE="${ZONE:-us-east5-a}"
REMOTE_USER="${REMOTE_USER:-sk7524_princeton_edu}"
SSH_KEY_FILE="${SSH_KEY_FILE:-$HOME/.ssh/jobman_tpu_ed25519}"
QR_NAME="${QR_NAME:-$TPU_NAME}"
JOBMAN_DIR="${JOBMAN_DIR:-/scratch/gpfs/ZHUANGL/sk7524/skyrltpu-resumable}"
JOBMAN_BIN="${JOBMAN_BIN:-jobman}"
API_PORT="${API_PORT:-8000}"
TUNNEL_SESSION="${TUNNEL_SESSION:-skyrl-tinker-tunnel-${RUN_NAME}}"
BRINGUP_MANUAL="${BRINGUP_MANUAL:-0}"

POLL_SEC="${POLL_SEC:-120}"
SETUP_GRACE_SEC="${SETUP_GRACE_SEC:-180}"
HEALTH_SEC="${HEALTH_SEC:-180}"
STALL_SEC="${STALL_SEC:-2400}"
# Grace between client launch and the first metrics.jsonl line (venv build +
# step 0 + eval can take a while on a cold uv cache).
NO_METRICS_TIMEOUT_SEC="${NO_METRICS_TIMEOUT_SEC:-3600}"
FIRST_BEHAVIOR="${FIRST_BEHAVIOR:-delete}"
MAX_STEPS="${MAX_STEPS:-180}"

WATCHER_LOG="${WATCHER_LOG:-/home/sk7524/skyrl-runs/autoresume-${RUN_NAME}.log}"
STATE_FILE="${STATE_FILE:-${LOG_PATH}/.autoresume_state}"
metrics_file="${LOG_PATH}/metrics.jsonl"

# Client-side placement (lesson: /scratch fills up; keep uv cache and run
# logs on /home which has room).
export UV_CACHE_DIR="${UV_CACHE_DIR:-/home/sk7524/.cache/uv-skyrl}"
export CLIENT_UV_CACHE_DIR="${CLIENT_UV_CACHE_DIR:-$UV_CACHE_DIR}"
export LOG_ROOT="${LOG_ROOT:-/home/sk7524/skyrl-runs/math_rl}"

[[ "$JOBMAN_YAML" = /* ]] || JOBMAN_YAML="${repo_root}/${JOBMAN_YAML}"
if [[ ! -f "$JOBMAN_YAML" ]]; then
  echo "JOBMAN_YAML not found: $JOBMAN_YAML" >&2
  exit 2
fi

mkdir -p "$(dirname "$WATCHER_LOG")" "$LOG_ROOT"

# Console copy goes to stderr so functions whose stdout is captured (e.g.
# health_loop) can still log freely.
log() {
  echo "$(date -u '+%Y-%m-%d %H:%M:%S') [autoresume ${RUN_NAME}] $*" >> "$WATCHER_LOG"
  echo "$(date -u '+%Y-%m-%d %H:%M:%S') [autoresume ${RUN_NAME}] $*" >&2
}

# ---------------------------------------------------------------------------
# Fire-and-forget: detach into tmux unless asked to stay in the foreground.
# ---------------------------------------------------------------------------
if [[ "${AUTORESUME_FOREGROUND:-0}" != "1" ]]; then
  session="autoresume-${RUN_NAME}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Supervisor session '$session' already exists; attach with: tmux attach -t $session" >&2
    exit 1
  fi
  tmux new-session -d -s "$session" \
    "AUTORESUME_FOREGROUND=1 '${BASH_SOURCE[0]}' '$env_file'"
  echo "Launched supervisor in tmux session: $session"
  echo "Watcher log: $WATCHER_LOG"
  exit 0
fi

# One supervisor per run.
lock_file="/home/sk7524/skyrl-runs/.autoresume-${RUN_NAME}.lock"
exec 9>"$lock_file"
if ! flock -n 9; then
  log "another supervisor holds $lock_file; exiting"
  exit 1
fi

# ---------------------------------------------------------------------------
# gcloud helpers
# ---------------------------------------------------------------------------

# READY / CREATING / ... / MISSING (definitively absent) / UNKNOWN (gcloud
# itself failed — never treat as a preemption).
node_state() {
  local out rc
  out="$(timeout 90 gcloud compute tpus tpu-vm list \
    --zone "$ZONE" --project "$PROJECT" \
    --filter="name.basename()=${TPU_NAME}" \
    --format="value(state)" 2>/dev/null)"
  rc=$?
  if (( rc != 0 )); then
    echo "UNKNOWN"
  elif [[ -z "$out" ]]; then
    echo "MISSING"
  else
    echo "$out" | head -1
  fi
}

# ACTIVE / SUSPENDED / FAILED / ... / MISSING / UNKNOWN
qr_state() {
  local out rc
  out="$(timeout 90 gcloud alpha compute tpus queued-resources describe "$QR_NAME" \
    --zone "$ZONE" --project "$PROJECT" \
    --format="value(state.state)" 2>&1)"
  rc=$?
  if (( rc == 0 )); then
    echo "${out:-UNKNOWN}" | head -1
  elif echo "$out" | grep -qi "NOT_FOUND\|was not found"; then
    echo "MISSING"
  else
    echo "UNKNOWN"
  fi
}

qr_delete() {
  timeout 120 gcloud alpha compute tpus queued-resources delete "$QR_NAME" \
    --zone "$ZONE" --project "$PROJECT" --force --quiet >>"$WATCHER_LOG" 2>&1 || true
}

ssh_ok() {
  timeout 60 gcloud alpha compute tpus tpu-vm ssh "${REMOTE_USER}@${TPU_NAME}" \
    --project="$PROJECT" --zone="$ZONE" --worker=0 \
    --ssh-key-file="$SSH_KEY_FILE" --quiet --command 'true' >/dev/null 2>&1
}

# Tinker API healthy on worker 0? (checked from the worker itself, so it
# works regardless of tunnel state)
api_healthy() {
  timeout 60 gcloud alpha compute tpus tpu-vm ssh "${REMOTE_USER}@${TPU_NAME}" \
    --project="$PROJECT" --zone="$ZONE" --worker=0 \
    --ssh-key-file="$SSH_KEY_FILE" --quiet \
    --command "curl -fsS -m 5 http://127.0.0.1:${API_PORT}/api/v1/get_server_capabilities >/dev/null" \
    >/dev/null 2>&1
}

jobman_create() {
  log "jobman create $JOBMAN_YAML (from $JOBMAN_DIR)"
  (cd "$JOBMAN_DIR" && timeout 600 "$JOBMAN_BIN" create "$JOBMAN_YAML") \
    >>"$WATCHER_LOG" 2>&1 || log "jobman create exited nonzero (will re-poll)"
}

# ---------------------------------------------------------------------------
# (a)+(b) QR ensure + wait READY
# ---------------------------------------------------------------------------
ensure_node() {
  local created=0 ns qs i
  while true; do
    ns="$(node_state)"
    case "$ns" in
      READY)
        if ssh_ok; then
          if (( created )); then
            log "node READY + ssh ok after (re)create; ${SETUP_GRACE_SEC}s grace for jobman base setup"
            sleep "$SETUP_GRACE_SEC"
          fi
          log "node ${TPU_NAME} READY"
          return 0
        fi
        log "node READY but ssh not up yet; waiting"
        ;;
      UNKNOWN)
        log "gcloud unavailable while checking node (NOT treating as preemption); retrying"
        ;;
      MISSING|SUSPENDED|PREEMPTED|TERMINATED|STOPPED|FAILED)
        qs="$(qr_state)"
        log "node state=${ns}, QR state=${qs}"
        case "$qs" in
          SUSPENDED|FAILED)
            log "deleting ${qs} queued-resource ${QR_NAME} (must be gone before jobman create)"
            qr_delete
            for i in $(seq 1 60); do
              [[ "$(qr_state)" == "MISSING" ]] && break
              (( i % 10 == 0 )) && qr_delete   # delete is async; re-nudge
              sleep 20
            done
            if [[ "$(qr_state)" == "MISSING" ]]; then
              created=1
              jobman_create
            else
              log "queued-resource still present after delete wait; looping"
            fi
            ;;
          MISSING)
            created=1
            jobman_create
            ;;
          UNKNOWN)
            log "gcloud unavailable while checking QR; retrying"
            ;;
          *)
            # ACTIVE / ACCEPTED / PROVISIONING / WAITING_FOR_RESOURCES /
            # SUSPENDING ... let it play out.
            ;;
        esac
        ;;
      *)
        log "node state=${ns}; waiting"
        ;;
    esac
    sleep "$POLL_SEC"
  done
}

# ---------------------------------------------------------------------------
# (c) bring-up
# ---------------------------------------------------------------------------
bring_up() {
  if api_healthy; then
    log "Tinker API already healthy on worker 0; skipping bring-up"
    return 0
  fi
  if [[ "$BRINGUP_MANUAL" == "1" ]]; then
    log "BRINGUP_MANUAL=1: waiting for a MANUALLY started server (see the env file's TODO-manual notes); polling API every ${POLL_SEC}s"
    while true; do
      api_healthy && { log "manual server is healthy"; return 0; }
      local ns
      ns="$(node_state)"
      if [[ "$ns" != "READY" && "$ns" != "UNKNOWN" ]]; then
        log "node lost while waiting for manual bring-up (state=${ns})"
        return 1
      fi
      sleep "$POLL_SEC"
    done
  fi
  log "running colocated bring-up (start_colocated_vllm_tinker.sh)"
  if (cd "$repo_root" && ./tpu/start_colocated_vllm_tinker.sh) >>"$WATCHER_LOG" 2>&1; then
    log "bring-up done"
    return 0
  fi
  # Bring-up failures are very often preemptions mid-flight: recheck node
  # state from the top rather than hammering a dead machine.
  log "bring-up FAILED (see $WATCHER_LOG); sleeping 120s then rechecking node state"
  sleep 120
  return 1
}

# ---------------------------------------------------------------------------
# (d) client
# ---------------------------------------------------------------------------
launched_this_session=0
last_launch_epoch=0

write_state_file() {
  if [[ -d "$LOG_PATH" && ! -f "$STATE_FILE" ]]; then
    {
      echo "run_name=${RUN_NAME}"
      echo "first_launch_utc=$(date -u '+%Y-%m-%d %H:%M:%S')"
      echo "log_path=${LOG_PATH}"
    } > "$STATE_FILE" 2>/dev/null && log "state file written: $STATE_FILE"
  fi
}

launch_client() {
  local behavior="$FIRST_BEHAVIOR"
  if [[ -f "$STATE_FILE" || -f "$metrics_file" || "$launched_this_session" == "1" ]]; then
    behavior="resume"
  fi

  # The tunnel targets worker 0's external IP, which changes on every spot
  # recreation — always drop the old tunnel so the client script rebuilds it.
  tmux kill-session -t "$TUNNEL_SESSION" 2>/dev/null || true

  log "launching client (session=${CLIENT_SESSION}, behavior_if_log_dir_exists=${behavior}, log_path=${LOG_PATH})"
  if ! (cd "$repo_root" && \
        TUNNEL_SESSION="$TUNNEL_SESSION" \
        REPLACE_CLIENT=1 \
        BEHAVIOR_IF_LOG_DIR_EXISTS="$behavior" \
        LOG_PATH="$LOG_PATH" \
        ./tpu/run_tinker_math_rl_qwen35_9b.sh) >>"$WATCHER_LOG" 2>&1; then
    log "client launch FAILED (API unreachable through tunnel?); will recheck node + server"
    return 1
  fi
  launched_this_session=1
  last_launch_epoch="$(date +%s)"
  # If we resumed, LOG_PATH persists and this sticks now; on a first 'delete'
  # launch the cookbook wipes LOG_PATH at startup, so the health loop rewrites
  # the state file once metrics.jsonl exists.
  if [[ "$behavior" == "resume" ]]; then
    write_state_file
  fi
  return 0
}

run_complete() {
  [[ -f "$metrics_file" ]] || return 1
  local last_step
  last_step="$(python3 - "$metrics_file" <<'PY' 2>/dev/null
import json, sys
last = -1
with open(sys.argv[1], "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            step = json.loads(line).get("step")
        except Exception:
            continue  # tolerate a torn trailing line
        if isinstance(step, int) and step > last:
            last = step
print(last)
PY
)"
  [[ -n "$last_step" ]] && (( last_step >= MAX_STEPS - 1 ))
}

metrics_age_sec() {
  local mtime now
  if [[ -f "$metrics_file" ]]; then
    mtime="$(stat -c %Y "$metrics_file" 2>/dev/null || echo 0)"
    now="$(date +%s)"
    echo $(( now - mtime ))
  else
    echo -1
  fi
}

# ---------------------------------------------------------------------------
# (e) health loop -> echoes NODE_LOST | CLIENT_DONE | CLIENT_DEAD | STALLED
# ---------------------------------------------------------------------------
health_loop() {
  local stale_count=0 ns age
  while true; do
    sleep "$HEALTH_SEC"

    ns="$(node_state)"
    if [[ "$ns" == "UNKNOWN" ]]; then
      log "health: gcloud unavailable; leaving client alone"
    elif [[ "$ns" != "READY" ]]; then
      log "health: node lost (state=${ns}); killing client session"
      tmux kill-session -t "$CLIENT_SESSION" 2>/dev/null || true
      tmux kill-session -t "$TUNNEL_SESSION" 2>/dev/null || true
      echo "NODE_LOST"
      return
    fi

    if ! tmux has-session -t "$CLIENT_SESSION" 2>/dev/null; then
      if run_complete; then
        echo "CLIENT_DONE"
      else
        log "health: client session gone before step $((MAX_STEPS - 1)); will relaunch with resume"
        echo "CLIENT_DEAD"
      fi
      return
    fi

    write_state_file

    age="$(metrics_age_sec)"
    if (( age < 0 )); then
      if (( $(date +%s) - last_launch_epoch > NO_METRICS_TIMEOUT_SEC )); then
        log "health: no metrics.jsonl ${NO_METRICS_TIMEOUT_SEC}s after launch; treating as stalled"
        tmux kill-session -t "$CLIENT_SESSION" 2>/dev/null || true
        echo "STALLED"
        return
      fi
      log "health: waiting for first metrics line (client alive)"
    elif (( age > STALL_SEC )); then
      stale_count=$((stale_count + 1))
      log "health: WARNING metrics.jsonl stale (${age}s > ${STALL_SEC}s), consecutive=${stale_count}"
      if (( stale_count >= 2 )); then
        log "health: 2 consecutive stale checks; killing client for a resume relaunch"
        tmux kill-session -t "$CLIENT_SESSION" 2>/dev/null || true
        echo "STALLED"
        return
      fi
    else
      stale_count=0
    fi
  done
}

# ---------------------------------------------------------------------------
# main supervision loop
# ---------------------------------------------------------------------------
log "supervisor starting: env=${env_file} tpu=${TPU_NAME} qr=${QR_NAME} yaml=${JOBMAN_YAML}"
log "log_path=${LOG_PATH} client_session=${CLIENT_SESSION} max_steps=${MAX_STEPS} first_behavior=${FIRST_BEHAVIOR}"

# A finished run should not be restarted by a stray supervisor launch.
if run_complete; then
  log "metrics already show step >= $((MAX_STEPS - 1)); run is complete. Exiting."
  exit 0
fi

while true; do
  ensure_node

  if ! bring_up; then
    continue
  fi

  if ! launch_client; then
    sleep 120
    continue
  fi

  reason="$(health_loop | tail -1)"
  case "$reason" in
    CLIENT_DONE)
      log "run complete: metrics reached step >= $((MAX_STEPS - 1)) and client exited. Supervisor done."
      exit 0
      ;;
    NODE_LOST)
      log "looping: node lost -> QR ensure"
      ;;
    CLIENT_DEAD|STALLED)
      log "looping: ${reason} -> recheck node/server, relaunch client with resume"
      ;;
    *)
      log "looping: unexpected health result '${reason}'"
      ;;
  esac
done
