#!/usr/bin/env bash
# Login-node supervisor for the TTD Erdos run on a spot v5p slice.
#
# jobman (loop: true) keeps the *reservation* alive across preemptions. This
# script keeps the *software stack* alive on top of it: whenever a fresh slice
# is READY but the Tinker API is not serving, it re-syncs SkyRL, relaunches the
# API + JAX backend, (re)opens the SSH tunnel, and (re)starts the discover
# client. The client resumes from the latest checkpoint (discover saves every
# save_every steps to the GCS-backed checkpoints dir), so training continues
# across preemptions.
#
# Run it in a persistent tmux on the login node:
#   tmux new-session -d -s ttd-supervise "bash tpu/supervise_ttd_run.sh 2>&1 | tee runs/ttd_erdos_v5p16/logs/supervise.log"
set -uo pipefail   # deliberately NOT -e: the loop must survive transient gcloud/ssh failures

PROJECT="${PROJECT:-vision-mix}"
ZONE="${ZONE:-us-east5-a}"
TPU_NAME="${TPU_NAME:-sk7524-ttd-erdos-v5p16-east5a_spot}"
REMOTE_USER="${REMOTE_USER:-sk7524_princeton_edu}"
SSH_KEY_FILE="${SSH_KEY_FILE:-$HOME/.ssh/jobman_tpu_ed25519}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

LOCAL_PORT="${LOCAL_PORT:-18000}"
REMOTE_PORT="${REMOTE_PORT:-8000}"
TUNNEL_SESSION="${TUNNEL_SESSION:-ttd-tunnel}"
CLIENT_SESSION="${CLIENT_SESSION:-ttd-client}"

# Training / mesh params (v5p-16 = 2 hosts, option B: whole slice trains + samples)
NUM_PROCESSES="${NUM_PROCESSES:-2}"
TP_SIZE="${TP_SIZE:-4}"
FSDP_SIZE="${FSDP_SIZE:-2}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-9B}"
SAMPLE_MAX_NUM_SEQUENCES="${SAMPLE_MAX_NUM_SEQUENCES:-64}"
MAX_LORA_RANK="${MAX_LORA_RANK:-32}"
TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-1}"

# Client params
RUN_DIR="${TTD_RUN_DIR:-${repo_root}/runs/ttd_erdos_v5p16}"
CONTEXT_WINDOW="${CONTEXT_WINDOW:-32768}"
GROUP_SIZE="${GROUP_SIZE:-16}"
GROUPS_PER_BATCH="${GROUPS_PER_BATCH:-64}"
TTD_EVAL_BACKEND="${TTD_EVAL_BACKEND:-submitit}"
LORA_RANK="${LORA_RANK:-32}"

POLL_SEC="${POLL_SEC:-60}"
API_READY_TRIES="${API_READY_TRIES:-60}"     # * API_READY_SLEEP seconds
API_READY_SLEEP="${API_READY_SLEEP:-15}"

mkdir -p "${RUN_DIR}/logs"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] supervise: $*"; }

ssh_worker0() {
  gcloud alpha compute tpus tpu-vm ssh "${REMOTE_USER}@${TPU_NAME}" \
    --project="${PROJECT}" --zone="${ZONE}" --worker=0 \
    --ssh-key-file="${SSH_KEY_FILE}" --quiet --command "$1" 2>/dev/null
}

vm_state() {
  gcloud alpha compute tpus tpu-vm describe "${TPU_NAME}" \
    --project="${PROJECT}" --zone="${ZONE}" --format='value(state)' 2>/dev/null
}

vm_ready() { local s; s="$(vm_state)"; [[ "$s" == "READY" || "$s" == "ACTIVE" ]]; }

base_ready() {  # uv installed and gcsfuse dir present -> jobman base setup done
  ssh_worker0 'export PATH="$HOME/.local/bin:$PATH"; command -v uv >/dev/null 2>&1 && echo BASE_OK' | grep -q BASE_OK
}

api_up() {  # Tinker API serving on worker0 (checked on the VM itself)
  ssh_worker0 'curl -fsS -m 6 http://localhost:'"${REMOTE_PORT}"'/api/v1/get_server_capabilities >/dev/null 2>&1 && echo API_UP' | grep -q API_UP
}

ensure_tunnel() {
  if tmux has-session -t "${TUNNEL_SESSION}" 2>/dev/null; then return 0; fi
  log "opening tunnel ${LOCAL_PORT}->${TPU_NAME}:${REMOTE_PORT} (auto-reconnect)"
  tmux new-session -d -s "${TUNNEL_SESSION}" "
while true; do
  gcloud alpha compute tpus tpu-vm ssh ${REMOTE_USER}@${TPU_NAME} \
    --project=${PROJECT} --zone=${ZONE} --worker=0 --ssh-key-file=${SSH_KEY_FILE} --quiet \
    -- -N -L 127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT} \
       -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes
  echo \"tunnel dropped \$(date -u +%H:%M:%SZ), reconnecting in 3s\"; sleep 3
done"
}

start_client() {
  local clog="${RUN_DIR}/logs/client-$(date -u +%Y%m%dT%H%M%SZ).log"
  log "starting discover client (log: ${clog})"
  tmux new-session -d -s "${CLIENT_SESSION}" "
cd '${repo_root}' && \
TTD_RUN_DIR='${RUN_DIR}' \
TINKER_BASE_URL='http://127.0.0.1:${LOCAL_PORT}' \
MODEL_NAME='${MODEL_NAME}' \
CONTEXT_WINDOW='${CONTEXT_WINDOW}' \
GROUP_SIZE='${GROUP_SIZE}' \
GROUPS_PER_BATCH='${GROUPS_PER_BATCH}' \
TTD_EVAL_BACKEND='${TTD_EVAL_BACKEND}' \
LORA_RANK='${LORA_RANK}' \
bash tpu/run_ttd_erdos_client.sh 2>&1 | tee '${clog}'"
}

deploy_stack() {
  log "deploying: sync SkyRL + launch API/backend on ${TPU_NAME}"
  TPU_NAME="${TPU_NAME}" ZONE="${ZONE}" PROJECT="${PROJECT}" REMOTE_USER="${REMOTE_USER}" \
    SSH_KEY_FILE="${SSH_KEY_FILE}" NUM_PROCESSES="${NUM_PROCESSES}" \
    bash "${script_dir}/sync_skyrl_to_tpu.sh" || { log "sync failed (VM may have died); will retry"; return 1; }

  TPU_NAME="${TPU_NAME}" ZONE="${ZONE}" PROJECT="${PROJECT}" REMOTE_USER="${REMOTE_USER}" \
    SSH_KEY_FILE="${SSH_KEY_FILE}" NUM_PROCESSES="${NUM_PROCESSES}" TP_SIZE="${TP_SIZE}" \
    FSDP_SIZE="${FSDP_SIZE}" MODEL_NAME="${MODEL_NAME}" INFERENCE_BACKEND=jax \
    SAMPLE_MAX_NUM_SEQUENCES="${SAMPLE_MAX_NUM_SEQUENCES}" MAX_LORA_RANK="${MAX_LORA_RANK}" \
    TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE}" API_PORT="${REMOTE_PORT}" \
    bash "${script_dir}/start_skyrl_tinker.sh" || { log "API launch failed; will retry"; return 1; }

  log "waiting for API to serve..."
  for ((i = 0; i < API_READY_TRIES; i++)); do
    if ! vm_ready; then log "VM left READY during API bringup; aborting deploy"; return 1; fi
    if api_up; then log "API is serving"; return 0; fi
    sleep "${API_READY_SLEEP}"
  done
  log "API did not come up within budget; will retry"
  return 1
}

log "supervisor started for ${TPU_NAME} (${NUM_PROCESSES} hosts, TP=${TP_SIZE} FSDP=${FSDP_SIZE})"
while true; do
  if ! vm_ready; then
    log "VM state='$(vm_state)' (not READY) — jobman is (re)provisioning; waiting"
    sleep "${POLL_SEC}"; continue
  fi
  if ! base_ready; then
    log "VM READY but jobman base setup (uv) not finished yet; waiting"
    sleep "${POLL_SEC}"; continue
  fi
  if ! api_up; then
    log "API down on a READY slice -> (re)deploy stack + fresh client"
    tmux kill-session -t "${CLIENT_SESSION}" 2>/dev/null || true   # old client is bound to a dead session
    if ! deploy_stack; then sleep "${POLL_SEC}"; continue; fi
  fi
  ensure_tunnel
  if ! tmux has-session -t "${CLIENT_SESSION}" 2>/dev/null; then
    start_client
  fi
  sleep "${POLL_SEC}"
done
