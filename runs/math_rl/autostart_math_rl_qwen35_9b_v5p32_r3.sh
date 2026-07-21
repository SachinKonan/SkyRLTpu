#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-vision-mix}"
ZONE="${ZONE:-us-east5-a}"
TPU_NAME="${TPU_NAME:-sk7524-math-qwen35-9b-v5p32-r3-east5a_spot}"
RUN_LABEL="${RUN_LABEL:-v5p32-r3}"
REMOTE_USER="${REMOTE_USER:-sk7524_princeton_edu}"
SSH_KEY_FILE="${SSH_KEY_FILE:-$HOME/.ssh/jobman_tpu_ed25519}"

SERVER_REPO="${SERVER_REPO:-/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu}"
RUN_DIR="${MATH_RL_RUN_DIR:-${SERVER_REPO}/runs/math_rl}"
LOG_DIR="${RUN_DIR}/logs"
REMOTE_SKYRL_DIR="${REMOTE_SKYRL_DIR:-/home/${REMOTE_USER}/SkyRLTpu}"

NUM_PROCESSES="${NUM_PROCESSES:-4}"
TP_SIZE="${TP_SIZE:-4}"
FSDP_SIZE="${FSDP_SIZE:-4}"
TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-8}"
SAMPLE_MAX_NUM_SEQUENCES="${SAMPLE_MAX_NUM_SEQUENCES:-256}"
LOCAL_PORT="${LOCAL_PORT:-18009}"
REMOTE_PORT="${REMOTE_PORT:-8000}"
TINKER_API_WORKER="${TINKER_API_WORKER:-0}"
TUNNEL_SESSION="${TUNNEL_SESSION:-skyrl-math-tinker-tunnel-${RUN_LABEL}}"
CLIENT_SESSION="${CLIENT_SESSION:-skyrl-math-rl-qwen35-9b-${RUN_LABEL}}"

mkdir -p "${LOG_DIR}"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

tpu_state() {
  gcloud compute tpus tpu-vm describe "${TPU_NAME}" \
    --project="${PROJECT}" \
    --zone="${ZONE}" \
    --format='value(state)' 2>/dev/null || true
}

all_workers_ready() {
  gcloud alpha compute tpus tpu-vm ssh "${REMOTE_USER}@${TPU_NAME}" \
    --project="${PROJECT}" \
    --zone="${ZONE}" \
    --worker=all \
    --batch-size="${NUM_PROCESSES}" \
    --ssh-key-file="${SSH_KEY_FILE}" \
    --quiet \
    --command='export PATH="$HOME/.local/bin:$PATH"; command -v uv >/dev/null && test -d "$HOME/gcs"' >/dev/null 2>&1
}

wait_for_tpu() {
  while true; do
    state="$(tpu_state)"
    log "TPU state: ${state:-unknown}"
    if [[ "${state}" == "READY" || "${state}" == "ACTIVE" ]]; then
      if all_workers_ready; then
        log "All workers are reachable and have uv/gcs ready."
        return 0
      fi
      log "TPU is up but workers are not fully initialized yet."
    fi
    sleep 30
  done
}

server_commit() {
  git --git-dir="${SERVER_REPO}/.git" --work-tree="${SERVER_REPO}" rev-parse HEAD
}

remote_commit_matches() {
  local commit="$1"
  gcloud alpha compute tpus tpu-vm ssh "${REMOTE_USER}@${TPU_NAME}" \
    --project="${PROJECT}" \
    --zone="${ZONE}" \
    --worker=all \
    --batch-size="${NUM_PROCESSES}" \
    --ssh-key-file="${SSH_KEY_FILE}" \
    --quiet \
    --command="test \"\$(cat '${REMOTE_SKYRL_DIR}/.skyrltpu_commit' 2>/dev/null)\" = '${commit}'" >/dev/null 2>&1
}

sync_server_repo() {
  local commit
  commit="$(server_commit)"
  if remote_commit_matches "${commit}"; then
    log "Server repo ${commit} is already present on all TPU workers."
    return 0
  fi

  local tmpdir archive archive_tar remote_archive
  tmpdir="$(mktemp -d)"
  archive="${tmpdir}/SkyRLTpu-${commit}.tar.gz"
  archive_tar="${tmpdir}/SkyRLTpu-${commit}.tar"
  remote_archive="/tmp/SkyRLTpu-${commit}.tar.gz"

  local git_cmd
  git_cmd=(git --git-dir="${SERVER_REPO}/.git" --work-tree="${SERVER_REPO}")
  if ! "${git_cmd[@]}" diff --quiet || ! "${git_cmd[@]}" diff --cached --quiet; then
    log "Server repo is dirty; refusing to sync ${SERVER_REPO}."
    return 1
  fi

  mapfile -t submodule_paths < <("${git_cmd[@]}" config --file "${SERVER_REPO}/.gitmodules" --get-regexp '^submodule\..*\.path$' | awk '{print $2}')
  "${git_cmd[@]}" archive --format=tar HEAD > "${archive_tar}"
  for submodule_path in "${submodule_paths[@]}"; do
    submodule_commit="$("${git_cmd[@]}" ls-tree HEAD "${submodule_path}" | awk '{print $3}')"
    if [[ -z "${submodule_commit}" ]]; then
      log "Could not find ${submodule_path} in server repo HEAD."
      return 1
    fi
    if ! git -C "${SERVER_REPO}/${submodule_path}" cat-file -e "${submodule_commit}^{commit}" 2>/dev/null; then
      log "Missing ${submodule_path} commit ${submodule_commit}."
      return 1
    fi
    submodule_tar="${tmpdir}/$(echo "${submodule_path}" | tr / _)-${submodule_commit}.tar"
    git -C "${SERVER_REPO}/${submodule_path}" archive --format=tar --prefix="${submodule_path}/" "${submodule_commit}" > "${submodule_tar}"
    tar --concatenate --file "${archive_tar}" "${submodule_tar}"
  done
  gzip -c "${archive_tar}" > "${archive}"

  log "Parallel syncing server repo ${commit} to all TPU workers."
  gcloud alpha compute tpus tpu-vm scp "${archive}" "${REMOTE_USER}@${TPU_NAME}:${remote_archive}" \
    --project="${PROJECT}" \
    --zone="${ZONE}" \
    --worker=all \
    --batch-size="${NUM_PROCESSES}" \
    --ssh-key-file="${SSH_KEY_FILE}" \
    --quiet

  local remote_extract_cmd
  remote_extract_cmd="
set -euo pipefail
mkdir -p \"\$(dirname '${REMOTE_SKYRL_DIR}')\"
old_dir='${REMOTE_SKYRL_DIR}.old.'\$(date +%s).\$\$
if [ -e '${REMOTE_SKYRL_DIR}' ]; then
  mv '${REMOTE_SKYRL_DIR}' \"\${old_dir}\"
fi
mkdir -p '${REMOTE_SKYRL_DIR}'
tar -xzf '${remote_archive}' -C '${REMOTE_SKYRL_DIR}'
printf '%s\n' '${commit}' > '${REMOTE_SKYRL_DIR}/.skyrltpu_commit'
rm -f '${remote_archive}'
if [ -n \"\${old_dir:-}\" ] && [ -e \"\${old_dir}\" ]; then
  (rm -rf \"\${old_dir}\" >/tmp/skyrltpu-sync-rm.log 2>&1 || true) &
fi
"
  gcloud alpha compute tpus tpu-vm ssh "${REMOTE_USER}@${TPU_NAME}" \
    --project="${PROJECT}" \
    --zone="${ZONE}" \
    --worker=all \
    --batch-size="${NUM_PROCESSES}" \
    --ssh-key-file="${SSH_KEY_FILE}" \
    --quiet \
    --command "${remote_extract_cmd}"
  rm -rf "${tmpdir}"
}

start_tinker() {
  sync_server_repo

  log "Starting ${RUN_LABEL} Qwen3.5-9B Tinker server."
  PROJECT="${PROJECT}" \
  ZONE="${ZONE}" \
  TPU_NAME="${TPU_NAME}" \
  REMOTE_USER="${REMOTE_USER}" \
  SSH_KEY_FILE="${SSH_KEY_FILE}" \
  REMOTE_SKYRL_DIR="${REMOTE_SKYRL_DIR}" \
  MODEL_NAME="Qwen/Qwen3.5-9B" \
  NUM_PROCESSES="${NUM_PROCESSES}" \
  TP_SIZE="${TP_SIZE}" \
  FSDP_SIZE="${FSDP_SIZE}" \
  TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE}" \
  SAMPLE_MAX_NUM_SEQUENCES="${SAMPLE_MAX_NUM_SEQUENCES}" \
  INFERENCE_BACKEND=jax \
  API_PORT="${REMOTE_PORT}" \
  SESSION_TIMEOUT_SEC=86400 \
    bash "${SERVER_REPO}/tpu/start_skyrl_tinker.sh"
}

api_worker_host() {
  gcloud alpha compute tpus tpu-vm describe "${TPU_NAME}" \
    --project="${PROJECT}" \
    --zone="${ZONE}" \
    --format="value(networkEndpoints[${TINKER_API_WORKER}].accessConfig.externalIp)"
}

start_tunnel() {
  local worker_host
  worker_host="$(api_worker_host)"
  if [[ -z "${worker_host}" ]]; then
    log "Could not resolve API worker ${TINKER_API_WORKER} external IP for ${TPU_NAME}."
    return 1
  fi

  tmux kill-session -t "${TUNNEL_SESSION}" 2>/dev/null || true
  log "Starting local tunnel on 127.0.0.1:${LOCAL_PORT}."
  tmux new-session -d -s "${TUNNEL_SESSION}" \
    "/usr/bin/ssh -T -i '${SSH_KEY_FILE}' \
      -o IdentitiesOnly=yes \
      -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=30 \
      -o ServerAliveCountMax=3 \
      -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      -N -L '127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}' \
      '${REMOTE_USER}@${worker_host}' 2>&1 | tee '${LOG_DIR}/tinker-tunnel-${RUN_LABEL}.log'"
}

wait_for_tinker_api() {
  for _ in $(seq 1 360); do
    if curl -fsS --max-time 5 "http://127.0.0.1:${LOCAL_PORT}/api/v1/get_server_capabilities" >>"${LOG_DIR}/api-ready-${RUN_LABEL}.log" 2>&1; then
      log "Tinker API is reachable on 127.0.0.1:${LOCAL_PORT}."
      return 0
    fi
    sleep 10
  done
  log "Tinker API did not become reachable in time."
  return 1
}

api_ready_once() {
  curl -fsS --max-time 5 "http://127.0.0.1:${LOCAL_PORT}/api/v1/get_server_capabilities" >/dev/null 2>&1
}

tmux_session_exists() {
  tmux has-session -t "$1" 2>/dev/null
}

cleanup_local_sessions() {
  tmux kill-session -t "${TUNNEL_SESSION}" 2>/dev/null || true
  tmux kill-session -t "${CLIENT_SESSION}" 2>/dev/null || true
}

monitor_active_run() {
  local failures state
  failures=0

  while true; do
    sleep 30

    state="$(tpu_state)"
    if [[ "${state}" != "READY" && "${state}" != "ACTIVE" ]]; then
      log "TPU state changed to ${state:-unknown}; cleaning local sessions and waiting for jobman loop."
      cleanup_local_sessions
      return 1
    fi

    if ! tmux_session_exists "${CLIENT_SESSION}"; then
      log "Client session ${CLIENT_SESSION} exited; leaving TPU reservation up and stopping autostart watcher."
      return 0
    fi

    if api_ready_once; then
      failures=0
    else
      failures=$((failures + 1))
      log "Tinker API health check failed (${failures}/6)."
      if (( failures >= 6 )); then
        log "Tinker API stayed down; cleaning local sessions and retrying startup."
        cleanup_local_sessions
        return 1
      fi
    fi
  done
}

start_math_client() {
  local ts log_path client_log
  ts="$(date -u +%Y-%m-%d-%H-%M)"
  log_path="${RUN_DIR}/math-Qwen-Qwen3.5-9B-32rank-2e-05lr-16group-64batch-importance_sampling-seed0-1024tok-${RUN_LABEL}-${ts}"
  client_log="${LOG_DIR}/math-rl-qwen35-9b-1024tok-${RUN_LABEL}-${ts}.log"

  log "Starting MathRL client in tmux session ${CLIENT_SESSION}."
  PROJECT="${PROJECT}" \
  ZONE="${ZONE}" \
  TPU_NAME="${TPU_NAME}" \
  REMOTE_USER="${REMOTE_USER}" \
  SSH_KEY_FILE="${SSH_KEY_FILE}" \
  LOCAL_PORT="${LOCAL_PORT}" \
  REMOTE_PORT="${REMOTE_PORT}" \
  TINKER_API_WORKER="${TINKER_API_WORKER}" \
  TUNNEL_SESSION="${TUNNEL_SESSION}" \
  CLIENT_SESSION="${CLIENT_SESSION}" \
  REPLACE_CLIENT=1 \
  MODEL_NAME="Qwen/Qwen3.5-9B" \
  GROUP_SIZE=16 \
  GROUPS_PER_BATCH=64 \
  LEARNING_RATE=2e-5 \
  MAX_TOKENS=1024 \
  LORA_RANK=32 \
  MAX_STEPS=180 \
  SAVE_EVERY=20 \
  EVAL_EVERY=20 \
  LOSS_FN=importance_sampling \
  SEED=0 \
  LOG_PATH="${log_path}" \
  CLIENT_LOG="${client_log}" \
    bash "${SERVER_REPO}/tpu/run_tinker_math_rl_qwen35_9b.sh"
}

while true; do
  wait_for_tpu
  if start_tinker; then
    start_tunnel
    if wait_for_tinker_api; then
      start_math_client
      log "Autostart complete; monitoring ${RUN_LABEL}."
      if monitor_active_run; then
        exit 0
      fi
    fi
  else
    log "Tinker startup failed; will retry after TPU is available again."
  fi
  sleep 30
done
