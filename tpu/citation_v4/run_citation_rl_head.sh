#!/usr/bin/env bash
set -euo pipefail
set -x

ROOT_DIR="${ROOT_DIR:-/scratch/gpfs/ZHUANGL/hk4638/SkyRLTpu}"
SETUP_DIR="${SETUP_DIR:-${ROOT_DIR}/tpu/citation_v4}"
GCLOUD_ENV_FILE="${GCLOUD_ENV_FILE:-${ROOT_DIR}/external/autoresearch-tpu-optimization/tpu_setup/gcloud_env.sh}"
GCLOUD_FALLBACK_BIN="${GCLOUD_FALLBACK_BIN:-/home/hk4638/SkyRL/.tools/google-cloud-sdk/bin/gcloud}"
RETRIEVER_CHAIN_SCRIPT="${RETRIEVER_CHAIN_SCRIPT:-${SETUP_DIR}/start_retriever_chain_gpu_test.sh}"
RETRIEVER_SLURM="${RETRIEVER_SLURM:-${SETUP_DIR}/start_retriever_v4_gpu_test.slurm}"
RUN_NAME="${RUN_NAME:-cit-v4-tinker-head-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-/scratch/gpfs/ZHUANGL/hk4638/tinker_outputs/citation_prediction_v4/${RUN_NAME}}"
HEAD_COMPLETE_FILE="${CITATION_RL_HEAD_COMPLETE_FILE:-${RUN_DIR}/.citation_rl_head_complete}"
TINKER_READY_WAIT_SECONDS="${TINKER_READY_WAIT_SECONDS:-43200}"
TINKER_POST_RETRIEVER_READY_WAIT_SECONDS="${TINKER_POST_RETRIEVER_READY_WAIT_SECONDS:-${TINKER_READY_WAIT_SECONDS}}"
RETRIEVER_READY_WAIT_SECONDS="${RETRIEVER_READY_WAIT_SECONDS:-43200}"
mkdir -p "${RUN_DIR}"

# A controller or terminal can be resumed after the training client has already
# finished. Treat that as success instead of silently launching a second RL run.
if [ "${CITATION_RL_FORCE_RERUN:-0}" != "1" ] && [ -s "${HEAD_COMPLETE_FILE}" ]; then
  echo "Citation RL head already completed: ${HEAD_COMPLETE_FILE}"
  exit 0
fi

pick_free_port() {
  python -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()'
}

wait_for_http() {
  local url="$1"
  local seconds="${2:-1800}"
  local start
  start="$(date +%s)"
  while true; do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      return 0
    fi
    if [ "$(( $(date +%s) - start ))" -ge "${seconds}" ]; then
      echo "Timed out waiting for ${url}" >&2
      return 1
    fi
    sleep 10
  done
}

wait_for_retriever() {
  local url="$1"
  local seconds="${2:-7200}"
  local start
  start="$(date +%s)"
  while true; do
    if curl -fsS -X POST "${url}" \
      -H "Content-Type: application/json" \
      -d '{"query":"retriever readiness probe","topk":1,"return_scores":true}' >/dev/null 2>&1; then
      return 0
    fi
    if [ "$(( $(date +%s) - start ))" -ge "${seconds}" ]; then
      echo "Timed out waiting for retriever ${url}" >&2
      return 1
    fi
    sleep 10
  done
}

wait_for_file() {
  local path="$1"
  local seconds="${2:-7200}"
  local start
  start="$(date +%s)"
  while [ ! -f "${path}" ]; do
    if [ "$(( $(date +%s) - start ))" -ge "${seconds}" ]; then
      echo "Timed out waiting for ${path}" >&2
      return 1
    fi
    sleep 10
  done
}

cleanup() {
  if [ -n "${TUNNEL_PID:-}" ]; then
    kill "${TUNNEL_PID}" 2>/dev/null || true
  fi
  if [ -n "${RETRIEVER_TUNNEL_PID:-}" ]; then
    kill "${RETRIEVER_TUNNEL_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

start_retriever_tunnel() {
  if [ -n "${RETRIEVER_TUNNEL_PID:-}" ]; then
    kill "${RETRIEVER_TUNNEL_PID}" 2>/dev/null || true
    wait "${RETRIEVER_TUNNEL_PID}" 2>/dev/null || true
    unset RETRIEVER_TUNNEL_PID
  fi
  RETRIEVER_LOCAL_PORT="${RETRIEVER_LOCAL_PORT:-$(pick_free_port)}"
  ssh -N -L "${RETRIEVER_LOCAL_PORT}:127.0.0.1:${RETRIEVER_PORT}" "${RETRIEVER_NODE}" &
  RETRIEVER_TUNNEL_PID="$!"
  sleep "${RETRIEVER_TUNNEL_SETTLE_SECONDS:-3}"
  kill -0 "${RETRIEVER_TUNNEL_PID}" 2>/dev/null
  CITATION_RETRIEVER_URL="http://127.0.0.1:${RETRIEVER_LOCAL_PORT}/retrieve"
  wait_for_retriever "${CITATION_RETRIEVER_URL}" "${RETRIEVER_TUNNEL_READY_WAIT_SECONDS:-120}"
  echo "Using tunneled gpu-test retriever: ${CITATION_RETRIEVER_URL} -> ${RETRIEVER_NODE}:${RETRIEVER_PORT}"
}

load_retriever_ready_file() {
  if [ -z "${CITATION_RETRIEVER_READY_FILE:-}" ] || [ -n "${CITATION_RETRIEVER_URL:-}" ]; then
    return 0
  fi

  set -a
  # shellcheck disable=SC1090
  source "${CITATION_RETRIEVER_READY_FILE}"
  set +a

  if [ "${CITATION_RETRIEVER_TUNNEL_FROM_READY_FILE:-1}" = "1" ] && [ -n "${RETRIEVER_NODE:-}" ] && [ -n "${RETRIEVER_PORT:-}" ]; then
    start_retriever_tunnel
    return 0
  fi

  if [ -n "${RETRIEVER_URL:-}" ]; then
    CITATION_RETRIEVER_URL="${RETRIEVER_URL}"
    wait_for_retriever "${CITATION_RETRIEVER_URL}" "${RETRIEVER_READY_WAIT_SECONDS:-7200}"
  fi
}

start_tinker_tunnel() {
  if [ -n "${TUNNEL_PID:-}" ]; then
    kill "${TUNNEL_PID}" 2>/dev/null || true
    wait "${TUNNEL_PID}" 2>/dev/null || true
    unset TUNNEL_PID
  fi
  if [ -n "${TINKER_TPU_SSH_HOST:-}" ]; then
    local ssh_args=(
      -i "${TINKER_TPU_SSH_KEY:-${HOME}/.ssh/google_compute_engine}"
      -o IdentitiesOnly=yes
      -o StrictHostKeyChecking=no
      -o "UserKnownHostsFile=${TINKER_TPU_KNOWN_HOSTS:-/tmp/tpu_known_hosts}"
      -N
      -L "${TINKER_LOCAL_PORT}:127.0.0.1:${TINKER_REMOTE_PORT}"
    )
    if [ "${USE_TPU_LOCAL_RETRIEVER:-0}" = "1" ] && [ -z "${CITATION_RETRIEVER_READY_FILE:-}" ] && [ -z "${CITATION_RETRIEVER_URL:-}" ]; then
      ssh_args+=(-L "${TPU_RETRIEVER_LOCAL_PORT}:127.0.0.1:${TPU_RETRIEVER_REMOTE_PORT}")
    fi
    ssh "${ssh_args[@]}" "${TINKER_TPU_SSH_USER:-${USER}}@${TINKER_TPU_SSH_HOST}" &
    TUNNEL_PID="$!"
    sleep "${TINKER_TUNNEL_SETTLE_SECONDS:-5}"
    kill -0 "${TUNNEL_PID}" 2>/dev/null
    return 0
  fi
  local ssh_args=(
    alpha compute tpus tpu-vm ssh "${TINKER_TPU_NAME}"
    --zone "${TINKER_TPU_ZONE}"
    --worker="${TINKER_API_WORKER_ID:-0}"
    --ssh-flag="-N"
    --ssh-flag="-L ${TINKER_LOCAL_PORT}:127.0.0.1:${TINKER_REMOTE_PORT}"
    --quiet
  )
  if [ "${USE_TPU_LOCAL_RETRIEVER:-0}" = "1" ] && [ -z "${CITATION_RETRIEVER_READY_FILE:-}" ] && [ -z "${CITATION_RETRIEVER_URL:-}" ]; then
    ssh_args+=(--ssh-flag="-L ${TPU_RETRIEVER_LOCAL_PORT}:127.0.0.1:${TPU_RETRIEVER_REMOTE_PORT}")
  fi
  gcloud "${ssh_args[@]}" &
  TUNNEL_PID="$!"
  sleep "${TINKER_TUNNEL_SETTLE_SECONDS:-5}"
  kill -0 "${TUNNEL_PID}" 2>/dev/null
}

ensure_tinker_api() {
  local seconds="${1:-7200}"
  if [ "${CAN_START_TINKER_TUNNEL:-0}" = "1" ]; then
    local start
    start="$(date +%s)"
    while true; do
      if start_tinker_tunnel && wait_for_http "${TINKER_BASE_URL}/openapi.json" "${TINKER_TUNNEL_PROBE_SECONDS:-120}"; then
        TINKER_READY_CONFIRMED=1
        return 0
      fi
      if [ "$(( $(date +%s) - start ))" -ge "${seconds}" ]; then
        echo "Timed out waiting for TPU Tinker API at ${TINKER_BASE_URL}" >&2
        return 1
      fi
      echo "Tinker TPU API not reachable yet; retrying SSH tunnel in ${TINKER_TUNNEL_RETRY_SECONDS:-60}s" >&2
      sleep "${TINKER_TUNNEL_RETRY_SECONDS:-60}"
    done
  fi

  wait_for_http "${TINKER_BASE_URL}/openapi.json" "${seconds}"
  TINKER_READY_CONFIRMED=1
}

TINKER_READY_CONFIRMED=0

if [ -z "${TINKER_BASE_URL:-}" ]; then
  TINKER_TPU_NAME="${TINKER_TPU_NAME:-hk4638-v5p-16-cit-v4-tinker_1}"
  TINKER_TPU_ZONE="${TINKER_TPU_ZONE:-us-east5-a}"
  TINKER_API_WORKER_ID="${TINKER_API_WORKER_ID:-0}"
  TINKER_REMOTE_PORT="${TINKER_REMOTE_PORT:-8000}"
  TINKER_LOCAL_PORT="${TINKER_LOCAL_PORT:-$(pick_free_port)}"
  TPU_RETRIEVER_REMOTE_PORT="${TPU_RETRIEVER_REMOTE_PORT:-8010}"
  TPU_RETRIEVER_LOCAL_PORT="${TPU_RETRIEVER_LOCAL_PORT:-$(pick_free_port)}"
  if [ -f "${GCLOUD_ENV_FILE}" ]; then
    # shellcheck disable=SC1090
    source "${GCLOUD_ENV_FILE}"
  elif ! command -v gcloud >/dev/null 2>&1; then
    if [ -x "${GCLOUD_FALLBACK_BIN}" ]; then
      export PATH="$(dirname "${GCLOUD_FALLBACK_BIN}"):${PATH}"
    else
      echo "gcloud is required to open the TPU API tunnel" >&2
      exit 1
    fi
  fi
  TINKER_BASE_URL="http://127.0.0.1:${TINKER_LOCAL_PORT}"
  CAN_START_TINKER_TUNNEL=1
fi

ensure_tinker_api "${TINKER_READY_WAIT_SECONDS}"

if [ "${USE_TPU_LOCAL_RETRIEVER:-0}" = "1" ] && [ -z "${CITATION_RETRIEVER_READY_FILE:-}" ] && [ -z "${CITATION_RETRIEVER_URL:-}" ]; then
  CITATION_RETRIEVER_URL="http://127.0.0.1:${TPU_RETRIEVER_LOCAL_PORT}/retrieve"
  wait_for_retriever "${CITATION_RETRIEVER_URL}" "${RETRIEVER_READY_WAIT_SECONDS}"
  echo "Using TPU-local retriever tunnel: ${CITATION_RETRIEVER_URL}"
fi

if [ -z "${CITATION_RETRIEVER_READY_FILE:-}" ] && [ -z "${CITATION_RETRIEVER_URL:-}" ]; then
  CITATION_RETRIEVER_READY_FILE="${RUN_DIR}/retriever_current.env"
  RETRIEVER_SESSION="${RETRIEVER_SESSION:-cit-v4-tinker-${RUN_NAME}-retriever}"
  RETRIEVER_CHAIN_SECONDS="${RETRIEVER_CHAIN_SECONDS:-86400}"
  RETRIEVER_SERVER_TOPK="${RETRIEVER_SERVER_TOPK:-50}"
  if [ ! -f "${RETRIEVER_CHAIN_SCRIPT}" ] || [ ! -f "${RETRIEVER_SLURM}" ]; then
    echo "Missing retriever launcher: ${RETRIEVER_CHAIN_SCRIPT} or ${RETRIEVER_SLURM}" >&2
    exit 1
  fi
  tmux new-session -d -s "${RETRIEVER_SESSION}" \
    "cd '${ROOT_DIR}' && RETRIEVER_SLURM='${RETRIEVER_SLURM}' AUTORESEARCH_RUN_DIR='${RUN_DIR}' AUTORESEARCH_RETRIEVER_CURRENT_READY_FILE='${CITATION_RETRIEVER_READY_FILE}' AUTORESEARCH_RETRIEVER_CHAIN_SECONDS='${RETRIEVER_CHAIN_SECONDS}' AUTORESEARCH_RETRIEVER_SERVER_TOPK='${RETRIEVER_SERVER_TOPK}' bash '${RETRIEVER_CHAIN_SCRIPT}'"
  echo "Started retriever chain tmux session: ${RETRIEVER_SESSION}"
fi

if [ -n "${CITATION_RETRIEVER_READY_FILE:-}" ]; then
  wait_for_file "${CITATION_RETRIEVER_READY_FILE}" "${RETRIEVER_READY_WAIT_SECONDS}"
  load_retriever_ready_file
fi

# The gpu-test retriever can take a while to come up. Reconfirm the TPU API after
# that wait so preemptions do not leave the trainer using a stale SSH tunnel.
ensure_tinker_api "${TINKER_POST_RETRIEVER_READY_WAIT_SECONDS}"

export TINKER_BASE_URL
export CITATION_RETRIEVER_READY_FILE="${CITATION_RETRIEVER_READY_FILE:-}"
export CITATION_RETRIEVER_URL="${CITATION_RETRIEVER_URL:-}"
export CITATION_RERANK_RETRIEVAL_TOPK="${CITATION_RERANK_RETRIEVAL_TOPK:-50}"
export CITATION_RERANK_ALPHA="${CITATION_RERANK_ALPHA:-0.1}"
export CITATION_RERANK_FINAL_TOPK="${CITATION_RERANK_FINAL_TOPK:-5}"
export CITATION_TOP_K="${CITATION_TOP_K:-5}"
export CITATION_COUNT_PATH="${CITATION_COUNT_PATH:-/scratch/gpfs/ZHUANGL/hk4638/SemanticScholar/arxiv_impact_counts_s2.csv}"
export CITATION_METRIC_BETA="${CITATION_METRIC_BETA:-0.5}"

CLIENT_SCRIPT="${CITATION_HEAD_CLIENT_SCRIPT:-${ROOT_DIR}/skyrl-agent/examples/run_tinker/tinker_citation_prediction_v4.sh}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}}" \
WANDB_NAME="${WANDB_NAME:-${RUN_NAME}}" \
bash "${CLIENT_SCRIPT}" "$@"

COMPLETE_TMP="${HEAD_COMPLETE_FILE}.tmp.$$"
printf 'run_name=%s\ncompleted_at=%s\n' "${RUN_NAME}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${COMPLETE_TMP}"
mv "${COMPLETE_TMP}" "${HEAD_COMPLETE_FILE}"
echo "Citation RL head completed: ${HEAD_COMPLETE_FILE}"
