#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TOOLS_ROOT="${TOOLS_ROOT:-/home/hk4638/SkyRL/.tools}"
export PATH="${TOOLS_ROOT}/google-cloud-sdk/bin:${TOOLS_ROOT}/jobman-venv/bin:${PATH}"
export CLOUDSDK_CONFIG="${CLOUDSDK_CONFIG:-${TOOLS_ROOT}/gcloud-config}"

: "${RESTORE_RESULT_PREFIX:?Set RESTORE_RESULT_PREFIX to a sanitized SFT/RL Tinker result prefix}"
: "${TINKER_INITIAL_STATE_PATH:?Set TINKER_INITIAL_STATE_PATH to the checkpoint URI inside that prefix}"

RUN_ID="${RUN_ID:-citation-rl-qwen35-9b-$(date -u +%Y%m%d-%H%M%S)}"
SERVER_RUN_ID="${SERVER_RUN_ID:-${RUN_ID}-server}"
RUN_NAME="${RUN_NAME:-${RUN_ID}}"
RUN_DIR="${RUN_DIR:-/scratch/gpfs/ZHUANGL/hk4638/tinker_outputs/citation_prediction_v4/${RUN_NAME}}"
TPU_BUCKET="${TPU_BUCKET:-hk4638-autoresearch-tpu-us-east5}"
TPU_ZONE="${TPU_ZONE:-us-east5-a}"
SERVER_RUN_PREFIX="gs://${TPU_BUCKET}/skyrl-tpu/citation-v4/rl/${SERVER_RUN_ID}"
SERVER_RESULT_PREFIX="${SERVER_RUN_PREFIX}/results"
SERVER_STATUS_URI="${SERVER_RESULT_PREFIX}/status.txt"
TPU_RESOURCE_NAME="hk4638-${SERVER_RUN_ID}_1"
RETRIEVER_SESSION="${RETRIEVER_SESSION:-cit-v4-tinker-${RUN_NAME}-retriever}"
SUBMIT_LOG="${RUN_DIR}/server-submit.log"
SERVER_JOB_ID=""
SERVER_SUBMITTED=0
HEAD_STATUS=1

mkdir -p "${RUN_DIR}"
command -v gcloud >/dev/null
command -v jobman >/dev/null

stop_server() {
  if [ "${SERVER_SUBMITTED}" = "1" ]; then
    printf 'stop\n' | gcloud storage cp - "${SERVER_RESULT_PREFIX}/stop.txt" >/dev/null 2>&1 || true
  fi
}

cleanup() {
  set +e
  stop_server
  if [ -f "${RUN_DIR}/retriever_current.env" ]; then
    # shellcheck disable=SC1090
    source "${RUN_DIR}/retriever_current.env"
    [ -z "${RETRIEVER_CHAIN_JOB_ID:-}" ] || scancel "${RETRIEVER_CHAIN_JOB_ID}" >/dev/null 2>&1 || true
  fi
  tmux kill-session -t "${RETRIEVER_SESSION}" 2>/dev/null || true
  if [ -n "${SERVER_JOB_ID}" ]; then
    jobman delete "${SERVER_JOB_ID}" >/dev/null 2>&1 || true
  fi
  if [ "${SERVER_SUBMITTED}" = "1" ]; then
    STATE="$(gcloud compute tpus queued-resources describe "${TPU_RESOURCE_NAME}" \
      --zone="${TPU_ZONE}" --format='value(state.state)' 2>/dev/null || true)"
    if [ "${STATE}" = "ACTIVE" ]; then
      gcloud compute tpus tpu-vm delete "${TPU_RESOURCE_NAME}" \
        --zone="${TPU_ZONE}" --quiet >/dev/null 2>&1 || true
    fi
    for _ in $(seq 1 120); do
      STATE="$(gcloud compute tpus queued-resources describe "${TPU_RESOURCE_NAME}" \
        --zone="${TPU_ZONE}" --format='value(state.state)' 2>/dev/null || true)"
      if [ -z "${STATE}" ] || { [ "${STATE}" != "ACTIVE" ] && [ "${STATE}" != "SUSPENDING" ]; }; then
        break
      fi
      sleep 5
    done
    gcloud compute tpus queued-resources delete "${TPU_RESOURCE_NAME}" \
      --zone="${TPU_ZONE}" --quiet >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

echo "Provisioning TPU server ${TPU_RESOURCE_NAME}"
MODE=canary \
WORKLOAD_MODE=server \
RUN_ID="${SERVER_RUN_ID}" \
TPU_BUCKET="${TPU_BUCKET}" \
RESULT_PREFIX_OVERRIDE="${SERVER_RUN_PREFIX}" \
RESTORE_RESULT_PREFIX="${RESTORE_RESULT_PREFIX}" \
SAMPLE_MAX_NUM_SEQUENCES="${SAMPLE_MAX_NUM_SEQUENCES:-32}" \
SAMPLE_HBM_UTILIZATION="${SAMPLE_HBM_UTILIZATION:-0.2}" \
CHECKPOINT_UPLOAD_SECONDS="${CHECKPOINT_UPLOAD_SECONDS:-60}" \
WAIT_AND_CLEANUP=0 \
bash "${SCRIPT_DIR}/submit_citation_sft.sh" | tee "${SUBMIT_LOG}"

SERVER_JOB_ID="$(sed -n 's/^Job ID: \([0-9][0-9]*\)$/\1/p' "${SUBMIT_LOG}" | tail -1)"
test -n "${SERVER_JOB_ID}"
SERVER_SUBMITTED=1

echo "TPU Jobman ID: ${SERVER_JOB_ID}"
echo "TPU resource: ${TPU_RESOURCE_NAME}"
echo "Server results: ${SERVER_RESULT_PREFIX}"
echo "RL output: ${RUN_DIR}"

set +e
ROOT_DIR="${REPO_ROOT}" \
RUN_NAME="${RUN_NAME}" \
RUN_DIR="${RUN_DIR}" \
RETRIEVER_SESSION="${RETRIEVER_SESSION}" \
TINKER_TPU_NAME="${TPU_RESOURCE_NAME}" \
TINKER_INITIAL_STATE_PATH="${TINKER_INITIAL_STATE_PATH}" \
MODEL_KEY="${MODEL_KEY:-qwen35_9b}" \
DATA_DIR="${DATA_DIR:-/scratch/gpfs/ZHUANGL/hk4638/data/citation_prediction_v4/rl_exclude_conservative_sft_prompts}" \
SWEEP_BP="${SWEEP_BP:-40}" \
SWEEP_N="${SWEEP_N:-30}" \
SWEEP_LR="${SWEEP_LR:-3e-6}" \
LORA_RANK="${LORA_RANK:-32}" \
MAX_STEPS="${MAX_STEPS:-100}" \
SAVE_EVERY="${SAVE_EVERY:-5}" \
EVAL_EVERY="${EVAL_EVERY:-10}" \
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-100}" \
TINKER_AGENT_MAX_PARALLEL="${TINKER_AGENT_MAX_PARALLEL:-32}" \
TINKER_AGENT_MAX_PROMPT_LENGTH="${TINKER_AGENT_MAX_PROMPT_LENGTH:-45000}" \
RETRIEVER_CHAIN_SECONDS="${RETRIEVER_CHAIN_SECONDS:-604800}" \
RETRIEVER_SERVER_TOPK="${RETRIEVER_SERVER_TOPK:-50}" \
bash "${SCRIPT_DIR}/run_citation_rl_head.sh" "$@"
HEAD_STATUS="$?"
set -e

if [ "${HEAD_STATUS}" -eq 0 ]; then
  HEAD_CHECKPOINTS="$(find "${RUN_DIR}" -type f -name checkpoints.jsonl -print -quit)"
  HEAD_RUNTIME_TASK="$(find "${RUN_DIR}" -type f -name runtime_task.yaml -print -quit)"
  [ -z "${HEAD_CHECKPOINTS}" ] || gcloud storage cp "${HEAD_CHECKPOINTS}" \
    "${SERVER_RESULT_PREFIX}/head-client-output/checkpoints.jsonl"
  [ -z "${HEAD_RUNTIME_TASK}" ] || gcloud storage cp "${HEAD_RUNTIME_TASK}" \
    "${SERVER_RESULT_PREFIX}/head-client-output/runtime_task.yaml"
  [ ! -s "${RUN_DIR}/.citation_rl_head_complete" ] || gcloud storage cp \
    "${RUN_DIR}/.citation_rl_head_complete" \
    "${SERVER_RESULT_PREFIX}/head-client-output/head-complete.txt"
fi

# The TPU mirrors checkpoints asynchronously. Allow one complete post-run
# interval before signaling shutdown so the final learner, sampler, and SQLite
# state are durable in GCS.
if [ "${HEAD_STATUS}" -eq 0 ]; then
  sleep "${CHECKPOINT_SETTLE_SECONDS:-75}"
fi
stop_server

for _ in $(seq 1 "${SERVER_STOP_POLL_ATTEMPTS:-120}"); do
  STATUS="$(gcloud storage cat "${SERVER_STATUS_URI}" 2>/dev/null || true)"
  [ "${STATUS}" = "passed" ] && break
  [ "${STATUS}" = "failed" ] && break
  sleep "${SERVER_STOP_POLL_SECONDS:-10}"
done

if [ "${HEAD_STATUS}" -ne 0 ]; then
  echo "Citation RL head failed with status ${HEAD_STATUS}" >&2
  exit "${HEAD_STATUS}"
fi
test "$(gcloud storage cat "${SERVER_STATUS_URI}" 2>/dev/null || true)" = "passed"
echo "Citation RL completed: ${RUN_NAME}"
echo "Durable server state: ${SERVER_RESULT_PREFIX}"
