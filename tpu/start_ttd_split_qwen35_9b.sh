#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-vision-mix}"
ZONE="${ZONE:-us-east5-a}"
TPU_NAME="${TPU_NAME:-sk7524-ttd-erdos-v5p64-east5a_spot}"
REMOTE_USER="${REMOTE_USER:-sk7524_princeton_edu}"
SSH_KEY_FILE="${SSH_KEY_FILE:-$HOME/.ssh/jobman_tpu_ed25519}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-9B}"
REMOTE_SKYRL_DIR="${REMOTE_SKYRL_DIR:-/home/${REMOTE_USER}/SkyRLTpu}"
VLLM_WORKERS="${VLLM_WORKERS:-6,7}"
TRAIN_NUM_PROCESSES="${TRAIN_NUM_PROCESSES:-6}"
TRAIN_WORKER_IDS="${TRAIN_WORKER_IDS:-0,1,2,3,4,5}"
VLLM_PORT="${VLLM_PORT:-8001}"
TINKER_API_PORT="${TINKER_API_PORT:-8000}"

VLLM_TP_SIZE="${VLLM_TP_SIZE:-4}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
VLLM_MAX_CONCURRENT_REQUESTS="${VLLM_MAX_CONCURRENT_REQUESTS:-64}"
VLLM_TPU_PROCESS_BOUNDS="${VLLM_TPU_PROCESS_BOUNDS:-1,1,1}"
VLLM_TPU_CHIPS_PER_PROCESS_BOUNDS="${VLLM_TPU_CHIPS_PER_PROCESS_BOUNDS:-1,4,1}"
VLLM_TPU_VISIBLE_CHIPS="${VLLM_TPU_VISIBLE_CHIPS:-0,1,2,3}"
TRAIN_TP_SIZE="${TRAIN_TP_SIZE:-4}"
TRAIN_FSDP_SIZE="${TRAIN_FSDP_SIZE:-6}"
TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-1}"
SAMPLE_MAX_NUM_SEQUENCES="${SAMPLE_MAX_NUM_SEQUENCES:-64}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mapfile -t endpoint_ips < <(
  gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --format='value(networkEndpoints.ipAddress)' | tr ';' '\n' | tr ' ' '\n' | sed '/^$/d'
)

if (( ${#endpoint_ips[@]} < 8 )); then
  echo "Expected 8 worker endpoint IPs for v5p-64, found ${#endpoint_ips[@]}." >&2
  exit 1
fi

IFS=',' read -r -a vllm_worker_array <<< "$VLLM_WORKERS"
vllm_urls=()
for worker in "${vllm_worker_array[@]}"; do
  worker="${worker//[[:space:]]/}"
  ip="${endpoint_ips[$worker]}"
  vllm_urls+=("http://${ip}:${VLLM_PORT}")
  echo "Starting vLLM on worker ${worker} (${ip})"
  PROJECT="$PROJECT" \
    ZONE="$ZONE" \
    TPU_NAME="$TPU_NAME" \
    REMOTE_USER="$REMOTE_USER" \
    SSH_KEY_FILE="$SSH_KEY_FILE" \
    REMOTE_SKYRL_DIR="$REMOTE_SKYRL_DIR" \
    MODEL_NAME="$MODEL_NAME" \
    SERVED_MODEL_NAME="$MODEL_NAME" \
    VLLM_WORKER="$worker" \
    VLLM_PORT="$VLLM_PORT" \
    VLLM_TP_SIZE="$VLLM_TP_SIZE" \
    VLLM_MAX_MODEL_LEN="$VLLM_MAX_MODEL_LEN" \
    VLLM_MAX_NUM_SEQS="$VLLM_MAX_NUM_SEQS" \
    VLLM_DISABLE_SHARDY=true \
    VLLM_TPU_PROCESS_BOUNDS="$VLLM_TPU_PROCESS_BOUNDS" \
    VLLM_TPU_CHIPS_PER_PROCESS_BOUNDS="$VLLM_TPU_CHIPS_PER_PROCESS_BOUNDS" \
    VLLM_TPU_VISIBLE_CHIPS="$VLLM_TPU_VISIBLE_CHIPS" \
    bash "${script_dir}/start_vllm_tpu.sh"
done

train_addresses=""
for ((worker = 0; worker < TRAIN_NUM_PROCESSES; worker++)); do
  if [[ -n "$train_addresses" ]]; then
    train_addresses+=","
  fi
  train_addresses+="${endpoint_ips[$worker]}:8476"
done

vllm_base_url="$(IFS=,; echo "${vllm_urls[*]}")"
echo "Starting Tinker API/backend on train workers ${TRAIN_WORKER_IDS}"
echo "vLLM URLs: ${vllm_base_url}"

PROJECT="$PROJECT" \
  ZONE="$ZONE" \
  TPU_NAME="$TPU_NAME" \
  REMOTE_USER="$REMOTE_USER" \
  SSH_KEY_FILE="$SSH_KEY_FILE" \
  REMOTE_SKYRL_DIR="$REMOTE_SKYRL_DIR" \
  MODEL_NAME="$MODEL_NAME" \
  NUM_PROCESSES="$TRAIN_NUM_PROCESSES" \
  TP_SIZE="$TRAIN_TP_SIZE" \
  FSDP_SIZE="$TRAIN_FSDP_SIZE" \
  ACTIVE_WORKER_IDS="$TRAIN_WORKER_IDS" \
  MESH_WORKER_IDS="$TRAIN_WORKER_IDS" \
  TRAIN_MICRO_BATCH_SIZE="$TRAIN_MICRO_BATCH_SIZE" \
  SAMPLE_MAX_NUM_SEQUENCES="$SAMPLE_MAX_NUM_SEQUENCES" \
  TRAIN_TPU_PROCESS_ADDRESSES="$train_addresses" \
  INFERENCE_BACKEND=vllm \
  VLLM_BASE_URL="$vllm_base_url" \
  VLLM_MODEL_NAME="$MODEL_NAME" \
  VLLM_MAX_CONCURRENT_REQUESTS="$VLLM_MAX_CONCURRENT_REQUESTS" \
  API_PORT="$TINKER_API_PORT" \
  bash "${script_dir}/start_skyrl_tinker.sh"

echo "Split Qwen3.5-9B Tinker/vLLM launch submitted."
