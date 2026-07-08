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
TRAIN_NUM_PROCESSES="${TRAIN_NUM_PROCESSES:-1}"
TRAIN_WORKER_IDS="${TRAIN_WORKER_IDS:-0}"
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
TRAIN_FSDP_SIZE="${TRAIN_FSDP_SIZE:-$TRAIN_NUM_PROCESSES}"
TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-1}"
SAMPLE_MAX_NUM_SEQUENCES="${SAMPLE_MAX_NUM_SEQUENCES:-64}"
TRAIN_TPU_PROCESS_BOUNDS="${TRAIN_TPU_PROCESS_BOUNDS:-}"
TRAIN_TPU_CHIPS_PER_PROCESS_BOUNDS="${TRAIN_TPU_CHIPS_PER_PROCESS_BOUNDS:-}"
TRAIN_TPU_VISIBLE_CHIPS="${TRAIN_TPU_VISIBLE_CHIPS:-0,1,2,3}"
TRAIN_TPU_PROCESS_PORT="${TRAIN_TPU_PROCESS_PORT:-8476}"
ALLOW_EXPERIMENTAL_MULTIHOST_TRAIN_SPLIT="${ALLOW_EXPERIMENTAL_MULTIHOST_TRAIN_SPLIT:-0}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

parse_csv_ints() {
  local raw="$1"
  local -n out_ref="$2"
  local entry

  out_ref=()
  IFS=',' read -r -a raw_entries <<< "$raw"
  for entry in "${raw_entries[@]}"; do
    entry="$(trim "$entry")"
    [[ -z "$entry" ]] && continue
    if [[ ! "$entry" =~ ^[0-9]+$ ]]; then
      echo "Expected integer in comma-separated list, got '${entry}' from '${raw}'." >&2
      exit 1
    fi
    out_ref+=("$entry")
  done
}

csv_join() {
  local IFS=,
  echo "$*"
}

bounds_product() {
  local bounds="$1"
  local -a dims
  parse_csv_ints "$bounds" dims
  if (( ${#dims[@]} != 3 )); then
    echo "Expected three comma-separated topology bounds, got '${bounds}'." >&2
    exit 1
  fi
  echo $(( dims[0] * dims[1] * dims[2] ))
}

default_train_process_bounds() {
  case "$1" in
    1) echo "1,1,1" ;;
    *)
      echo "No safe same-reservation TPU_PROCESS_BOUNDS default for TRAIN_NUM_PROCESSES=$1." >&2
      echo "Multi-host JAX sub-slices inside an existing v5p-64 failed our init probes; use ALLOW_EXPERIMENTAL_MULTIHOST_TRAIN_SPLIT=1 and set TRAIN_TPU_PROCESS_BOUNDS explicitly to experiment." >&2
      exit 1
      ;;
  esac
}

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

parse_csv_ints "$VLLM_WORKERS" vllm_worker_array
parse_csv_ints "$TRAIN_WORKER_IDS" train_worker_array
if (( ${#vllm_worker_array[@]} == 0 )); then
  echo "VLLM_WORKERS must contain at least one worker." >&2
  exit 1
fi
if (( ${#train_worker_array[@]} != TRAIN_NUM_PROCESSES )); then
  echo "TRAIN_WORKER_IDS has ${#train_worker_array[@]} workers but TRAIN_NUM_PROCESSES=${TRAIN_NUM_PROCESSES}." >&2
  exit 1
fi
if (( TRAIN_NUM_PROCESSES > 1 )) && [[ "$ALLOW_EXPERIMENTAL_MULTIHOST_TRAIN_SPLIT" != "1" ]]; then
  echo "Refusing unsupported same-reservation multi-host train split: TRAIN_NUM_PROCESSES=${TRAIN_NUM_PROCESSES}, VLLM_WORKERS=${VLLM_WORKERS}." >&2
  echo "The 6-host and 2-host JAX init probes failed with SLICE_FAILURE_INIT_ERROR while vLLM owned separate workers." >&2
  echo "Use the default single-host train split, put vLLM on a separate TPU job, or set ALLOW_EXPERIMENTAL_MULTIHOST_TRAIN_SPLIT=1 to test explicit topology bounds." >&2
  exit 1
fi

expected_train_workers=()
for ((worker = 0; worker < TRAIN_NUM_PROCESSES; worker++)); do
  expected_train_workers+=("$worker")
done
expected_train_worker_ids="$(csv_join "${expected_train_workers[@]}")"
actual_train_worker_ids="$(csv_join "${train_worker_array[@]}")"
if [[ "$actual_train_worker_ids" != "$expected_train_worker_ids" ]]; then
  echo "This launcher currently requires TRAIN_WORKER_IDS=${expected_train_worker_ids} so JAX process IDs match TPU worker IDs." >&2
  echo "Got TRAIN_WORKER_IDS=${actual_train_worker_ids}." >&2
  exit 1
fi

if [[ -z "$TRAIN_TPU_PROCESS_BOUNDS" ]]; then
  TRAIN_TPU_PROCESS_BOUNDS="$(default_train_process_bounds "$TRAIN_NUM_PROCESSES")"
fi
if [[ -z "$TRAIN_TPU_CHIPS_PER_PROCESS_BOUNDS" ]]; then
  TRAIN_TPU_CHIPS_PER_PROCESS_BOUNDS="2,2,1"
fi
if (( "$(bounds_product "$TRAIN_TPU_PROCESS_BOUNDS")" != TRAIN_NUM_PROCESSES )); then
  echo "TRAIN_TPU_PROCESS_BOUNDS=${TRAIN_TPU_PROCESS_BOUNDS} does not match TRAIN_NUM_PROCESSES=${TRAIN_NUM_PROCESSES}." >&2
  exit 1
fi

declare -A assigned_workers=()
for worker in "${train_worker_array[@]}" "${vllm_worker_array[@]}"; do
  if (( worker < 0 || worker >= ${#endpoint_ips[@]} )); then
    echo "Worker ${worker} is out of range for ${#endpoint_ips[@]} TPU endpoints." >&2
    exit 1
  fi
  if [[ -n "${assigned_workers[$worker]:-}" ]]; then
    echo "Worker ${worker} appears in both train/vLLM assignments or is duplicated." >&2
    exit 1
  fi
  assigned_workers[$worker]=1
done

vllm_urls=()
for worker in "${vllm_worker_array[@]}"; do
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
for worker in "${train_worker_array[@]}"; do
  if [[ -n "$train_addresses" ]]; then
    train_addresses+=","
  fi
  train_addresses+="${endpoint_ips[$worker]}:${TRAIN_TPU_PROCESS_PORT}"
done

vllm_base_url="$(IFS=,; echo "${vllm_urls[*]}")"
echo "Starting Tinker API/backend on train workers ${TRAIN_WORKER_IDS}"
echo "Training TPU topology: TPU_PROCESS_BOUNDS=${TRAIN_TPU_PROCESS_BOUNDS}, TPU_CHIPS_PER_PROCESS_BOUNDS=${TRAIN_TPU_CHIPS_PER_PROCESS_BOUNDS}, TPU_PROCESS_ADDRESSES=${train_addresses}"
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
  TRAIN_TPU_PROCESS_BOUNDS="$TRAIN_TPU_PROCESS_BOUNDS" \
  TRAIN_TPU_CHIPS_PER_PROCESS_BOUNDS="$TRAIN_TPU_CHIPS_PER_PROCESS_BOUNDS" \
  TRAIN_TPU_VISIBLE_CHIPS="$TRAIN_TPU_VISIBLE_CHIPS" \
  TRAIN_TPU_PROCESS_PORT="$TRAIN_TPU_PROCESS_PORT" \
  TRAIN_TPU_PROCESS_ADDRESSES="$train_addresses" \
  INFERENCE_BACKEND=vllm \
  VLLM_BASE_URL="$vllm_base_url" \
  VLLM_MODEL_NAME="$MODEL_NAME" \
  VLLM_MAX_CONCURRENT_REQUESTS="$VLLM_MAX_CONCURRENT_REQUESTS" \
  API_PORT="$TINKER_API_PORT" \
  bash "${script_dir}/start_skyrl_tinker.sh"

echo "Split Qwen3.5-9B Tinker/vLLM launch submitted."
