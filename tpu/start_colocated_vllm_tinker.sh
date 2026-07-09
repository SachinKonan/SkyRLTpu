#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-vision-mix}"
ZONE="${ZONE:-us-east5-a}"
TPU_NAME="${TPU_NAME:-sk7524-coloc-vllm-qwen35-4b-v5p32-r1-east5a_spot}"
REMOTE_USER="${REMOTE_USER:-sk7524_princeton_edu}"
SSH_KEY_FILE="${SSH_KEY_FILE:-$HOME/.ssh/jobman_tpu_ed25519}"

TRAIN_WORKER="${TRAIN_WORKER:-0}"
VLLM_WORKER="${VLLM_WORKER:-1}"
TRAIN_WORKERS="${TRAIN_WORKERS:-$TRAIN_WORKER}"
VLLM_WORKERS="${VLLM_WORKERS:-$VLLM_WORKER}"
SYNC_SKYRL="${SYNC_SKYRL:-1}"
SYNC_MODE="${SYNC_MODE:-worktree}"
START_VLLM="${START_VLLM:-1}"
START_TINKER="${START_TINKER:-1}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-4B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_NAME}"
REMOTE_SKYRL_DIR="${REMOTE_SKYRL_DIR:-/home/${REMOTE_USER}/SkyRLTpu}"
REMOTE_HF_HOME="${REMOTE_HF_HOME:-/home/${REMOTE_USER}/.cache/huggingface}"
REMOTE_CHECKPOINTS="${REMOTE_CHECKPOINTS:-/home/${REMOTE_USER}/gcs/skyrl-checkpoints}"
REMOTE_LORA_BASE="${REMOTE_LORA_BASE:-/home/${REMOTE_USER}/gcs/skyrl-lora-models}"
TINKER_API_KEY="${TINKER_API_KEY:-tml-dummy}"

API_PORT="${API_PORT:-8000}"
SESSION_TIMEOUT_SEC="${SESSION_TIMEOUT_SEC:-86400}"
VLLM_PORT="${VLLM_PORT:-8001}"
VLLM_TPU_VERSION="${VLLM_TPU_VERSION:-0.23.0}"
VLLM_MODEL_IMPL_TYPE="${VLLM_MODEL_IMPL_TYPE:-vllm}"
VLLM_TPU_BACKEND_TYPE="${VLLM_TPU_BACKEND_TYPE:-torchax}"
VLLM_DISABLE_SHARDY="${VLLM_DISABLE_SHARDY:-auto}"
VLLM_SKIP_JAX_PRECOMPILE="${VLLM_SKIP_JAX_PRECOMPILE:-0}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-auto}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-2048}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
VLLM_MAX_LORAS="${VLLM_MAX_LORAS:-8}"
VLLM_MAX_LORA_RANK="${VLLM_MAX_LORA_RANK:-32}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
VLLM_TPU_PROCESS_BOUNDS="${VLLM_TPU_PROCESS_BOUNDS:-auto}"
VLLM_TPU_CHIPS_PER_PROCESS_BOUNDS="${VLLM_TPU_CHIPS_PER_PROCESS_BOUNDS:-auto}"
VLLM_TPU_PROCESS_PORT="${VLLM_TPU_PROCESS_PORT:-8476}"
VLLM_TPU_VISIBLE_CHIPS="${VLLM_TPU_VISIBLE_CHIPS:-}"
VLLM_RAY_EXECUTOR="${VLLM_RAY_EXECUTOR:-0}"
VLLM_USE_RAY_V2_EXECUTOR_BACKEND="${VLLM_USE_RAY_V2_EXECUTOR_BACKEND:-1}"

JAX_COORD_PORT="${JAX_COORD_PORT:-7777}"
TRAIN_TPU_PROCESS_BOUNDS="${TRAIN_TPU_PROCESS_BOUNDS:-auto}"
TRAIN_TPU_CHIPS_PER_PROCESS_BOUNDS="${TRAIN_TPU_CHIPS_PER_PROCESS_BOUNDS:-2,2,1}"
TRAIN_TPU_PROCESS_PORT="${TRAIN_TPU_PROCESS_PORT:-8477}"
TRAIN_TPU_VISIBLE_CHIPS="${TRAIN_TPU_VISIBLE_CHIPS:-}"

TP_SIZE="${TP_SIZE:-4}"
FSDP_SIZE="${FSDP_SIZE:-auto}"
TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-1}"
SAMPLE_MAX_NUM_SEQUENCES="${SAMPLE_MAX_NUM_SEQUENCES:-256}"
MAX_LORA_ADAPTERS="${MAX_LORA_ADAPTERS:-8}"
MAX_LORA_RANK="${MAX_LORA_RANK:-32}"
VLLM_LORA_LOAD_ENDPOINT="${VLLM_LORA_LOAD_ENDPOINT:-/v1/load_lora_adapter}"
VLLM_LORA_UNLOAD_ENDPOINT="${VLLM_LORA_UNLOAD_ENDPOINT:-/v1/unload_lora_adapter}"
VLLM_LORA_LOAD_RETRIES="${VLLM_LORA_LOAD_RETRIES:-3}"
VLLM_LORA_LOAD_RETRY_SLEEP_SEC="${VLLM_LORA_LOAD_RETRY_SLEEP_SEC:-2}"
VLLM_REQUEST_TIMEOUT_SEC="${VLLM_REQUEST_TIMEOUT_SEC:-300}"
VLLM_MAX_CONCURRENT_REQUESTS="${VLLM_MAX_CONCURRENT_REQUESTS:-256}"

READY_ATTEMPTS="${READY_ATTEMPTS:-720}"
READY_SLEEP_SEC="${READY_SLEEP_SEC:-5}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

parse_worker_list() {
  python3 - "$1" <<'PY'
import sys

raw = sys.argv[1].strip()
if not raw:
    raise SystemExit("worker list is empty")
workers = []
for part in raw.replace(" ", "").split(","):
    if not part:
        continue
    if "-" in part:
        start_s, end_s = part.split("-", 1)
        start, end = int(start_s), int(end_s)
        if end < start:
            raise SystemExit(f"invalid worker range: {part}")
        workers.extend(range(start, end + 1))
    else:
        workers.append(int(part))
if not workers:
    raise SystemExit("worker list is empty")
if len(set(workers)) != len(workers):
    raise SystemExit(f"worker list contains duplicates: {workers}")
for worker in workers:
    print(worker)
PY
}

join_csv() {
  local IFS=,
  echo "$*"
}

default_process_bounds() {
  local process_count="$1"
  case "$process_count" in
    1) echo "1,1,1" ;;
    2) echo "1,1,2" ;;
    4) echo "1,1,4" ;;
    8) echo "1,2,4" ;;
    *) echo "1,1,${process_count}" ;;
  esac
}

product_csv() {
  python3 - "$1" <<'PY'
import math
import sys

print(math.prod(int(x) for x in sys.argv[1].split(",")))
PY
}

mapfile -t train_workers < <(parse_worker_list "$TRAIN_WORKERS")
mapfile -t vllm_workers < <(parse_worker_list "$VLLM_WORKERS")
train_worker_count="${#train_workers[@]}"
vllm_worker_count="${#vllm_workers[@]}"
train_workers_csv="$(join_csv "${train_workers[@]}")"
vllm_workers_csv="$(join_csv "${vllm_workers[@]}")"
train_coord_worker="${train_workers[0]}"
vllm_coord_worker="${vllm_workers[0]}"

for train_worker in "${train_workers[@]}"; do
  for vllm_worker in "${vllm_workers[@]}"; do
    if [[ "$train_worker" == "$vllm_worker" ]]; then
      echo "TRAIN_WORKERS and VLLM_WORKERS must be disjoint; worker ${train_worker} appears in both." >&2
      exit 1
    fi
  done
done

declare -A seen_workers=()
all_workers=()
for worker in "${train_workers[@]}" "${vllm_workers[@]}"; do
  if [[ -z "${seen_workers[$worker]+x}" ]]; then
    all_workers+=("$worker")
    seen_workers[$worker]=1
  fi
done

if [[ "$TRAIN_TPU_PROCESS_BOUNDS" == "auto" ]]; then
  TRAIN_TPU_PROCESS_BOUNDS="$(default_process_bounds "$train_worker_count")"
fi
if [[ "$VLLM_TPU_PROCESS_BOUNDS" == "auto" ]]; then
  if [[ "$VLLM_RAY_EXECUTOR" == "0" ]]; then
    VLLM_TPU_PROCESS_BOUNDS="1,1,1"
  elif [[ "$VLLM_EXTRA_ARGS" == *"--pipeline-parallel-size"* ]]; then
    VLLM_TPU_PROCESS_BOUNDS=""
  else
    VLLM_TPU_PROCESS_BOUNDS="$(default_process_bounds "$vllm_worker_count")"
  fi
fi
if [[ "$VLLM_TPU_CHIPS_PER_PROCESS_BOUNDS" == "auto" ]]; then
  if [[ "$VLLM_RAY_EXECUTOR" != "0" && "$VLLM_EXTRA_ARGS" == *"--pipeline-parallel-size"* ]]; then
    VLLM_TPU_CHIPS_PER_PROCESS_BOUNDS=""
  else
    VLLM_TPU_CHIPS_PER_PROCESS_BOUNDS="2,2,1"
  fi
fi
if [[ "$VLLM_RAY_EXECUTOR" == "0" && "$VLLM_EXTRA_ARGS" == *"--pipeline-parallel-size"* ]]; then
  echo "VLLM_RAY_EXECUTOR=0 launches duplicate per-worker vLLM engines; remove --pipeline-parallel-size from VLLM_EXTRA_ARGS." >&2
  exit 1
fi
if [[ "$FSDP_SIZE" == "auto" ]]; then
  FSDP_SIZE="$train_worker_count"
fi
if [[ "$VLLM_TP_SIZE" == "auto" ]]; then
  chips_per_vllm_worker="$(product_csv "$VLLM_TPU_CHIPS_PER_PROCESS_BOUNDS")"
  if [[ "$VLLM_RAY_EXECUTOR" == "0" ]]; then
    VLLM_TP_SIZE="$chips_per_vllm_worker"
  else
    VLLM_TP_SIZE="$((vllm_worker_count * chips_per_vllm_worker))"
  fi
fi

expected_train_devices="$((train_worker_count * $(product_csv "$TRAIN_TPU_CHIPS_PER_PROCESS_BOUNDS")))"
mesh_devices="$((FSDP_SIZE * TP_SIZE))"
if (( mesh_devices != expected_train_devices )); then
  echo "SkyRL mesh mismatch: FSDP_SIZE(${FSDP_SIZE}) * TP_SIZE(${TP_SIZE}) = ${mesh_devices}, but TRAIN_WORKERS=${train_workers_csv} exposes ${expected_train_devices} devices." >&2
  echo "For workers 0,1 on v5p-32, use TP_SIZE=4 FSDP_SIZE=2." >&2
  exit 1
fi

endpoint_ips="$(
  gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --format='value(networkEndpoints.ipAddress)'
)"

worker_internal_ip() {
  local worker="$1"
  python3 - "$worker" "$endpoint_ips" <<'PY'
import sys

worker = int(sys.argv[1])
ips = sys.argv[2].replace(";", " ").split()
if worker < 0 or worker >= len(ips):
    raise SystemExit(f"TPU worker {worker} is out of range for {len(ips)} endpoints")
print(ips[worker])
PY
}

worker_external_ip() {
  local worker="$1"
  gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --format="value(networkEndpoints[${worker}].accessConfig.externalIp)"
}

process_addresses_for_workers() {
  local port="$1"
  shift
  local addresses=""
  local worker
  for worker in "$@"; do
    if [[ -n "$addresses" ]]; then
      addresses+=","
    fi
    addresses+="$(worker_internal_ip "$worker"):${port}"
  done
  echo "$addresses"
}

base_urls_for_workers() {
  local port="$1"
  shift
  local urls=""
  local worker
  for worker in "$@"; do
    if [[ -n "$urls" ]]; then
      urls+=","
    fi
    urls+="http://$(worker_internal_ip "$worker"):${port}"
  done
  echo "$urls"
}

train_internal_ip="$(worker_internal_ip "$train_coord_worker")"
vllm_internal_ip="$(worker_internal_ip "$vllm_coord_worker")"
train_external_ip="$(worker_external_ip "$train_coord_worker")"
train_process_addresses="$(process_addresses_for_workers "$TRAIN_TPU_PROCESS_PORT" "${train_workers[@]}")"
vllm_process_addresses="$(process_addresses_for_workers "$VLLM_TPU_PROCESS_PORT" "${vllm_workers[@]}")"
if [[ "$VLLM_RAY_EXECUTOR" == "0" ]]; then
  vllm_start_process_addresses=""
  vllm_base_url="$(base_urls_for_workers "$VLLM_PORT" "${vllm_workers[@]}")"
else
  vllm_start_process_addresses="$vllm_process_addresses"
  vllm_base_url="http://${vllm_internal_ip}:${VLLM_PORT}"
fi

if [[ "$SYNC_SKYRL" == "1" ]]; then
  if [[ "$SYNC_MODE" == "git" ]]; then
    echo "SYNC_MODE=git is intentionally limited to simple full-slice syncs; use SYNC_MODE=worktree for colocated worker subsets." >&2
    exit 1
  elif [[ "$SYNC_MODE" == "worktree" ]]; then
    archive="${tmpdir}/SkyRLTpu-worktree.tar.gz"
    tar -czf "$archive" \
      --exclude='.git' \
      --exclude='.venv' \
      --exclude='.pytest_cache' \
      --exclude='__pycache__' \
      --exclude='*/__pycache__' \
      --exclude='runs' \
      --exclude='benchmark_artifacts' \
      -C "$repo_root" .

    remote_archive="/tmp/SkyRLTpu-worktree-$(date +%s)-$$.tar.gz"
    remote_extract_cmd="
set -euo pipefail
mkdir -p \"\$(dirname '${REMOTE_SKYRL_DIR}')\"
old_dir='${REMOTE_SKYRL_DIR}.old.'\"\$(date +%s).\$\$\"
if [ -e '${REMOTE_SKYRL_DIR}' ]; then
  mv '${REMOTE_SKYRL_DIR}' \"\${old_dir}\"
fi
mkdir -p '${REMOTE_SKYRL_DIR}'
tar -xzf '${remote_archive}' -C '${REMOTE_SKYRL_DIR}'
rm -f '${remote_archive}'
if [ -n \"\${old_dir:-}\" ] && [ -e \"\${old_dir}\" ]; then
  (rm -rf \"\${old_dir}\" >/tmp/skyrltpu-worktree-sync-rm.log 2>&1 || true) &
fi
"
    for worker in "${all_workers[@]}"; do
      echo "Syncing SkyRLTpu worktree to worker ${worker}:${REMOTE_SKYRL_DIR}"
      gcloud alpha compute tpus tpu-vm scp "$archive" "${REMOTE_USER}@${TPU_NAME}:${remote_archive}" \
        --project="$PROJECT" --zone="$ZONE" --worker="$worker" --ssh-key-file="$SSH_KEY_FILE" --quiet
      gcloud alpha compute tpus tpu-vm ssh "${REMOTE_USER}@${TPU_NAME}" \
        --project="$PROJECT" --zone="$ZONE" --worker="$worker" --ssh-key-file="$SSH_KEY_FILE" --quiet \
        --command "$remote_extract_cmd"
    done
  else
    echo "Unsupported SYNC_MODE=${SYNC_MODE}; expected 'worktree' or 'git'." >&2
    exit 1
  fi
fi

if [[ "$START_VLLM" == "1" ]]; then
  PROJECT="$PROJECT" \
    ZONE="$ZONE" \
    TPU_NAME="$TPU_NAME" \
    REMOTE_USER="$REMOTE_USER" \
    SSH_KEY_FILE="$SSH_KEY_FILE" \
    VLLM_WORKERS="$vllm_workers_csv" \
    MODEL_NAME="$MODEL_NAME" \
    SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
    VLLM_TPU_VERSION="$VLLM_TPU_VERSION" \
    VLLM_MODEL_IMPL_TYPE="$VLLM_MODEL_IMPL_TYPE" \
    VLLM_TPU_BACKEND_TYPE="$VLLM_TPU_BACKEND_TYPE" \
    VLLM_DISABLE_SHARDY="$VLLM_DISABLE_SHARDY" \
    VLLM_SKIP_JAX_PRECOMPILE="$VLLM_SKIP_JAX_PRECOMPILE" \
    VLLM_PORT="$VLLM_PORT" \
    VLLM_TP_SIZE="$VLLM_TP_SIZE" \
    VLLM_MAX_MODEL_LEN="$VLLM_MAX_MODEL_LEN" \
    VLLM_MAX_NUM_SEQS="$VLLM_MAX_NUM_SEQS" \
    VLLM_MAX_LORAS="$VLLM_MAX_LORAS" \
    VLLM_MAX_LORA_RANK="$VLLM_MAX_LORA_RANK" \
    VLLM_EXTRA_ARGS="$VLLM_EXTRA_ARGS" \
    VLLM_TPU_PROCESS_BOUNDS="$VLLM_TPU_PROCESS_BOUNDS" \
    VLLM_TPU_CHIPS_PER_PROCESS_BOUNDS="$VLLM_TPU_CHIPS_PER_PROCESS_BOUNDS" \
    VLLM_TPU_PROCESS_ADDRESSES="$vllm_start_process_addresses" \
    VLLM_TPU_PROCESS_PORT="$VLLM_TPU_PROCESS_PORT" \
    VLLM_TPU_VISIBLE_CHIPS="$VLLM_TPU_VISIBLE_CHIPS" \
    VLLM_RAY_EXECUTOR="$VLLM_RAY_EXECUTOR" \
    VLLM_USE_RAY_V2_EXECUTOR_BACKEND="$VLLM_USE_RAY_V2_EXECUTOR_BACKEND" \
    REMOTE_HF_HOME="$REMOTE_HF_HOME" \
    REMOTE_LORA_BASE="$REMOTE_LORA_BASE" \
    "${repo_root}/tpu/start_vllm_tpu.sh"
fi

wait_from_worker() {
  local worker="$1"
  local url="$2"
  local label="$3"
  local remote_cmd="
set -euo pipefail
for i in \$(seq 1 '${READY_ATTEMPTS}'); do
  if curl -fsS --max-time 5 '${url}' >/dev/null 2>&1; then
    echo '${label} ready at ${url}'
    exit 0
  fi
  sleep '${READY_SLEEP_SEC}'
done
echo '${label} did not become ready at ${url}' >&2
exit 1
"
  gcloud alpha compute tpus tpu-vm ssh "${REMOTE_USER}@${TPU_NAME}" \
    --project="$PROJECT" --zone="$ZONE" --worker="$worker" --ssh-key-file="$SSH_KEY_FILE" --quiet \
    --command "$remote_cmd"
}

for vllm_worker in "${vllm_workers[@]}"; do
  wait_from_worker "$train_coord_worker" "http://$(worker_internal_ip "$vllm_worker"):${VLLM_PORT}/v1/models" "vLLM worker ${vllm_worker}"
done

backend_config="$(
  python3 - <<PY
import json

train_worker_count = int("${train_worker_count}")
logical_worker_ids = list(range(train_worker_count))
cfg = {
    "train_micro_batch_size": int("${TRAIN_MICRO_BATCH_SIZE}"),
    "sample_max_num_sequences": int("${SAMPLE_MAX_NUM_SEQUENCES}"),
    "tensor_parallel_size": int("${TP_SIZE}"),
    "fully_sharded_data_parallel_size": int("${FSDP_SIZE}"),
    "max_lora_adapters": int("${MAX_LORA_ADAPTERS}"),
    "max_lora_rank": int("${MAX_LORA_RANK}"),
    "inference_backend": "vllm",
    "vllm_base_url": "${vllm_base_url}",
    "vllm_model_name": "${SERVED_MODEL_NAME}",
    "vllm_lora_base_dir": "${REMOTE_LORA_BASE}",
    "vllm_lora_load_endpoint": "${VLLM_LORA_LOAD_ENDPOINT}",
    "vllm_lora_unload_endpoint": "${VLLM_LORA_UNLOAD_ENDPOINT}",
    "vllm_lora_load_retries": int("${VLLM_LORA_LOAD_RETRIES}"),
    "vllm_lora_load_retry_sleep_sec": float("${VLLM_LORA_LOAD_RETRY_SLEEP_SEC}"),
    "vllm_request_timeout_sec": float("${VLLM_REQUEST_TIMEOUT_SEC}"),
    "vllm_max_concurrent_requests": int("${VLLM_MAX_CONCURRENT_REQUESTS}"),
}
if train_worker_count > 1:
    cfg.update(
        {
            "coordinator_address": "${train_internal_ip}:${JAX_COORD_PORT}",
            "num_processes": train_worker_count,
            "active_worker_ids": logical_worker_ids,
            "mesh_worker_ids": logical_worker_ids,
        }
    )
print(json.dumps(cfg))
PY
)"

if [[ "$START_TINKER" == "1" ]]; then
  cleanup_cmd='mkdir -p ~/skyrl-logs; tmux kill-session -t skyrl-tinker 2>/dev/null || true; tmux list-sessions -F "#{session_name}" 2>/dev/null | awk "/^skyrl-tinker-worker-/ {print}" | xargs -r -n1 tmux kill-session -t; pkill -TERM -u "$USER" -f "[s]kyrl\\.tinker|[s]kyrl\\.backends\\.jax" || true; sleep 5; pkill -KILL -u "$USER" -f "[s]kyrl\\.tinker|[s]kyrl\\.backends\\.jax" || true'
  for worker in "${train_workers[@]}"; do
    gcloud alpha compute tpus tpu-vm ssh "${REMOTE_USER}@${TPU_NAME}" \
      --project="$PROJECT" --zone="$ZONE" --worker="$worker" --ssh-key-file="$SSH_KEY_FILE" --quiet \
      --command "$cleanup_cmd"
  done

  api_script="$tmpdir/start_colocated_skyrl_api.sh"
  cat > "$api_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
export HF_HOME="${REMOTE_HF_HOME}"
export TRANSFORMERS_CACHE="\${HF_HOME}/hub"
export TINKER_API_KEY="${TINKER_API_KEY}"
export TPU_PROCESS_BOUNDS="${TRAIN_TPU_PROCESS_BOUNDS}"
export TPU_CHIPS_PER_PROCESS_BOUNDS="${TRAIN_TPU_CHIPS_PER_PROCESS_BOUNDS}"
export TPU_PROCESS_ADDRESSES="${train_process_addresses}"
export TPU_PROCESS_PORT="${TRAIN_TPU_PROCESS_PORT}"
if [[ -n "${TRAIN_TPU_VISIBLE_CHIPS}" ]]; then
  export TPU_VISIBLE_CHIPS="${TRAIN_TPU_VISIBLE_CHIPS}"
else
  unset TPU_VISIBLE_CHIPS
fi
mkdir -p "${REMOTE_HF_HOME}" "${REMOTE_CHECKPOINTS}" "${REMOTE_LORA_BASE}" "\$HOME/skyrl-logs"
cd "${REMOTE_SKYRL_DIR}"
if [[ -f uv.lock ]]; then
  python3 - <<'PY'
from pathlib import Path

lockfile = Path("uv.lock")
lockfile.write_text(
    lockfile.read_text().replace(
        "https://download-r2.pytorch.org",
        "https://download.pytorch.org",
    )
)
PY
fi

exec uv run --extra tpu --extra tinker --extra jax -m skyrl.tinker.api \\
  --base-model "${MODEL_NAME}" \\
  --host 0.0.0.0 \\
  --port "${API_PORT}" \\
  --session-timeout-sec "${SESSION_TIMEOUT_SEC}" \\
  --checkpoints-base "${REMOTE_CHECKPOINTS}" \\
  --external-inference-lora-base "${REMOTE_LORA_BASE}" \\
  --backend-config '${backend_config}'
EOF
  chmod +x "$api_script"

  gcloud alpha compute tpus tpu-vm scp "$api_script" "${REMOTE_USER}@${TPU_NAME}:~/start_colocated_skyrl_api.sh" \
    --project="$PROJECT" --zone="$ZONE" --worker="$train_coord_worker" --ssh-key-file="$SSH_KEY_FILE" --quiet

  for ((process_id = 1; process_id < train_worker_count; process_id++)); do
    worker="${train_workers[$process_id]}"
    worker_script="$tmpdir/start_colocated_skyrl_worker_${process_id}.sh"
    cat > "$worker_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
export HF_HOME="${REMOTE_HF_HOME}"
export TRANSFORMERS_CACHE="\${HF_HOME}/hub"
export TPU_PROCESS_BOUNDS="${TRAIN_TPU_PROCESS_BOUNDS}"
export TPU_CHIPS_PER_PROCESS_BOUNDS="${TRAIN_TPU_CHIPS_PER_PROCESS_BOUNDS}"
export TPU_PROCESS_ADDRESSES="${train_process_addresses}"
export TPU_PROCESS_PORT="${TRAIN_TPU_PROCESS_PORT}"
if [[ -n "${TRAIN_TPU_VISIBLE_CHIPS}" ]]; then
  export TPU_VISIBLE_CHIPS="${TRAIN_TPU_VISIBLE_CHIPS}"
else
  unset TPU_VISIBLE_CHIPS
fi
cd "${REMOTE_SKYRL_DIR}"
exec uv run --extra tpu --extra tinker --extra jax -m skyrl.backends.jax \\
  --coordinator-address "${train_internal_ip}:${JAX_COORD_PORT}" \\
  --num-processes "${train_worker_count}" \\
  --process-id "${process_id}"
EOF
    chmod +x "$worker_script"
    gcloud alpha compute tpus tpu-vm scp "$worker_script" "${REMOTE_USER}@${TPU_NAME}:~/start_colocated_skyrl_worker_${process_id}.sh" \
      --project="$PROJECT" --zone="$ZONE" --worker="$worker" --ssh-key-file="$SSH_KEY_FILE" --quiet
    gcloud alpha compute tpus tpu-vm ssh "${REMOTE_USER}@${TPU_NAME}" \
      --project="$PROJECT" --zone="$ZONE" --worker="$worker" --ssh-key-file="$SSH_KEY_FILE" --quiet \
      --command "mkdir -p ~/skyrl-logs; tmux new-session -d -c \"\$HOME\" -s skyrl-tinker-worker-${process_id} \"bash ~/start_colocated_skyrl_worker_${process_id}.sh 2>&1 | tee ~/skyrl-logs/tinker-worker-${process_id}.log\""
  done

  gcloud alpha compute tpus tpu-vm ssh "${REMOTE_USER}@${TPU_NAME}" \
    --project="$PROJECT" --zone="$ZONE" --worker="$train_coord_worker" --ssh-key-file="$SSH_KEY_FILE" --quiet \
    --command 'mkdir -p ~/skyrl-logs; tmux new-session -d -c "$HOME" -s skyrl-tinker "bash ~/start_colocated_skyrl_api.sh 2>&1 | tee ~/skyrl-logs/tinker-api.log"'

  wait_from_worker "$train_coord_worker" "http://127.0.0.1:${API_PORT}/api/v1/get_server_capabilities" "Tinker API"
fi

echo "Colocated vLLM/Tinker split is up."
echo "Train workers: ${train_workers_csv}; Tinker API: http://127.0.0.1:${API_PORT} on worker ${train_coord_worker}"
echo "Train TPU_PROCESS_BOUNDS=${TRAIN_TPU_PROCESS_BOUNDS}; TRAIN_TPU_PROCESS_ADDRESSES=${train_process_addresses}; mesh fsdp=${FSDP_SIZE}, tp=${TP_SIZE}"
echo "vLLM workers: ${vllm_workers_csv}; vLLM URL from train workers: ${vllm_base_url}; vLLM tp=${VLLM_TP_SIZE}"
echo "vLLM TPU_PROCESS_BOUNDS=${VLLM_TPU_PROCESS_BOUNDS}; VLLM_TPU_PROCESS_ADDRESSES=${vllm_start_process_addresses}"
echo "Tinker log: gcloud alpha compute tpus tpu-vm ssh ${REMOTE_USER}@${TPU_NAME} --project=${PROJECT} --zone=${ZONE} --worker=${train_coord_worker} --ssh-key-file=${SSH_KEY_FILE} --command 'tail -f ~/skyrl-logs/tinker-api.log'"
echo "vLLM log: gcloud alpha compute tpus tpu-vm ssh ${REMOTE_USER}@${TPU_NAME} --project=${PROJECT} --zone=${ZONE} --worker=${vllm_coord_worker} --ssh-key-file=${SSH_KEY_FILE} --command 'tail -f ~/skyrl-logs/vllm-tpu.log'"
if [[ -n "$train_external_ip" ]]; then
  echo "Local tunnel: ssh -i ${SSH_KEY_FILE} -L 127.0.0.1:18025:127.0.0.1:${API_PORT} ${REMOTE_USER}@${train_external_ip}"
fi
