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
VLLM_WORKERS="${VLLM_WORKERS-$VLLM_WORKER}"
# Comma-separated already-running vLLM URLs. When set, VLLM_WORKERS may be
# empty and START_VLLM must be 0; this is the training-only v6e-32 path.
VLLM_BASE_URL_OVERRIDE="${VLLM_BASE_URL_OVERRIDE:-}"
SYNC_SKYRL="${SYNC_SKYRL:-1}"
SYNC_MODE="${SYNC_MODE:-worktree}"
START_VLLM="${START_VLLM:-1}"
START_TINKER="${START_TINKER:-1}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-4B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_NAME}"
REMOTE_SKYRL_DIR="${REMOTE_SKYRL_DIR:-/home/${REMOTE_USER}/SkyRLTpu}"
REMOTE_HF_HOME="${REMOTE_HF_HOME:-/home/${REMOTE_USER}/.cache/huggingface}"
# Optional shared HF weight cache on GCS, forwarded to start_vllm_tpu.sh (see
# there). Empty = vLLM downloads from HuggingFace as before.
HF_CACHE_GCS="${HF_CACHE_GCS:-}"
REMOTE_CHECKPOINTS="${REMOTE_CHECKPOINTS:-/home/${REMOTE_USER}/gcs/skyrl-checkpoints}"
REMOTE_LORA_BASE="${REMOTE_LORA_BASE:-/home/${REMOTE_USER}/gcs/skyrl-lora-models}"
TINKER_API_KEY="${TINKER_API_KEY:-tml-dummy}"

API_PORT="${API_PORT:-8000}"
# Seconds without a client heartbeat before the engine expires a session and
# unloads its models. This is the ONLY path that reclaims a dead client's LoRA
# adapter slot, so 86400 (the old default) meant every crashed client leaked one
# HBM slot for a full day — with a restart loop that exhausts the adapter pool
# long before the day is up. The tinker SDK heartbeats every 10s from a
# background thread for the life of the client process, so a live-but-idle
# client is never at risk; 1800s is ~180 missed heartbeats of slack.
SESSION_TIMEOUT_SEC="${SESSION_TIMEOUT_SEC:-1800}"
VLLM_PORT="${VLLM_PORT:-8001}"
VLLM_TPU_VERSION="${VLLM_TPU_VERSION:-0.23.0}"
VLLM_MODEL_IMPL_TYPE="${VLLM_MODEL_IMPL_TYPE:-vllm}"
VLLM_TPU_BACKEND_TYPE="${VLLM_TPU_BACKEND_TYPE:-torchax}"
VLLM_DISABLE_SHARDY="${VLLM_DISABLE_SHARDY:-auto}"
VLLM_SKIP_JAX_PRECOMPILE="${VLLM_SKIP_JAX_PRECOMPILE:-0}"
VLLM_XLA_CACHE_PATH="${VLLM_XLA_CACHE_PATH:-/home/${REMOTE_USER}/gcs/vllm-xla-cache}"
VLLM_XLA_CACHE_GCS="${VLLM_XLA_CACHE_GCS:-}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-auto}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-2048}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
VLLM_MAX_LORAS="${VLLM_MAX_LORAS:-8}"
VLLM_MAX_LORA_RANK="${VLLM_MAX_LORA_RANK:-32}"
VLLM_DATA_PARALLEL_SIZE="${VLLM_DATA_PARALLEL_SIZE:-auto}"
VLLM_DATA_PARALLEL_BACKEND="${VLLM_DATA_PARALLEL_BACKEND:-auto}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
VLLM_TPU_PROCESS_BOUNDS="${VLLM_TPU_PROCESS_BOUNDS:-auto}"
VLLM_TPU_CHIPS_PER_PROCESS_BOUNDS="${VLLM_TPU_CHIPS_PER_PROCESS_BOUNDS:-auto}"
VLLM_TPU_PROCESS_PORT="${VLLM_TPU_PROCESS_PORT:-8476}"
VLLM_TPU_VISIBLE_CHIPS="${VLLM_TPU_VISIBLE_CHIPS:-}"
VLLM_RAY_EXECUTOR="${VLLM_RAY_EXECUTOR:-auto}"
VLLM_USE_RAY_V2_EXECUTOR_BACKEND="${VLLM_USE_RAY_V2_EXECUTOR_BACKEND:-1}"
# Independent vLLM servers per serving host (see start_vllm_tpu.sh, where the
# per-engine chip/port isolation lives). 1 = historical behavior. N>1 forces
# the no-ray / DP=1 path (the independent servers ARE the data parallelism),
# splits each host into N engines of VLLM_TP_SIZE chips on HTTP ports
# VLLM_PORT..VLLM_PORT+N-1, and puts every engine URL into the client CSV
# (skyrl/backends/vllm_sampling.py splits on comma at :41, round-robins at
# :116, and broadcasts adapter loads to every server at :167/:210/:218).
VLLM_ENGINES_PER_HOST="${VLLM_ENGINES_PER_HOST:-1}"
# Forwarded to start_vllm_tpu.sh: extra pip specs (shell-quoted, verbatim)
# installed into the serving venv after the tpu-inference fork overlay.
VLLM_EXTRA_PIP_SPECS="${VLLM_EXTRA_PIP_SPECS:-}"

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
# Empty disables HTTP adapter push (falls back to shared-storage publish).
VLLM_LORA_UPLOAD_ENDPOINT="${VLLM_LORA_UPLOAD_ENDPOINT:-/skyrl/v1/upload_lora_adapter}"
VLLM_LORA_LOAD_RETRIES="${VLLM_LORA_LOAD_RETRIES:-3}"
VLLM_LORA_LOAD_RETRY_SLEEP_SEC="${VLLM_LORA_LOAD_RETRY_SLEEP_SEC:-2}"
VLLM_REQUEST_TIMEOUT_SEC="${VLLM_REQUEST_TIMEOUT_SEC:-300}"
# Release the pristine base params once the LoRA template exists (see
# tunix_backend.free_base_state_after_template). Off by default: it makes a
# second LoRA config an error, which single-config RL cells never need.
TUNIX_FREE_BASE_STATE="${TUNIX_FREE_BASE_STATE:-0}"
# Prefix-hash engine routing so phase 2 returns to the engine holding phase-1 KV.
VLLM_ROUTE_BY_PROMPT_PREFIX="${VLLM_ROUTE_BY_PROMPT_PREFIX:-0}"
VLLM_MAX_CONCURRENT_REQUESTS="${VLLM_MAX_CONCURRENT_REQUESTS:-256}"
VLLM_CLIENT_SIDE_ROUND_ROBIN="${VLLM_CLIENT_SIDE_ROUND_ROBIN:-0}"

READY_ATTEMPTS="${READY_ATTEMPTS:-720}"
READY_SLEEP_SEC="${READY_SLEEP_SEC:-5}"

# Training backend for the Tinker engine: "jax" (skyrl/tx) or "tunix".
TINKER_BACKEND="${TINKER_BACKEND:-jax}"
TUNIX_MODEL_SOURCE="${TUNIX_MODEL_SOURCE:-maxtext}"
TUNIX_MAX_TARGET_LENGTH="${TUNIX_MAX_TARGET_LENGTH:-4096}"
# >0 enables token-budget micro-batch packing in the tunix backend.
TUNIX_TRAIN_TOKEN_BUDGET="${TUNIX_TRAIN_TOKEN_BUDGET:-0}"
# >0 enables fused-linear cross-entropy target-logprob computation (project the
# flattened B*T token axis in tiles of this many TOKENS, never forming [B*T,V]).
# Requires maxtext_kwargs.num_vocab_tiling>1 so the decoder returns hidden.
TUNIX_FLCE_TILE_SIZE="${TUNIX_FLCE_TILE_SIZE:-0}"
TUNIX_REPLAY_DIAGNOSTICS="${TUNIX_REPLAY_DIAGNOSTICS:-0}"
TUNIX_MAXTEXT_MODEL_NAME="${TUNIX_MAXTEXT_MODEL_NAME:-}"
# Converted HF->orbax MaxText checkpoints are cached on LOCAL SSD (reading
# orbax through the gcsfuse mount is slow, and writing during conversion to a
# fuse mount flakes). Persistence across spot recreation is handled by the
# GCS mirror below: restored local before the engine loads, self-seeded back
# to GCS the first time it is converted.
TUNIX_MAXTEXT_CKPT_CACHE="${TUNIX_MAXTEXT_CKPT_CACHE:-/home/${REMOTE_USER}/skyrl-maxtext-ckpts-local}"
# Shared GCS mirror of the converted orbax checkpoint. Restored to the LOCAL
# cache above before the tinker engine starts, and self-seeded from local the
# first time (one-time, best-effort, backgrounded). Empty disables both.
TUNIX_MAXTEXT_CKPT_CACHE_GCS="${TUNIX_MAXTEXT_CKPT_CACHE_GCS:-gs://sk7524-tinker-tpu-us-east5/skyrl-maxtext-ckpts}"
# Pip spec for MaxText in the train venv. Override with a fork/commit spec
# (e.g. "maxtext @ git+https://github.com/<user>/maxtext.git@<sha>") when the
# model needs unreleased MaxText support; the api script reinstalls it after
# every `uv sync` (exact sync removes packages not in the lock).
TUNIX_MAXTEXT_PIP_SPEC="${TUNIX_MAXTEXT_PIP_SPEC:-maxtext}"
# Extra MaxText pyconfig overrides as JSON (e.g. remat_policy / mesh knobs).
# Defaulted so `set -u` doesn't abort when the caller omits it.
TUNIX_MAXTEXT_KWARGS="${TUNIX_MAXTEXT_KWARGS:-}"
TUNIX_UNIFORM_SEQ_LEN="${TUNIX_UNIFORM_SEQ_LEN:-0}"
TUNIX_MINIMAL_FB_OUTPUT="${TUNIX_MINIMAL_FB_OUTPUT:-0}"
TUNIX_SEQ_BUCKETS="${TUNIX_SEQ_BUCKETS:-}"
TUNIX_ROW_SHARD="${TUNIX_ROW_SHARD:-0}"
# MaxText writes its persistent compilation cache here (base.yml overrides the
# generic JAX cache env). Every trainer host restores the same seed; only
# process 0 publishes it back to GCS.
TUNIX_JAX_CACHE_LOCAL="${TUNIX_JAX_CACHE_LOCAL:-/home/${REMOTE_USER}/jax_cache}"
TUNIX_JAX_CACHE_GCS="${TUNIX_JAX_CACHE_GCS:-}"
if [[ "$TINKER_BACKEND" == "tunix" ]]; then
  TINKER_ENGINE_EXTRA="tunix"
else
  TINKER_ENGINE_EXTRA="jax"
fi
# Sampling around the engine: the engine loop is single-threaded, so
# concurrent train+sample (pipelined RL) requires the API-side external
# inference path. Default on for tunix.
EXTERNAL_SAMPLING="${EXTERNAL_SAMPLING:-auto}"
if [[ "$EXTERNAL_SAMPLING" == "auto" ]]; then
  if [[ "$TINKER_BACKEND" == "tunix" ]]; then EXTERNAL_SAMPLING=1; else EXTERNAL_SAMPLING=0; fi
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

# shellcheck source=tpu/tpu_ssh_lib.sh
source "${script_dir}/tpu_ssh_lib.sh"

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
vllm_workers=()
if [[ -n "${VLLM_WORKERS//[[:space:]]/}" ]]; then
  mapfile -t vllm_workers < <(parse_worker_list "$VLLM_WORKERS")
elif [[ -z "$VLLM_BASE_URL_OVERRIDE" ]]; then
  echo "VLLM_WORKERS may be empty only when VLLM_BASE_URL_OVERRIDE is set." >&2
  exit 1
fi
train_worker_count="${#train_workers[@]}"
vllm_worker_count="${#vllm_workers[@]}"
train_workers_csv="$(join_csv "${train_workers[@]}")"
vllm_workers_csv="$(join_csv "${vllm_workers[@]}")"
train_coord_worker="${train_workers[0]}"
vllm_coord_worker="${vllm_workers[0]:-$train_coord_worker}"

if [[ "$START_VLLM" == "1" && "$vllm_worker_count" -eq 0 ]]; then
  echo "START_VLLM=1 requires at least one VLLM_WORKERS entry." >&2
  exit 1
fi

if ! [[ "$VLLM_ENGINES_PER_HOST" =~ ^[0-9]+$ ]] || (( VLLM_ENGINES_PER_HOST < 1 )); then
  echo "VLLM_ENGINES_PER_HOST must be a positive integer, got '${VLLM_ENGINES_PER_HOST}'." >&2
  exit 1
fi
if (( VLLM_ENGINES_PER_HOST > 1 )); then
  # Independent per-host engines: no cross-host executor, no vLLM-level DP.
  if [[ "$VLLM_RAY_EXECUTOR" == "auto" ]]; then
    VLLM_RAY_EXECUTOR="0"
  elif [[ "$VLLM_RAY_EXECUTOR" == "1" ]]; then
    echo "VLLM_ENGINES_PER_HOST=${VLLM_ENGINES_PER_HOST} requires the no-ray path (VLLM_RAY_EXECUTOR=0/auto)." >&2
    exit 1
  fi
  if [[ "$VLLM_CLIENT_SIDE_ROUND_ROBIN" != "1" && "$VLLM_CLIENT_SIDE_ROUND_ROBIN" != "true" ]]; then
    echo "VLLM_ENGINES_PER_HOST=${VLLM_ENGINES_PER_HOST}: enabling VLLM_CLIENT_SIDE_ROUND_ROBIN so every engine URL is used."
    VLLM_CLIENT_SIDE_ROUND_ROBIN=1
  fi
fi

if [[ "$VLLM_RAY_EXECUTOR" == "auto" ]]; then
  if (( vllm_worker_count > 1 )); then
    VLLM_RAY_EXECUTOR="1"
  else
    VLLM_RAY_EXECUTOR="0"
  fi
fi

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
if [[ -n "$VLLM_TPU_CHIPS_PER_PROCESS_BOUNDS" ]]; then
  chips_per_vllm_worker="$(product_csv "$VLLM_TPU_CHIPS_PER_PROCESS_BOUNDS")"
else
  chips_per_vllm_worker=1
fi
if [[ "$VLLM_DATA_PARALLEL_SIZE" == "auto" ]]; then
  if [[ "$VLLM_RAY_EXECUTOR" != "0" && "$vllm_worker_count" -gt 1 ]]; then
    VLLM_DATA_PARALLEL_SIZE="$vllm_worker_count"
  else
    VLLM_DATA_PARALLEL_SIZE="1"
  fi
fi
if [[ "$VLLM_DATA_PARALLEL_BACKEND" == "auto" ]]; then
  if [[ "$VLLM_DATA_PARALLEL_SIZE" != "1" && "$VLLM_RAY_EXECUTOR" != "0" ]]; then
    VLLM_DATA_PARALLEL_BACKEND="ray"
  else
    VLLM_DATA_PARALLEL_BACKEND=""
  fi
fi
if [[ "$VLLM_DATA_PARALLEL_SIZE" != "1" && "$VLLM_RAY_EXECUTOR" == "0" ]]; then
  echo "VLLM_DATA_PARALLEL_SIZE=${VLLM_DATA_PARALLEL_SIZE} requires VLLM_RAY_EXECUTOR=1/auto in the colocated launcher." >&2
  exit 1
fi
if [[ "$VLLM_RAY_EXECUTOR" == "0" && "$VLLM_EXTRA_ARGS" == *"--pipeline-parallel-size"* ]]; then
  echo "VLLM_RAY_EXECUTOR=0 launches duplicate per-worker vLLM engines; remove --pipeline-parallel-size from VLLM_EXTRA_ARGS." >&2
  exit 1
fi
if [[ "$FSDP_SIZE" == "auto" ]]; then
  FSDP_SIZE="$train_worker_count"
fi
if [[ "$TINKER_BACKEND" == "tunix" && "$train_worker_count" -gt 1 ]]; then
  if [[ "$TUNIX_ROW_SHARD" == "0" ]]; then
    TUNIX_ROW_SHARD="$FSDP_SIZE"
  elif [[ "$TUNIX_ROW_SHARD" != "$FSDP_SIZE" ]]; then
    echo "TUNIX_ROW_SHARD(${TUNIX_ROW_SHARD}) must equal FSDP_SIZE(${FSDP_SIZE}) for multi-host Tunix." >&2
    exit 1
  fi
fi
if [[ "$VLLM_TP_SIZE" == "auto" ]]; then
  if (( VLLM_ENGINES_PER_HOST > 1 )); then
    if (( chips_per_vllm_worker % VLLM_ENGINES_PER_HOST != 0 )); then
      echo "VLLM_ENGINES_PER_HOST=${VLLM_ENGINES_PER_HOST} does not divide the ${chips_per_vllm_worker} chips per serving host." >&2
      exit 1
    fi
    VLLM_TP_SIZE="$((chips_per_vllm_worker / VLLM_ENGINES_PER_HOST))"
  elif [[ "$VLLM_RAY_EXECUTOR" == "0" || "$VLLM_DATA_PARALLEL_SIZE" != "1" ]]; then
    VLLM_TP_SIZE="$chips_per_vllm_worker"
  else
    VLLM_TP_SIZE="$((vllm_worker_count * chips_per_vllm_worker))"
  fi
fi
if (( VLLM_ENGINES_PER_HOST > 1 )) && (( VLLM_TP_SIZE * VLLM_ENGINES_PER_HOST != chips_per_vllm_worker )); then
  echo "VLLM_TP_SIZE(${VLLM_TP_SIZE}) x VLLM_ENGINES_PER_HOST(${VLLM_ENGINES_PER_HOST}) must equal the ${chips_per_vllm_worker} chips per serving host." >&2
  exit 1
fi

expected_train_devices="$((train_worker_count * $(product_csv "$TRAIN_TPU_CHIPS_PER_PROCESS_BOUNDS")))"
mesh_devices="$((FSDP_SIZE * TP_SIZE))"
if (( mesh_devices != expected_train_devices )); then
  echo "SkyRL mesh mismatch: FSDP_SIZE(${FSDP_SIZE}) * TP_SIZE(${TP_SIZE}) = ${mesh_devices}, but TRAIN_WORKERS=${train_workers_csv} exposes ${expected_train_devices} devices." >&2
  echo "For workers 0,1 on v5p-32, use TP_SIZE=4 FSDP_SIZE=2." >&2
  exit 1
fi

endpoint_ips="$(tpu_vm_internal_ips)"

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
  tpu_vm_external_ip "$1"
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
  # One URL per ENGINE, not per worker: with VLLM_ENGINES_PER_HOST=N every
  # serving host contributes N consecutive ports starting at the base port.
  # At N=1 the output is unchanged from the historical per-worker list.
  local port="$1"
  shift
  local urls=""
  local worker engine
  for worker in "$@"; do
    for ((engine = 0; engine < VLLM_ENGINES_PER_HOST; engine++)); do
      if [[ -n "$urls" ]]; then
        urls+=","
      fi
      urls+="http://$(worker_internal_ip "$worker"):$((port + engine))"
    done
  done
  echo "$urls"
}

train_internal_ip="$(worker_internal_ip "$train_coord_worker")"
vllm_internal_ip="$(worker_internal_ip "$vllm_coord_worker")"
train_external_ip="$(worker_external_ip "$train_coord_worker")"
train_process_addresses="$(process_addresses_for_workers "$TRAIN_TPU_PROCESS_PORT" "${train_workers[@]}")"
vllm_process_addresses="$(process_addresses_for_workers "$VLLM_TPU_PROCESS_PORT" "${vllm_workers[@]}")"
if [[ -n "$VLLM_BASE_URL_OVERRIDE" ]]; then
  vllm_start_process_addresses=""
  vllm_base_url="$VLLM_BASE_URL_OVERRIDE"
elif [[ "$VLLM_RAY_EXECUTOR" == "0" ]]; then
  vllm_start_process_addresses=""
  if [[ "$VLLM_CLIENT_SIDE_ROUND_ROBIN" == "1" || "$VLLM_CLIENT_SIDE_ROUND_ROBIN" == "true" ]]; then
    vllm_base_url="$(base_urls_for_workers "$VLLM_PORT" "${vllm_workers[@]}")"
  else
    vllm_base_url="http://${vllm_internal_ip}:${VLLM_PORT}"
  fi
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
      --exclude='third_party/discover' \
      --exclude='third_party/jobman' \
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
      tpu_vm_scp "$worker" "$archive" "$remote_archive"
      tpu_vm_ssh "$worker" "$remote_extract_cmd"
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
    TPU_INFERENCE_FORK_URL="${TPU_INFERENCE_FORK_URL:-}" \
    TPU_INFERENCE_FORK_REF="${TPU_INFERENCE_FORK_REF:-}" \
    VLLM_TPU_BACKEND_TYPE="$VLLM_TPU_BACKEND_TYPE" \
    VLLM_DISABLE_SHARDY="$VLLM_DISABLE_SHARDY" \
    VLLM_SKIP_JAX_PRECOMPILE="$VLLM_SKIP_JAX_PRECOMPILE" \
    VLLM_XLA_CACHE_PATH="$VLLM_XLA_CACHE_PATH" \
    VLLM_XLA_CACHE_GCS="$VLLM_XLA_CACHE_GCS" \
    VLLM_PORT="$VLLM_PORT" \
    VLLM_TP_SIZE="$VLLM_TP_SIZE" \
    VLLM_MAX_MODEL_LEN="$VLLM_MAX_MODEL_LEN" \
    VLLM_MAX_NUM_SEQS="$VLLM_MAX_NUM_SEQS" \
    VLLM_MAX_LORAS="$VLLM_MAX_LORAS" \
    VLLM_MAX_LORA_RANK="$VLLM_MAX_LORA_RANK" \
    VLLM_DATA_PARALLEL_SIZE="$VLLM_DATA_PARALLEL_SIZE" \
    VLLM_DATA_PARALLEL_BACKEND="$VLLM_DATA_PARALLEL_BACKEND" \
    VLLM_EXTRA_ARGS="$VLLM_EXTRA_ARGS" \
    VLLM_TPU_PROCESS_BOUNDS="$VLLM_TPU_PROCESS_BOUNDS" \
    VLLM_TPU_CHIPS_PER_PROCESS_BOUNDS="$VLLM_TPU_CHIPS_PER_PROCESS_BOUNDS" \
    VLLM_TPU_PROCESS_ADDRESSES="$vllm_start_process_addresses" \
    VLLM_TPU_PROCESS_PORT="$VLLM_TPU_PROCESS_PORT" \
    VLLM_TPU_VISIBLE_CHIPS="$VLLM_TPU_VISIBLE_CHIPS" \
    VLLM_ENGINES_PER_HOST="$VLLM_ENGINES_PER_HOST" \
    VLLM_EXTRA_PIP_SPECS="$VLLM_EXTRA_PIP_SPECS" \
    VLLM_RAY_EXECUTOR="$VLLM_RAY_EXECUTOR" \
    VLLM_USE_RAY_V2_EXECUTOR_BACKEND="$VLLM_USE_RAY_V2_EXECUTOR_BACKEND" \
    REMOTE_HF_HOME="$REMOTE_HF_HOME" \
    HF_CACHE_GCS="$HF_CACHE_GCS" \
    REMOTE_LORA_BASE="$REMOTE_LORA_BASE" \
    "${repo_root}/tpu/start_vllm_tpu.sh"
fi

wait_from_worker() {
  local worker="$1"
  local url="$2"
  local label="$3"
  # Fail fast when the engine is already dead: a vLLM whose EngineCore failed
  # (commonly "No space left on device" from a partial HF-cache shard) never
  # recovers, but the plain poll would still burn the full READY_ATTEMPTS
  # window -- ~60min, i.e. an entire spot slice's median lifetime -- before the
  # attempt fails and the guardian can recycle.
  local remote_cmd="
set -euo pipefail
vlog=\"\$HOME/skyrl-logs/vllm-tpu.log\"
for i in \$(seq 1 '${READY_ATTEMPTS}'); do
  if curl -fsS --max-time 5 '${url}' >/dev/null 2>&1; then
    echo '${label} ready at ${url}'
    exit 0
  fi
  if [ -f \"\$vlog\" ] && grep -qE 'Engine core initialization failed|EngineCore failed to start|No space left on device' \"\$vlog\" 2>/dev/null; then
    echo '${label} FATAL: vLLM engine core died -- not waiting out the poll window' >&2
    grep -aE 'RuntimeError|No space left on device' \"\$vlog\" 2>/dev/null | tail -3 >&2
    exit 1
  fi
  sleep '${READY_SLEEP_SEC}'
done
echo '${label} did not become ready at ${url}' >&2
exit 1
"
  tpu_vm_ssh "$worker" "$remote_cmd"
}

if [[ "$START_VLLM" == "1" || "$START_TINKER" == "1" ]]; then
  for vllm_worker in "${vllm_workers[@]}"; do
    for ((engine = 0; engine < VLLM_ENGINES_PER_HOST; engine++)); do
      engine_label="vLLM worker ${vllm_worker}"
      if (( VLLM_ENGINES_PER_HOST > 1 )); then
        engine_label+=" engine ${engine}"
      fi
      wait_from_worker "$train_coord_worker" "http://$(worker_internal_ip "$vllm_worker"):$((VLLM_PORT + engine))/v1/models" "$engine_label"
    done
  done
fi

backend_config="$(
  python3 - <<PY
import json

backend = "${TINKER_BACKEND}"
train_worker_count = int("${train_worker_count}")
logical_worker_ids = list(range(train_worker_count))

vllm_cfg = {
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
    "vllm_client_side_round_robin": "${VLLM_CLIENT_SIDE_ROUND_ROBIN}".lower() in ("1", "true", "yes", "on"),
    "vllm_route_by_prompt_prefix": "${VLLM_ROUTE_BY_PROMPT_PREFIX}".lower() in ("1", "true", "yes", "on"),
}

if backend == "tunix":
    cfg = {
        "vllm_lora_upload_endpoint": "${VLLM_LORA_UPLOAD_ENDPOINT}",
        "model_source": "${TUNIX_MODEL_SOURCE}",
        "max_lora_rank": int("${MAX_LORA_RANK}"),
        "train_micro_batch_size": int("${TRAIN_MICRO_BATCH_SIZE}"),
        "sample_max_num_sequences": int("${SAMPLE_MAX_NUM_SEQUENCES}"),
        "param_dtype": "bfloat16",
        "maxtext_max_target_length": int("${TUNIX_MAX_TARGET_LENGTH}"),
        "train_token_budget": int("${TUNIX_TRAIN_TOKEN_BUDGET}"),
        "flce_tile_size": int("${TUNIX_FLCE_TILE_SIZE}"),
        "maxtext_ckpt_cache_dir": "${TUNIX_MAXTEXT_CKPT_CACHE}",
        "free_base_state_after_template": "${TUNIX_FREE_BASE_STATE}".lower() in ("1", "true", "yes", "on"),
        **vllm_cfg,
    }
    if "${TUNIX_MAXTEXT_MODEL_NAME}":
        cfg["maxtext_model_name"] = "${TUNIX_MAXTEXT_MODEL_NAME}"
    # Extra MaxText pyconfig overrides (JSON), e.g. remat_policy for activation
    # rematerialization / mesh parallelism knobs. Merged over the defaults.
    _mt_kwargs = json.loads('${TUNIX_MAXTEXT_KWARGS}' or "{}")
    if train_worker_count > 1:
        requested = {
            "ici_tensor_parallelism": int("${TP_SIZE}"),
            "ici_fsdp_parallelism": int("${FSDP_SIZE}"),
        }
        for key, value in requested.items():
            if key in _mt_kwargs and int(_mt_kwargs[key]) != value:
                raise SystemExit(
                    f"Tunix mesh mismatch: {key}={_mt_kwargs[key]} but launcher requests {value}"
                )
            _mt_kwargs[key] = value
        cfg.update(
            {
                "coordinator_address": "${train_internal_ip}:${JAX_COORD_PORT}",
                "num_processes": train_worker_count,
            }
        )
    if _mt_kwargs:
        cfg["maxtext_kwargs"] = _mt_kwargs
else:
    cfg = {
        "train_micro_batch_size": int("${TRAIN_MICRO_BATCH_SIZE}"),
        "sample_max_num_sequences": int("${SAMPLE_MAX_NUM_SEQUENCES}"),
        "tensor_parallel_size": int("${TP_SIZE}"),
        "fully_sharded_data_parallel_size": int("${FSDP_SIZE}"),
        "max_lora_adapters": int("${MAX_LORA_ADAPTERS}"),
        "max_lora_rank": int("${MAX_LORA_RANK}"),
        **vllm_cfg,
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

external_inference_flag=""
if [[ "$EXTERNAL_SAMPLING" == "1" ]]; then
  # Round-robin sampling across every vLLM worker (the client splits this
  # comma-separated list). With one vLLM worker it's just the single URL.
  if [[ -n "$VLLM_BASE_URL_OVERRIDE" ]]; then
    external_inference_urls="$VLLM_BASE_URL_OVERRIDE"
    external_inference_flag="--external-inference-url ${external_inference_urls}"
  elif [[ "$VLLM_CLIENT_SIDE_ROUND_ROBIN" == "1" || "$VLLM_CLIENT_SIDE_ROUND_ROBIN" == "true" ]]; then
    # Same per-engine URL list the backend CSV uses.
    external_inference_urls="$(base_urls_for_workers "$VLLM_PORT" "${vllm_workers[@]}")"
    external_inference_flag="--external-inference-url ${external_inference_urls}"
  else
    external_inference_flag="--external-inference-url http://$(worker_internal_ip "${vllm_workers[0]}"):${VLLM_PORT}"
  fi
fi

if [[ "$START_TINKER" == "1" ]]; then
  cleanup_cmd='mkdir -p ~/skyrl-logs; tmux kill-session -t skyrl-tinker 2>/dev/null || true; tmux list-sessions -F "#{session_name}" 2>/dev/null | awk "/^skyrl-tinker-worker-/ {print}" | xargs -r -n1 tmux kill-session -t; pkill -TERM -u "$USER" -f "[s]kyrl\\.tinker|[s]kyrl\\.backends\\.(jax|rpc)" || true; sleep 5; pkill -KILL -u "$USER" -f "[s]kyrl\\.tinker|[s]kyrl\\.backends\\.(jax|rpc)" || true'
  for worker in "${train_workers[@]}"; do
    tpu_vm_ssh "$worker" "$cleanup_cmd"
  done

  api_script="$tmpdir/start_colocated_skyrl_api.sh"
  cat > "$api_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
export HF_HOME="${REMOTE_HF_HOME}"
export TRANSFORMERS_CACHE="\${HF_HOME}/hub"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TINKER_API_KEY="${TINKER_API_KEY}"
export SKYRL_TRAIN_PROCESS_ID="\${SKYRL_TRAIN_PROCESS_ID:-0}"
# Persist the TRAINER's JAX compiles (fb at the uniform length, optimizer
# graphs). Without this every fresh node re-JITs them from scratch -- 10-20 min
# of every bring-up, on every model, thrown away at each preemption. The vLLM
# side has had a GCS-backed cache for a while; the trainer never did, though the
# muse study scripts (rs_tpu.sh, e2e_tpu.sh) set exactly this. Cache misses are
# free (JAX just compiles), so a stale or absent entry costs nothing.
export JAX_COMPILATION_CACHE_DIR="${TUNIX_JAX_CACHE_LOCAL}"
mkdir -p "\$JAX_COMPILATION_CACHE_DIR"
if [[ -n "${TUNIX_JAX_CACHE_GCS}" ]]; then
  gcloud storage rsync -r "${TUNIX_JAX_CACHE_GCS}" "\$JAX_COMPILATION_CACHE_DIR" >/dev/null 2>&1 \
    && echo "trainer JAX cache restored from ${TUNIX_JAX_CACHE_GCS}" \
    || echo "trainer JAX cache empty/miss on process \$SKYRL_TRAIN_PROCESS_ID"
fi
export TUNIX_UNIFORM_SEQ_LEN="${TUNIX_UNIFORM_SEQ_LEN}"
export TUNIX_MINIMAL_FB_OUTPUT="${TUNIX_MINIMAL_FB_OUTPUT}"
export TUNIX_REPLAY_DIAGNOSTICS="${TUNIX_REPLAY_DIAGNOSTICS}"
# Length-bucket ladder for training microbatches (see tunix_backend._bucket_len).
# Empty => exact per-datum rounding (many XLA shapes). UNIFORM, if set, wins.
export TUNIX_SEQ_BUCKETS="${TUNIX_SEQ_BUCKETS}"
# Row-padding multiple = the MaxText batch-sharding axis. Empty/0 => chip count
# (correct under pure FSDP). Set to the fsdp/data axis when using tensor
# parallelism, e.g. TUNIX_ROW_SHARD=2 alongside
# TUNIX_MAXTEXT_KWARGS='{"ici_fsdp_parallelism":2,"ici_tensor_parallelism":2}'.
export TUNIX_ROW_SHARD="${TUNIX_ROW_SHARD}"
export TPU_PROCESS_BOUNDS="${TRAIN_TPU_PROCESS_BOUNDS}"
export TPU_CHIPS_PER_PROCESS_BOUNDS="${TRAIN_TPU_CHIPS_PER_PROCESS_BOUNDS}"
export TPU_PROCESS_ADDRESSES="${train_process_addresses}"
export TPU_PROCESS_PORT="${TRAIN_TPU_PROCESS_PORT}"
# Task id must be the process index within the train group, not the VM
# metadata worker number, or non-{0..n-1} TRAIN_WORKERS selections fail with
# "Invalid task id specified by 'CLOUD_TPU_TASK_ID'".
export CLOUD_TPU_TASK_ID=0
if [[ -n "${TRAIN_TPU_VISIBLE_CHIPS}" ]]; then
  export TPU_VISIBLE_CHIPS="${TRAIN_TPU_VISIBLE_CHIPS}"
else
  unset TPU_VISIBLE_CHIPS
fi
mkdir -p "${REMOTE_HF_HOME}" "${REMOTE_CHECKPOINTS}" "${REMOTE_LORA_BASE}" "\$HOME/skyrl-logs"
cd "${REMOTE_SKYRL_DIR}"
if [[ "\${SKYRL_TRAIN_SKIP_PROVISION:-0}" != "1" ]]; then
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

if [[ "${TINKER_BACKEND}" == "tunix" ]]; then
  # The root project depends on HF peft, which pulls CUDA PyTorch and several
  # GiB of NVIDIA wheels on Linux. Tunix uses Qwix LoRA, so provision its
  # dedicated TPU-only lock and install SkyRL editable without root deps.
  UV_PROJECT_ENVIRONMENT="${REMOTE_SKYRL_DIR}/.venv" \\
    uv sync --project "${REMOTE_SKYRL_DIR}/tpu/tunix_runtime" --frozen
  uv pip install --python "${REMOTE_SKYRL_DIR}/.venv/bin/python" \\
    --no-deps --editable "${REMOTE_SKYRL_DIR}"
  # MaxText declares no base dependencies. Install the caller-pinned revision
  # into the same environment from \$HOME to avoid root build configuration.
  (cd "\$HOME" && uv pip install --python "${REMOTE_SKYRL_DIR}/.venv/bin/python" \\
      "${TUNIX_MAXTEXT_PIP_SPEC}" aqtp pathwaysutils tokamax tiktoken)
fi

# Converted orbax MaxText checkpoint cache. The engine reads a LOCAL dir
# (TUNIX_MAXTEXT_CKPT_CACHE) because reading orbax through the gcsfuse mount is
# slow and writing during conversion to fuse flakes. Restore the local cache
# from its shared GCS mirror before the engine loads (best effort: a miss just
# means the tunix backend re-converts from HF), then self-seed the mirror the
# first time the local cache is populated -- one-time (guarded on the GCS prefix
# being empty), backgrounded, and best-effort so it never blocks/fails launch.
#
# Scoped to THIS model's subdir only: the backend caches under
# <cache_dir>/<mt_name>/0/items (tunix_backend.py _ensure_maxtext_orbax_checkpoint),
# and mt_name == maxtext_model_name (== TUNIX_MAXTEXT_MODEL_NAME) when set. The
# shared GCS prefix holds several models (>100 GB); pulling the whole prefix
# would overflow the host disk, so we sync only <subdir>. When
# TUNIX_MAXTEXT_MODEL_NAME is empty the backend derives mt_name via tunix naming
# (unknowable in shell), so we skip the GCS restore/seed and let it convert.
if [[ "${TINKER_BACKEND}" == "tunix" && -n "${TUNIX_MAXTEXT_CKPT_CACHE_GCS}" && -n "${TUNIX_MAXTEXT_MODEL_NAME}" ]]; then
  export gsutil_bin="\$(command -v gsutil || echo "\$HOME/google-cloud-sdk/bin/gsutil")"
  mkdir -p "${TUNIX_MAXTEXT_CKPT_CACHE}/${TUNIX_MAXTEXT_MODEL_NAME}"
  # Retry + partial purge, same discipline as the HF-cache restore. An rsync cut
  # short (preemption mid-bring-up is routine on spot) leaves *_.gstmp partials:
  # every filename present, almost no bytes. The trainer then reads them and dies
  # with "DATA_LOSS: Error reading shard entry" from TensorStore -- observed live
  # on muse (132K local against a 40.58 GiB checkpoint). A partial cache is worse
  # than none: absent, the backend just re-converts from HF.
  _ck_dir="${TUNIX_MAXTEXT_CKPT_CACHE}/${TUNIX_MAXTEXT_MODEL_NAME}"
  # tpu/jobman/ensure_orbax_ckpt.sh (prepare hook) OWNS checkpoint acquisition.
  # If it already put something here, do not touch it: two restore paths that
  # each purge-on-incomplete race and delete each other's progress -- observed
  # live as a sawtooth (10.5 GB -> purged -> 7.7 GB -> purged) that never
  # converged. Presence is enough; prepare verified completeness.
  if [ -n "\$(ls -A "\$_ck_dir" 2>/dev/null)" ]; then
    echo "MaxText ckpt present at \$_ck_dir -- prepare hook owns it, skipping inline restore"
  else
  _ck_ok=0
  for _try in 1 2 3; do
    "\$gsutil_bin" -m -q rsync -r "${TUNIX_MAXTEXT_CKPT_CACHE_GCS}/${TUNIX_MAXTEXT_MODEL_NAME}" "\$_ck_dir" 2>/dev/null && _ck_ok=1
    _parts="\$(find "\$_ck_dir" \( -name '*_.gstmp' -o -name '*.gstmp' \) 2>/dev/null | head -1)"
    if [ "\$_ck_ok" = "1" ] && [ -z "\$_parts" ]; then
      echo "restored MaxText ckpt cache from ${TUNIX_MAXTEXT_CKPT_CACHE_GCS}/${TUNIX_MAXTEXT_MODEL_NAME} (attempt \$_try)"
      break
    fi
    echo "MaxText ckpt restore incomplete (attempt \$_try): partial=\${_parts:-none}"
    _ck_ok=0
    sleep 15
  done
  if [ "\$_ck_ok" != "1" ]; then
    _n="\$(find "\$_ck_dir" -mindepth 1 -delete -print 2>/dev/null | wc -l)"
    echo "MaxText ckpt cache restore FAILED after 3 tries; purged \$_n partial path(s) so the backend converts from HF instead of reading corrupt shards"
  fi
  fi
  # Only process 0 may seed a missing shared prefix. Every trainer restores its
  # own local copy above, but concurrent first-writer rsyncs can corrupt an
  # otherwise valid one-time publication.
  if [[ "\${SKYRL_TRAIN_PROCESS_ID}" == "0" ]]; then
  nohup bash -c '
    for _try in \$(seq 1 240); do
      if [ -n "\$(ls -A "${TUNIX_MAXTEXT_CKPT_CACHE}/${TUNIX_MAXTEXT_MODEL_NAME}" 2>/dev/null)" ]; then break; fi
      sleep 15
    done
    if [ -n "\$(ls -A "${TUNIX_MAXTEXT_CKPT_CACHE}/${TUNIX_MAXTEXT_MODEL_NAME}" 2>/dev/null)" ] &&
       [ -z "\$("\$gsutil_bin" ls "${TUNIX_MAXTEXT_CKPT_CACHE_GCS}/${TUNIX_MAXTEXT_MODEL_NAME}/**" 2>/dev/null)" ]; then
      "\$gsutil_bin" -m rsync -r "${TUNIX_MAXTEXT_CKPT_CACHE}/${TUNIX_MAXTEXT_MODEL_NAME}" "${TUNIX_MAXTEXT_CKPT_CACHE_GCS}/${TUNIX_MAXTEXT_MODEL_NAME}" >/dev/null 2>&1 &&
        echo "seeded MaxText ckpt cache to ${TUNIX_MAXTEXT_CKPT_CACHE_GCS}/${TUNIX_MAXTEXT_MODEL_NAME}" || true
    fi
  ' >"\$HOME/skyrl-logs/maxtext-ckpt-seed.log" 2>&1 &
  fi
fi

# FLCE needs MaxText's Transformer.__call__ to surface hidden when
# num_vocab_tiling>1 (the decoder already skips the output head there but the
# wrapper returns the None logits). Re-apply this one-liner after every reinstall
# above. Idempotent; only when FLCE is enabled.
if [[ "${TUNIX_FLCE_TILE_SIZE}" -gt 0 ]]; then
  "${REMOTE_SKYRL_DIR}/.venv/bin/python" - <<'PYPATCH'
import glob, os, sys
for mt in glob.glob(os.path.join(sys.prefix, "lib/python*/site-packages/maxtext/models/models.py")):
    src = open(mt).read()
    if "num_vocab_tiling > 1 and model_mode == MODEL_MODE_TRAIN" in src:
        continue
    old = ('    if self.config.attention == "vllm_rpa":\n'
           '      # In vLLM, logits are computed separately after updating the KV cache.\n'
           '      return hidden_state, kv_caches\n\n'
           '    return logits')
    new = ('    if self.config.attention == "vllm_rpa":\n'
           '      # In vLLM, logits are computed separately after updating the KV cache.\n'
           '      return hidden_state, kv_caches\n\n'
           '    if self.config.num_vocab_tiling > 1 and model_mode == MODEL_MODE_TRAIN:\n'
           '      return hidden_state\n\n'
           '    return logits')
    if old in src:
        open(mt, "w").write(src.replace(old, new))
        print("FLCE-patched", mt, flush=True)
PYPATCH
fi
fi

if [[ "\${SKYRL_TRAIN_SETUP_ONLY:-0}" == "1" ]]; then
  echo "trainer process \$SKYRL_TRAIN_PROCESS_ID provisioning complete"
  exit 0
fi

runner=(uv run --extra tpu --extra tinker --extra "${TINKER_ENGINE_EXTRA}")
if [[ "${TINKER_BACKEND}" == "tunix" ]]; then
  export SKYRL_TINKER_ENGINE_DIRECT_PYTHON=1
  runner=("${REMOTE_SKYRL_DIR}/.venv/bin/python")
fi
exec "\${runner[@]}" -m skyrl.tinker.api \\
  --base-model "${MODEL_NAME}" \\
  --host 0.0.0.0 \\
  --port "${API_PORT}" \\
  --session-timeout-sec "${SESSION_TIMEOUT_SEC}" \\
  --checkpoints-base "${REMOTE_CHECKPOINTS}" \\
  --external-inference-lora-base "${REMOTE_LORA_BASE}" \\
  --backend "${TINKER_BACKEND}" \\
  ${external_inference_flag} \\
  --backend-config '${backend_config}'
EOF
  chmod +x "$api_script"

  # Provision every JAX process from the exact same generated program. Run the
  # expensive uv/MaxText/cache work concurrently: serial setup costs several
  # spot-window minutes per VM and made a four-process v6e-16 unnecessarily
  # fragile. The real API/worker starts below set SKYRL_TRAIN_SKIP_PROVISION=1.
  for worker in "${train_workers[@]}"; do
    tpu_vm_scp "$worker" "$api_script" "~/start_colocated_skyrl_api.sh"
  done
  provision_pids=()
  for ((process_id = 0; process_id < train_worker_count; process_id++)); do
    worker="${train_workers[$process_id]}"
    tpu_vm_ssh "$worker" "SKYRL_TRAIN_PROCESS_ID=${process_id} SKYRL_TRAIN_SETUP_ONLY=1 bash ~/start_colocated_skyrl_api.sh" &
    provision_pids+=("$!")
  done
  provision_failed=0
  for pid in "${provision_pids[@]}"; do
    wait "$pid" || provision_failed=1
  done
  if (( provision_failed )); then
    echo "At least one trainer process failed provisioning; collective start aborted." >&2
    exit 1
  fi

  for ((process_id = 1; process_id < train_worker_count; process_id++)); do
    worker="${train_workers[$process_id]}"
    worker_script="$tmpdir/start_colocated_skyrl_worker_${process_id}.sh"
    cat > "$worker_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
export HF_HOME="${REMOTE_HF_HOME}"
export TRANSFORMERS_CACHE="\${HF_HOME}/hub"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export JAX_COMPILATION_CACHE_DIR="${TUNIX_JAX_CACHE_LOCAL}"
export TUNIX_UNIFORM_SEQ_LEN="${TUNIX_UNIFORM_SEQ_LEN}"
export TUNIX_MINIMAL_FB_OUTPUT="${TUNIX_MINIMAL_FB_OUTPUT}"
export TUNIX_REPLAY_DIAGNOSTICS="${TUNIX_REPLAY_DIAGNOSTICS}"
export TUNIX_SEQ_BUCKETS="${TUNIX_SEQ_BUCKETS}"
export TUNIX_ROW_SHARD="${TUNIX_ROW_SHARD}"
export TPU_PROCESS_BOUNDS="${TRAIN_TPU_PROCESS_BOUNDS}"
export TPU_CHIPS_PER_PROCESS_BOUNDS="${TRAIN_TPU_CHIPS_PER_PROCESS_BOUNDS}"
export TPU_PROCESS_ADDRESSES="${train_process_addresses}"
export TPU_PROCESS_PORT="${TRAIN_TPU_PROCESS_PORT}"
# See the api script: group-relative task id, not the VM worker number.
export CLOUD_TPU_TASK_ID=${process_id}
if [[ -n "${TRAIN_TPU_VISIBLE_CHIPS}" ]]; then
  export TPU_VISIBLE_CHIPS="${TRAIN_TPU_VISIBLE_CHIPS}"
else
  unset TPU_VISIBLE_CHIPS
fi
cd "${REMOTE_SKYRL_DIR}"
runner=(uv run --extra tpu --extra tinker --extra "${TINKER_ENGINE_EXTRA}")
if [[ "${TINKER_BACKEND}" == "tunix" ]]; then
  runner=("${REMOTE_SKYRL_DIR}/.venv/bin/python")
fi
exec "\${runner[@]}" -m skyrl.backends.rpc \\
  --coordinator-address "${train_internal_ip}:${JAX_COORD_PORT}" \\
  --num-processes "${train_worker_count}" \\
  --process-id "${process_id}" \\
  --backend "${TINKER_BACKEND}"
EOF
    chmod +x "$worker_script"
    tpu_vm_scp "$worker" "$worker_script" "~/start_colocated_skyrl_worker_${process_id}.sh"
    tpu_vm_ssh "$worker" "mkdir -p ~/skyrl-logs; tmux new-session -d -c \"\$HOME\" -s skyrl-tinker-worker-${process_id} \"SKYRL_TRAIN_SKIP_PROVISION=1 bash ~/start_colocated_skyrl_worker_${process_id}.sh 2>&1 | tee ~/skyrl-logs/tinker-worker-${process_id}.log\""
  done

  tpu_vm_ssh "$train_coord_worker" 'mkdir -p ~/skyrl-logs; tmux new-session -d -c "$HOME" -s skyrl-tinker "SKYRL_TRAIN_SKIP_PROVISION=1 bash ~/start_colocated_skyrl_api.sh 2>&1 | tee ~/skyrl-logs/tinker-api.log"'

  wait_from_worker "$train_coord_worker" "http://127.0.0.1:${API_PORT}/api/v1/get_server_capabilities" "Tinker API"
fi

echo "Colocated vLLM/Tinker split is up."
echo "Train workers: ${train_workers_csv}; Tinker API: http://127.0.0.1:${API_PORT} on worker ${train_coord_worker}"
echo "Train TPU_PROCESS_BOUNDS=${TRAIN_TPU_PROCESS_BOUNDS}; TRAIN_TPU_PROCESS_ADDRESSES=${train_process_addresses}; mesh fsdp=${FSDP_SIZE}, tp=${TP_SIZE}"
echo "vLLM workers: ${vllm_workers_csv:-external}; vLLM URL from train workers: ${vllm_base_url}; vLLM tp=${VLLM_TP_SIZE}; engines/host=${VLLM_ENGINES_PER_HOST}"
echo "vLLM data parallel: size=${VLLM_DATA_PARALLEL_SIZE}; backend=${VLLM_DATA_PARALLEL_BACKEND:-none}"
echo "vLLM client-side round-robin: ${VLLM_CLIENT_SIDE_ROUND_ROBIN}"
echo "vLLM TPU_PROCESS_BOUNDS=${VLLM_TPU_PROCESS_BOUNDS}; VLLM_TPU_PROCESS_ADDRESSES=${vllm_start_process_addresses}"
echo "Tinker log: gcloud alpha compute tpus tpu-vm ssh ${REMOTE_USER}@${TPU_NAME} --project=${PROJECT} --zone=${ZONE} --worker=${train_coord_worker} --ssh-key-file=${SSH_KEY_FILE} --command 'tail -f ~/skyrl-logs/tinker-api.log'"
if (( vllm_worker_count > 0 )); then
  echo "vLLM log: gcloud alpha compute tpus tpu-vm ssh ${REMOTE_USER}@${TPU_NAME} --project=${PROJECT} --zone=${ZONE} --worker=${vllm_coord_worker} --ssh-key-file=${SSH_KEY_FILE} --command 'tail -f ~/skyrl-logs/vllm-tpu.log'"
fi
if [[ -n "$train_external_ip" ]]; then
  echo "Local tunnel: ssh -i ${SSH_KEY_FILE} -L 127.0.0.1:18025:127.0.0.1:${API_PORT} ${REMOTE_USER}@${train_external_ip}"
fi
