#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-vision-mix}"
ZONE="${ZONE:-us-east5-a}"
TPU_NAME="${TPU_NAME:-sk7524-vllm-qwen3-4b-v5p8-east5a_spot}"
REMOTE_USER="${REMOTE_USER:-sk7524_princeton_edu}"
SSH_KEY_FILE="${SSH_KEY_FILE:-$HOME/.ssh/jobman_tpu_ed25519}"
VLLM_WORKER="${VLLM_WORKER:-0}"
VLLM_WORKERS="${VLLM_WORKERS:-$VLLM_WORKER}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_NAME}"
VLLM_TPU_VERSION="${VLLM_TPU_VERSION:-0.23.0}"
VLLM_MODEL_IMPL_TYPE="${VLLM_MODEL_IMPL_TYPE:-vllm}"
VLLM_TPU_BACKEND_TYPE="${VLLM_TPU_BACKEND_TYPE:-torchax}"
VLLM_DISABLE_SHARDY="${VLLM_DISABLE_SHARDY:-auto}"
VLLM_SKIP_JAX_PRECOMPILE="${VLLM_SKIP_JAX_PRECOMPILE:-0}"
VLLM_PORT="${VLLM_PORT:-8001}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-1}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-2048}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
VLLM_MAX_LORAS="${VLLM_MAX_LORAS:-8}"
VLLM_MAX_LORA_RANK="${VLLM_MAX_LORA_RANK:-32}"
VLLM_DATA_PARALLEL_SIZE="${VLLM_DATA_PARALLEL_SIZE:-1}"
VLLM_DATA_PARALLEL_BACKEND="${VLLM_DATA_PARALLEL_BACKEND:-}"
VLLM_DATA_PARALLEL_SIZE_LOCAL="${VLLM_DATA_PARALLEL_SIZE_LOCAL:-}"
VLLM_DATA_PARALLEL_START_RANK="${VLLM_DATA_PARALLEL_START_RANK:-}"
VLLM_DATA_PARALLEL_ADDRESS="${VLLM_DATA_PARALLEL_ADDRESS:-}"
VLLM_DATA_PARALLEL_RPC_PORT="${VLLM_DATA_PARALLEL_RPC_PORT:-}"
VLLM_DATA_PARALLEL_HYBRID_LB="${VLLM_DATA_PARALLEL_HYBRID_LB:-0}"
VLLM_API_SERVER_COUNT="${VLLM_API_SERVER_COUNT:-}"
VLLM_HEADLESS="${VLLM_HEADLESS:-0}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
VLLM_TPU_PROCESS_BOUNDS="${VLLM_TPU_PROCESS_BOUNDS:-}"
VLLM_TPU_CHIPS_PER_PROCESS_BOUNDS="${VLLM_TPU_CHIPS_PER_PROCESS_BOUNDS:-}"
VLLM_TPU_PROCESS_ADDRESSES="${VLLM_TPU_PROCESS_ADDRESSES:-}"
VLLM_TPU_PROCESS_PORT="${VLLM_TPU_PROCESS_PORT:-}"
VLLM_TPU_VISIBLE_CHIPS="${VLLM_TPU_VISIBLE_CHIPS:-}"
VLLM_RAY_EXECUTOR="${VLLM_RAY_EXECUTOR:-auto}"
VLLM_USE_RAY_V2_EXECUTOR_BACKEND="${VLLM_USE_RAY_V2_EXECUTOR_BACKEND:-1}"
VLLM_PARALLEL_PREINSTALL="${VLLM_PARALLEL_PREINSTALL:-1}"
VLLM_RAY_PORT="${VLLM_RAY_PORT:-6379}"
VLLM_RAY_DASHBOARD_PORT="${VLLM_RAY_DASHBOARD_PORT:-8265}"
VLLM_VENV="${VLLM_VENV:-/home/${REMOTE_USER}/.venvs/vllm-tpu}"
REMOTE_HF_HOME="${REMOTE_HF_HOME:-/home/${REMOTE_USER}/.cache/huggingface}"
REMOTE_LORA_BASE="${REMOTE_LORA_BASE:-/home/${REMOTE_USER}/gcs/skyrl-lora-models}"
# Persist the JAX/XLA compile cache on the GCS mount so precompiled kernels
# survive spot VM recreation (vLLM defaults to local ~/.cache/vllm/xla_cache).
VLLM_XLA_CACHE_PATH="${VLLM_XLA_CACHE_PATH:-/home/${REMOTE_USER}/gcs/vllm-xla-cache}"
# Optional shared XLA cache on GCS: restored to the local cache path before
# vLLM starts, so the first host to compile a shape warms it for all others
# (a fresh 22528 compile is ~55min; the 4GB same-region restore is ~1min).
# Kept OFF the gcsfuse mount deliberately — writing the cache during compile
# to a fuse mount flaked ("transport endpoint not connected").
VLLM_XLA_CACHE_GCS="${VLLM_XLA_CACHE_GCS:-}"
# Forked tpu-inference with the runtime-LoRA forwarders committed (tracked as
# the third_party/tpu-inference submodule). Overlaid on the vllm-tpu wheel
# install; replaces the old deploy-time apply_vllm_tpu_lora_patch.sh flow.
TPU_INFERENCE_FORK_URL="${TPU_INFERENCE_FORK_URL:-https://github.com/SachinKonan/tpu-inference.git}"
TPU_INFERENCE_FORK_REF="${TPU_INFERENCE_FORK_REF:-skyrl/v0.23.0-lora}"
# 1 = run tpu/vllm_tpu_server.py (adds /skyrl/v1/upload_lora_adapter; adapters
# land on local disk over HTTP). 0 = plain `vllm serve`.
VLLM_UPLOAD_SERVER="${VLLM_UPLOAD_SERVER:-1}"
VLLM_LOCAL_LORA_DIR="${VLLM_LOCAL_LORA_DIR:-/home/${REMOTE_USER}/skyrl-local-loras}"

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

mapfile -t vllm_workers < <(parse_worker_list "$VLLM_WORKERS")
primary_vllm_worker="${vllm_workers[0]}"
vllm_worker_count="${#vllm_workers[@]}"

if [[ "$VLLM_RAY_EXECUTOR" == "auto" ]]; then
  if (( vllm_worker_count > 1 )); then
    VLLM_RAY_EXECUTOR="1"
  else
    VLLM_RAY_EXECUTOR="0"
  fi
fi

split_process_addresses() {
  python3 - "$1" <<'PY'
import sys

raw = sys.argv[1].strip()
if not raw:
    raise SystemExit(0)
for item in raw.split(","):
    item = item.strip()
    if item:
        print(item)
PY
}

mapfile -t vllm_process_addresses < <(split_process_addresses "$VLLM_TPU_PROCESS_ADDRESSES")
if (( ${#vllm_process_addresses[@]} > 0 && ${#vllm_process_addresses[@]} != vllm_worker_count )); then
  echo "VLLM_TPU_PROCESS_ADDRESSES has ${#vllm_process_addresses[@]} entries but VLLM_WORKERS has ${vllm_worker_count} workers." >&2
  exit 1
fi

process_address_ip() {
  local index="$1"
  if (( ${#vllm_process_addresses[@]} > 0 )); then
    echo "${vllm_process_addresses[$index]%%:*}"
  else
    echo ""
  fi
}

ray_head_ip="$(process_address_ip 0)"
ray_head_address="${ray_head_ip}:${VLLM_RAY_PORT}"

bootstrap_script="${tmpdir}/start_vllm_tpu_bootstrap.sh"
runner_script="${tmpdir}/run_vllm_tpu_server.sh"

cat > "$bootstrap_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="\$HOME/.local/bin:\$PATH"
fi
export HF_HOME="${REMOTE_HF_HOME}"
export TRANSFORMERS_CACHE="\$HF_HOME/hub"
export SKIP_JAX_PRECOMPILE="${VLLM_SKIP_JAX_PRECOMPILE}"
if [[ -n "\${VLLM_RELATIVE_WORKER_ID:-}" ]]; then
  export CLOUD_TPU_TASK_ID="\${VLLM_RELATIVE_WORKER_ID}"
fi
mkdir -p "${REMOTE_HF_HOME}" "${REMOTE_LORA_BASE}" "$(dirname "${VLLM_VENV}")" "\$HOME/skyrl-logs"

if [ ! -x "${VLLM_VENV}/bin/vllm" ]; then
  uv venv --python 3.12 "${VLLM_VENV}"
fi

uv pip install --python "${VLLM_VENV}/bin/python" "vllm-tpu==${VLLM_TPU_VERSION}"
# Overlay the forked tpu-inference (runtime LoRA forwarders + Ray env
# allowlist). vllm-tpu is a meta-package depending on tpu-inference, so a
# --no-deps force-reinstall cleanly swaps in the fork at the pinned ref.
uv pip install --python "${VLLM_VENV}/bin/python" --no-deps --force-reinstall \\
  "tpu-inference @ git+${TPU_INFERENCE_FORK_URL}@${TPU_INFERENCE_FORK_REF}"
"${VLLM_VENV}/bin/python" - <<'PY'
from tpu_inference.worker.tpu_worker import TPUWorker

missing = [n for n in ("add_lora", "remove_lora", "list_loras", "pin_lora") if not hasattr(TPUWorker, n)]
assert not missing, f"forked tpu-inference missing LoRA forwarders: {missing}"
print("tpu-inference fork overlay verified (runtime LoRA forwarders present)")
PY

if [[ "\${VLLM_CLEANUP:-1}" == "1" ]]; then
  tmux kill-session -t vllm-tpu 2>/dev/null || true
  pkill -TERM -u "\$USER" -f "[V]LLM::EngineCore|[v]llm serve|[a]pi_server" || true
  "${VLLM_VENV}/bin/ray" stop --force >/tmp/vllm-ray-stop.log 2>&1 || true
  sleep 5
  pkill -KILL -u "\$USER" -f "[V]LLM::EngineCore|[v]llm serve|[a]pi_server" || true
fi

if [[ "\${VLLM_USE_RAY_EXECUTOR:-0}" == "1" ]]; then
  if [[ "\${VLLM_RAY_ROLE:-}" == "head" ]]; then
    "${VLLM_VENV}/bin/ray" start \\
      --head \\
      --node-ip-address="\${VLLM_RAY_NODE_IP}" \\
      --port="${VLLM_RAY_PORT}" \\
      --dashboard-host=0.0.0.0 \\
      --dashboard-port="${VLLM_RAY_DASHBOARD_PORT}" \\
      --num-cpus=200
  elif [[ "\${VLLM_RAY_ROLE:-}" == "worker" ]]; then
    "${VLLM_VENV}/bin/ray" start \\
      --address="\${VLLM_RAY_HEAD_ADDRESS}" \\
      --node-ip-address="\${VLLM_RAY_NODE_IP}" \\
      --num-cpus=200
  fi
fi

if [[ "\${VLLM_START_SERVER:-1}" == "1" ]]; then
  tmux new-session -d -s vllm-tpu "VLLM_RELATIVE_WORKER_ID='\${VLLM_RELATIVE_WORKER_ID:-}' bash \$HOME/run_vllm_tpu_server.sh"
fi
EOF

cat > "$runner_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source "${VLLM_VENV}/bin/activate"
export HF_HOME="${REMOTE_HF_HOME}"
export TRANSFORMERS_CACHE="${REMOTE_HF_HOME}/hub"
export MODEL_IMPL_TYPE="${VLLM_MODEL_IMPL_TYPE}"
export TPU_BACKEND_TYPE="${VLLM_TPU_BACKEND_TYPE}"
export SKIP_JAX_PRECOMPILE="${VLLM_SKIP_JAX_PRECOMPILE}"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
# Resolve unknown adapter names from the shared lora dir (external
# inference path references adapters by name without an explicit load).
export VLLM_PLUGINS="\${VLLM_PLUGINS:-lora_filesystem_resolver}"
if [[ "${VLLM_UPLOAD_SERVER}" == "1" ]]; then
  export VLLM_LORA_RESOLVER_CACHE_DIR="${VLLM_LOCAL_LORA_DIR}"
else
  export VLLM_LORA_RESOLVER_CACHE_DIR="${REMOTE_LORA_BASE}"
fi
# The resolver plugin refuses to register unless the dir already exists
# (checked at CLI-arg parse time, before the upload server's own mkdir).
mkdir -p "\${VLLM_LORA_RESOLVER_CACHE_DIR}"
export VLLM_XLA_CACHE_PATH="${VLLM_XLA_CACHE_PATH}"
mkdir -p "${VLLM_XLA_CACHE_PATH}"
# Restore shared compiled-program cache from GCS (skips cold compile). Best
# effort: a miss or partial just means vLLM recompiles what's absent.
if [[ -n "${VLLM_XLA_CACHE_GCS}" ]]; then
  gsutil -m -q rsync -r "${VLLM_XLA_CACHE_GCS}" "${VLLM_XLA_CACHE_PATH}" 2>/dev/null && echo "restored XLA cache from ${VLLM_XLA_CACHE_GCS}" || echo "XLA cache restore skipped/failed (will compile)"
fi
if [[ -n "\${VLLM_RELATIVE_WORKER_ID:-}" ]]; then
  export CLOUD_TPU_TASK_ID="\${VLLM_RELATIVE_WORKER_ID}"
fi
ray_args=()
if [[ "${VLLM_RAY_EXECUTOR}" == "1" ]]; then
  export TPU_MULTIHOST_BACKEND=ray
  export VLLM_USE_RAY_V2_EXECUTOR_BACKEND="${VLLM_USE_RAY_V2_EXECUTOR_BACKEND}"
  ray_args=(--distributed-executor-backend ray)
fi
dp_args=()
if [[ -n "${VLLM_DATA_PARALLEL_SIZE}" && "${VLLM_DATA_PARALLEL_SIZE}" != "1" ]]; then
  dp_args+=(--data-parallel-size "${VLLM_DATA_PARALLEL_SIZE}")
fi
if [[ -n "${VLLM_DATA_PARALLEL_BACKEND}" ]]; then
  dp_args+=(--data-parallel-backend "${VLLM_DATA_PARALLEL_BACKEND}")
fi
if [[ -n "${VLLM_DATA_PARALLEL_SIZE_LOCAL}" ]]; then
  dp_args+=(--data-parallel-size-local "${VLLM_DATA_PARALLEL_SIZE_LOCAL}")
fi
if [[ -n "${VLLM_DATA_PARALLEL_START_RANK}" ]]; then
  dp_args+=(--data-parallel-start-rank "${VLLM_DATA_PARALLEL_START_RANK}")
fi
if [[ -n "${VLLM_DATA_PARALLEL_ADDRESS}" ]]; then
  dp_args+=(--data-parallel-address "${VLLM_DATA_PARALLEL_ADDRESS}")
fi
if [[ -n "${VLLM_DATA_PARALLEL_RPC_PORT}" ]]; then
  dp_args+=(--data-parallel-rpc-port "${VLLM_DATA_PARALLEL_RPC_PORT}")
fi
if [[ "${VLLM_DATA_PARALLEL_HYBRID_LB}" == "1" || "${VLLM_DATA_PARALLEL_HYBRID_LB}" == "true" ]]; then
  dp_args+=(--data-parallel-hybrid-lb)
fi
if [[ -n "${VLLM_API_SERVER_COUNT}" ]]; then
  dp_args+=(--api-server-count "${VLLM_API_SERVER_COUNT}")
fi
if [[ "${VLLM_HEADLESS}" == "1" || "${VLLM_HEADLESS}" == "true" ]]; then
  dp_args+=(--headless)
fi
read -r -a extra_args <<< "${VLLM_EXTRA_ARGS}"
if [[ -n "${VLLM_TPU_PROCESS_BOUNDS}" ]]; then
  export TPU_PROCESS_BOUNDS="${VLLM_TPU_PROCESS_BOUNDS}"
fi
if [[ -n "${VLLM_TPU_CHIPS_PER_PROCESS_BOUNDS}" ]]; then
  export TPU_CHIPS_PER_PROCESS_BOUNDS="${VLLM_TPU_CHIPS_PER_PROCESS_BOUNDS}"
fi
if [[ -n "${VLLM_TPU_PROCESS_ADDRESSES}" ]]; then
  export TPU_PROCESS_ADDRESSES="${VLLM_TPU_PROCESS_ADDRESSES}"
fi
if [[ -n "${VLLM_TPU_PROCESS_PORT}" ]]; then
  export TPU_PROCESS_PORT="${VLLM_TPU_PROCESS_PORT}"
fi
if [[ -n "${VLLM_TPU_VISIBLE_CHIPS}" ]]; then
  export TPU_VISIBLE_CHIPS="${VLLM_TPU_VISIBLE_CHIPS}"
else
  unset TPU_VISIBLE_CHIPS
fi
if [[ "${VLLM_DISABLE_SHARDY}" == "1" || "${VLLM_DISABLE_SHARDY}" == "true" || \\
      ( "${VLLM_DISABLE_SHARDY}" == "auto" && "${MODEL_NAME}" == *"Qwen3.5-4B"* ) ]]; then
  export JAX_USE_SHARDY_PARTITIONER=false
  export LIBTPU_INIT_ARGS="--xla_use_shardy=false --xla_tpu_scoped_vmem_limit_kib=131072 \${LIBTPU_INIT_ARGS:-}"
fi

if [[ "${VLLM_UPLOAD_SERVER}" == "1" ]]; then
  server_cmd=(python "\$HOME/vllm_tpu_server.py" "${MODEL_NAME}" --skyrl-lora-dir "${VLLM_LOCAL_LORA_DIR}")
else
  server_cmd=(vllm serve "${MODEL_NAME}")
fi
exec "\${server_cmd[@]}" \\
  --served-model-name "${SERVED_MODEL_NAME}" \\
  --host 0.0.0.0 \\
  --port "${VLLM_PORT}" \\
  --tensor-parallel-size "${VLLM_TP_SIZE}" \\
  --max-model-len "${VLLM_MAX_MODEL_LEN}" \\
  --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \\
  --enable-lora \\
  --max-loras "${VLLM_MAX_LORAS}" \\
  --max-lora-rank "${VLLM_MAX_LORA_RANK}" \\
  --download-dir "${REMOTE_HF_HOME}/hub" \\
  "\${ray_args[@]}" \\
  "\${dp_args[@]}" \\
  "\${extra_args[@]}" \\
  2>&1 | tee "\$HOME/skyrl-logs/vllm-tpu.log"
EOF

chmod +x "$bootstrap_script" "$runner_script"

for worker in "${vllm_workers[@]}"; do
  tpu_vm_scp "$worker" "$bootstrap_script" "~/start_vllm_tpu_bootstrap.sh"
  tpu_vm_scp "$worker" "$runner_script" "~/run_vllm_tpu_server.sh"
  tpu_vm_scp "$worker" "${repo_root}/tpu/vllm_tpu_server.py" "~/vllm_tpu_server.py"
done

if [[ "$VLLM_PARALLEL_PREINSTALL" == "1" && "$vllm_worker_count" -gt 1 ]]; then
  echo "Preinstalling vLLM TPU environment on workers ${VLLM_WORKERS} in parallel."
  preinstall_pids=()
  for i in "${!vllm_workers[@]}"; do
    worker="${vllm_workers[$i]}"
    (
      tpu_vm_ssh "$worker" "VLLM_RELATIVE_WORKER_ID=${i} VLLM_USE_RAY_EXECUTOR=0 VLLM_START_SERVER=0 VLLM_CLEANUP=1 bash ~/start_vllm_tpu_bootstrap.sh"
    ) &
    preinstall_pids+=("$!")
  done
  for pid in "${preinstall_pids[@]}"; do
    wait "$pid"
  done
fi

if [[ "$VLLM_RAY_EXECUTOR" == "1" ]]; then
  if [[ -z "$ray_head_ip" ]]; then
    echo "VLLM_RAY_EXECUTOR=1 requires VLLM_TPU_PROCESS_ADDRESSES so the Ray head IP is known." >&2
    exit 1
  fi

  for i in "${!vllm_workers[@]}"; do
    worker="${vllm_workers[$i]}"
    node_ip="$(process_address_ip "$i")"
    if (( i == 0 )); then
      role="head"
    else
      role="worker"
    fi
    tpu_vm_ssh "$worker" "VLLM_RELATIVE_WORKER_ID=${i} VLLM_USE_RAY_EXECUTOR=1 VLLM_RAY_ROLE=${role} VLLM_RAY_NODE_IP=${node_ip} VLLM_RAY_HEAD_ADDRESS=${ray_head_address} VLLM_START_SERVER=0 VLLM_CLEANUP=1 bash ~/start_vllm_tpu_bootstrap.sh"
  done

  tpu_vm_ssh "$primary_vllm_worker" "VLLM_RELATIVE_WORKER_ID=0 VLLM_USE_RAY_EXECUTOR=1 VLLM_START_SERVER=1 VLLM_CLEANUP=0 bash ~/start_vllm_tpu_bootstrap.sh"
else
  for i in "${!vllm_workers[@]}"; do
    worker="${vllm_workers[$i]}"
    tpu_vm_ssh "$worker" "VLLM_RELATIVE_WORKER_ID=0 VLLM_USE_RAY_EXECUTOR=0 bash ~/start_vllm_tpu_bootstrap.sh"
  done
fi

echo "vLLM TPU start command submitted on workers ${VLLM_WORKERS}."
echo "Log: gcloud alpha compute tpus tpu-vm ssh ${REMOTE_USER}@${TPU_NAME} --project=${PROJECT} --zone=${ZONE} --worker=${primary_vllm_worker} --ssh-key-file=${SSH_KEY_FILE} --command 'tail -f ~/skyrl-logs/vllm-tpu.log'"
echo "URL from worker ${primary_vllm_worker}: http://localhost:${VLLM_PORT}"
