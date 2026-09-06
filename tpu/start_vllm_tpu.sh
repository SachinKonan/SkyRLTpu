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
# Serving-venv transformers, per model. EMPTY = do not pin: vllm-tpu 0.23.0
# already pulls the released 5.15.0, which is the first version shipping
# muse_glimmer's modeling/config code -- the checkpoint has no .py and there is
# no trust_remote_code path, so downgrading here yields a transformers that
# cannot parse the architecture. gemma-4 is the opposite case: 5.15.0 raises
# AmbiguousGlobalPerLayerAttributeError on its per-layer head_dim, so it must be
# held at 5.8.0. The repo-wide <=5.8.0 override (there for Megatron) is NOT
# touched by any of this; the train side needs no HF muse code (MaxText carries
# its own, and 5.8.0 tokenizes muse correctly).
VLLM_TRANSFORMERS_VERSION="${VLLM_TRANSFORMERS_VERSION-5.8.0}"
VLLM_MODEL_IMPL_TYPE="${VLLM_MODEL_IMPL_TYPE:-vllm}"
VLLM_TPU_BACKEND_TYPE="${VLLM_TPU_BACKEND_TYPE:-torchax}"
VLLM_DISABLE_SHARDY="${VLLM_DISABLE_SHARDY:-auto}"
VLLM_SKIP_JAX_PRECOMPILE="${VLLM_SKIP_JAX_PRECOMPILE:-0}"
VLLM_PORT="${VLLM_PORT:-8001}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-1}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-2048}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
VLLM_ENABLE_LORA="${VLLM_ENABLE_LORA:-1}"
VLLM_USE_BATCHED_RPA_KERNEL="${VLLM_USE_BATCHED_RPA_KERNEL:-0}"
VLLM_USE_JAX_RAGGED_CONV1D="${VLLM_USE_JAX_RAGGED_CONV1D:-0}"
VLLM_CUSTOM_NUM_TOKENS_BUCKETS="${VLLM_CUSTOM_NUM_TOKENS_BUCKETS:-}"
VLLM_SERIALIZE_MODEL_AND_SAMPLING="${VLLM_SERIALIZE_MODEL_AND_SAMPLING:-0}"
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
VLLM_LIMIT_MM_PER_PROMPT="${VLLM_LIMIT_MM_PER_PROMPT:-}"
if [[ "$VLLM_EXTRA_ARGS" == *"--no-enable-prefix-caching"* ]]; then
  echo "prefix caching is required; remove --no-enable-prefix-caching from VLLM_EXTRA_ARGS" >&2
  exit 2
fi
# Independent vLLM servers per serving host. 1 = one engine on the whole
# host, exactly the historical behavior (the generated remote scripts are
# byte-identical). N>1 = N fully independent servers per worker: engine e
# serves HTTP on VLLM_PORT+e, sees only the chips
# [e*VLLM_TP_SIZE, (e+1)*VLLM_TP_SIZE) via TPU_VISIBLE_CHIPS, and owns a
# distinct libtpu coordination port (base + first chip); the XLA compile
# cache is shared between siblings ON PURPOSE so the second engine boots off
# the first one's compile (79s measured). Pattern hardware-validated for
# Muse-Glimmer-30B 2xTP=2 on v5p in tpu/muse_glimmer/vllm_tp4_bench_tpu.sh
# arm B + TP-BENCHMARK-22K.md. Independent servers ARE the data parallelism:
# this requires the no-ray single-host-server path and DP size 1.
VLLM_ENGINES_PER_HOST="${VLLM_ENGINES_PER_HOST:-1}"
# Extra pip specs installed into the serving venv with --no-deps
# --force-reinstall AFTER the tpu-inference fork overlay, in order and
# verbatim. The value must be SHELL-QUOTED where a spec contains spaces,
# because the quoting is preserved into the generated remote install line.
# Needed when the served model requires an unreleased transformers, e.g.
# muse_glimmer (see tpu/muse_glimmer/vllm_tp4_bench_tpu.sh). Empty = no
# change to the historical install set.
VLLM_EXTRA_PIP_SPECS="${VLLM_EXTRA_PIP_SPECS:-}"
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
VLLM_RAY_TEMP_DIR="${VLLM_RAY_TEMP_DIR:-/tmp/ray_tpuswarm_vllm}"
VLLM_VENV="${VLLM_VENV:-/home/${REMOTE_USER}/.venvs/vllm-tpu}"
REMOTE_HF_HOME="${REMOTE_HF_HOME:-/home/${REMOTE_USER}/.cache/huggingface}"
# Scope cache restore to THIS model's dir: the shared prefix holds several
# models (qwen 4G + muse 55G); pulling everything doubles restore time and can
# fill the boot disk.
HF_MODEL_DIR="models--${MODEL_NAME//\//--}"
# Optional shared HF weight cache on GCS: restored to the local HF hub dir
# (REMOTE_HF_HOME/hub, vLLM's --download-dir) before serve so vLLM finds the
# weights already on local SSD instead of re-downloading from HuggingFace. The
# first host to download self-seeds it (one-time, best-effort, backgrounded).
# Kept OFF the gcsfuse mount deliberately (matches VLLM_XLA_CACHE_GCS): the
# engine only ever reads a LOCAL path; GCS is purely the seed/restore source.
HF_CACHE_GCS="${HF_CACHE_GCS:-}"
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

if ! [[ "$VLLM_ENGINES_PER_HOST" =~ ^[0-9]+$ ]] || (( VLLM_ENGINES_PER_HOST < 1 )); then
  echo "VLLM_ENGINES_PER_HOST must be a positive integer, got '${VLLM_ENGINES_PER_HOST}'." >&2
  exit 1
fi
if (( VLLM_ENGINES_PER_HOST > 1 )) && [[ "$VLLM_RAY_EXECUTOR" == "auto" ]]; then
  # Per-host independent engines never coordinate across hosts.
  VLLM_RAY_EXECUTOR="0"
fi

if [[ "$VLLM_RAY_EXECUTOR" == "auto" ]]; then
  if (( vllm_worker_count > 1 )); then
    VLLM_RAY_EXECUTOR="1"
  else
    VLLM_RAY_EXECUTOR="0"
  fi
fi

if (( VLLM_ENGINES_PER_HOST > 1 )); then
  if [[ "$VLLM_RAY_EXECUTOR" == "1" ]]; then
    echo "VLLM_ENGINES_PER_HOST=${VLLM_ENGINES_PER_HOST} starts independent per-host servers and requires the no-ray path (VLLM_RAY_EXECUTOR=0/auto)." >&2
    exit 1
  fi
  if [[ -n "$VLLM_DATA_PARALLEL_SIZE" && "$VLLM_DATA_PARALLEL_SIZE" != "1" ]]; then
    echo "VLLM_ENGINES_PER_HOST=${VLLM_ENGINES_PER_HOST} requires VLLM_DATA_PARALLEL_SIZE=1: the independent servers ARE the data parallelism." >&2
    exit 1
  fi
  if [[ -n "$VLLM_TPU_PROCESS_ADDRESSES" ]]; then
    echo "VLLM_ENGINES_PER_HOST>1 is single-host-per-engine only; VLLM_TPU_PROCESS_ADDRESSES must be empty." >&2
    exit 1
  fi
  if [[ -n "$VLLM_TPU_VISIBLE_CHIPS" ]]; then
    echo "VLLM_ENGINES_PER_HOST>1 computes TPU_VISIBLE_CHIPS per engine from VLLM_TP_SIZE; leave VLLM_TPU_VISIBLE_CHIPS empty." >&2
    exit 1
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

# Multi-engine interpolation blocks. All four render EMPTY at
# VLLM_ENGINES_PER_HOST=1 and are appended to the tail of existing generated
# lines, so the single-engine remote scripts stay byte-identical to the
# historical ones. The blocks land inside unquoted-EOF heredocs below: every
# dollar that must survive to remote-eval time is backslash-escaped here, and
# the generated comments deliberately contain no quotes, no backticks and no
# command substitution.
extra_engine_cleanup=""
extra_engine_start=""
engine_env_block=""
extra_pip_block=""
runner_http_port="$VLLM_PORT"
runner_log_name="vllm-tpu.log"
if (( VLLM_ENGINES_PER_HOST > 1 )); then
  for ((engine = 1; engine < VLLM_ENGINES_PER_HOST; engine++)); do
    extra_engine_cleanup+=$'\n'"  tmux kill-session -t =vllm-tpu-e${engine} 2>/dev/null || true"
    extra_engine_start+=$'\n'"  tmux new-session -d -s vllm-tpu-e${engine} \"VLLM_RELATIVE_WORKER_ID='\${VLLM_RELATIVE_WORKER_ID:-}' VLLM_ENGINE_INDEX=${engine} bash \$HOME/run_vllm_tpu_server.sh\""
  done
  engine_coord_base="${VLLM_TPU_PROCESS_PORT:-8476}"
  engine_env_block+=$'\n'"# Per-engine isolation, VLLM_ENGINES_PER_HOST=${VLLM_ENGINES_PER_HOST}: each engine sees only"
  engine_env_block+=$'\n'"# its own ${VLLM_TP_SIZE} chips, owns a distinct libtpu coordination port, and"
  engine_env_block+=$'\n'"# shares the XLA cache with its siblings on purpose. Hardware-validated"
  engine_env_block+=$'\n'"# pattern: tpu/muse_glimmer/vllm_tp4_bench_tpu.sh arm B."
  engine_env_block+=$'\n'"engine_index=\"\${VLLM_ENGINE_INDEX:-0}\""
  engine_env_block+=$'\n'"engine_port=\$(( ${VLLM_PORT} + engine_index ))"
  engine_env_block+=$'\n'"engine_first_chip=\$(( engine_index * ${VLLM_TP_SIZE} ))"
  engine_env_block+=$'\n'"engine_chips=\"\${engine_first_chip}\""
  engine_env_block+=$'\n'"engine_chip_step=1"
  engine_env_block+=$'\n'"while [ \"\${engine_chip_step}\" -lt ${VLLM_TP_SIZE} ]; do"
  engine_env_block+=$'\n'"  engine_chips=\"\${engine_chips},\$(( engine_first_chip + engine_chip_step ))\""
  engine_env_block+=$'\n'"  engine_chip_step=\$(( engine_chip_step + 1 ))"
  engine_env_block+=$'\n'"done"
  engine_env_block+=$'\n'"engine_log_suffix=\"\""
  engine_env_block+=$'\n'"if [ \"\${engine_index}\" != \"0\" ]; then engine_log_suffix=\"-e\${engine_index}\"; fi"
  engine_env_block+=$'\n'"export TPU_VISIBLE_CHIPS=\"\${engine_chips}\""
  engine_env_block+=$'\n'"export TPU_CHIPS_PER_PROCESS_BOUNDS=\"1,${VLLM_TP_SIZE},1\""
  engine_env_block+=$'\n'"export TPU_PROCESS_BOUNDS=\"1,1,1\""
  engine_env_block+=$'\n'"export TPU_PROCESS_PORT=\$(( ${engine_coord_base} + engine_first_chip ))"
  engine_env_block+=$'\n'"export TPU_PROCESS_ADDRESSES=\"localhost:\${TPU_PROCESS_PORT}\""
  engine_env_block+=$'\n'"export CLOUD_TPU_TASK_ID=0"
  engine_env_block+=$'\n'"export JAX_COMPILATION_CACHE_DIR=\"${VLLM_XLA_CACHE_PATH}\""
  runner_http_port='${engine_port}'
  runner_log_name='vllm-tpu${engine_log_suffix}.log'
fi
if [[ -n "$VLLM_EXTRA_PIP_SPECS" ]]; then
  # UV_NO_CONFIG so no uv config on the host can re-resolve these; --no-deps
  # --force-reinstall and last position mirror the proven muse venv recipe.
  extra_pip_block+=$'\n'"UV_NO_CONFIG=1 uv pip install --python \"${VLLM_VENV}/bin/python\" --no-deps --force-reinstall ${VLLM_EXTRA_PIP_SPECS}"
fi

cat > "$bootstrap_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export TPUSWARM_BUNDLE_ID="${TPUSWARM_BUNDLE_ID:-}"
export PATH="\$HOME/.local/bin:\$PATH"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="\$HOME/.local/bin:\$PATH"
fi
export HF_HOME="${REMOTE_HF_HOME}"
export TRANSFORMERS_CACHE="\$HF_HOME/hub"
export SKIP_JAX_PRECOMPILE="${VLLM_SKIP_JAX_PRECOMPILE}"
export USE_BATCHED_RPA_KERNEL="${VLLM_USE_BATCHED_RPA_KERNEL}"
export USE_JAX_RAGGED_CONV1D="${VLLM_USE_JAX_RAGGED_CONV1D}"
export CUSTOM_NUM_TOKENS_BUCKETS="${VLLM_CUSTOM_NUM_TOKENS_BUCKETS}"
export SERIALIZE_MODEL_AND_SAMPLING="${VLLM_SERIALIZE_MODEL_AND_SAMPLING}"
if [[ -n "\${VLLM_RELATIVE_WORKER_ID:-}" ]]; then
  export CLOUD_TPU_TASK_ID="\${VLLM_RELATIVE_WORKER_ID}"
fi
mkdir -p "${REMOTE_HF_HOME}" "${REMOTE_LORA_BASE}" "$(dirname "${VLLM_VENV}")" "\$HOME/skyrl-logs"

if [ ! -x "${VLLM_VENV}/bin/vllm" ]; then
  uv venv --python 3.12 "${VLLM_VENV}"
fi

uv pip install --python "${VLLM_VENV}/bin/python" "vllm-tpu==${VLLM_TPU_VERSION}"
# Pin transformers to the ONE version both sides accept: tpu-inference requires
# >=5.8.0, and gemma-4's heterogeneous config (per-layer head_dim) breaks on
# newer releases -- 5.15.0 raised AmbiguousGlobalPerLayerAttributeError at model
# load and killed every gemma vLLM worker (verified live 2026-08-15; 5.8.0 loads
# Gemma4Config fine). vllm-tpu leaves transformers unpinned, so a fresh venv
# silently drifts to whatever is latest.
if [[ -n "${VLLM_TRANSFORMERS_VERSION}" ]]; then
  uv pip install --python "${VLLM_VENV}/bin/python" "transformers==${VLLM_TRANSFORMERS_VERSION}"
fi
# Overlay the forked tpu-inference (runtime LoRA forwarders + Ray env
# allowlist). vllm-tpu is a meta-package depending on tpu-inference, so a
# --no-deps force-reinstall cleanly swaps in the fork at the pinned ref.
if [[ -f "${REMOTE_SKYRL_DIR}/third_party/tpu-inference/pyproject.toml" ]]; then
  uv pip install --python "${VLLM_VENV}/bin/python" --no-deps --force-reinstall \\
    "${REMOTE_SKYRL_DIR}/third_party/tpu-inference"${extra_pip_block}
else
  uv pip install --python "${VLLM_VENV}/bin/python" --no-deps --force-reinstall \\
    "tpu-inference @ git+${TPU_INFERENCE_FORK_URL}@${TPU_INFERENCE_FORK_REF}"${extra_pip_block}
fi
"${VLLM_VENV}/bin/python" - <<'PY'
from tpu_inference.worker.tpu_worker import TPUWorker

missing = [n for n in ("add_lora", "remove_lora", "list_loras", "pin_lora") if not hasattr(TPUWorker, n)]
assert not missing, f"forked tpu-inference missing LoRA forwarders: {missing}"
print("tpu-inference fork overlay verified (runtime LoRA forwarders present)")
PY

if [[ "\${VLLM_CLEANUP:-1}" == "1" ]]; then
  tmux kill-session -t =vllm-tpu 2>/dev/null || true${extra_engine_cleanup}
  pkill -TERM -u "\$USER" -f "[V]LLM::EngineCore|[v]llm serve|[a]pi_server" || true
  if [[ "\${VLLM_USE_RAY_EXECUTOR:-0}" == "1" ]]; then
    RAY_BIN="${VLLM_VENV}/bin/ray" \\
      VLLM_RAY_ADDRESS="\${VLLM_RAY_HEAD_ADDRESS}" \\
      VLLM_RAY_TEMP_DIR="${VLLM_RAY_TEMP_DIR}" \\
      bash "\$HOME/vllm_ray.sh" stop
  fi
  sleep 5
  pkill -KILL -u "\$USER" -f "[V]LLM::EngineCore|[v]llm serve|[a]pi_server" || true
fi

if [[ "\${VLLM_USE_RAY_EXECUTOR:-0}" == "1" ]]; then
  RAY_BIN="${VLLM_VENV}/bin/ray" \\
    VLLM_RAY_ADDRESS="\${VLLM_RAY_HEAD_ADDRESS}" \\
    VLLM_RAY_NODE_IP="\${VLLM_RAY_NODE_IP}" \\
    VLLM_RAY_TEMP_DIR="${VLLM_RAY_TEMP_DIR}" \\
    bash "\$HOME/vllm_ray.sh" "\${VLLM_RAY_ROLE}"
fi

if [[ "\${VLLM_START_SERVER:-1}" == "1" ]]; then
  tmux new-session -d -s vllm-tpu "VLLM_RELATIVE_WORKER_ID='\${VLLM_RELATIVE_WORKER_ID:-}' bash \$HOME/run_vllm_tpu_server.sh"${extra_engine_start}
fi
EOF

printf -v vllm_limit_mm_per_prompt_q '%q' "$VLLM_LIMIT_MM_PER_PROMPT"
cat > "$runner_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source "${VLLM_VENV}/bin/activate"
# The per-engine log suffix is needed by the log setup that follows, which
# runs BEFORE the multi-engine env block further down. Under set -u the
# VLLM_ENGINES_PER_HOST>1 runner (muse, 2xTP=2) otherwise dies on its fourth
# line with "engine_log_suffix: unbound variable" -- before any log exists --
# and the head waits out its whole readiness window (jobs 241/254, 2026-09-05).
engine_log_suffix=""
if [ "\${VLLM_ENGINE_INDEX:-0}" != "0" ]; then engine_log_suffix="-e\${VLLM_ENGINE_INDEX}"; fi
runner_log_path="\$HOME/skyrl-logs/${runner_log_name}"
runner_status_path="\${runner_log_path%.log}.exits.log"
runner_history_dir="\$HOME/skyrl-logs/vllm-history"
mkdir -p "\$runner_history_dir"
if [[ -s "\$runner_log_path" ]]; then
  runner_started_at="\$(date -u +%Y%m%dT%H%M%SZ)"
  mv "\$runner_log_path" \
    "\$runner_history_dir/\$(basename "\$runner_log_path").\${runner_started_at}.\$\$.log"
fi
mapfile -t old_runner_logs < <(
  find "\$runner_history_dir" -maxdepth 1 -type f \
    -name "\$(basename "\$runner_log_path").*.log" -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | awk 'NR > 8 {sub(/^[^ ]+ /, ""); print}'
)
if (( \${#old_runner_logs[@]} > 0 )); then
  rm -f -- "\${old_runner_logs[@]}"
fi
printf 'start=%s pid=%s worker=%s engine=%s bundle=%s\n' \
  "\$(date -u +%Y-%m-%dT%H:%M:%SZ)" "\$\$" \
  "\${VLLM_RELATIVE_WORKER_ID:-unknown}" "\${VLLM_ENGINE_INDEX:-0}" \
  "\${TPUSWARM_BUNDLE_ID:-unversioned}" >> "\$runner_status_path"
exec > >(tee "\$runner_log_path") 2>&1
echo "vLLM runner started; status=\$runner_status_path history=\$runner_history_dir"
export HF_HOME="${REMOTE_HF_HOME}"
export TRANSFORMERS_CACHE="${REMOTE_HF_HOME}/hub"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export MODEL_IMPL_TYPE="${VLLM_MODEL_IMPL_TYPE}"
export TPU_BACKEND_TYPE="${VLLM_TPU_BACKEND_TYPE}"
export SKIP_JAX_PRECOMPILE="${VLLM_SKIP_JAX_PRECOMPILE}"
export USE_BATCHED_RPA_KERNEL="${VLLM_USE_BATCHED_RPA_KERNEL}"
export USE_JAX_RAGGED_CONV1D="${VLLM_USE_JAX_RAGGED_CONV1D}"
export CUSTOM_NUM_TOKENS_BUCKETS="${VLLM_CUSTOM_NUM_TOKENS_BUCKETS}"
export SERIALIZE_MODEL_AND_SAMPLING="${VLLM_SERIALIZE_MODEL_AND_SAMPLING}"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
# Cloud SDK 428's parallel/sliced downloads have repeatedly stalled ext4
# writeback or produced hash mismatches on these TPU VM boot disks. Keep cache
# restores resumable but single-stream; they are one-time worker warmups.
export CLOUDSDK_STORAGE_SLICED_OBJECT_DOWNLOAD_THRESHOLD=0
export CLOUDSDK_STORAGE_PROCESS_COUNT=1
export CLOUDSDK_STORAGE_THREAD_COUNT=1
# VLLM_PLUGINS is an ALLOW-LIST: unset loads every installed general plugin,
# set loads ONLY the named ones. Models that exist solely in the
# tpu-inference fork (muse) NEED the fork's vllm.general_plugins entry point
# (tpu_inference.layers.vllm:register_layers) -- it is what injects the OOT
# architecture into vLLM's ModelRegistry. Pinning the list to the resolver
# excluded it, and vLLM silently served the generic transformers fallback,
# which passes every health check and kills EngineCore on the first
# generate (NonConcreteBooleanIndexError). The validated muse smoke ran
# with the variable UNSET, which also still loads the resolver below.
$( if [[ "${VLLM_UNSET_PLUGINS:-0}" == "1" ]]; then
     echo 'unset VLLM_PLUGINS'
   else
     # Resolve unknown adapter names from the shared lora dir (external
     # inference path references adapters by name without an explicit load).
     echo 'export VLLM_PLUGINS="${VLLM_PLUGINS:-lora_filesystem_resolver}"'
   fi )
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
  bash "\$HOME/gcs_rsync.sh" -r "${VLLM_XLA_CACHE_GCS}" "${VLLM_XLA_CACHE_PATH}" 2>/dev/null && echo "restored XLA cache from ${VLLM_XLA_CACHE_GCS}" || echo "XLA cache restore skipped/failed (will compile)"
  # Seed-back: restore alone is ONE-WAY, so entries compiled on this node died
  # with it (the muse rs-study run repopulated nothing). Delayed additive rsync
  # (no -d, checksum-skips existing) publishes fresh compiles once the boot
  # compile window has passed; near-free when the cache was already warm.
  if [[ "${VLLM_XLA_SEED_BACK:-1}" == "1" ]]; then
    # Publish EARLY and REPEATEDLY, not once-at-1h: on spot capacity nodes die
    # well inside an hour, so a single delayed publish loses every compile the
    # node paid for and the next node starts cold again (lived this on muse,
    # four bring-ups with zero cache accumulation). Additive rsync (no -d,
    # checksum-skips existing) is near-free once warm, so a 10-min cadence
    # costs nothing and each cycle preserves whatever compiled since the last.
    ( for _i in \$(seq 1 24); do
        sleep 600
        bash "\$HOME/gcs_rsync.sh" -r "${VLLM_XLA_CACHE_PATH}" "${VLLM_XLA_CACHE_GCS}" >/dev/null 2>&1 \
          && echo "\$(date -u +%H:%M) XLA cache seeded back (cycle \$_i)"
      done ) >> "\$HOME/xla-seedback.log" 2>&1 &
  fi
fi
# Restore HF weights from the shared GCS cache onto local SSD so vLLM finds
# them already present under --download-dir (below) instead of pulling from
# HuggingFace. Validate first: gsutil rsync compares composite GCS objects
# poorly on restart and can replace complete 50GB shards with duplicate
# <shard>_.gstmp downloads until the boot disk fills.
hf_snapshot_ready() {
  HF_HOME="${REMOTE_HF_HOME}" HF_HUB_OFFLINE=1 \
    "${VLLM_VENV}/bin/python" - "${MODEL_NAME}" <<'PY'
import sys
import json
from pathlib import Path

from huggingface_hub import snapshot_download

try:
    snapshot = Path(snapshot_download(sys.argv[1], local_files_only=True))
except Exception:
    raise SystemExit(1)

# snapshot_download() returns an existing snapshot directory without proving
# that a previous selective restore populated its weights or processor assets.
index_path = snapshot / "model.safetensors.index.json"
if index_path.is_file():
    index = json.loads(index_path.read_text())
    required = set(index.get("weight_map", {}).values())
    if not required or any(not (snapshot / name).is_file() for name in required):
        raise SystemExit(1)
elif not any(path.is_file() for path in snapshot.glob("*.safetensors")):
    raise SystemExit(1)

if sys.argv[1].startswith("Qwen/Qwen3.5"):
    required = ("preprocessor_config.json", "video_preprocessor_config.json")
    if any(not (snapshot / name).is_file() for name in required):
        raise SystemExit(1)
PY
}

# Some cache producers store a compact trees/<commit>.json manifest instead of
# snapshots/<commit> symlinks. Materialize the standard offline snapshot after
# the blobs have passed GCS CRC verification.
materialize_hf_tree_manifest() {
  "${VLLM_VENV}/bin/python" - "${REMOTE_HF_HOME}/hub/${HF_MODEL_DIR}" <<'PY'
import json
import os
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
trees = model_dir / "trees"
if not trees.is_dir():
    raise SystemExit(0)
for manifest_path in trees.glob("*.json"):
    manifest = json.loads(manifest_path.read_text())
    snapshot = model_dir / "snapshots" / manifest_path.stem
    snapshot.mkdir(parents=True, exist_ok=True)
    for name, metadata in manifest["files"].items():
        blob_id = metadata.get("lfs_sha256") or metadata["blob_id"]
        blob = model_dir / "blobs" / blob_id
        if not blob.is_file() or blob.stat().st_size != metadata["size"]:
            raise SystemExit(f"missing or wrong-size HF blob for {name}: {blob}")
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.unlink(missing_ok=True)
        target.symlink_to(os.path.relpath(blob, target.parent))
PY
}

mkdir -p "${REMOTE_HF_HOME}/hub/${HF_MODEL_DIR}"
# Engines on the same host share one HF cache. With VLLM_ENGINES_PER_HOST>1
# every engine runner reaches this point at the same moment on a fresh host;
# unserialized, both ran the 55 GB \`gcloud storage cp --no-clobber\` into the
# same hub, the loser saw the winner's in-flight partials, purged them, and
# exited before its log tee opened -- one silent engine per host (muse v5p-32
# pool bring-up, 2026-09-04). Hold a per-model lock across the whole restore so
# the second engine simply waits and then finds the snapshot ready.
exec 9>"${REMOTE_HF_HOME}/hub/.${HF_MODEL_DIR}.restore.lock"
flock 9
if hf_snapshot_ready; then
  echo "HF cache already complete for ${MODEL_NAME}; skipping restore"
elif [[ -n "${HF_CACHE_GCS}" ]]; then
  hf_ok=0
  for _try in 1 2 3; do
    if gcloud storage cp --recursive --no-clobber \
        "${HF_CACHE_GCS}/${HF_MODEL_DIR}" "${REMOTE_HF_HOME}/hub"; then
      hf_ok=1
    fi
    _partials="\$(find "${REMOTE_HF_HOME}/hub" -name '*_.gstmp' -o -name '*.incomplete' 2>/dev/null | head -1)"
    if [[ "\$hf_ok" == "1" && -z "\$_partials" ]]; then
      echo "restored HF cache from ${HF_CACHE_GCS} (attempt \$_try)"
      break
    fi
    echo "HF cache restore incomplete (attempt \$_try): partial=\${_partials:-none}" >&2
    hf_ok=0
    sleep 15
  done
  if [[ "\$hf_ok" == "1" ]]; then
    # Best effort: some caches ship a trees/<commit>.json that references blobs
    # never uploaded (gemma4: .eval_results/*), while snapshots/ already holds
    # real files. A fatal exit here killed every gemma engine on a warm host
    # before its log opened (job 194, 2026-09-05); the readiness check below is
    # the real gate.
    materialize_hf_tree_manifest \\
      || echo "HF tree manifest not materialized (rc=\$?); relying on the snapshot readiness check" >&2
    # GCS cannot hold the symlink a native HF cache uses from snapshots/ to
    # blobs/, so the restore materializes weight shards TWICE (gemma-4-31B-it:
    # 11.9 GB in both, 71 GB total on a 97 GB engine disk -> 0 bytes free, LoRA
    # uploads 500, cell dead: job 245, 2026-09-05). Collapse duplicates to
    # hardlinks; vLLM reads snapshots/ and the bytes are unchanged.
    bash "\$HOME/dedupe_hf_snapshot.sh" "${REMOTE_HF_HOME}/hub/${HF_MODEL_DIR}" 2>/dev/null | tail -1
  else
    _n="\$(find "${REMOTE_HF_HOME}/hub" \( -name '*_.gstmp' -o -name '*.incomplete' \) -delete -print 2>/dev/null | wc -l)"
    echo "HF cache restore FAILED after 3 tries; purged \$_n partial file(s)" >&2
  fi
  if ! hf_snapshot_ready && [[ "\${HF_HUB_OFFLINE}" == "1" ]]; then
    echo "offline HF snapshot is incomplete for ${MODEL_NAME}" >&2
    exit 1
  fi
elif [[ "\${HF_HUB_OFFLINE}" == "1" ]]; then
  echo "offline HF snapshot is absent for ${MODEL_NAME} and HF_CACHE_GCS is unset" >&2
  exit 1
fi
flock -u 9
exec 9>&-
if [[ -n "\${VLLM_RELATIVE_WORKER_ID:-}" ]]; then
  export CLOUD_TPU_TASK_ID="\${VLLM_RELATIVE_WORKER_ID}"
fi
ray_args=()
if [[ "${VLLM_RAY_EXECUTOR}" == "1" ]]; then
  export TPU_MULTIHOST_BACKEND=ray
  export VLLM_USE_RAY_V2_EXECUTOR_BACKEND="${VLLM_USE_RAY_V2_EXECUTOR_BACKEND}"
  export RAY_ADDRESS="${ray_head_address}"
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
VLLM_LIMIT_MM_PER_PROMPT=${vllm_limit_mm_per_prompt_q}
limit_mm_args=()
if [[ -n "\${VLLM_LIMIT_MM_PER_PROMPT}" ]]; then
  limit_mm_args+=(--limit-mm-per-prompt "\${VLLM_LIMIT_MM_PER_PROMPT}")
fi
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
fi${engine_env_block}
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
lora_args=()
if [[ "${VLLM_ENABLE_LORA}" == "1" || "${VLLM_ENABLE_LORA}" == "true" ]]; then
  lora_args=(
    --enable-lora
    --max-loras "${VLLM_MAX_LORAS}"
    --max-lora-rank "${VLLM_MAX_LORA_RANK}"
  )
fi
# One-time self-seed of the shared HF cache: once vLLM has pulled the weights
# onto local SSD, stage them back to GCS so the next spot VM restores from
# HF_CACHE_GCS instead of re-downloading from HuggingFace. Fully backgrounded
# and best-effort (guarded on the GCS prefix currently being empty) — it must
# never block or fail the serve launch, so it runs before the exec below.
if [[ -n "${HF_CACHE_GCS}" ]]; then
  nohup bash -c '
    hub="${REMOTE_HF_HOME}/hub"
    for _try in \$(seq 1 60); do
      if compgen -G "\${hub}/models--*/snapshots/*/*.safetensors" >/dev/null 2>&1; then break; fi
      sleep 10
    done
    if compgen -G "\${hub}/models--*/snapshots/*/*.safetensors" >/dev/null 2>&1 &&
       [[ -z "\$(gcloud storage ls "${HF_CACHE_GCS}/**" 2>/dev/null)" ]]; then
      bash "\$HOME/gcs_rsync.sh" -r "\${hub}" "${HF_CACHE_GCS}" >/dev/null 2>&1 &&
        echo "seeded HF cache to ${HF_CACHE_GCS}" || true
    fi
  ' >"\$HOME/skyrl-logs/hf-cache-seed.log" 2>&1 &
fi
set +e
"\${server_cmd[@]}" \\
  --served-model-name "${SERVED_MODEL_NAME}" \\
  --host 0.0.0.0 \\
  --port "${runner_http_port}" \\
  --tensor-parallel-size "${VLLM_TP_SIZE}" \\
  --max-model-len "${VLLM_MAX_MODEL_LEN}" \\
  --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \\
  --enable-prefix-caching \\
  "\${lora_args[@]}" \\
  --download-dir "${REMOTE_HF_HOME}/hub" \\
  "\${ray_args[@]}" \\
  "\${dp_args[@]}" \\
  "\${limit_mm_args[@]}" \\
  "\${extra_args[@]}"
server_rc=\$?
set -e
printf 'end=%s pid=%s worker=%s engine=%s bundle=%s exit_code=%s\n' \
  "\$(date -u +%Y-%m-%dT%H:%M:%SZ)" "\$\$" \
  "\${VLLM_RELATIVE_WORKER_ID:-unknown}" "\${VLLM_ENGINE_INDEX:-0}" \
  "\${TPUSWARM_BUNDLE_ID:-unversioned}" "\$server_rc" >> "\$runner_status_path"
echo "vLLM server exited with code \$server_rc"
exit "\$server_rc"
EOF

chmod +x "$bootstrap_script" "$runner_script"

for worker in "${vllm_workers[@]}"; do
  tpu_vm_scp "$worker" "$bootstrap_script" "~/start_vllm_tpu_bootstrap.sh"
  tpu_vm_scp "$worker" "$runner_script" "~/run_vllm_tpu_server.sh"
  tpu_vm_scp "$worker" "${repo_root}/tpu/vllm_tpu_server.py" "~/vllm_tpu_server.py"
  tpu_vm_scp "$worker" "${repo_root}/tpu/vllm_ray.sh" "~/vllm_ray.sh"
  tpu_vm_scp "$worker" "${repo_root}/tpu/gcs_rsync.sh" "~/gcs_rsync.sh"
  tpu_vm_scp "$worker" "${repo_root}/tpu/dedupe_hf_snapshot.sh" "~/dedupe_hf_snapshot.sh"
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
if (( VLLM_ENGINES_PER_HOST > 1 )); then
  echo "Engines per host: ${VLLM_ENGINES_PER_HOST} (HTTP ports ${VLLM_PORT}..$((VLLM_PORT + VLLM_ENGINES_PER_HOST - 1)), TP=${VLLM_TP_SIZE} each, per-engine logs ~/skyrl-logs/vllm-tpu[-eN].log)"
fi
echo "Log: gcloud alpha compute tpus tpu-vm ssh ${REMOTE_USER}@${TPU_NAME} --project=${PROJECT} --zone=${ZONE} --worker=${primary_vllm_worker} --ssh-key-file=${SSH_KEY_FILE} --command 'tail -f ~/skyrl-logs/vllm-tpu.log'"
echo "URL from worker ${primary_vllm_worker}: http://localhost:${VLLM_PORT}"
