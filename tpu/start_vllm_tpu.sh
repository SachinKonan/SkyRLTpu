#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-vision-mix}"
ZONE="${ZONE:-us-east5-a}"
TPU_NAME="${TPU_NAME:-sk7524-vllm-qwen3-4b-v5p8-east5a_spot}"
REMOTE_USER="${REMOTE_USER:-sk7524_princeton_edu}"
SSH_KEY_FILE="${SSH_KEY_FILE:-$HOME/.ssh/jobman_tpu_ed25519}"
VLLM_WORKER="${VLLM_WORKER:-0}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_NAME}"
VLLM_TPU_VERSION="${VLLM_TPU_VERSION:-0.23.0}"
VLLM_MODEL_IMPL_TYPE="${VLLM_MODEL_IMPL_TYPE:-vllm}"
VLLM_PORT="${VLLM_PORT:-8001}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-1}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-2048}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
VLLM_MAX_LORAS="${VLLM_MAX_LORAS:-8}"
VLLM_MAX_LORA_RANK="${VLLM_MAX_LORA_RANK:-32}"
VLLM_VENV="${VLLM_VENV:-/home/${REMOTE_USER}/.venvs/vllm-tpu}"
REMOTE_HF_HOME="${REMOTE_HF_HOME:-/home/${REMOTE_USER}/.cache/huggingface}"
REMOTE_LORA_BASE="${REMOTE_LORA_BASE:-/home/${REMOTE_USER}/gcs/skyrl-lora-models}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
apply_patch_script="${repo_root}/tpu/apply_vllm_tpu_lora_patch.sh"
patch_file="${repo_root}/third_party/patches/tpu-inference-tpu-worker-lora-forwarders.patch"

if [[ ! -f "${apply_patch_script}" || ! -f "${patch_file}" ]]; then
  echo "Missing vLLM TPU patch assets. Expected:" >&2
  echo "  ${apply_patch_script}" >&2
  echo "  ${patch_file}" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

bootstrap_script="${tmpdir}/start_vllm_tpu_bootstrap.sh"
runner_script="${tmpdir}/run_vllm_tpu_server.sh"

cat > "$bootstrap_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
export HF_HOME="${REMOTE_HF_HOME}"
export TRANSFORMERS_CACHE="\$HF_HOME/hub"
mkdir -p "${REMOTE_HF_HOME}" "${REMOTE_LORA_BASE}" "$(dirname "${VLLM_VENV}")" "\$HOME/skyrl-logs"

if [ ! -x "${VLLM_VENV}/bin/vllm" ]; then
  uv venv --python 3.12 "${VLLM_VENV}"
fi

uv pip install --python "${VLLM_VENV}/bin/python" "vllm-tpu==${VLLM_TPU_VERSION}"
PATCH_FILE="\$HOME/tpu-inference-tpu-worker-lora-forwarders.patch" \\
  PYTHON="${VLLM_VENV}/bin/python" \\
  bash "\$HOME/apply_vllm_tpu_lora_patch.sh"

tmux kill-session -t vllm-tpu 2>/dev/null || true
tmux new-session -d -s vllm-tpu "bash \$HOME/run_vllm_tpu_server.sh"
EOF

cat > "$runner_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source "${VLLM_VENV}/bin/activate"
export HF_HOME="${REMOTE_HF_HOME}"
export TRANSFORMERS_CACHE="${REMOTE_HF_HOME}/hub"
export MODEL_IMPL_TYPE="${VLLM_MODEL_IMPL_TYPE}"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

exec vllm serve "${MODEL_NAME}" \\
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
  --disable-log-requests \\
  2>&1 | tee "\$HOME/skyrl-logs/vllm-tpu.log"
EOF

chmod +x "$bootstrap_script" "$runner_script"

gcloud alpha compute tpus tpu-vm scp "$bootstrap_script" "${REMOTE_USER}@${TPU_NAME}:~/start_vllm_tpu_bootstrap.sh" \
  --project="$PROJECT" --zone="$ZONE" --worker="$VLLM_WORKER" --ssh-key-file="$SSH_KEY_FILE" --quiet

gcloud alpha compute tpus tpu-vm scp "$runner_script" "${REMOTE_USER}@${TPU_NAME}:~/run_vllm_tpu_server.sh" \
  --project="$PROJECT" --zone="$ZONE" --worker="$VLLM_WORKER" --ssh-key-file="$SSH_KEY_FILE" --quiet

gcloud alpha compute tpus tpu-vm scp "$apply_patch_script" "${REMOTE_USER}@${TPU_NAME}:~/apply_vllm_tpu_lora_patch.sh" \
  --project="$PROJECT" --zone="$ZONE" --worker="$VLLM_WORKER" --ssh-key-file="$SSH_KEY_FILE" --quiet

gcloud alpha compute tpus tpu-vm scp "$patch_file" "${REMOTE_USER}@${TPU_NAME}:~/tpu-inference-tpu-worker-lora-forwarders.patch" \
  --project="$PROJECT" --zone="$ZONE" --worker="$VLLM_WORKER" --ssh-key-file="$SSH_KEY_FILE" --quiet

gcloud alpha compute tpus tpu-vm ssh "${REMOTE_USER}@${TPU_NAME}" \
  --project="$PROJECT" --zone="$ZONE" --worker="$VLLM_WORKER" --ssh-key-file="$SSH_KEY_FILE" --quiet \
  --command "bash ~/start_vllm_tpu_bootstrap.sh"

echo "vLLM TPU start command submitted on worker ${VLLM_WORKER}."
echo "Log: gcloud alpha compute tpus tpu-vm ssh ${REMOTE_USER}@${TPU_NAME} --project=${PROJECT} --zone=${ZONE} --worker=${VLLM_WORKER} --ssh-key-file=${SSH_KEY_FILE} --command 'tail -f ~/skyrl-logs/vllm-tpu.log'"
echo "URL from worker ${VLLM_WORKER}: http://localhost:${VLLM_PORT}"
