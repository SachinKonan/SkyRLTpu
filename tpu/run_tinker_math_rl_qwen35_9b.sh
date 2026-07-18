#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

PROJECT="${PROJECT:-vision-mix}"
ZONE="${ZONE:-us-east5-a}"
TPU_NAME="${TPU_NAME:-sk7524-tinker-v5p64-east5a_spot}"
REMOTE_USER="${REMOTE_USER:-sk7524_princeton_edu}"
SSH_KEY_FILE="${SSH_KEY_FILE:-$HOME/.ssh/jobman_tpu_ed25519}"

LOCAL_PORT="${LOCAL_PORT:-18000}"
REMOTE_PORT="${REMOTE_PORT:-8000}"
TINKER_API_WORKER="${TINKER_API_WORKER:-0}"
TUNNEL_SESSION="${TUNNEL_SESSION:-skyrl-tinker-tunnel}"
CLIENT_SESSION="${CLIENT_SESSION:-skyrl-math-rl}"
REPLACE_CLIENT="${REPLACE_CLIENT:-0}"

TINKER_API_KEY="${TINKER_API_KEY:-tml-dummy}"
LOCAL_HF_HOME="${LOCAL_HF_HOME:-$HOME/.cache/huggingface}"
LOCAL_PYTHON="${LOCAL_PYTHON:-3.12}"
LOG_ROOT="${LOG_ROOT:-${repo_root}/runs/math_rl}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-9B}"
ENV_NAME="${ENV_NAME:-math}"
GROUP_SIZE="${GROUP_SIZE:-16}"
GROUPS_PER_BATCH="${GROUPS_PER_BATCH:-64}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
MAX_TOKENS="${MAX_TOKENS:-512}"
MAX_TURNS="${MAX_TURNS:-1}"
LORA_RANK="${LORA_RANK:-32}"
MAX_STEPS="${MAX_STEPS:-180}"
SAVE_EVERY="${SAVE_EVERY:-20}"
EVAL_EVERY="${EVAL_EVERY:-20}"
LOSS_FN="${LOSS_FN:-importance_sampling}"
# >0 enables on-policy pipelined training (stream minibatches).
STREAM_NUM_MINIBATCHES="${STREAM_NUM_MINIBATCHES:-0}"
SEED="${SEED:-0}"
BEHAVIOR_IF_LOG_DIR_EXISTS="${BEHAVIOR_IF_LOG_DIR_EXISTS:-delete}"

worker0_host="${TPU_WORKER0_HOST:-}"
if [[ -z "$worker0_host" ]]; then
  worker0_host="$(
    gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" \
      --project="$PROJECT" \
      --zone="$ZONE" \
      --format="value(networkEndpoints[${TINKER_API_WORKER}].accessConfig.externalIp)"
  )"
fi
if [[ -z "$worker0_host" ]]; then
  echo "Could not resolve API worker ${TINKER_API_WORKER} external IP for ${TPU_NAME}" >&2
  exit 1
fi

mkdir -p "$LOG_ROOT"
date_stamp="$(date -u +%Y-%m-%d-%H-%M)"
log_path="${LOG_PATH:-${LOG_ROOT}/math-Qwen-Qwen3.5-9B-32rank-2e-05lr-16group-64batch-importance_sampling-seed0-${date_stamp}}"
client_log="${CLIENT_LOG:-${LOG_ROOT}/math-rl-qwen35-9b-${date_stamp}.log}"
run_script="${LOG_ROOT}/run-client-qwen35-9b-${date_stamp}.sh"
base_url="http://127.0.0.1:${LOCAL_PORT}"
stream_arg=""
if [[ "${STREAM_NUM_MINIBATCHES}" -gt 0 ]]; then
  stream_arg="stream_minibatch_config.num_minibatches='${STREAM_NUM_MINIBATCHES}' stream_minibatch_config.groups_per_batch='${GROUPS_PER_BATCH}'"
fi
# A single failed rollout group otherwise hangs the stream-minibatch loop
# (queue consumer waits forever for the dead group's slot).
tolerance_arg=""
if [[ "${ROLLOUT_ERROR_TOLERANCE:-0}" == "1" ]]; then
  tolerance_arg="rollout_error_tolerance='True'"
fi
cookbook_spec="tinker-cookbook[math-rl] @ file://${repo_root}/third_party/tinker-cookbook"

if ! tmux has-session -t "$TUNNEL_SESSION" 2>/dev/null; then
  tmux new-session -d -s "$TUNNEL_SESSION" \
    "/usr/bin/ssh -T -i '$SSH_KEY_FILE' \
      -o IdentitiesOnly=yes \
      -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=30 \
      -o ServerAliveCountMax=3 \
      -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      -N -L '127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}' \
      '${REMOTE_USER}@${worker0_host}'"
fi

for _ in $(seq 1 30); do
  if curl -fsS -m 2 "${base_url}/api/v1/get_server_capabilities" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
curl -fsS -m 5 "${base_url}/api/v1/get_server_capabilities" >/dev/null

cat > "$run_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail

export TINKER_API_KEY='${TINKER_API_KEY}'
export HF_HOME='${LOCAL_HF_HOME}'
export TRANSFORMERS_CACHE="\${HF_HOME}/hub"
if [[ '${MAX_TURNS}' == '1' && -z "\${TINKER_COOKBOOK_GROUP_COALESCE_SAMPLING:-}" ]]; then
  export TINKER_COOKBOOK_GROUP_COALESCE_SAMPLING=1
fi

cd '${repo_root}'
exec uv run --python '${LOCAL_PYTHON}' --no-project --with '${cookbook_spec}' \\
  --reinstall-package tinker-cookbook \\
  python -m tinker_cookbook.recipes.math_rl.train \\
    base_url='${base_url}' \\
    env='${ENV_NAME}' \\
    model_name='${MODEL_NAME}' \\
    group_size='${GROUP_SIZE}' \\
    groups_per_batch='${GROUPS_PER_BATCH}' \\
    learning_rate='${LEARNING_RATE}' \\
    max_tokens='${MAX_TOKENS}' \\
    max_turns='${MAX_TURNS}' \\
    lora_rank='${LORA_RANK}' \\
    max_steps='${MAX_STEPS}' \\
    save_every='${SAVE_EVERY}' \\
    eval_every='${EVAL_EVERY}' \\
    loss_fn='${LOSS_FN}' \\
    ${stream_arg} ${tolerance_arg} seed='${SEED}' \\
    log_path='${log_path}' \\
    behavior_if_log_dir_exists='${BEHAVIOR_IF_LOG_DIR_EXISTS}'
EOF
chmod +x "$run_script"

if tmux has-session -t "$CLIENT_SESSION" 2>/dev/null; then
  if [[ "$REPLACE_CLIENT" != "1" ]]; then
    echo "tmux session ${CLIENT_SESSION} already exists. Set REPLACE_CLIENT=1 to replace it." >&2
    exit 1
  fi
  tmux kill-session -t "$CLIENT_SESSION"
fi

tmux new-session -d -s "$CLIENT_SESSION" "bash '$run_script' 2>&1 | tee '$client_log'"

echo "Tunnel tmux session: ${TUNNEL_SESSION}"
echo "Client tmux session: ${CLIENT_SESSION}"
echo "Base URL: ${base_url}"
echo "Run script: ${run_script}"
echo "Client log: ${client_log}"
echo "Metrics dir: ${log_path}"
