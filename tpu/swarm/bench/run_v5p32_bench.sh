#!/usr/bin/env bash
# vLLM serving benchmark on one tpu-v5p-32 pool worker in the production cell
# shape: rank 0 is the (idle) trainer/driver host, ranks 1-3 serve engines.
# No trainer is started.  Rank 0 brings the engines up with the SAME
# start_vllm_tpu.sh the cells use, ships realistic_bench.py + states.json to
# engine host 1 (it has the vLLM venv and the HF tokenizer), runs the bench in
# a tmux session there, and publishes the result JSON to GCS.
#
# Required env (set by the job YAML): BENCH_NAME, BENCH_MODEL (qwen35|gemma4|muse),
# BENCH_ENGINES_PER_HOST (1|2), BENCH_MAX_NUM_SEQS, BENCH_GPU_UTIL, BENCH_STATES_URL,
# BENCH_RESULT_GCS_PREFIX.  Optional: BENCH_MODE (realistic|capacity),
# BENCH_REQUEST_SIZE (32; 1 = exact per-sequence latency), BENCH_TIMEOUT_SECONDS,
# BENCH_CONCURRENCIES, BENCH_KEEP_ENGINES.
set -euo pipefail
: "${SKYPILOT_NODE_RANK:=0}"
: "${SKYPILOT_NODE_IPS:?SkyPilot must provide all v5p-32 TPU VM IPs}"
: "${BENCH_NAME:?}"; : "${BENCH_MODEL:?qwen35|gemma4|muse}"
: "${BENCH_ENGINES_PER_HOST:?}"; : "${BENCH_MAX_NUM_SEQS:?}"
: "${BENCH_STATES_URL:?}"; : "${BENCH_RESULT_GCS_PREFIX:?}"
BENCH_MODE="${BENCH_MODE:-realistic}"
BENCH_GPU_UTIL="${BENCH_GPU_UTIL:-0.92}"
BENCH_REQUEST_SIZE="${BENCH_REQUEST_SIZE:-32}"
BENCH_TIMEOUT_SECONDS="${BENCH_TIMEOUT_SECONDS:-14400}"
BENCH_KEEP_ENGINES="${BENCH_KEEP_ENGINES:-0}"
REPO="${SKYRL_REPO_DIR:-$PWD}"

ips=$(printf '%s\n' "$SKYPILOT_NODE_IPS" | awk 'NF' | paste -sd, -)
node_count=$(awk -F, '{print NF}' <<<"$ips")
if [ "$node_count" -ne 4 ]; then
  echo "a v5p-32 bench requires 4 TPU VMs; SkyPilot supplied $node_count" >&2
  exit 2
fi
export REMOTE_USER="${REMOTE_USER:-$(id -un)}"
export SSH_KEY_FILE="${SSH_KEY_FILE:-$HOME/ray_bootstrap_key.pem}"
mkdir -p "$HOME/.ssh" "$HOME/skyrl-logs" "$HOME/bench"
chmod 700 "$HOME/.ssh"

# Pool workers are reused across jobs and models; a 97 GB boot disk cannot hold
# two models' HF snapshots. Drop the other models' caches on EVERY rank before
# bring-up, exactly as the cells do (reconcile keys the model off a CELL-style
# prefix: g-* gemma, m-* muse, else qwen). Best effort.
case "$BENCH_MODEL" in
  gemma4) reconcile_cell="g-bench" ;;
  muse)   reconcile_cell="m-bench" ;;
  *)      reconcile_cell="bench-qwen" ;;
esac
CELL="$reconcile_cell" bash "$REPO/tpu/swarm/reconcile_v5p32_worker.sh" \
  || echo "worker reconcile failed (continuing)" >&2

if [ "$SKYPILOT_NODE_RANK" != "0" ]; then
  echo "rank $SKYPILOT_NODE_RANK ready; rank 0 drives bench $BENCH_NAME"
  exit 0
fi
[ -f "$SSH_KEY_FILE" ] || { echo "SkyPilot TPU pod key is missing: $SSH_KEY_FILE" >&2; exit 2; }
chmod 600 "$SSH_KEY_FILE"
ln -sfn "$SSH_KEY_FILE" "$HOME/.ssh/jobman_tpu_ed25519"

# Rank 0 must be the first address (the cell's w0 = trainer host = driver);
# SkyPilot does not promise that ordering, so rotate until this VM leads.
local_ips=$(hostname -I 2>/dev/null || true)
IFS=, read -r -a all_ips <<<"$ips"
self_idx=-1
for i in "${!all_ips[@]}"; do for l in $local_ips; do [ "${all_ips[$i]}" = "$l" ] && self_idx=$i; done; done
if [ "$self_idx" -lt 0 ]; then echo "this VM ($local_ips) is not among SKYPILOT_NODE_IPS ($ips)" >&2; exit 2; fi
ordered=("${all_ips[$self_idx]}")
for i in "${!all_ips[@]}"; do [ "$i" -ne "$self_idx" ] && ordered+=("${all_ips[$i]}"); done
INT=$(IFS=,; echo "${ordered[*]}")
engine_hosts=("${ordered[@]:1}")
echo "bench $BENCH_NAME: driver ${ordered[0]} (idle trainer host); engine hosts ${engine_hosts[*]}"

SSHO=(-F /dev/null -i "$SSH_KEY_FILE" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no
      -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o LogLevel=ERROR)
rssh() { ssh "${SSHO[@]}" "$REMOTE_USER@$1" "$2"; }

# ---- per-model serving env: byte-for-byte the cell_worker.sh model blocks ---
TP_SIZE=$(( 4 / BENCH_ENGINES_PER_HOST ))
case "$BENCH_MODEL" in
  qwen35)
    MODEL_NAME=Qwen/Qwen3.5-27B; RENDERER=qwen3
    VLLM_LEN=22528; PHASE1=13824; CTX=22528
    TF_VERSION=5.8.0; TPU_BACKEND=torchax; UNSET_PLUGINS=0; SKIP_PRECOMPILE=0; REQ_TIMEOUT=300
    TPUINF_REF=skyrl/v0.23.0-lora; EXTRA_PIP=""; LIMIT_MM=""
    VLLM_XARGS="--max-num-batched-tokens 8192 --gpu-memory-utilization $BENCH_GPU_UTIL"
    HF_GCS=gs://sk7524-tinker-tpu-us-east5/hf-cache-qwen35-v1 ;;
  gemma4)
    MODEL_NAME=google/gemma-4-31B-it; RENDERER=gemma4
    VLLM_LEN=16384; PHASE1=6656; CTX=10240
    TF_VERSION=5.8.0; TPU_BACKEND=torchax; UNSET_PLUGINS=0; SKIP_PRECOMPILE=0; REQ_TIMEOUT=300
    TPUINF_REF=skyrl/v0.23.0-lora; EXTRA_PIP=""; LIMIT_MM='{"image":0,"audio":0,"video":0}'
    VLLM_XARGS="--max-num-batched-tokens 8192 --disable-chunked-mm-input --gpu-memory-utilization $BENCH_GPU_UTIL"
    HF_GCS=gs://sk7524-tinker-tpu-us-east5/hf-cache-gemma4 ;;
  muse)
    MODEL_NAME=meta-models/Muse-Glimmer-30B; RENDERER=muse_glimmer_high_reasoning
    VLLM_LEN=22528; PHASE1=13824; CTX=22528
    TF_VERSION=""; TPU_BACKEND=jax; UNSET_PLUGINS=1; SKIP_PRECOMPILE=1; REQ_TIMEOUT=1800
    TPUINF_REF=afe0cb9e9bf259a072242c6f3279d92b702f9f2a
    EXTRA_PIP="'transformers @ git+https://github.com/huggingface/transformers@main' 'tokenizers>=0.23.1,<0.24.0'"
    LIMIT_MM=""
    VLLM_XARGS="--max-num-batched-tokens 8192 --gpu-memory-utilization $BENCH_GPU_UTIL"
    HF_GCS=gs://sk7524-tinker-tpu-us-east5/hf-cache ;;
  *) echo "unknown BENCH_MODEL=$BENCH_MODEL" >&2; exit 2 ;;
esac
# Production client windows by default; override for what-if runs.  Note gemma's
# 6656 phase-1 cap leaves only ~50-600 reasoning tokens for the longest
# jssp/ac1 prompts (~6.0-6.6k tokens) -- that IS the production shape.
PHASE1="${BENCH_PHASE1_MAX_TOKENS:-$PHASE1}"
CTX="${BENCH_CONTEXT_WINDOW:-$CTX}"
# Bench-only XLA cache prefixes (keyed by model/len/TP) so the sweep never
# perturbs the production caches.
XLA_GCS="gs://sk7524-tinker-tpu-us-east5/vllm-xla-cache-bench-${BENCH_MODEL}-${VLLM_LEN}-tp${TP_SIZE}"
XLA_LOCAL="$HOME/vllm-xla-cache-bench-${BENCH_MODEL}-tp${TP_SIZE}"
HF_MODEL_DIR="models--${MODEL_NAME//\//--}"

# ---- engines: clean slate on the three serving hosts, then bring up --------
stop_engines() {
  local ip
  for ip in "${engine_hosts[@]}"; do
    rssh "$ip" 'for s in $(tmux list-sessions -F "#{session_name}" 2>/dev/null | grep -E "^vllm-tpu(-e[0-9]+)?$"); do tmux kill-session -t "=$s" 2>/dev/null || true; done;
      pkill -TERM -u "$USER" -f "[V]LLM::EngineCore|[v]llm_tpu_server\.py" 2>/dev/null || true; sleep 3;
      pkill -KILL -u "$USER" -f "[V]LLM::EngineCore|[v]llm_tpu_server\.py" 2>/dev/null || true; true' || true
  done
}
# The durable copy of the bench XLA cache is GCS (start_vllm_tpu.sh restores it
# on the next boot). The LOCAL copy, 5-6 GB per model/TP on every engine host,
# is what tipped reused pool workers to 0 bytes free for the training cells
# (2026-09-05), so publish it and drop it whenever this run ends -- success,
# failure, or timeout -- unless the engines are deliberately kept.
publish_and_drop_xla_caches() {
  local ip
  for ip in "${engine_hosts[@]}"; do
    rssh "$ip" "bash ~/gcs_rsync.sh -r '$XLA_LOCAL' '$XLA_GCS' >/dev/null 2>&1; rm -rf -- '$XLA_LOCAL'; df -Ph \$HOME | tail -1" || true
  done
}
bench_teardown() {
  if [ "$BENCH_KEEP_ENGINES" != "1" ]; then
    stop_engines
    publish_and_drop_xla_caches
  fi
}
trap bench_teardown EXIT
echo "stopping any stale engines on ${engine_hosts[*]}"
stop_engines

bringup_log="$HOME/bench/${BENCH_NAME}.engine-bringup.log"
# start_vllm_tpu.sh is normally invoked by the cells' colocated launcher, which
# supplies REMOTE_SKYRL_DIR (the checkout that carries third_party/tpu-inference
# for the fork overlay). Calling it directly without that died at
# "REMOTE_SKYRL_DIR: unbound variable" before any engine started (jobs 201/203).
env TPU_SSH_MODE=direct TPU_EXTERNAL_IPS="$INT" TPU_INTERNAL_IPS="$INT" TPU_NAME="bench-$BENCH_NAME" \
  PROJECT="${PROJECT:-vision-mix}" ZONE="${ZONE:-us-east5-a}" REMOTE_USER="$REMOTE_USER" SSH_KEY_FILE="$SSH_KEY_FILE" \
  REMOTE_SKYRL_DIR="$REPO" REMOTE_HF_HOME="/home/${REMOTE_USER}/.cache/huggingface" \
  REMOTE_LORA_BASE="/home/${REMOTE_USER}/gcs/skyrl-lora-models" VLLM_TPU_PROCESS_PORT=8476 \
  VLLM_TPU_PROCESS_BOUNDS=1,1,1 VLLM_TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 VLLM_TPU_PROCESS_ADDRESSES="" \
  VLLM_WORKERS=1,2,3 VLLM_RAY_EXECUTOR=0 \
  MODEL_NAME="$MODEL_NAME" SERVED_MODEL_NAME="$MODEL_NAME" \
  VLLM_MODEL_IMPL_TYPE=vllm TPU_INFERENCE_FORK_REF="$TPUINF_REF" HF_HUB_OFFLINE=1 \
  VLLM_TRANSFORMERS_VERSION="$TF_VERSION" VLLM_EXTRA_PIP_SPECS="$EXTRA_PIP" \
  VLLM_TPU_BACKEND_TYPE="$TPU_BACKEND" VLLM_UNSET_PLUGINS="$UNSET_PLUGINS" VLLM_SKIP_JAX_PRECOMPILE="$SKIP_PRECOMPILE" \
  VLLM_TP_SIZE="$TP_SIZE" VLLM_ENGINES_PER_HOST="$BENCH_ENGINES_PER_HOST" \
  VLLM_MAX_MODEL_LEN="$VLLM_LEN" VLLM_MAX_NUM_SEQS="$BENCH_MAX_NUM_SEQS" \
  VLLM_ENABLE_LORA=1 VLLM_MAX_LORA_RANK=32 VLLM_REQUEST_TIMEOUT_SEC="$REQ_TIMEOUT" \
  VLLM_EXTRA_ARGS="$VLLM_XARGS" VLLM_LIMIT_MM_PER_PROMPT="$LIMIT_MM" \
  VLLM_XLA_CACHE_PATH="$XLA_LOCAL" VLLM_XLA_CACHE_GCS="$XLA_GCS" HF_CACHE_GCS="$HF_GCS" \
  bash "$REPO/tpu/start_vllm_tpu.sh" > "$bringup_log" 2>&1 || { echo "start_vllm_tpu.sh failed" >&2; tail -40 "$bringup_log" >&2; exit 1; }

bases=()
for ip in "${engine_hosts[@]}"; do
  for (( e=0; e<BENCH_ENGINES_PER_HOST; e++ )); do bases+=("http://$ip:$((8001 + e))"); done
done
echo "waiting for ${#bases[@]} engines: ${bases[*]}"
deadline=$(( $(date +%s) + 5400 ))
for base in "${bases[@]}"; do
  until curl -fsS --max-time 5 "$base/v1/models" >/dev/null 2>&1; do
    if [ "$(date +%s)" -gt "$deadline" ]; then echo "engine $base never became ready" >&2; exit 1; fi
    ip=${base#http://}; ip=${ip%%:*}; port=${base##*:}; suffix=""
    [ "$port" != "8001" ] && suffix="-e$((port - 8001))"
    if rssh "$ip" "grep -qE 'Engine core initialization failed|EngineCore failed to start|No space left on device' ~/skyrl-logs/vllm-tpu$suffix.log 2>/dev/null"; then
      echo "engine $base failed during startup" >&2; rssh "$ip" "tail -60 ~/skyrl-logs/vllm-tpu$suffix.log" >&2 || true; exit 1
    fi
    sleep 15
  done
  echo "  ready: $base"
done

# ---- boot-log facts: KV cache size and the engine's own max concurrency -----
engine_info="$HOME/bench/${BENCH_NAME}.engine-info.json"
{
  echo '{"engines": ['
  first=1
  for base in "${bases[@]}"; do
    ip=${base#http://}; ip=${ip%%:*}; port=${base##*:}; suffix=""; [ "$port" != "8001" ] && suffix="-e$((port - 8001))"
    kv=$(rssh "$ip" "grep -oE 'KV cache size: [0-9,]+ tokens' ~/skyrl-logs/vllm-tpu$suffix.log 2>/dev/null | tail -1" || true)
    mc=$(rssh "$ip" "grep -oE 'Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x' ~/skyrl-logs/vllm-tpu$suffix.log 2>/dev/null | tail -1" || true)
    [ "$first" = 1 ] || echo ','
    first=0
    printf '{"base":"%s","kv_cache":"%s","max_concurrency":"%s"}' "$base" "$kv" "$mc"
  done
  echo '], "engines_per_host": '"$BENCH_ENGINES_PER_HOST"', "tp_size": '"$TP_SIZE"', "max_num_seqs": '"$BENCH_MAX_NUM_SEQS"', "gpu_memory_utilization": '"$BENCH_GPU_UTIL"'}'
} > "$engine_info"
echo "engine info: $(tr -d '\n' < "$engine_info" | cut -c1-400)"

# ---- run the bench on engine host 1 (vLLM venv + tokenizer live there) ------
driver_ip="${engine_hosts[0]}"
gcloud storage cp "$BENCH_STATES_URL" "$HOME/bench/states.json"
scp "${SSHO[@]}" "$REPO/tpu/swarm/bench/realistic_bench.py" "$HOME/bench/states.json" "$engine_info" "$REMOTE_USER@$driver_ip:~/" >/dev/null
bases_csv=$(IFS=,; echo "${bases[*]}")
result_remote="/home/$REMOTE_USER/bench-${BENCH_NAME}.json"
log_remote="/home/$REMOTE_USER/bench-${BENCH_NAME}.log"
if [ "$BENCH_MODE" = "capacity" ]; then
  # Cap the ladder at engines x the boot log's per-engine max concurrency (rounded down).
  per_engine_max=$(grep -oE 'per request: [0-9.]+x' "$engine_info" | head -1 | grep -oE '[0-9.]+' | cut -d. -f1 || true)
  max_c=0; [ -n "${per_engine_max:-}" ] && max_c=$(( per_engine_max * ${#bases[@]} ))
  bench_args="--mode capacity --max-concurrency $max_c --concurrencies ${BENCH_CONCURRENCIES:-16,32,64,96,128,192,256,384,512}"
else
  bench_args="--mode realistic --states ~/states.json --renderer $RENDERER --group-size 32 --request-size $BENCH_REQUEST_SIZE --phase1-max-tokens $PHASE1 --context-window $CTX --temperature 1.0"
fi
rssh "$driver_ip" "tmux kill-session -t '=bench' 2>/dev/null; rm -f '$result_remote'; tmux new-session -d -s bench \"HF_HOME=\\\$HOME/.cache/huggingface HF_HUB_OFFLINE=1 \\\$HOME/.venvs/vllm-tpu/bin/python ~/realistic_bench.py $bench_args --out '$result_remote' --bases '$bases_csv' --model '$MODEL_NAME' --model-dir '$MODEL_NAME' --max-num-seqs $BENCH_MAX_NUM_SEQS --gpu-memory-utilization $BENCH_GPU_UTIL --engines-per-host $BENCH_ENGINES_PER_HOST --tp-size $TP_SIZE --engine-info-json ~/$(basename "$engine_info") > '$log_remote' 2>&1; echo BENCH_EXIT=\\\$? >> '$log_remote'\""
echo "bench started on $driver_ip (tmux 'bench'); polling for $result_remote"
deadline=$(( $(date +%s) + BENCH_TIMEOUT_SECONDS ))
while true; do
  if rssh "$driver_ip" "test -s '$result_remote'"; then break; fi
  if rssh "$driver_ip" "grep -q '^BENCH_EXIT=' '$log_remote' 2>/dev/null"; then
    echo "bench process exited without a result; log tail:" >&2; rssh "$driver_ip" "tail -40 '$log_remote'" >&2 || true; exit 1
  fi
  if [ "$(date +%s)" -gt "$deadline" ]; then echo "bench timed out after ${BENCH_TIMEOUT_SECONDS}s" >&2; rssh "$driver_ip" "tail -20 '$log_remote'" >&2 || true; exit 1; fi
  rssh "$driver_ip" "grep -E 'HEARTBEAT|CAPACITY|WARMUP|WORKLOAD' '$log_remote' 2>/dev/null | tail -1" || true
  sleep 60
done
scp "${SSHO[@]}" "$REMOTE_USER@$driver_ip:$result_remote" "$HOME/bench/${BENCH_NAME}.json" >/dev/null
scp "${SSHO[@]}" "$REMOTE_USER@$driver_ip:$log_remote" "$HOME/bench/${BENCH_NAME}.log" >/dev/null
grep -E 'BENCHMARK_RESULT' "$HOME/bench/${BENCH_NAME}.log" | tail -1
gcloud storage cp "$HOME/bench/${BENCH_NAME}.json" "$BENCH_RESULT_GCS_PREFIX/${BENCH_NAME}.json"
gcloud storage cp "$HOME/bench/${BENCH_NAME}.log" "$BENCH_RESULT_GCS_PREFIX/${BENCH_NAME}.log"
gcloud storage cp "$bringup_log" "$BENCH_RESULT_GCS_PREFIX/${BENCH_NAME}.engine-bringup.log"
# Engines stop and the XLA cache is published (so the next setting on this
# model/TP boots warm) and dropped locally by the EXIT trap (bench_teardown).
echo "benchmark complete: $BENCH_RESULT_GCS_PREFIX/${BENCH_NAME}.json"
