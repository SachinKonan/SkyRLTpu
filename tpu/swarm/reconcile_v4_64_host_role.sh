#!/usr/bin/env bash
# Stop stale opposite-role work and reconcile one v4-64 host's local caches.
set -euo pipefail

: "${V4_64_HOST_ROLE:?expected trainer or vllm}"
: "${SKYRL_REPO_DIR:?}"
: "${HF_MODEL_CACHE_DIR:?}"
: "${MAXTEXT_MODEL_CACHE_DIR:?}"

stop_pids() {
  local raw_pids="$1" pid
  local -a pids=() active=()
  [[ -z "$raw_pids" ]] && return 0
  read -r -a pids <<< "$raw_pids"
  kill -TERM "${pids[@]}" 2>/dev/null || true
  for _ in $(seq 1 10); do
    active=()
    for pid in "${pids[@]}"; do
      kill -0 "$pid" 2>/dev/null && active+=("$pid")
    done
    (( ${#active[@]} == 0 )) && return 0
    sleep 1
  done
  kill -KILL "${active[@]}" 2>/dev/null || true
  sleep 1
  for pid in "${active[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "opposite-role process $pid survived SIGKILL" >&2
      return 1
    fi
  done
}

case "$V4_64_HOST_ROLE" in
  trainer)
    tmux kill-session -t =vllm-tpu 2>/dev/null || true
    while read -r session; do
      [[ -z "$session" ]] || tmux kill-session -t "=$session" 2>/dev/null || true
    done < <(tmux list-sessions -F '#{session_name}' 2>/dev/null | grep '^vllm-tpu-e[0-9]\+$' || true)
    stale_pids=$(ps -eo pid=,comm=,args= | awk -v model="$HF_MODEL_CACHE_DIR" '
      $2 ~ /^python/ && ($0 ~ /([v]llm_tpu_server\.py|[v]llm serve)/ || ($0 ~ /[g]cloud\.py storage cp/ && index($0, model))) {print $1}
    ')
    stop_pids "$stale_pids"
    python3 "$SKYRL_REPO_DIR/tpu/swarm/prune_hf_weight_cache.py" "$HF_MODEL_CACHE_DIR"
    python3 "$SKYRL_REPO_DIR/tpu/swarm/stage_hf_metadata_cache.py" \
      "$HF_MODEL_CACHE_GCS" "$HF_MODEL_CACHE_DIR"
    ;;
  vllm)
    tmux kill-session -t =skyrl-tinker 2>/dev/null || true
    while read -r session; do
      [[ -z "$session" ]] || tmux kill-session -t "=$session" 2>/dev/null || true
    done < <(tmux list-sessions -F '#{session_name}' 2>/dev/null | grep '^skyrl-tinker-worker-' || true)
    stale_pids=$(ps -eo pid=,comm=,args= | awk '
      $2 ~ /^python/ && $0 ~ /[s]kyrl\.(tinker|backends\.(jax|rpc))/ {print $1}
    ')
    stop_pids "$stale_pids"
    rm -rf -- "$MAXTEXT_MODEL_CACHE_DIR"
    if ! curl -fsS --max-time 5 http://127.0.0.1:8001/v1/models >/dev/null 2>&1; then
      # An interrupted GCS restore may outlive its tmux parent in uninterruptible
      # ext4 writeback. Stop it before start_vllm_tpu.sh resumes the model cache;
      # complete blobs remain reusable, while partials are always disposable.
      tmux kill-session -t =vllm-tpu 2>/dev/null || true
      while read -r session; do
        [[ -z "$session" ]] || tmux kill-session -t "=$session" 2>/dev/null || true
      done < <(tmux list-sessions -F '#{session_name}' 2>/dev/null | grep '^vllm-tpu-e[0-9]\+$' || true)
      stale_pids=$(ps -eo pid=,comm=,args= | awk -v model="$HF_MODEL_CACHE_DIR" '
        $2 ~ /^python/ && ($0 ~ /([v]llm_tpu_server\.py|[v]llm serve|[V]LLM::EngineCore)/ || ($0 ~ /[g]cloud\.py storage cp/ && index($0, model))) {print $1}
      ')
      stop_pids "$stale_pids"
      find "$HF_MODEL_CACHE_DIR" -type f \
        \( -name '*.incomplete' -o -name '*_.gstmp' \) -delete 2>/dev/null || true
    fi
    ;;
  *)
    echo "invalid V4_64_HOST_ROLE: $V4_64_HOST_ROLE" >&2
    exit 2
    ;;
esac

echo "v4-64 host cache reconciled for role=$V4_64_HOST_ROLE"
