#!/usr/bin/env bash
# Stop stale opposite-role work and reconcile one v4-64 host's local caches.
set -euo pipefail

: "${V4_64_HOST_ROLE:?expected trainer or vllm}"
: "${SKYRL_REPO_DIR:?}"
: "${HF_MODEL_CACHE_DIR:?}"
: "${MAXTEXT_MODEL_CACHE_DIR:?}"

HF_HUB_DIR="$(dirname "$HF_MODEL_CACHE_DIR")"
HF_MODEL_DIR="$(basename "$HF_MODEL_CACHE_DIR")"
MAXTEXT_CACHE_ROOT="$(dirname "$MAXTEXT_MODEL_CACHE_DIR")"
MAXTEXT_MODEL_DIR="$(basename "$MAXTEXT_MODEL_CACHE_DIR")"

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

evict_tree() {
  local reason="$1" path="$2"
  [[ -e "$path" ]] || return 0
  echo "v4-64 reconcile: $reason -- removing $path ($(du -sh "$path" 2>/dev/null | cut -f1))"
  rm -rf -- "$path"
}

stop_vllm() {
  tmux kill-session -t =vllm-tpu 2>/dev/null || true
  while read -r session; do
    [[ -z "$session" ]] || tmux kill-session -t "=$session" 2>/dev/null || true
  done < <(tmux list-sessions -F '#{session_name}' 2>/dev/null | grep '^vllm-tpu-e[0-9]\+$' || true)
  stale_pids=$(ps -eo pid=,comm=,args= | awk \
    -v hf_root="$HF_HUB_DIR" -v orbax_root="$MAXTEXT_CACHE_ROOT" '
    ($0 ~ /([v]llm_tpu_server\.py|[v]llm serve|[V]LLM::EngineCore)/ ||
     ($2 ~ /^python/ && $0 ~ /[g]cloud\.py storage cp/ &&
      (index($0, hf_root) || index($0, orbax_root)))) {print $1}
  ')
  stop_pids "$stale_pids"
}

evict_foreign_hf_models() {
  local path
  for path in "$HF_HUB_DIR"/models--*; do
    [[ -d "$path" ]] || continue
    [[ "$(basename "$path")" == "$HF_MODEL_DIR" ]] || \
      evict_tree "foreign HF model cache" "$path"
  done
}

evict_orbax_models() {
  local keep_target="$1" path
  for path in "$MAXTEXT_CACHE_ROOT"/*/; do
    path="${path%/}"
    [[ -d "$path" ]] || continue
    if [[ "$keep_target" != "1" || "$(basename "$path")" != "$MAXTEXT_MODEL_DIR" ]]; then
      evict_tree "unused Orbax checkpoint" "$path"
    fi
  done
}

case "$V4_64_HOST_ROLE" in
  trainer)
    stop_vllm
    evict_foreign_hf_models
    evict_orbax_models 1
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
    # A pool worker may still be serving the previous job's model. It is always
    # restarted by start_vllm_tpu.sh below, so stop it before deleting its cache.
    stop_vllm
    evict_foreign_hf_models
    evict_orbax_models 0
    find "$HF_MODEL_CACHE_DIR" -type f \
      \( -name '*.incomplete' -o -name '*_.gstmp' \) -delete 2>/dev/null || true
    ;;
  *)
    echo "invalid V4_64_HOST_ROLE: $V4_64_HOST_ROLE" >&2
    exit 2
    ;;
esac

echo "v4-64 host cache reconciled for role=$V4_64_HOST_ROLE"
