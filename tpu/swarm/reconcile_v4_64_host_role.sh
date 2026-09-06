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
HOST_RANK="${V4_64_HOST_RANK:-0}"
HOST_HOME="${V4_64_HOST_HOME:-$HOME}"
BUNDLES="$HOST_HOME/.cache/tpuswarm/bundles"
CURRENT_BUNDLE="$(readlink -f "$SKYRL_REPO_DIR" 2>/dev/null || true)"
JAX_CACHE_DIR="${TUNIX_JAX_CACHE_LOCAL:-$HOST_HOME/jax_cache}"
VLLM_CACHE_DIR="${VLLM_XLA_CACHE_PATH:-$HOST_HOME/vllm-xla-cache-local}"

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

stop_stale_workload() {
  local session stale_pids
  while read -r session; do
    [[ -z "$session" ]] && continue
    case "$session" in
      cell|cell-backup|skyrl-tinker|skyrl-tinker-worker-*|vllm-tpu|vllm-tpu-e*)
        tmux kill-session -t "=$session" 2>/dev/null || true ;;
    esac
  done < <(tmux list-sessions -F '#{session_name}' 2>/dev/null || true)

  # Fence restore parents as well as gcloud children. Killing only the copy and
  # deleting a foreign checkpoint lets its old parent start another attempt and
  # recreate it underneath the new model's restore (job 240, 2026-09-06).
  stale_pids=$(ps -eo pid=,comm=,args= | awk '
    /[e]nsure_orbax_ckpt\.sh|[c]ell_worker\.sh|[c]ell_monitor\.sh|[s]tart_colocated_vllm_tinker\.sh|[s]tart_vllm_tpu\.sh|[r]un_ttd_ensemble\.py|[s]kyrl\.tinker\.(api|engine)|[v]llm_tpu_server\.py|[V]LLM::EngineCore|[r]ay_tpuswarm_grader/ {print $1; next}
    $2 ~ /^python/ && /[g]cloud\.py storage (cp|rsync)/ {print $1}
  ')
  stop_pids "$stale_pids"

  find "$HF_HUB_DIR" "$MAXTEXT_CACHE_ROOT" "$HOST_HOME/gcs/skyrl-checkpoints" \
    \( -name '*_.gstmp' -o -name '*.gstmp' -o -name '*.incomplete' -o -name '*.partial' \) \
    -delete 2>/dev/null || true
  rm -rf "$HOST_HOME/.config/gcloud/surface_data/storage/tracker_files" 2>/dev/null || true
  rm -f "$HOST_HOME/ENGINE-SICK" 2>/dev/null || true
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

reconcile_compile_caches() {
  local keep="$1" path marker last
  for path in "$HOST_HOME"/jax-cache-* "$HOST_HOME"/jax_cache \
              "$HOST_HOME"/jax_cache_* "$HOST_HOME"/vllm-xla-cache-* \
              "$HOST_HOME"/vllm-xla-cache-local; do
    [[ -d "$path" ]] || continue
    if [[ "$path" != "$keep" ]]; then
      evict_tree "compile cache for another role or model" "$path"
      continue
    fi
    marker="$path/.tpuswarm-model"
    last="$(cat "$marker" 2>/dev/null || true)"
    if [[ "$last" != "$MAXTEXT_MODEL_DIR" ]]; then
      evict_tree "unmarked or foreign compile cache (${last:-unknown})" "$path"
    fi
  done
  mkdir -p "$keep"
  printf '%s\n' "$MAXTEXT_MODEL_DIR" > "$keep/.tpuswarm-model"
}

prune_superseded_bundles() {
  local path
  for path in "$BUNDLES"/*/ "$BUNDLES"/.extract.*/; do
    path="${path%/}"
    [[ -d "$path" ]] || continue
    [[ "$(readlink -f "$path")" == "$CURRENT_BUNDLE" ]] || \
      evict_tree "superseded bundle generation" "$path"
  done
}

stop_stale_workload

case "$V4_64_HOST_ROLE" in
  trainer)
    evict_foreign_hf_models
    evict_orbax_models 1
    if [[ "${V4_64_FAILURE_CLEANUP:-0}" == "1" &&
          -d "$MAXTEXT_MODEL_CACHE_DIR" &&
          ! -s "$MAXTEXT_MODEL_CACHE_DIR/.tpuswarm-complete" ]]; then
      evict_tree "incomplete Orbax checkpoint from failed attempt" "$MAXTEXT_MODEL_CACHE_DIR"
    fi
    python3 "$SKYRL_REPO_DIR/tpu/swarm/prune_hf_weight_cache.py" "$HF_MODEL_CACHE_DIR"
    python3 "$SKYRL_REPO_DIR/tpu/swarm/stage_hf_metadata_cache.py" \
      "$HF_MODEL_CACHE_GCS" "$HF_MODEL_CACHE_DIR"
    reconcile_compile_caches "$JAX_CACHE_DIR"
    ;;
  vllm)
    evict_foreign_hf_models
    evict_orbax_models 0
    find "$HF_MODEL_CACHE_DIR" -type f \
      \( -name '*.incomplete' -o -name '*_.gstmp' \) -delete 2>/dev/null || true
    if [[ -d "$HF_MODEL_CACHE_DIR" ]]; then
      bash "$SKYRL_REPO_DIR/tpu/dedupe_hf_snapshot.sh" "$HF_MODEL_CACHE_DIR" \
        2>/dev/null | tail -1 || true
    fi
    reconcile_compile_caches "$VLLM_CACHE_DIR"
    ;;
  *)
    echo "invalid V4_64_HOST_ROLE: $V4_64_HOST_ROLE" >&2
    exit 2
    ;;
esac

prune_superseded_bundles
[[ -x "$HOST_HOME/.local/bin/uv" ]] && "$HOST_HOME/.local/bin/uv" cache prune -q >/dev/null 2>&1 || true
echo "v4-64 host cache reconciled for role=$V4_64_HOST_ROLE rank=$HOST_RANK; free=$(df -Pk / | awk 'NR==2 {print int($4/1048576)}') GB"
