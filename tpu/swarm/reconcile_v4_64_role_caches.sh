#!/usr/bin/env bash
# Reconcile role-specific caches after the physical v4 topology is selected.
set -euo pipefail

: "${JOBMAN_TPU_INTERNAL_IPS:?}"
: "${TRAIN_WORKERS:?}"
: "${VLLM_WORKERS:?}"
: "${SSH_KEY_FILE:?}"

REMOTE_USER="${REMOTE_USER:-gcpuser}"
REPO="${SKYRL_REPO_DIR:-$HOME/SkyRLTpu-tpuswarm}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-27B}"
MAXTEXT_MODEL="${TUNIX_MAXTEXT_MODEL_NAME:-qwen3.5-27b}"
CKPT_ROOT="${TUNIX_MAXTEXT_CKPT_CACHE:-$HOME/skyrl-maxtext-ckpts-local}"
HF_ROOT="${REMOTE_HF_HOME:-$HOME/.cache/huggingface}"
HF_MODEL_DIR="models--${MODEL_NAME//\//--}"
HF_CACHE_GCS="${HF_CACHE_GCS:?v4-64 offline startup requires HF_CACHE_GCS}"
JAX_CACHE_LOCAL="${TUNIX_JAX_CACHE_LOCAL:-$HOME/jax_cache}"
VLLM_CACHE_LOCAL="${VLLM_XLA_CACHE_PATH:-$HOME/vllm-xla-cache-local}"

IFS=',' read -r -a worker_ips <<< "$JOBMAN_TPU_INTERNAL_IPS"
IFS=',' read -r -a train_workers <<< "$TRAIN_WORKERS"
IFS=',' read -r -a vllm_workers <<< "$VLLM_WORKERS"

if (( ${#worker_ips[@]} != 8 || ${#train_workers[@]} != 4 || ${#vllm_workers[@]} != 4 )); then
  echo "v4-64 cache reconciliation requires 8 hosts split into 4 trainer and 4 vLLM ranks" >&2
  exit 2
fi

declare -A roles=()
for rank in "${train_workers[@]}"; do
  [[ "$rank" =~ ^[0-7]$ ]] || { echo "invalid trainer rank: $rank" >&2; exit 2; }
  [[ -z "${roles[$rank]+x}" ]] || { echo "duplicate rank: $rank" >&2; exit 2; }
  roles[$rank]=trainer
done
for rank in "${vllm_workers[@]}"; do
  [[ "$rank" =~ ^[0-7]$ ]] || { echo "invalid vLLM rank: $rank" >&2; exit 2; }
  [[ -z "${roles[$rank]+x}" ]] || { echo "rank $rank appears in both roles" >&2; exit 2; }
  roles[$rank]=vllm
done
(( ${#roles[@]} == 8 )) || { echo "worker roles do not cover all eight ranks" >&2; exit 2; }

SSHO=(
  -F /dev/null
  -i "$SSH_KEY_FILE"
  -o IdentitiesOnly=yes
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o ConnectTimeout=30
)

remote() {
  local rank="$1" command="$2"
  ssh "${SSHO[@]}" "$REMOTE_USER@${worker_ips[$rank]}" "$command"
}

cleanup_pids=()
for rank in "${train_workers[@]}"; do
  printf -v command \
    'V4_64_HOST_ROLE=trainer V4_64_HOST_RANK=%q SKYRL_REPO_DIR=%q HF_MODEL_CACHE_DIR=%q HF_MODEL_CACHE_GCS=%q MAXTEXT_MODEL_CACHE_DIR=%q TUNIX_JAX_CACHE_LOCAL=%q VLLM_XLA_CACHE_PATH=%q bash %q' \
    "$rank" "$REPO" "$HF_ROOT/hub/$HF_MODEL_DIR" "$HF_CACHE_GCS/$HF_MODEL_DIR" \
    "$CKPT_ROOT/$MAXTEXT_MODEL" "$JAX_CACHE_LOCAL" "$VLLM_CACHE_LOCAL" \
    "$REPO/tpu/swarm/reconcile_v4_64_host_role.sh"
  remote "$rank" "$command" &
  cleanup_pids+=("$!")
done
for rank in "${vllm_workers[@]}"; do
  printf -v command \
    'V4_64_HOST_ROLE=vllm V4_64_HOST_RANK=%q SKYRL_REPO_DIR=%q HF_MODEL_CACHE_DIR=%q HF_MODEL_CACHE_GCS=%q MAXTEXT_MODEL_CACHE_DIR=%q TUNIX_JAX_CACHE_LOCAL=%q VLLM_XLA_CACHE_PATH=%q bash %q' \
    "$rank" "$REPO" "$HF_ROOT/hub/$HF_MODEL_DIR" "$HF_CACHE_GCS/$HF_MODEL_DIR" \
    "$CKPT_ROOT/$MAXTEXT_MODEL" "$JAX_CACHE_LOCAL" "$VLLM_CACHE_LOCAL" \
    "$REPO/tpu/swarm/reconcile_v4_64_host_role.sh"
  remote "$rank" "$command" &
  cleanup_pids+=("$!")
done

failed=0
for pid in "${cleanup_pids[@]}"; do
  wait "$pid" || failed=1
done
if (( failed )); then
  echo "v4-64 role cache cleanup failed" >&2
  exit 1
fi

# Checkpoint acquisition is intentionally after cleanup. Each trainer needs a
# complete local Orbax copy; a partial copy must never reach collective setup.
stage_pids=()
for rank in "${train_workers[@]}"; do
  printf -v command \
    'export JOBMAN_WORKER_ID=%q TRAIN_WORKERS=%q CELL=%q MODEL_NAME=%q TUNIX_MAXTEXT_MODEL_NAME=%q TUNIX_MAXTEXT_CKPT_CACHE=%q TUNIX_MAXTEXT_CKPT_CACHE_GCS=%q; bash %q' \
    "$rank" "$TRAIN_WORKERS" "${CELL:-grpo-n}" "$MODEL_NAME" "$MAXTEXT_MODEL" "$CKPT_ROOT" \
    "${TUNIX_MAXTEXT_CKPT_CACHE_GCS:-gs://sk7524-tinker-tpu-us-central2/skyrl-maxtext-ckpts}" \
    "$REPO/tpu/jobman/ensure_orbax_ckpt.sh"
  remote "$rank" "$command" &
  stage_pids+=("$!")
done

failed=0
for pid in "${stage_pids[@]}"; do
  wait "$pid" || failed=1
done
if (( failed )); then
  echo "v4-64 trainer checkpoint staging failed" >&2
  exit 1
fi

# ensure_orbax_ckpt.sh may reclaim a trainer host's complete HF model cache
# when the root disk cannot hold both it and Orbax. Restore the small offline
# tokenizer/config snapshot after that reclamation, before trainer startup.
metadata_pids=()
for rank in "${train_workers[@]}"; do
  printf -v command \
    'python3 %q %q %q' \
    "$REPO/tpu/swarm/stage_hf_metadata_cache.py" \
    "$HF_CACHE_GCS/$HF_MODEL_DIR" "$HF_ROOT/hub/$HF_MODEL_DIR"
  remote "$rank" "$command" &
  metadata_pids+=("$!")
done

failed=0
for pid in "${metadata_pids[@]}"; do
  wait "$pid" || failed=1
done
if (( failed )); then
  echo "v4-64 trainer HF metadata restaging failed" >&2
  exit 1
fi

echo "v4-64 role caches reconciled and trainer checkpoints verified"
