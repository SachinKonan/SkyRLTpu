#!/usr/bin/env bash
# Best-effort cleanup after a failed eight-host v4-64 cell attempt.
set -uo pipefail

rc="${1:-1}"
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
HF_CACHE_GCS="${HF_CACHE_GCS:-}"
JAX_CACHE_LOCAL="${TUNIX_JAX_CACHE_LOCAL:-$HOME/jax_cache}"
VLLM_CACHE_LOCAL="${VLLM_XLA_CACHE_PATH:-$HOME/vllm-xla-cache-local}"

IFS=',' read -r -a worker_ips <<< "$JOBMAN_TPU_INTERNAL_IPS"
declare -A roles=()
for rank in ${TRAIN_WORKERS//,/ }; do roles[$rank]=trainer; done
for rank in ${VLLM_WORKERS//,/ }; do roles[$rank]=vllm; done

SSHO=(-F /dev/null -i "$SSH_KEY_FILE" -o IdentitiesOnly=yes \
  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -o ConnectTimeout=20 -o LogLevel=ERROR)

echo "v4-64 cleanup: attempt exited $rc; fencing stale work on all eight hosts"
pids=()
for rank in "${!worker_ips[@]}"; do
  role="${roles[$rank]:-}"
  if [[ -z "$role" ]]; then
    echo "v4-64 cleanup: rank $rank has no role; skipping" >&2
    continue
  fi
  printf -v command \
    'V4_64_FAILURE_CLEANUP=1 V4_64_HOST_ROLE=%q V4_64_HOST_RANK=%q SKYRL_REPO_DIR=%q HF_MODEL_CACHE_DIR=%q HF_MODEL_CACHE_GCS=%q MAXTEXT_MODEL_CACHE_DIR=%q TUNIX_JAX_CACHE_LOCAL=%q VLLM_XLA_CACHE_PATH=%q bash %q' \
    "$role" "$rank" "$REPO" "$HF_ROOT/hub/$HF_MODEL_DIR" \
    "$HF_CACHE_GCS/$HF_MODEL_DIR" "$CKPT_ROOT/$MAXTEXT_MODEL" \
    "$JAX_CACHE_LOCAL" "$VLLM_CACHE_LOCAL" \
    "$REPO/tpu/swarm/reconcile_v4_64_host_role.sh"
  timeout 300 ssh "${SSHO[@]}" "$REMOTE_USER@${worker_ips[$rank]}" "$command" \
    2>&1 | sed "s/^/[rank $rank] /" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if (( failed )); then
  echo "v4-64 cleanup: one or more hosts could not be cleaned" >&2
else
  echo "v4-64 cleanup: all hosts fenced"
fi
exit 0
