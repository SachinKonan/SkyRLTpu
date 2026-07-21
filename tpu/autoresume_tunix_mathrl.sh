#!/usr/bin/env bash
# Autoresume watcher for the tunix math-RL machine: recreate the spot TPU via
# jobman when it disappears, run the colocated bring-up, resume the RL client,
# and loop on preemption. Miniature of the PR#2 autostart pattern.
set -uo pipefail

TPU_NAME="sk7524-tunix-mathrl-v5p16-r12-east5a_spot"
ZONE="us-east5-a"
PROJECT="vision-mix"
JOBMAN_CONFIG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/skyrl_tunix_v5p16_spot.yaml"
REPO="/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-tunix-backend"
RESUME_LOG_PATH="$REPO/runs/math_rl/math-Qwen3-8B-tunix-take5b-stream-2026-07-15-17-23"

log() { echo "$(date -u +%H:%M:%S) [watcher] $*"; }

node_state() {
  timeout 90 gcloud compute tpus tpu-vm list --zone "$ZONE" --project "$PROJECT" \
    --filter="name=${TPU_NAME}" --format="value(state)" 2>/dev/null | head -1
}

ssh_ok() {
  timeout 60 gcloud alpha compute tpus tpu-vm ssh "sk7524_princeton_edu@${TPU_NAME}" \
    --project="$PROJECT" --zone="$ZONE" --worker=0 --ssh-key-file="$HOME/.ssh/jobman_tpu_ed25519" \
    --quiet --command 'true' >/dev/null 2>&1
}

attempt=0
while true; do
  state="$(node_state)"
  if [[ "$state" != "READY" ]]; then
    attempt=$((attempt+1))
    log "node absent/not-ready (state='${state:-none}'); creating via jobman (attempt $attempt)"
    jobman create "$JOBMAN_CONFIG" 2>/dev/null | grep -E "Created job|error" || true
    # wait for READY + ssh (QR fulfillment can take a while)
    for i in $(seq 1 240); do
      sleep 30
      [[ "$(node_state)" == "READY" ]] && ssh_ok && break
    done
    if [[ "$(node_state)" != "READY" ]]; then
      log "still not ready after wait; looping"
      continue
    fi
    # let jobman's setup command finish (gcsfuse etc. run right after create)
    sleep 120
  fi

  log "node READY; running colocated bring-up"
  cd "$REPO"
  if ! TPU_NAME="$TPU_NAME" TINKER_BACKEND=tunix MODEL_NAME=Qwen/Qwen3-8B \
       TRAIN_WORKERS=0 VLLM_WORKERS=1 TRAIN_MICRO_BATCH_SIZE=32 \
       ./tpu/start_colocated_vllm_tinker.sh >> "$REPO/runs/math_rl/r12-bringup.log" 2>&1; then
    log "bring-up failed (see r12-bringup.log); rechecking node in 60s"
    sleep 60
    continue
  fi

  log "bring-up done; resuming RL client (resume from last checkpoint)"
  TPU_NAME="$TPU_NAME" MODEL_NAME=Qwen/Qwen3-8B CLIENT_SESSION=skyrl-math-rl-tunix \
    REPLACE_CLIENT=1 STREAM_NUM_MINIBATCHES=8 BEHAVIOR_IF_LOG_DIR_EXISTS=resume \
    LOG_PATH="$RESUME_LOG_PATH" \
    ./tpu/run_tinker_math_rl_qwen35_9b.sh >> "$REPO/runs/math_rl/r12-bringup.log" 2>&1 \
    || log "client launch failed"

  log "watching node health"
  while [[ "$(node_state)" == "READY" ]]; do
    sleep 120
  done
  log "node lost (preemption); tearing down client session and looping"
  tmux kill-session -t skyrl-math-rl-tunix 2>/dev/null || true
done
