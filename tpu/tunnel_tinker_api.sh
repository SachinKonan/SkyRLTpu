#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-vision-mix}"
ZONE="${ZONE:-us-east5-a}"
TPU_NAME="${TPU_NAME:-sk7524-ttd-erdos-v5p64-east5a_spot}"
REMOTE_USER="${REMOTE_USER:-sk7524_princeton_edu}"
SSH_KEY_FILE="${SSH_KEY_FILE:-$HOME/.ssh/jobman_tpu_ed25519}"
WORKER="${WORKER:-0}"
LOCAL_PORT="${LOCAL_PORT:-18000}"
REMOTE_PORT="${REMOTE_PORT:-8000}"

exec gcloud alpha compute tpus tpu-vm ssh "${REMOTE_USER}@${TPU_NAME}" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --worker="$WORKER" \
  --ssh-key-file="$SSH_KEY_FILE" \
  --quiet \
  -- -N -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}"
