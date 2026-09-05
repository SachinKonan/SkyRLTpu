#!/usr/bin/env bash
# jobman sync hook (workers: 0): push the cell's run dir to GCS. Runs from the
# monitor loop every SYNC_EVERY_SECONDS AND from job.py's finally-block on every
# incomplete attempt -- the state that weight-resume + tree-resume need is
# durable even when an attempt dies mid-step. Same exclusion filter as the
# sidecar (wandb + partial tmp files).
set -euo pipefail
: "${CELL:?}"
RUN="${RUN_DIR_NAME:-stageA-$CELL}"
GCS_RUN="${GCS_RUN:-gs://sk7524-tinker-tpu-us-east5/skyrl-runs/$RUN}"
[ -d "$HOME/skyrl-runs/$RUN" ] || { echo "no local run dir yet"; exit 0; }
gsutil -m rsync -r -x '.*wandb/.*|.*\.tmp$|.*\.gstmp$' \
  "$HOME/skyrl-runs/$RUN" "$GCS_RUN" >> "$HOME/cell-sync.log" 2>&1
echo "sync-rc=$? $(date -u +%H:%M:%S)" >> "$HOME/cell-sync.log"
# LoRA/optimizer checkpoints: durable for free under a gcsfuse mount (jobman),
# local disk on pool workers -- publish additively so a resume elsewhere can
# find them (see launch_cell.sh).
CKPT_ROOT="${REMOTE_CHECKPOINTS:-$HOME/gcs/skyrl-checkpoints}"
_bucket=${GCS_RUN#gs://}; _bucket=${_bucket%%/*}
SKYRL_CKPT_GCS="${SKYRL_CKPT_GCS:-gs://$_bucket/skyrl-checkpoints}"
if [ -d "$CKPT_ROOT" ] && ! mountpoint -q "$CKPT_ROOT" 2>/dev/null && ! mountpoint -q "$(dirname "$CKPT_ROOT")" 2>/dev/null; then
  # gcloud rsync (multi-GB tarballs make gsutil fall back to pure-Python CRC).
  bash "$(dirname "$0")/../gcs_rsync.sh" -r --exclude='.*\.tmp$|.*\.gstmp$|.*\.partial$' \
    "$CKPT_ROOT" "$SKYRL_CKPT_GCS" >> "$HOME/cell-sync.log" 2>&1
  echo "ckpt-writeback-rc=$? $(date -u +%H:%M:%S)" >> "$HOME/cell-sync.log"
  # Drop tarballs already in GCS (newest two per model stay): local disk is not
  # the durable store and ~3 GB/step would fill the boot disk before step 15.
  bash "$(dirname "$0")/prune_local_checkpoints.sh" "$CKPT_ROOT" "$SKYRL_CKPT_GCS" >> "$HOME/cell-sync.log" 2>&1
fi
