#!/usr/bin/env bash
# jobman sync hook (workers: 0): push the cell's run dir to GCS. Runs from the
# monitor loop every SYNC_EVERY_SECONDS AND from job.py's finally-block on every
# incomplete attempt -- the state that weight-resume + tree-resume need is
# durable even when an attempt dies mid-step. Same exclusion filter as the
# sidecar (wandb + partial tmp files).
set -euo pipefail
: "${CELL:?}"
RUN="stageA-$CELL"
GCS_RUN="${GCS_RUN:-gs://sk7524-tinker-tpu-us-east5/skyrl-runs/$RUN}"
[ -d "$HOME/skyrl-runs/$RUN" ] || { echo "no local run dir yet"; exit 0; }
gsutil -m rsync -r -x '.*wandb/.*|.*\.tmp$|.*\.gstmp$' \
  "$HOME/skyrl-runs/$RUN" "$GCS_RUN" >> "$HOME/cell-sync.log" 2>&1
echo "sync-rc=$? $(date -u +%H:%M:%S)" >> "$HOME/cell-sync.log"
