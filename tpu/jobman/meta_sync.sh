#!/usr/bin/env bash
# Push all three member run dirs to GCS (sidecars also do this; this is the
# synchronous push the monitor calls on exit paths).
set -uo pipefail
: "${ARM:?}"; : "${GEN:?}"
for tag in qwen gemma muse; do
  run="${ARM}-g${GEN}-${tag}"
  [ -d ~/skyrl-runs/"$run" ] || continue
  gsutil -m rsync -r -x '.*wandb/.*|.*\.tmp$|.*\.gstmp$' \
    ~/skyrl-runs/"$run" "gs://sk7524-tinker-tpu-us-east5/skyrl-runs/$run" >> ~/meta-sync.log 2>&1 \
    || echo "sync $run failed (see ~/meta-sync.log)"
done
