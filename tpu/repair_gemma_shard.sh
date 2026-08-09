#!/bin/bash
# Repair a gemma vLLM worker whose local HF cache holds only a partial
# (.gstmp) shard-1: kill the failed server, resume the GCS copy, drop the
# partial, restart vLLM with xet disabled. Idempotent.
set -u
GREV=842da3794eaa0b77d5f08bae87a17459d91ff475
SRC="gs://sk7524-tinker-tpu-us-east5/hf-cache-gemma4/models--google--gemma-4-31B-it/snapshots/$GREV/model-00001-of-00002.safetensors"
SNAP="$HOME/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/$GREV"
DST="$SNAP/model-00001-of-00002.safetensors"

echo "host=$(hostname) free-before=$(df -h / | tail -1 | awk '{print $4}')"
tmux kill-session -t vllm-tpu 2>/dev/null
pkill -f vllm 2>/dev/null
sleep 3

if [ -s "$DST" ]; then
  echo "shard1 already present: $(stat -c %s "$DST")"
else
  echo "resuming shard1 download..."
  gsutil -q cp "$SRC" "$DST" 2>&1 | tail -3
  echo "cp-rc=$?"
fi
rm -f "$SNAP"/*_.gstmp
sz=$(stat -c %s "$DST" 2>/dev/null || echo 0)
echo "shard1-bytes=$sz free-after=$(df -h / | tail -1 | awk '{print $4}')"
if [ "$sz" -lt 40000000000 ]; then
  echo "REPAIR-FAILED: shard1 too small"
  exit 1
fi

export HF_HUB_DISABLE_XET=1
bash "$HOME/start_vllm_tpu_bootstrap.sh" >> "$HOME/skyrl-logs/repair-restart.log" 2>&1
sleep 5
tmux ls 2>/dev/null | grep -c vllm-tpu
echo "REPAIR-DONE"
