#!/usr/bin/env bash
# Make the trainer's MaxText/orbax checkpoint VALID before any engine starts.
#
# Runs from the jobman `prepare` hook on the train coordinator (worker 0) only.
# Engine bring-up may then assume a complete checkpoint, or a deliberate absence.
#
# Why this exists. The restore used to live inside engine bring-up as one
# best-effort rsync. On spot capacity that rsync gets cut short constantly, and
# a half-restore is WORSE than none:
#
#   * partial shards -> the trainer reads them and dies with
#     "ValueError: DATA_LOSS: Error reading shard entry {0,0}/{1,1} in
#      params.params.decoder.layers.layers_0.post_ffw_norm.scale"
#   * purge the partials and the backend silently falls back to converting from
#     HF, which needs the 56 GB safetensors AND writes 40 GB of orbax on a 97 GB
#     disk -> "RESOURCE_EXHAUSTED: Error writing ... in OCDBT database"
#
# Both were observed live on muse-glimmer, two layers below where the real
# problem was (a truncated download), and each cost a full bring-up cycle.
#
# The space rule this encodes: on the TRAIN host the HF safetensors and the
# orbax checkpoint are ALTERNATIVES, not co-residents. The trainer reads orbax;
# it only needs HF weights to convert, which is exactly what a good checkpoint
# makes unnecessary. So if the checkpoint will not fit, the weights go.
set -uo pipefail

[ "${JOBMAN_WORKER_ID:-0}" = "0" ] || { echo "ckpt: worker ${JOBMAN_WORKER_ID} no-op"; exit 0; }

# The prepare hook's env carries CELL but not the model vars (those are set in
# cell_worker.sh, which runs later), so derive them here from the same prefixes.
case "${CELL:-}" in
  g-*) _MT=gemma4-31b;       _HF=google/gemma-4-31B-it ;;
  m-*) _MT=muse-glimmer-30b; _HF=meta-models/Muse-Glimmer-30B ;;
  *)   _MT=qwen3.5-27b;      _HF=Qwen/Qwen3.5-27B ;;
esac
MT_NAME="${TUNIX_MAXTEXT_MODEL_NAME:-$_MT}"
CACHE="${TUNIX_MAXTEXT_CKPT_CACHE:-$HOME/skyrl-maxtext-ckpts-local}"
CACHE_GCS="${TUNIX_MAXTEXT_CKPT_CACHE_GCS:-gs://sk7524-tinker-tpu-us-east5/skyrl-maxtext-ckpts}"
HF_MODEL="${MODEL_NAME:-$_HF}"
MARGIN_GB="${CKPT_MARGIN_GB:-12}"      # venvs, logs, XLA cache, room to breathe

[ -n "$MT_NAME" ] || { echo "ckpt: TUNIX_MAXTEXT_MODEL_NAME unset -- backend will derive/convert"; exit 0; }
GS="$(command -v gsutil || echo "$HOME/google-cloud-sdk/bin/gsutil")"
SRC="$CACHE_GCS/$MT_NAME"
DST="$CACHE/$MT_NAME"

want=$(timeout 300 "$GS" du -s "$SRC" 2>/dev/null | awk '{print $1}')
if [ -z "$want" ] || [ "$want" = "0" ]; then
  echo "ckpt: no checkpoint at $SRC -- leaving conversion-from-HF as the path"
  exit 0
fi
want_gb=$(( want / 1024 / 1024 / 1024 ))

have=$(du -sb "$DST" 2>/dev/null | awk '{print $1}'); have="${have:-0}"
parts=$(find "$DST" \( -name '*_.gstmp' -o -name '*.gstmp' \) 2>/dev/null | head -1)
# 2% tolerance: du and gsutil du disagree slightly on block accounting.
if [ -z "$parts" ] && [ "$have" -ge $(( want * 98 / 100 )) ]; then
  echo "ckpt: already complete ($(( have / 1024 / 1024 / 1024 )) GB >= 98% of ${want_gb} GB)"
  exit 0
fi
[ "$have" != "0" ] && echo "ckpt: local copy incomplete ($(( have / 1024 / 1024 / 1024 ))/${want_gb} GB, partial=${parts:-none}) -- refetching"

mkdir -p "$DST"
for try in 1 2 3; do
  free_kb=$(df -Pk "$DST" | awk 'NR==2 {print $4}')
  need_kb=$(( (want / 1024) + (MARGIN_GB * 1024 * 1024) ))
  if [ "$free_kb" -lt "$need_kb" ] && [ -n "$HF_MODEL" ]; then
    # Reclaim from the alternative, not from the thing we need.
    hf_dir="$HOME/.cache/huggingface/hub/models--${HF_MODEL//\//--}"
    if [ -d "$hf_dir" ]; then
      echo "ckpt: need $(( need_kb / 1024 / 1024 )) GB, free $(( free_kb / 1024 / 1024 )) GB -- dropping HF weights ($hf_dir); the trainer reads orbax, and vLLM hosts keep their own copy"
      rm -rf "$hf_dir"
    fi
  fi

  timeout 3600 "$GS" -m -q rsync -r "$SRC" "$DST" 2>/dev/null
  have=$(du -sb "$DST" 2>/dev/null | awk '{print $1}'); have="${have:-0}"
  parts=$(find "$DST" \( -name '*_.gstmp' -o -name '*.gstmp' \) 2>/dev/null | head -1)
  if [ -z "$parts" ] && [ "$have" -ge $(( want * 98 / 100 )) ]; then
    echo "ckpt: restored $(( have / 1024 / 1024 / 1024 ))/${want_gb} GB from $SRC (attempt $try)"
    exit 0
  fi
  echo "ckpt: attempt $try incomplete ($(( have / 1024 / 1024 / 1024 ))/${want_gb} GB, partial=${parts:-none})"
  sleep 20
done

# Never hand the trainer a torn checkpoint: absent is recoverable, corrupt is not.
n=$(find "$DST" -mindepth 1 -delete -print 2>/dev/null | wc -l)
echo "ckpt: FAILED to restore $SRC after 3 attempts; purged $n path(s). Engine bring-up will convert from HF (slow) or fail loudly." >&2
exit 0
