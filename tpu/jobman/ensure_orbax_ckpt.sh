#!/usr/bin/env bash
# Make the trainer's MaxText/orbax checkpoint VALID before any engine starts.
#
# Runs from the jobman `prepare` hook on every selected trainer worker.
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

if ! python3 - "${JOBMAN_WORKER_ID:-0}" "${TRAIN_WORKERS:-0}" <<'PY'
import sys

worker = int(sys.argv[1])
selected = set()
for part in sys.argv[2].replace(" ", "").split(","):
    if not part:
        continue
    if "-" in part:
        lo, hi = map(int, part.split("-", 1))
        selected.update(range(lo, hi + 1))
    else:
        selected.add(int(part))
raise SystemExit(0 if worker in selected else 1)
PY
then
  echo "ckpt: worker ${JOBMAN_WORKER_ID:-0} is not in TRAIN_WORKERS=${TRAIN_WORKERS:-0}; no-op"
  exit 0
fi

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
REQUIRE_MARKER="${TUNIX_MAXTEXT_CKPT_REQUIRE_MARKER:-0}"

[ -n "$MT_NAME" ] || { echo "ckpt: TUNIX_MAXTEXT_MODEL_NAME unset -- backend will derive/convert"; exit 0; }
# gcloud storage, NOT gsutil. The checkpoint objects are composite (that is how
# `gsutil cp -r` seeds them), and composite downloads force CRC32c validation;
# without compiled crcmod on the node gsutil falls back to python hashing and
# crawls -- measured 78 KiB in 90 s against 40 GB, i.e. never finishes. The same
# transfer under `gcloud storage rsync` took 90 s flat. rs_tpu.sh already uses
# gcloud storage for exactly this reason.
GS="$(command -v gsutil || echo "$HOME/google-cloud-sdk/bin/gsutil")"
GCS_CLI="$(command -v gcloud || echo "$HOME/google-cloud-sdk/bin/gcloud")"
SRC="$CACHE_GCS/$MT_NAME"
DST="$CACHE/$MT_NAME"

if [ "$REQUIRE_MARKER" = "1" ] && ! timeout 60 "$GCS_CLI" storage objects describe \
  "$SRC/CHECKPOINT_COMPLETE" >/dev/null 2>&1; then
  echo "ckpt: required completion marker is missing at $SRC/CHECKPOINT_COMPLETE" >&2
  exit 3
fi

want=$(timeout 300 "$GS" du -s "$SRC" 2>/dev/null | awk '{print $1}')
if [ -z "$want" ] || [ "$want" = "0" ]; then
  echo "ckpt: no checkpoint at $SRC -- leaving conversion-from-HF as the path"
  exit 0
fi
want_gb=$(( want / 1024 / 1024 / 1024 ))

valid_layout() {
  [ -s "$DST/0/_CHECKPOINT_METADATA" ] && \
    [ -s "$DST/0/items/_METADATA" ] && \
    [ -s "$DST/0/items/_sharding" ] && \
    [ -s "$DST/0/items/manifest.ocdbt" ] && \
    { [ "$REQUIRE_MARKER" != "1" ] || [ -s "$DST/CHECKPOINT_COMPLETE" ]; }
}

have=$(du -sb "$DST" 2>/dev/null | awk '{print $1}'); have="${have:-0}"
parts=$(find "$DST" \( -name '*_.gstmp' -o -name '*.gstmp' \) 2>/dev/null | head -1)
# Object bytes plus local directory entries make a complete `du -sb` at least
# as large as `gsutil du`.  A percentage tolerance is unsafe for 100+ GB
# checkpoints: even 2% missing can be several corrupt tensor chunks.
if [ -z "$parts" ] && [ "$have" -ge "$want" ] && valid_layout; then
  echo "ckpt: already complete ($(( have / 1024 / 1024 / 1024 ))/${want_gb} GB)"
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

  rsync_extra=()
  if [ "$REQUIRE_MARKER" = "1" ]; then
    # The remote marker means its upload is complete. Do not copy it locally
    # until this host's data rsync also completes, or an interrupted transfer
    # could leave a false-ready local cache.
    rm -f -- "$DST/CHECKPOINT_COMPLETE"
    rsync_extra+=(--exclude='(^|/)CHECKPOINT_COMPLETE$')
  fi
  if timeout 3600 "$GCS_CLI" storage rsync -r "${rsync_extra[@]}" \
    "$SRC" "$DST" >/dev/null 2>>"$HOME/ckpt-prep-errors.log"; then
    if [ "$REQUIRE_MARKER" = "1" ]; then
      timeout 60 "$GCS_CLI" storage cp "$SRC/CHECKPOINT_COMPLETE" \
        "$DST/CHECKPOINT_COMPLETE" >/dev/null 2>>"$HOME/ckpt-prep-errors.log" || true
    fi
  fi
  have=$(du -sb "$DST" 2>/dev/null | awk '{print $1}'); have="${have:-0}"
  parts=$(find "$DST" \( -name '*_.gstmp' -o -name '*.gstmp' \) 2>/dev/null | head -1)
  if [ -z "$parts" ] && [ "$have" -ge "$want" ] && valid_layout; then
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
