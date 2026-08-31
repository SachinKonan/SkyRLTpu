#!/usr/bin/env bash
# One-time Qwen cache seed from a Neuronic Slurm CPU allocation.
set -euo pipefail

: "${SLURM_JOB_ID:?run this script with srun on the cpu partition}"

CACHE_GCS="${HF_CACHE_GCS:-gs://sk7524-tinker-tpu-asia-northeast1/hf-cache-qwen35-v1}"
MODEL_NAME="Qwen/Qwen3.5-27B"
if gcloud storage objects describe "$CACHE_GCS/HF_CACHE_COMPLETE" \
  --format='value(name)' >/dev/null 2>&1; then
  echo "Qwen HF cache is already complete at $CACHE_GCS"
  exit 0
fi

scratch_parent="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
mkdir -p "$scratch_parent"
stage_root=$(mktemp -d "$scratch_parent/qwen35-hf-cache.XXXXXX")
cleanup() {
  rm -rf -- "$stage_root"
}
trap cleanup EXIT

export HF_HOME="$stage_root/huggingface"
export HF_XET_HIGH_PERFORMANCE=1
venv="$stage_root/venv"
uv venv "$venv" --python 3.11
uv pip install --python "$venv/bin/python" \
  'huggingface_hub>=0.34,<2' hf_transfer

"$venv/bin/python" - "$stage_root/snapshot-path" <<'PY'
import pathlib
import sys

from huggingface_hub import snapshot_download

snapshot = snapshot_download("Qwen/Qwen3.5-27B")
pathlib.Path(sys.argv[1]).write_text(snapshot + "\n", encoding="utf-8")
PY

snapshot=$(cat "$stage_root/snapshot-path")
"$venv/bin/python" - "$snapshot" <<'PY'
import json
import pathlib
import sys

snapshot = pathlib.Path(sys.argv[1])
index_path = snapshot / "model.safetensors.index.json"
index = json.loads(index_path.read_text(encoding="utf-8"))
shards = sorted(set(index["weight_map"].values()))
missing = [name for name in shards if not (snapshot / name).is_file()]
if missing:
    raise SystemExit(f"incomplete Qwen snapshot; missing shards: {missing}")
print(f"validated Qwen snapshot: {len(shards)} safetensor shards")
PY

# Upload the standard Hugging Face hub layout.  gcloud follows the snapshot
# symlinks, so restored TPU caches remain usable even though GCS has no symlink
# object type.
gcloud storage rsync -r "$HF_HOME/hub" "$CACHE_GCS"
{
  printf 'model=%s\n' "$MODEL_NAME"
  printf 'snapshot=%s\n' "$(basename "$snapshot")"
  printf 'seeded_at=%s\n' "$(date -u +%FT%TZ)"
} > "$stage_root/HF_CACHE_COMPLETE"
gcloud storage cp "$stage_root/HF_CACHE_COMPLETE" \
  "$CACHE_GCS/HF_CACHE_COMPLETE"
echo "Qwen HF cache seeded at $CACHE_GCS"
