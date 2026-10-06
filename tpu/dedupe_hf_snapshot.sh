#!/usr/bin/env bash
# Reclaim the blob/snapshot duplication a GCS-restored HF cache carries.
#
#   dedupe_hf_snapshot.sh <hub>/models--<org>--<name> [...]
#
# A native HF cache stores each weight file once under blobs/<sha> and points
# snapshots/<rev>/<file> at it with a symlink. GCS has no symlinks, so the
# seeded cache holds both as real objects and `gcloud storage cp` restores both:
# gemma-4-31B-it came back as 71 GB (46.4 + 11.9 in snapshots, 11.9 again in
# blobs) on a 97 GB engine disk that also carries the vLLM venv, the compile
# cache and /var -- 0 bytes free, LoRA uploads 500, cell dead (job 245,
# 2026-09-05). vLLM only reads snapshots/, so every blob whose size matches a
# regular snapshot file is replaced by a hardlink to that file: same paths,
# same bytes, one copy on disk. Safe to run repeatedly and on a live engine.
set -uo pipefail
freed=0
for model_dir in "$@"; do
  [ -d "$model_dir/blobs" ] && [ -d "$model_dir/snapshots" ] || continue
  while IFS= read -r -d '' blob; do
    size=$(stat -c %s "$blob" 2>/dev/null) || continue
    [ "$size" -ge 1048576 ] || continue            # only weight-sized files matter
    match=$(find "$model_dir/snapshots" -type f -size "${size}c" -print -quit 2>/dev/null)
    [ -n "$match" ] || continue
    if [ "$(stat -c %i "$match")" = "$(stat -c %i "$blob")" ]; then
      continue                                      # already one inode
    fi
    if ln -f "$match" "$blob"; then
      freed=$(( freed + size ))
      echo "dedupe: $(basename "$blob") -> hardlink of snapshots/.../$(basename "$match") ($(( size / 1048576 )) MB)"
    fi
  done < <(find "$model_dir/blobs" -type f -print0 2>/dev/null)
done
echo "dedupe: reclaimed $(( freed / 1048576 )) MB"
