#!/usr/bin/env bash
# Delete local LoRA/optimizer checkpoint tarballs that are already published.
#
#   prune_local_checkpoints.sh CKPT_ROOT CKPT_GCS [KEEP]
#
# On jobman cells CKPT_ROOT was a gcsfuse mount (the bucket itself), so there
# was nothing to prune. On SkyPilot pool workers it is local disk: the sidecar
# and cell_sync.sh publish every tarball to CKPT_GCS, but each muse step still
# leaves ~3 GB behind (2.3 GB weights + 0.77 GB sampler), which fills a 97 GB
# boot disk holding a 41 GB orbax checkpoint before step 15 (jobs 188/161,
# 2026-09-05). Nothing reads old local tarballs: a resume restores the newest
# rows from GCS (launch_cell.sh), so every tarball except the newest KEEP per
# model is deleted -- but only after its GCS object is confirmed to exist at
# the same size. Anything unpublished, partial, or still being written stays.
set -uo pipefail
CKPT_ROOT="${1:?CKPT_ROOT}"
CKPT_GCS="${2:?CKPT_GCS (gs://bucket/skyrl-checkpoints)}"
KEEP="${3:-2}"
GCLOUD="${GCLOUD_STORAGE_CLI:-$(command -v gcloud || echo "$HOME/google-cloud-sdk/bin/gcloud")}"

[ -d "$CKPT_ROOT" ] || exit 0
if mountpoint -q "$CKPT_ROOT" 2>/dev/null || mountpoint -q "$(dirname "$CKPT_ROOT")" 2>/dev/null; then
  exit 0
fi

pruned=0; kept=0; unpublished=0; freed=0
for model_dir in "$CKPT_ROOT"/model_*; do
  [ -d "$model_dir" ] || continue
  model=$(basename "$model_dir")
  for family in "" "sampler_weights/"; do
    dir="$model_dir/$family"
    [ -d "$dir" ] || continue
    # Zero-padded step names sort correctly; "final" sorts after every digit
    # name and is therefore always kept. Seed sampler snapshots (ss*) are tiny
    # and referenced for the life of the run: leave them alone.
    mapfile -t tarballs < <(cd "$dir" && ls -1 [0-9]*.tar.gz final.tar.gz 2>/dev/null | sort)
    total=${#tarballs[@]}
    (( total > KEEP )) || { kept=$((kept + total)); continue; }
    for (( i = 0; i < total - KEEP; i++ )); do
      name="${tarballs[$i]}"
      local_path="$dir$name"
      [ -f "$local_path" ] || continue
      local_size=$(stat -c %s "$local_path" 2>/dev/null || echo -1)
      remote_size=$(timeout 60 "$GCLOUD" storage ls -l "$CKPT_GCS/$model/$family$name" 2>/dev/null \
        | awk 'NR==1 {print $1}')
      if [ -n "$remote_size" ] && [ "$remote_size" = "$local_size" ] && [ "$local_size" -gt 0 ]; then
        rm -f -- "$local_path"
        pruned=$((pruned + 1)); freed=$((freed + local_size))
      else
        unpublished=$((unpublished + 1))
      fi
    done
    kept=$((kept + KEEP))
  done
done
echo "ckpt-prune: removed=$pruned freed=$(( freed / 1000000000 ))GB kept=$kept unpublished-or-mismatched=$unpublished $(date -u +%H:%M:%S)"
