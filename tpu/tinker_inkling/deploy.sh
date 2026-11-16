#!/usr/bin/env bash
# Deploy a pinned snapshot of the Inkling Tinker client for the supervised runs on della-vis1.
#   deploy.sh SKYRL_COMMIT DISCOVER_COMMIT
# Unpacks each commit into $BASE/code/{skyrl,discover}-<sha12> (git archive, read-only copy), points the
# $BASE/code/{skyrl,discover} symlinks at them, copies these ops scripts to $BASE/ops, and appends a manifest line.
# Runs (and their restarts) execute whatever the symlinks point at; do not redeploy mid-experiment unless intended.
set -euo pipefail
BASE=/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling
SKYRL_REPO=/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-multihost
DISCOVER_REPO=/scratch/gpfs/ZHUANGL/sk7524/discover-tinker-inkling
SK=$(git -C "$SKYRL_REPO" rev-parse "$1"); DC=$(git -C "$DISCOVER_REPO" rev-parse "$2")
mkdir -p "$BASE/code" "$BASE/ops"
for pair in "skyrl:$SKYRL_REPO:$SK" "discover:$DISCOVER_REPO:$DC"; do
  IFS=: read -r name repo sha <<<"$pair"
  dest="$BASE/code/$name-${sha:0:12}"
  if [ ! -d "$dest" ]; then
    mkdir -p "$dest.tmp"
    git -C "$repo" archive "$sha" | tar -x -C "$dest.tmp"
    chmod -R a-w "$dest.tmp"
    mv "$dest.tmp" "$dest"
  fi
  ln -sfn "$dest" "$BASE/code/$name"
done
cp "$BASE/code/skyrl/tpu/tinker_inkling/"*.sh "$BASE/code/skyrl/tpu/tinker_inkling/"*.py "$BASE/ops/"
chmod u+w "$BASE/ops/"*
echo "$(date -u +%FT%TZ) skyrl=$SK discover=$DC" >> "$BASE/ops/DEPLOYED"
tail -1 "$BASE/ops/DEPLOYED"
