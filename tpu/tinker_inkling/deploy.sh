#!/usr/bin/env bash
# Deploy a pinned snapshot of the Inkling Tinker client for the supervised runs on della-vis1.
#   deploy.sh SKYRL_COMMIT DISCOVER_COMMIT     (explicit SHAs or branch names; "HEAD" is refused: it would resolve
#                                              in the main checkout, not the worktree the change was made in)
# Unpacks each commit into $BASE/code/{skyrl,discover}-<sha12> (git archive, read-only copy), points the
# $BASE/code/{skyrl,discover} symlinks at them, copies these ops scripts to $BASE/ops, and appends a manifest line.
# Runs (and their restarts) execute whatever the symlinks point at; do not redeploy mid-experiment unless intended.
set -euo pipefail
BASE=/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling
SKYRL_REPO=/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-multihost
DISCOVER_REPO=/scratch/gpfs/ZHUANGL/sk7524/discover-tinker-inkling
case "$1$2" in *HEAD*) echo "refusing HEAD: pass explicit commits"; exit 2;; esac
SK=$(git -C "$SKYRL_REPO" rev-parse "$1^{commit}"); DC=$(git -C "$DISCOVER_REPO" rev-parse "$2^{commit}")
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
  # Sanity: the snapshot must contain what the runs execute before the link is switched.
  if [ "$name" = skyrl ] && [ ! -f "$dest/tpu/tinker_inkling/launch_run.sh" ]; then echo "skyrl $sha has no tpu/tinker_inkling"; exit 1; fi
  if [ "$name" = discover ] && [ ! -f "$dest/ttt_discover/rl/ensemble.py" ]; then echo "discover $sha is not the client"; exit 1; fi
  ln -sfn "$dest" "$BASE/code/$name"
done
cp "$BASE/code/skyrl/tpu/tinker_inkling/"*.sh "$BASE/code/skyrl/tpu/tinker_inkling/"*.py "$BASE/ops/"
chmod u+w "$BASE/ops/"*
echo "$(date -u +%FT%TZ) skyrl=$SK discover=$DC" >> "$BASE/ops/DEPLOYED"
tail -1 "$BASE/ops/DEPLOYED"
