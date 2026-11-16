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
done
# Publish the PAIR in one atomic step: release-<sk>-<dc>/{skyrl,discover} and a single rename of the `current` link,
# so a run launched mid-deploy can never pin new skyrl with old discover (launch_run.sh pins through `current`).
REL="$BASE/code/release-${SK:0:12}-${DC:0:12}"
if [ ! -d "$REL" ]; then
  mkdir -p "$REL.tmp"
  ln -sfn "$BASE/code/skyrl-${SK:0:12}" "$REL.tmp/skyrl"
  ln -sfn "$BASE/code/discover-${DC:0:12}" "$REL.tmp/discover"
  mv "$REL.tmp" "$REL"
fi
ln -sfn "$REL" "$BASE/code/.current.tmp" && mv -Tf "$BASE/code/.current.tmp" "$BASE/code/current"
# convenience links for people (runs do not use them)
ln -sfn "$BASE/code/skyrl-${SK:0:12}" "$BASE/code/skyrl"
ln -sfn "$BASE/code/discover-${DC:0:12}" "$BASE/code/discover"
# Replace each ops file atomically (new inode): a running bash script keeps reading its old copy instead of
# bytes of the new one (an in-place cp corrupted the running launch_run.sh wrappers on 2026-11-16).
for f in "$BASE/code/skyrl/tpu/tinker_inkling/"*.sh "$BASE/code/skyrl/tpu/tinker_inkling/"*.py; do
  cp "$f" "$BASE/ops/.$(basename "$f").tmp" && mv -f "$BASE/ops/.$(basename "$f").tmp" "$BASE/ops/$(basename "$f")"
done
chmod u+w "$BASE/ops/"*
echo "$(date -u +%FT%TZ) skyrl=$SK discover=$DC" >> "$BASE/ops/DEPLOYED"
tail -1 "$BASE/ops/DEPLOYED"
