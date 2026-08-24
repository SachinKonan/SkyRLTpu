#!/usr/bin/env bash
# THE code-bundle builder. One implementation, whole-tree, SHA-identified.
#
# Why this exists: bundles were built ad-hoc in 7+ places (five league
# bringup_*.sh inline tars, the stagea jobman bundle, the gemma-lenprobe
# bundle), each with its own hand-curated file list. Hand-curated manifests
# re-derive `git ls-files` one missing file at a time -- the lenprobe bundle
# cost one TPU boot each for skyrl-gym (editable path dep), .python-version
# (uv falls back to system 3.11 -> maxtext>=3.12 unsatisfiable), and nearly
# the tpu/ scripts. The league never hit these because it tars the WHOLE
# tree; this does the same.
#
# Identity is the git SHA, not Content-Length: the old `.bundle-size`
# staleness check cannot tell two same-sized bundles apart (a stale-serve of
# exactly that kind bit the league's tile-fix rollout). The tar contains
# `.bundle-sha`; consumers compare it against `bundle.sha` uploaded next to
# the tarball. A dirty tree appends -dirty.<hash-of-diff> so an uncommitted
# change still changes the identity.
#
# Usage:
#   tpu/make_bundle.sh gs://bucket/code-bundles/NAME.tar.gz [repo_root]
#
# Consumer (prepare hook) pattern:
#   want=$(gsutil cat "${BUNDLE_URL%.tar.gz}.sha")
#   have=$(cat "$DEST/.bundle-sha" 2>/dev/null || echo none)
#   [ "$want" != "$have" ] && { fetch + unpack; }
set -euo pipefail

BUNDLE_URL="${1:?usage: make_bundle.sh gs://.../name.tar.gz [repo_root]}"
REPO="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
case "$BUNDLE_URL" in gs://*.tar.gz) ;; *) echo "BUNDLE_URL must be gs://...tar.gz" >&2; exit 1 ;; esac

cd "$REPO"
sha="$(git rev-parse HEAD 2>/dev/null || echo nogit)"
if ! git diff --quiet HEAD 2>/dev/null; then
  sha="${sha}-dirty.$(git diff HEAD 2>/dev/null | sha256sum | cut -c1-12)"
fi
echo "$sha" > .bundle-sha

tar_tmp="$(mktemp /tmp/bundle-XXXXXX.tar.gz)"
trap 'rm -f "$tar_tmp" .bundle-sha' EXIT
# Whole tree. Excludes are OUTPUTS and CACHES only -- never source, configs,
# or dotfiles (.python-version is load-bearing).
tar -czf "$tar_tmp" \
  --exclude='.git' --exclude='.venv' --exclude='.venv*' \
  --exclude='runs' --exclude='wandb' --exclude='*.log' \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' \
  --exclude='node_modules' --exclude='third_party/tokamax' \
  --exclude='third_party/jobman/jobs' \
  .

gsutil -q cp "$tar_tmp" "$BUNDLE_URL"
echo "$sha" | gsutil -q cp - "${BUNDLE_URL%.tar.gz}.sha"
size=$(stat -c%s "$tar_tmp")
echo "bundle: $BUNDLE_URL"
echo "sha:    $sha"
echo "size:   $(( size / 1024 / 1024 ))MB"
