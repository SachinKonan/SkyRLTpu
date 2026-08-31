#!/usr/bin/env bash
# Build the SkyRL worker bundle used only by TPUSwarm pools.
set -euo pipefail

REPO=$(git rev-parse --show-toplevel)
BUNDLE_URL=${1:-gs://sk7524-tinker-tpu-us-east5/code-bundles/tpuswarm-skyrl-v1.tar.gz}
STAGING=$(mktemp -d)
ARCHIVE="$STAGING/tpuswarm-skyrl.tar.gz"

cleanup() {
  rm -rf -- "$STAGING"
}
trap cleanup EXIT

# Keep credentials and local runtime state out of the shared worker artifact.
# Runtime credentials belong in SkyPilot `secrets`, not in a checked-out .env.
tar -czf "$ARCHIVE" -C "$REPO" \
  --exclude='.git' \
  --exclude='*/.git' \
  --exclude='.venv*' \
  --exclude='*/.venv*' \
  --exclude='.env' \
  --exclude='*/.env' \
  --exclude='.env.*' \
  --exclude='*/.env.*' \
  --exclude='__pycache__' \
  --exclude='*/__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='*/.pytest_cache' \
  --exclude='.ruff_cache' \
  --exclude='*/.ruff_cache' \
  --exclude='skyrl.egg-info' \
  --exclude='runs' \
  --exclude='*/runs' \
  --exclude='results' \
  --exclude='benchmark_artifacts' \
  --exclude='wandb' \
  --exclude='*/wandb' \
  --exclude='*.log' \
  --exclude='*.tar.gz' \
  .

SHA256=$(sha256sum "$ARCHIVE" | awk '{print $1}')
if [ "${TPUSWARM_BUNDLE_DRY_RUN:-0}" = "1" ]; then
  tar -tzf "$ARCHIVE" >/dev/null
  echo "TPUSwarm SkyRL bundle validated without upload"
  echo "sha256=$SHA256"
  echo "bytes=$(stat -c %s "$ARCHIVE")"
  exit 0
fi
gcloud storage cp "$ARCHIVE" "$BUNDLE_URL"
GENERATION=$(gcloud storage objects describe "$BUNDLE_URL" \
  --format='value(generation)')

echo "TPUSwarm SkyRL bundle uploaded"
echo "url=$BUNDLE_URL"
echo "sha256=$SHA256"
echo "generation=$GENERATION"
