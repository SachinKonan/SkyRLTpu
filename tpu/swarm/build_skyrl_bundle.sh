#!/usr/bin/env bash
# Build the SkyRL worker bundle used only by TPUSwarm pools.
set -euo pipefail

REPO=$(git rev-parse --show-toplevel)
if [ "$#" -ne 1 ]; then
  echo "usage: $0 gs://BUCKET/code-bundles/BUNDLE.tar.gz" >&2
  exit 2
fi
BUNDLE_URL=$1
STAGING=$(mktemp -d)
ARCHIVE="$STAGING/tpuswarm-skyrl.tar.gz"
MANIFEST="$STAGING/.tpuswarm-bundle-manifest"

cleanup() {
  rm -rf -- "$STAGING"
}
trap cleanup EXIT

PARENT_COMMIT=$(git -C "$REPO" rev-parse HEAD)
if [ -n "$(git -C "$REPO" status --porcelain --untracked-files=normal)" ]; then
  if [ "${TPUSWARM_BUNDLE_ALLOW_DIRTY:-0}" != "1" ]; then
    echo "refusing to publish an uncommitted source bundle" >&2
    echo "commit the parent repository, or set TPUSWARM_BUNDLE_ALLOW_DIRTY=1 for a local dry run" >&2
    exit 2
  fi
  SOURCE_DIRTY=true
else
  SOURCE_DIRTY=false
fi
{
  printf 'parent_commit=%s\n' "$PARENT_COMMIT"
  printf 'source_dirty=%s\n' "$SOURCE_DIRTY"
  printf 'built_at=%s\n' "$(date -u +%FT%TZ)"
  git -C "$REPO" submodule status --recursive | sed 's/^/submodule=/'
} > "$MANIFEST"

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
  --exclude='.mypy_cache' \
  --exclude='*/.mypy_cache' \
  --exclude='skyrl.egg-info' \
  --exclude='runs' \
  --exclude='*/runs' \
  --exclude='results' \
  --exclude='benchmark_artifacts' \
  --exclude='wandb' \
  --exclude='*/wandb' \
  --exclude='*.log' \
  --exclude='*.tar.gz' \
  . \
  -C "$STAGING" .tpuswarm-bundle-manifest

# Validate the files consumed by pool setup from the archive itself.  Runtime
# invokes these through bash or python, so readability is the relevant mode.
REQUIRED_BUNDLE_FILES=(
  .tpuswarm-bundle-manifest
  third_party/TPUSwarm/pyproject.toml
  tpu/gcs_rsync.sh
  tpu/launch_cell.sh
  tpu/jobman/cell_monitor.sh
  tpu/jobman/cell_sync.sh
  tpu/jobman/cell_worker.sh
  tpu/jobman/grader_ray.sh
  tpu/jobman/v6e_tunix_smoke_worker.sh
  tpu/probe_topology.py
  tpu/start_colocated_vllm_tinker.sh
  tpu/swarm/discover_v4_64_topology.sh
  tpu/swarm/prepare_qwen35_v6e32.sh
  tpu/swarm/prune_hf_weight_cache.py
  tpu/swarm/reconcile_v4_64_host_role.sh
  tpu/swarm/reconcile_v4_64_role_caches.sh
  tpu/swarm/run_erdos_min_overlap.sh
  tpu/swarm/run_qwen35_v4_64_grpo.sh
  tpu/swarm/run_qwen35_v6e32_grpo.sh
  tpu/swarm/run_v5p32_cell.sh
  tpu/swarm/select_v4_64_topology.py
  tpu/swarm/stage_hf_metadata_cache.py
)
VERIFY_ROOT="$STAGING/verify"
mkdir -p "$VERIFY_ROOT"
archive_paths=()
for required in "${REQUIRED_BUNDLE_FILES[@]}"; do
  if [ "$required" = ".tpuswarm-bundle-manifest" ]; then
    archive_paths+=("$required")
  else
    archive_paths+=("./$required")
  fi
done
tar -xzf "$ARCHIVE" -C "$VERIFY_ROOT" -- "${archive_paths[@]}"
for required in "${REQUIRED_BUNDLE_FILES[@]}"; do
  if [ ! -r "$VERIFY_ROOT/$required" ]; then
    echo "bundle validation failed: missing or unreadable $required" >&2
    exit 1
  fi
done

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
