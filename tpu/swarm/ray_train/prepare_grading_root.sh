#!/usr/bin/env bash
# Build the offline runtime for della CPU grading farms (FARM_CAPABILITIES.md).
#
# Run on a host WITH network (della-milan). Farm hosts (della compute nodes) have none: their SkyPilot setup
# only points at what this leaves under SCRATCH:
#   $SCRATCH/envs/controller        Ray 2.58 + farm pins (profile controller_env)
#   $SCRATCH/code/<bundle sha12>/   extracted farm science bundle = $code = SCIENCE_WORKER_ROOT, with
#       .science/venv               routing grader venv     .science/cargo, rustup  Rust toolchain + vendored crates
#       .science/router-target      router build cache      .science/venv-placement, candidate-venv
#       .science/placement-inputs   verified inputs         tpu/science/fast_proxy/libproxy.so (built helper)
#   $SCRATCH/farms/<node>/          per-node farm roots (profile root), created by the farm itself
# Qubit candidate builds run `cargo build --offline` against the vendored crates; nothing fetches pip, cargo,
# GCS or HF at runtime.
#
# Usage: prepare_grading_root.sh BUNDLE [SCRATCH] [BUCKET]
#   BUNDLE   science-training-<sha>.tar.gz of an RQ4 farm profile (tpu.science.package_training), same tree
#   SCRATCH  default /scratch/gpfs/ZHUANGL/sk7524/skypilot-slurm-cpu-scratch
#   BUCKET   where the routing CPU runtime bundle lives (default us-east5)
# Needs a uv that can install Python 3.12.12 (uv >= 0.9) and gcloud.
set -euo pipefail
bundle=$(realpath "$1") scratch=${2:-/scratch/gpfs/ZHUANGL/sk7524/skypilot-slurm-cpu-scratch}
bucket=${3:-gs://sk7524-tinker-tpu-us-east5}
code="$scratch/code/$(sha256sum "$bundle" | cut -c1-12)"
mkdir -p "$code" "$scratch/farms" && cd "$code"
export PATH="$HOME/.local/bin:$PATH" SCIENCE_WORKER_ROOT="$code" SCIENCE_GRADING_ISOLATION=bwrap
[ -f .bundle-extracted ] || { tar -xzf "$bundle" -C "$code" && sha256sum "$bundle" > .bundle-extracted; }
export PYTHONPATH="$code:$code/tpu"
python3 -c 'import sys; from pathlib import Path; from tpu.swarm.ray_train.bootstrap import ensure_controller_runtime
ensure_controller_runtime(Path(sys.argv[1]), runtime=Path(sys.argv[1]) / "envs/controller")' "$scratch"
py="$scratch/envs/controller/bin/python"
if [ ! -f .science/routing-prepared ]; then
  sha=$(sed -n "s/^CPU_BUNDLE_SHA256 = '\(.*\)'/\1/p" tpu/science/package_training.py)
  CLOUDSDK_STORAGE_CHECK_HASHES=never gcloud storage cp "$bucket/code-bundles/science-cpu-$sha.tar.gz" cpu-runtime.tar.gz
  echo "$sha  cpu-runtime.tar.gz" | sha256sum -c -
  tar -xzf cpu-runtime.tar.gz -C "$code" .science/routing-task .science/rustup .science/cargo && rm cpu-runtime.tar.gz
  bash tpu/science/prepare_cpu_host.sh
  touch .science/routing-prepared
fi
# Placement on a host that also grades routing: its own venv (the locks disagree) and the farms' helper.
SCIENCE_PLACEMENT_VENV=venv-placement SCIENCE_PLACEMENT_BACKEND=cpu SCIENCE_PLACEMENT_RUNTIME=cpu300-4g-v1 \
SCIENCE_PLACEMENT_HELPER=fast_proxy_v1 SCIENCE_PLACEMENT_SLOTS_PER_HOST=1 SCIENCE_PLACEMENT_CPUS_PER_CASE=2 \
SCIENCE_PLACEMENT_MEMORY_GIB=2 SKYPILOT_NODE_RANK=0 PLACEMENT_TPU_RANKS=0 "$py" -m tpu.science.prepare_placement_host
"$py" -c 'import os; from tpu.swarm.ray_train.grading_service import grader_sha256; print("code", os.getcwd(), "grader_sha256", grader_sha256())'
