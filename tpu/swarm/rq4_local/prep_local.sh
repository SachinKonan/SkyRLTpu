#!/bin/bash
# prep_local.sh <run_id>: build the run's package and its runtime source (base bundle + overlays), like Host.source_ready.
set -euo pipefail
run=$1
L=/scratch/gpfs/ZHUANGL/sk7524/rq4-local
W=/scratch/gpfs/ZHUANGL/sk7524/SkyRL-Main-infer-release
T=/scratch/gpfs/ZHUANGL/sk7524/rq4-local/scripts
root=$L/runs/$run
# The client imports its runtime (grading transport included) from this base, so the source key names it.
base=$L/base-v27.tar.gz
$T/build.sh "$run" > /dev/null
rm -rf "$root/pkg"; mkdir -p "$root/pkg"
# Science runs (qubit/circuit) build science-training.tar.gz: the generic package plus the grader/environment
# modules (tpu/science/training_env.py, ...). Use it whenever this build produced it.
out="$W/.rq4/builds/$run"; bundle="$out/ray-training.tar.gz"
[ -f "$out/science-training.tar.gz" ] && [ ! "$out/science-training.tar.gz" -ot "$bundle" ] && bundle="$out/science-training.tar.gz"
tar -xzf "$bundle" -C "$root/pkg"
profile="$root/pkg/tpu/swarm/ray_train/profiles/$run.json"
key=$(cd "$root/pkg/tpu/swarm/ray_train" && { basename "$base"; find source_overlay thinking_budget -type f | sort | xargs sha256sum; \
      python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['model_preset'])" "$profile"; } | sha256sum | cut -c1-16)
src=$L/sources/$key
if [ ! -f "$src/.source-complete" ]; then
  rm -rf "$src-partial"; mkdir -p "$src-partial"
  tar -xzf "$base" -C "$src-partial"
  PYTHONPATH="$root/pkg" $L/envs/controller/bin/python - "$profile" "$root/pkg" "$src-partial" <<'EOF'
import sys
from pathlib import Path
from tpu.swarm.ray_train.config import Config
from tpu.swarm.ray_train.overlay import install
from tpu.swarm.ray_train.thinking_budget.install import install as install_thinking
config, package, staging = Config.load(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
if config.requires_source_overlay:
    install(package / 'tpu/swarm/ray_train/source_overlay', staging)
if config.inference.native_thinking_budget:
    install_thinking(staging, model=config.native_thinking_format, require_client=True)
for path in ('tpu/run_ttd_ensemble.py', 'third_party/discover/ttt_discover/rl/ensemble.py'):
    assert (staging / path).is_file(), path
print('source ready', staging)
EOF
  rm -rf "$src"; mv "$src-partial" "$src"; touch "$src/.source-complete"
fi
ln -sfn "$src" "$root/source"
echo "$run source=$key bundle=$(basename "$bundle")"
