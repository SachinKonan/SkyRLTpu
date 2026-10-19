#!/bin/bash
# run_local.sh <run_id> <spec.json>: build and start one SimpleTES baseline run on della, in the background.
# Same machinery as an RQ4 no-train run (rq4-local/scripts/prep_local.sh + run_local.sh): a local ingress
# leases this model's farms from rq4-local/farms.json and grades on them; the child is the SimpleTES driver.
# Everything (package, source cache, outputs) lives under rq4-local/simpletes/. Optional env:
# FARM_TARGET (farms leased), FARM_GROUP_CAP (requests in flight per farm).
set -euo pipefail
run=$1; spec=$(readlink -f "$2")
S=$(cd "$(dirname "$0")/../.." && pwd)
L=/scratch/gpfs/ZHUANGL/sk7524/rq4-local
root=$L/simpletes/$run
mkdir -p "$root"
if [ -f "$root/runner.pid" ] && kill -0 "$(cat "$root/runner.pid")" 2>/dev/null; then
  echo "$run already running (pid $(cat "$root/runner.pid"))"; exit 1
fi
cd "$S"
/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-multihost/third_party/TPUSwarm/.venv/bin/python -m tpu.swarm.ray_train.build \
  "tpu/swarm/ray_train/profiles/$run.json" --output "$root/build" > "$root/build.log"
bundle="$root/build/ray-training.tar.gz"
[ -f "$root/build/science-training.tar.gz" ] && [ ! "$root/build/science-training.tar.gz" -ot "$bundle" ] && bundle="$root/build/science-training.tar.gz"
rm -rf "$root/pkg"; mkdir -p "$root/pkg"; tar -xzf "$bundle" -C "$root/pkg"
profile="$root/pkg/tpu/swarm/ray_train/profiles/$run.json"
key=$(cd "$root/pkg/tpu/swarm/ray_train" && { find source_overlay thinking_budget -type f | sort | xargs sha256sum; \
      python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['model_preset'])" "$profile"; } | sha256sum | cut -c1-16)
src=$L/simpletes/sources/$key
if [ ! -f "$src/.source-complete" ]; then
  rm -rf "$src-partial"; mkdir -p "$src-partial"
  tar -xzf $L/base-v22.tar.gz -C "$src-partial"
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
print('source ready', staging)
EOF
  rm -rf "$src"; mv "$src-partial" "$src"; touch "$src/.source-complete"
fi
ln -sfn "$src" "$root/source"
model=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['model_preset'])" "$profile")
case $model in
  qwen3.5-27b) hf=models--Qwen--Qwen3.5-27B;;
  gemma4-31b) hf=models--google--gemma-4-31B-it;;
  muse-glimmer-30b) hf=models--meta-models--Muse-Glimmer-30B;;
esac
snapshot=$L/hf/hub/$hf/snapshots/$(cat $L/hf/hub/$hf/refs/main)
cd "$root/pkg"
PYTHONPATH="$root/pkg" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 setsid nohup $L/envs/controller/bin/python -u \
  -m tpu.swarm.ray_train.local_notrain --config "$profile" --registry $L/farms.json --package "$root/pkg" \
  --source "$(readlink -f "$root/source")" --python $L/envs/client/bin/python --hf-home $L/hf --snapshot "$snapshot" \
  --driver simpletes --simpletes-spec "$spec" --simpletes-root "$S" \
  ${FARM_GROUP_CAP:+--farm-group-cap $FARM_GROUP_CAP} ${FARM_TARGET:+--farm-target $FARM_TARGET} \
  >> "$root/runner.log" 2>&1 < /dev/null &
echo $! > "$root/runner.pid"
echo "$run pid=$(cat "$root/runner.pid")"
