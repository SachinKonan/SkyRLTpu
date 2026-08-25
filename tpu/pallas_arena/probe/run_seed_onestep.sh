#!/usr/bin/env bash
# Orchestrate the seeded one-step test end-to-end:
#   1. gate on CPU seed validation (seedval4: both seeds must grade correct)
#   2. gate on the parity fleet's REAL-silicon verdicts for the seeds
#   3. build per-task observation files (the judge's actual feedback) and
#      measured reward strings
#   4. launch the four 32-gen arms (qwen/gemma x rg_lru/splash)
set -uo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
cd "$REPO"
out=runs/pallas_arena

LOCK=/tmp/seed-onestep-runner.lock
exec 9>"$LOCK"
flock -n 9 || { echo "another runner holds $LOCK"; exit 0; }

echo "=== [1] waiting for CPU seed validation (seedval4) ==="
until [ -s "$out/evolve-smoke-graded-seedval4.json" ]; do sleep 60; done
python3 - <<'PY' || { echo "FATAL: a seed failed CPU validation"; exit 1; }
import json, sys
g=json.load(open("runs/pallas_arena/evolve-smoke-graded-seedval4.json"))
for cell,v in g.items():
    for r in v["rows"]:
        print(f"[seedval4] {cell}: {str(r['outcome'])[:150]}")
        if str(r["outcome"])!="correct": sys.exit(1)
PY

echo "=== [2] waiting for real-silicon parity verdicts ==="
until [ -s "$out/seed-parity-results.json" ]; do sleep 120; done

echo "=== [3] observations + measured rewards ==="
python3 - <<'PY' || { echo "FATAL: a seed failed on real silicon"; exit 1; }
import json, sys, pathlib
res=json.load(open("runs/pallas_arena/seed-parity-results.json"))
for name, task, tag in [("SEED-rglru-timeblocked","rg_lru","rglru"),
                        ("SEED-splash-flash","splash_attention","splash")]:
    r=res.get(name) or {}
    if not r.get("passed"):
        print(f"[parity] {name}: NOT passed on TPU: gate={r.get('gate')} "
              f"violations={str(r.get('violations'))[:300]}")
        sys.exit(1)
    rw=r.get("reward_with_bwd") or r.get("reward")
    obs=str(r.get("observation") or "").strip()
    if not obs:
        obs=f"passed; reward={rw}"
    pathlib.Path(f"runs/pallas_arena/seed-obs-{tag}.txt").write_text(obs)
    pathlib.Path(f"runs/pallas_arena/seed-reward-{tag}.txt").write_text(
        f"{float(rw):.3f}x vs the production kernel -- reward accrues only ABOVE this")
    print(f"[parity] {name}: reward={rw}; obs {len(obs)} chars banked")
PY

echo "=== [4] launching the four arms ==="
S=$REPO/tpu/pallas_arena/probe
launch() { # launch <tag> <cells> <seedfile> <obstag> [gemma]
  ( export SUP_TAG="$1" SMOKE_CELLS="$2" SEED_FILE="$3"
    export SEED_OBS="$REPO/runs/pallas_arena/seed-obs-$4.txt"
    export SEED_REWARD="$(cat "$REPO/runs/pallas_arena/seed-reward-$4.txt")"
    if [ "${5:-}" = "gemma" ]; then
      export QWEN_MODEL=google/gemma-4-31B-it QWEN_MAXLEN=16384
      export QWEN_HF_GCS=gs://sk7524-tinker-tpu-us-east5/hf-cache-gemma4
      export QWEN_XLA_GCS=gs://sk7524-tinker-tpu-us-east5/vllm-xla-cache-gemma4-31b-16k
      export EXTRA_PIP="transformers==5.14.0" ENABLE_THINKING_KWARG=1
    fi
    nohup bash "$S/bench_supervisor.sh" > "runs/pallas_arena/bench-sup-$1.log" 2>&1 &
    echo "launched $1" )
}
launch qwen-seed2-rglru   "rg_lru:rf3s"           "$S/seed_rglru_timeblocked.py" rglru
launch qwen-seed2-splash  "splash_attention:rf3s" "$S/seed_splash_flash.py"      splash
launch gemma-seed2-rglru  "rg_lru:rf3s"           "$S/seed_rglru_timeblocked.py" rglru  gemma
launch gemma-seed2-splash "splash_attention:rf3s" "$S/seed_splash_flash.py"      splash gemma
echo "=== all four arms launched $(date +%H:%M:%S) ==="
