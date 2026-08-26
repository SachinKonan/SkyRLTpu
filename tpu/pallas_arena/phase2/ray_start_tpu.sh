#!/usr/bin/env bash
# Start a Ray head on a TPU judge host and GUARANTEE a usable "TPU" resource.
#
# Ray auto-detects TPU chips on a TPU VM and publishes them as the "TPU"
# resource; but detection depends on the Ray version and on libtpu being
# visible to the raylet, and passing --resources='{"TPU": N}' when Ray has
# already detected the accelerator can collide. This script tries detection
# first and only declares the resource explicitly if detection came up empty
# -- so the pool gets its chips either way, on any Ray version, and we never
# double-declare.
#
# Usage: RAY_CHIPS=4 bash ray_start_tpu.sh   (RAY_CHIPS = chips on this host)
set -uo pipefail

VENV="${VENV:-$HOME/arena-venv}"
RAY="${RAY:-$VENV/bin/ray}"
CHIPS="${RAY_CHIPS:-4}"

# IDEMPOTENT: the fleet's supervisor re-runs this on every ssh reconnect,
# and the old stop/start KILLED every in-flight grading task each time --
# leases expired, items requeued, duplicate verdicts (measured 2026-08-26).
# A healthy head with a TPU resource is reused, never restarted.
have=$("$VENV/bin/python" - <<'EOF' 2>/dev/null
try:
    import ray
    ray.init(address="auto", ignore_reinit_error=True, logging_level="ERROR")
    print(int(ray.cluster_resources().get("TPU", 0)))
except Exception:
    print(0)
EOF
)
if [ "${have:-0}" -ge 1 ]; then
  echo "[ray] head already up with TPU=${have}; reusing"
  exit 0
fi

"$RAY" stop --force >/dev/null 2>&1
pkill -f 'raylet|ray::' >/dev/null 2>&1
sleep 2

tpu_resource() { # -> the TPU resource count Ray currently reports, or empty
  "$VENV/bin/python" - <<'PY' 2>/dev/null
import json, subprocess, sys
try:
    import ray
    ray.init(address="auto", ignore_reinit_error=True, logging_level="ERROR")
    print(int(ray.cluster_resources().get("TPU", 0)))
except Exception:
    print(0)
PY
}

echo "[ray] starting head (auto-detect first)"
"$RAY" start --head --num-cpus="$(nproc)" --disable-usage-stats >/dev/null 2>&1
sleep 5
have="$(tpu_resource)"
if [ "${have:-0}" -ge 1 ]; then
  echo "[ray] auto-detected TPU resource: ${have}"
  exit 0
fi

echo "[ray] no TPU resource detected; declaring ${CHIPS} explicitly"
"$RAY" stop --force >/dev/null 2>&1
sleep 2
"$RAY" start --head --num-cpus="$(nproc)" \
  --resources="{\"TPU\": ${CHIPS}}" --disable-usage-stats >/dev/null 2>&1
sleep 5
have="$(tpu_resource)"
if [ "${have:-0}" -ge 1 ]; then
  echo "[ray] TPU resource declared: ${have}"
  exit 0
fi
echo "[ray] FATAL: no TPU resource after both attempts" >&2
exit 1
