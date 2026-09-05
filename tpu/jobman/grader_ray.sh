#!/usr/bin/env bash
# Start TPUSwarm's grading Ray without touching SkyPilot's control-plane Ray.
set -euo pipefail

mode="${1:?usage: grader_ray.sh head|worker}"
: "${RAY_BIN:?}"
: "${GRADER_RAY_ADDRESS:?}"
: "${GRADER_RAY_NODE_IP:?}"

GRADER_RAY_TEMP_DIR="${GRADER_RAY_TEMP_DIR:-/tmp/ray_tpuswarm_grader}"
GRADER_RAY_NUM_CPUS="${GRADER_RAY_NUM_CPUS:-0}"
GRADER_RAY_LOG="${GRADER_RAY_LOG:-/tmp/ray-tpuswarm-grader.log}"

raylet_joined() {
  ps -eo args= | grep -F '/ray/raylet/raylet' | \
    grep -F -- "--gcs-address=$GRADER_RAY_ADDRESS" >/dev/null
}

grader_ray_ready() {
  "$RAY_BIN" health-check --address="$GRADER_RAY_ADDRESS" >/dev/null 2>&1 &&
    raylet_joined
}

grader_ray_pids() {
  ps -eo pid=,args= | awk \
    -v root="$GRADER_RAY_TEMP_DIR/" \
    -v address="--gcs-address=$GRADER_RAY_ADDRESS" \
    'index($0, root) || index($0, address) { print $1 }'
}

stop_stale_grader_ray() {
  local pids
  pids=$(grader_ray_pids)
  [ -n "$pids" ] || return 0
  # Match this grading cluster's unique temp root or exact GCS address.
  # SkyPilot uses /tmp/ray_skypilot on port 6380 and cannot match this list.
  kill $pids 2>/dev/null || true
  sleep 2
  pids=$(grader_ray_pids)
  [ -z "$pids" ] || kill -KILL $pids 2>/dev/null || true
}

shared_ports=(
  --object-manager-port=8077
  --node-manager-port=8078
  --dashboard-agent-listen-port=52366
  --min-worker-port=20000
  --max-worker-port=29999
)

start_with_retries() {
  local attempt
  for attempt in 1 2 3; do
    if "$@" >"$GRADER_RAY_LOG" 2>&1; then
      return 0
    fi
    echo "grading ray start failed (attempt $attempt/3)" >&2
    tail -30 "$GRADER_RAY_LOG" >&2 || true
    stop_stale_grader_ray
    sleep 2
  done
  return 1
}

case "$mode" in
  head)
    if grader_ray_ready; then
      echo "grading ray head already ready at $GRADER_RAY_ADDRESS"
      exit 0
    fi
    stop_stale_grader_ray
    mkdir -p "$GRADER_RAY_TEMP_DIR"
    if ! start_with_retries "$RAY_BIN" start --head \
      --node-ip-address="$GRADER_RAY_NODE_IP" \
      --port="${GRADER_RAY_ADDRESS##*:}" \
      --dashboard-port=8265 \
      --ray-client-server-port=10002 \
      --temp-dir="$GRADER_RAY_TEMP_DIR" \
      --num-cpus="$GRADER_RAY_NUM_CPUS" \
      --disable-usage-stats --include-log-monitor=false \
      "${shared_ports[@]}"; then
      echo "grading ray head failed to start at $GRADER_RAY_ADDRESS" >&2
      exit 1
    fi
    for _attempt in $(seq 1 30); do
      if grader_ray_ready; then
        echo "grading ray head started at $GRADER_RAY_ADDRESS"
        exit 0
      fi
      sleep 1
    done
    echo "grading ray head failed to become ready at $GRADER_RAY_ADDRESS" >&2
    tail -30 "$GRADER_RAY_LOG" >&2 || true
    exit 1
    ;;
  worker)
    if raylet_joined; then
      echo "grading ray worker already joined at $GRADER_RAY_ADDRESS"
      exit 0
    fi
    stop_stale_grader_ray
    if ! start_with_retries "$RAY_BIN" start \
      --address="$GRADER_RAY_ADDRESS" \
      --node-ip-address="$GRADER_RAY_NODE_IP" \
      --num-cpus="$GRADER_RAY_NUM_CPUS" \
      --disable-usage-stats --include-log-monitor=false \
      "${shared_ports[@]}"; then
      echo "grading ray worker failed to start for $GRADER_RAY_ADDRESS" >&2
      exit 1
    fi
    if ! raylet_joined; then
      echo "grading ray worker failed to join $GRADER_RAY_ADDRESS" >&2
      tail -30 "$GRADER_RAY_LOG" >&2 || true
      exit 1
    fi
    echo "grading ray worker joined at $GRADER_RAY_ADDRESS"
    ;;
  *)
    echo "usage: grader_ray.sh head|worker" >&2
    exit 2
    ;;
esac
