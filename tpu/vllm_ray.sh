#!/usr/bin/env bash
# Manage vLLM's private Ray runtime without touching SkyPilot's Ray cluster.
set -euo pipefail

mode="${1:?usage: vllm_ray.sh head|worker|stop}"
: "${RAY_BIN:?}"
: "${VLLM_RAY_ADDRESS:?}"

VLLM_RAY_TEMP_DIR="${VLLM_RAY_TEMP_DIR:-/tmp/ray_tpuswarm_vllm}"
VLLM_RAY_NUM_CPUS="${VLLM_RAY_NUM_CPUS:-200}"
VLLM_RAY_LOG="${VLLM_RAY_LOG:-/tmp/ray-tpuswarm-vllm.log}"

vllm_ray_pids() {
  ps -eo pid=,args= | awk \
    -v root="$VLLM_RAY_TEMP_DIR/" \
    -v address="--gcs-address=$VLLM_RAY_ADDRESS" \
    'index($0, root) || index($0, address) { print $1 }'
}

stop_vllm_ray() {
  local pids
  pids=$(vllm_ray_pids)
  [ -n "$pids" ] || return 0
  # SkyPilot uses /tmp/ray_skypilot and port 6380. This list contains only
  # vLLM's private temp root or processes joined to vLLM's private GCS port.
  kill $pids 2>/dev/null || true
  sleep 2
  pids=$(vllm_ray_pids)
  [ -z "$pids" ] || kill -KILL $pids 2>/dev/null || true
}

raylet_joined() {
  ps -eo args= | grep -F '/ray/raylet/raylet' | \
    grep -F -- "--gcs-address=$VLLM_RAY_ADDRESS" >/dev/null
}

shared_ports=(
  --object-manager-port=8079
  --node-manager-port=8080
  --dashboard-agent-listen-port=52370
  --dashboard-agent-grpc-port=52371
  --runtime-env-agent-port=52372
  --metrics-export-port=52373
  --min-worker-port=30000
  --max-worker-port=39999
)

case "$mode" in
  head)
    : "${VLLM_RAY_NODE_IP:?}"
    if "$RAY_BIN" status --address="$VLLM_RAY_ADDRESS" >/dev/null 2>&1; then
      echo "vLLM Ray head already ready at $VLLM_RAY_ADDRESS"
      exit 0
    fi
    stop_vllm_ray
    mkdir -p "$VLLM_RAY_TEMP_DIR"
    "$RAY_BIN" start --head \
      --node-ip-address="$VLLM_RAY_NODE_IP" \
      --port="${VLLM_RAY_ADDRESS##*:}" \
      --dashboard-port=8267 \
      --ray-client-server-port=10003 \
      --temp-dir="$VLLM_RAY_TEMP_DIR" \
      --num-cpus="$VLLM_RAY_NUM_CPUS" \
      --disable-usage-stats --include-log-monitor=false \
      "${shared_ports[@]}" >"$VLLM_RAY_LOG" 2>&1
    for _attempt in $(seq 1 60); do
      if "$RAY_BIN" status --address="$VLLM_RAY_ADDRESS" >/dev/null 2>&1; then
        echo "vLLM Ray head started at $VLLM_RAY_ADDRESS"
        exit 0
      fi
      sleep 1
    done
    echo "vLLM Ray head failed to become ready at $VLLM_RAY_ADDRESS" >&2
    tail -50 "$VLLM_RAY_LOG" >&2 || true
    exit 1
    ;;
  worker)
    : "${VLLM_RAY_NODE_IP:?}"
    if raylet_joined; then
      echo "vLLM Ray worker already joined at $VLLM_RAY_ADDRESS"
      exit 0
    fi
    stop_vllm_ray
    "$RAY_BIN" start \
      --address="$VLLM_RAY_ADDRESS" \
      --node-ip-address="$VLLM_RAY_NODE_IP" \
      --num-cpus="$VLLM_RAY_NUM_CPUS" \
      --disable-usage-stats --include-log-monitor=false \
      "${shared_ports[@]}" >"$VLLM_RAY_LOG" 2>&1
    if ! raylet_joined; then
      echo "vLLM Ray worker failed to join $VLLM_RAY_ADDRESS" >&2
      tail -50 "$VLLM_RAY_LOG" >&2 || true
      exit 1
    fi
    echo "vLLM Ray worker joined at $VLLM_RAY_ADDRESS"
    ;;
  stop)
    stop_vllm_ray
    ;;
  *)
    echo "usage: vllm_ray.sh head|worker|stop" >&2
    exit 2
    ;;
esac
