#!/usr/bin/env bash
set -uo pipefail

STATE=${TPUSWARM_STATE_ROOT:-/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state}
REPO=${SKYRL_REPO_DIR:-/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-multihost}
VENV=$REPO/third_party/TPUSwarm/.venv
GCLOUD=${GCLOUD:-/scratch/gpfs/ZHUANGL/sk7524/google-cloud-sdk/bin/gcloud}
TPUSWARM_SERVER=${TPUSWARM_SERVER:-http://127.0.0.1:8787}
SKY_API_SERVER_URL=${SKY_API_SERVER_URL:-http://127.0.0.1:46580}
TOKEN_FILE=$STATE/token

POOLS=(
  tpuswarm-v6e32-east5b-qwen35
  tpuswarm-v6e32-asia-qwen35
  tpuswarm-v6e32-europe-w4a-qwen35
  tpuswarm-v5p32-east5a-erdos
  tpuswarm-v4-64-central2-qwen35-erdos
)
TASK_IDS=(
  qwen35-v6e32-grpo-001-east5b
  qwen35-v6e32-grpo-001-asia-ne1b
  qwen35-v6e32-grpo-002-asia-ne1b
  qwen35-v6e32-grpo-001-europe-w4a
)
ZONES=(us-east5-b asia-northeast1-b europe-west4-a us-east5-a us-central2-b)

export HOME=$STATE/sky-home-v6e32
export SKYPILOT_CONFIG=$STATE/skypilot-config.yaml
export CLOUDSDK_CONFIG=/home/sk7524/.config/gcloud-tpuswarm-compute-sa-v6e32
export GOOGLE_APPLICATION_CREDENTIALS=/home/sk7524/.config/gcloud/vision-mix-compute-sa-key.json
export SKYPILOT_API_SERVER_LOCAL_PORT=46580
export SKYPILOT_DISABLE_USAGE_COLLECTION=1
export SKY_API_SERVER_URL
export TPUSWARM_SERVER
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

require_token() {
  if [ ! -s "$TOKEN_FILE" ]; then
    log "missing TPUSwarm token file: $TOKEN_FILE"
    return 1
  fi
  TPUSWARM_TOKEN="$(<"$TOKEN_FILE")"
  export TPUSWARM_TOKEN
}

task_summary() {
  local task_id=$1
  local response=$STATE/.monitor-$task_id.json
  local code

  code="$(
    curl -sS -o "$response" -w '%{http_code}' \
      -H "Authorization: Bearer $TPUSWARM_TOKEN" \
      "$TPUSWARM_SERVER/v1/tasks/$task_id" || true
  )"
  if [ "$code" != 200 ]; then
    printf '{"task_id":"%s","http_status":"%s"}\n' "$task_id" "$code"
    return
  fi
  "$VENV/bin/python" -c '
import json, sys

row = json.load(open(sys.argv[1]))
keys = ("task_id", "status", "job_id", "executor_status", "recovery_count",
        "last_error", "updated_at")
print(json.dumps({key: row.get(key) for key in keys}, sort_keys=True))
' "$response"
}

gcp_queued_resources() {
  local zone=$1
  "$GCLOUD" compute tpus queued-resources list \
    --project=vision-mix \
    --zone="$zone" \
    --format='table(name.basename(),state.state,state.initiator,createTime)' |
    awk 'NR == 1 || /tpuswarm-v(4|5p|6e)/' || true
}

main() {
  cd "$REPO" || exit 1
  require_token || exit 1
  log "starting multi-region monitor pools=${POOLS[*]}"

  while true; do
    log "TPUSwarm health"
    curl -fsS "$TPUSWARM_SERVER/healthz" || true
    printf '\n'

    for zone in "${ZONES[@]}"; do
      log "GCP queued resources zone=$zone"
      gcp_queued_resources "$zone"
    done

    log "SkyPilot pool status"
    timeout 180s "$VENV/bin/sky" jobs pool status --all --verbose || true

    log "TPUSwarm task summaries"
    for task_id in "${TASK_IDS[@]}"; do
      task_summary "$task_id"
    done

    log "SkyPilot jobs queue"
    timeout 180s "$VENV/bin/sky" jobs queue --skip-finished --verbose || true

    log "sleeping 300s"
    sleep 300
  done
}

main "$@"
