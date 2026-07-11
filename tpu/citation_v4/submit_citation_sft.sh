#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TOOLS_ROOT="/home/hk4638/SkyRL/.tools"
export PATH="${TOOLS_ROOT}/google-cloud-sdk/bin:${TOOLS_ROOT}/jobman-venv/bin:${PATH}"
export CLOUDSDK_CONFIG="${TOOLS_ROOT}/gcloud-config"

MODE="${MODE:-canary}"
WORKLOAD_MODE="${WORKLOAD_MODE:-sft}"
if [ "$WORKLOAD_MODE" = "server" ]; then
  SAMPLE_MAX_NUM_SEQUENCES="${SAMPLE_MAX_NUM_SEQUENCES:-32}"
  SAMPLE_HBM_UTILIZATION="${SAMPLE_HBM_UTILIZATION:-0.2}"
else
  SAMPLE_MAX_NUM_SEQUENCES="${SAMPLE_MAX_NUM_SEQUENCES:-2}"
  SAMPLE_HBM_UTILIZATION="${SAMPLE_HBM_UTILIZATION:-0.05}"
fi
SAMPLE_MAX_MODEL_LEN="${SAMPLE_MAX_MODEL_LEN:-60000}"
case "$MODE" in
  canary)
    LOCAL_DATA="${LOCAL_DATA:-/scratch/gpfs/ZHUANGL/hk4638/data/citation_prediction_v4/tpu_stage/conservative_qwen35_ctx60k_canary.jsonl.gz}"
    BATCH_SIZE="${BATCH_SIZE:-1}"
    SAVE_EVERY_EXAMPLES="${SAVE_EVERY_EXAMPLES:-0}"
    ;;
  full)
    LOCAL_DATA="${LOCAL_DATA:-/scratch/gpfs/ZHUANGL/hk4638/data/citation_prediction_v4/tpu_stage/conservative_qwen35_ctx60k_train.jsonl.gz}"
    BATCH_SIZE="${BATCH_SIZE:-2}"
    SAVE_EVERY_EXAMPLES="${SAVE_EVERY_EXAMPLES:-2000}"
    ;;
  *) echo "MODE must be canary or full" >&2; exit 64 ;;
esac

RUN_ID="${RUN_ID:-citation-sft-qwen35-9b-${MODE}-$(date -u +%Y%m%d-%H%M%S)}"
BUCKET="${TPU_BUCKET:-hk4638-autoresearch-tpu-us-east5}"
PREFIX="${RESULT_PREFIX_OVERRIDE:-gs://${BUCKET}/skyrl-tpu/citation-v4/sft/${RUN_ID}}"
RESTORE_RESULT_PREFIX="${RESTORE_RESULT_PREFIX:-${PREFIX}/results}"
RESTORE_MODEL_ID="${RESTORE_MODEL_ID:-}"
RESTORE_CHECKPOINT_IDS="${RESTORE_CHECKPOINT_IDS:-}"
DATA_URI="gs://${BUCKET}/citation-v4-data/$(basename "$LOCAL_DATA")"
STAGE="${TMPDIR:-/tmp}/${RUN_ID}"
ARCHIVE="${STAGE}/source.tar.gz"
CONFIG="${STAGE}/jobman.yaml"
TPU_PRICING="${TPU_PRICING:-spot}"
CHECKPOINT_UPLOAD_SECONDS="${CHECKPOINT_UPLOAD_SECONDS:-300}"
TPU_VALID_UNTIL_DURATION="${TPU_VALID_UNTIL_DURATION:-}"
TPU_FLAGS_YAML='[]'
if [ -n "$TPU_VALID_UNTIL_DURATION" ]; then
  TPU_FLAGS_YAML="[\"--valid-until-duration=${TPU_VALID_UNTIL_DURATION}\"]"
fi

command -v gcloud >/dev/null
command -v jobman >/dev/null
test -f "$LOCAL_DATA"
mkdir -p "$STAGE"

cd "$REPO_ROOT"
tar --exclude='*/.venv' --exclude='*/__pycache__' --exclude='*/.git' -czf "$ARCHIVE" \
  .python-version LICENSE README.md pyproject.toml uv.lock skyrl skyrl-agent skyrl-gym tpu
gcloud storage cp "$ARCHIVE" "${PREFIX}/source.tar.gz"
if ! gcloud storage ls "$DATA_URI" >/dev/null 2>&1; then
  gcloud storage cp "$LOCAL_DATA" "$DATA_URI"
fi

sed \
  -e "s#__SOURCE_ARCHIVE_URI__#${PREFIX}/source.tar.gz#g" \
  -e "s#__DATA_ARCHIVE_URI__#${DATA_URI}#g" \
  -e "s#__RESULT_PREFIX__#${PREFIX}/results#g" \
  -e "s#__RESTORE_RESULT_PREFIX__#${RESTORE_RESULT_PREFIX}#g" \
  -e "s#__RESTORE_MODEL_ID__#${RESTORE_MODEL_ID}#g" \
  -e "s#__RESTORE_CHECKPOINT_IDS__#${RESTORE_CHECKPOINT_IDS}#g" \
  -e "s#__RUN_ID__#${RUN_ID}#g" \
  -e "s#__WORKLOAD_MODE__#${WORKLOAD_MODE}#g" \
  -e "s#__SAMPLE_MAX_NUM_SEQUENCES__#${SAMPLE_MAX_NUM_SEQUENCES}#g" \
  -e "s#__SAMPLE_HBM_UTILIZATION__#${SAMPLE_HBM_UTILIZATION}#g" \
  -e "s#__SAMPLE_MAX_MODEL_LEN__#${SAMPLE_MAX_MODEL_LEN}#g" \
  -e "s#__BATCH_SIZE__#${BATCH_SIZE}#g" \
  -e "s#__SAVE_EVERY_EXAMPLES__#${SAVE_EVERY_EXAMPLES}#g" \
  -e "s#__CHECKPOINT_UPLOAD_SECONDS__#${CHECKPOINT_UPLOAD_SECONDS}#g" \
  -e "s#__TPU_PRICING__#${TPU_PRICING}#g" \
  -e "s#__TPU_FLAGS__#${TPU_FLAGS_YAML}#g" \
  "${SCRIPT_DIR}/jobman_easydel_citation_sft.template.yaml" \
  | sed 's/${/\\${/g' > "$CONFIG"

echo "Run ID: ${RUN_ID}"
echo "Data: ${DATA_URI}"
echo "Results: ${PREFIX}/results"
CREATE_OUTPUT="$(jobman create "$CONFIG")"
printf '%s\n' "$CREATE_OUTPUT"
JOB_ID="$(printf '%s\n' "$CREATE_OUTPUT" | sed -n 's/.*Created job \([0-9][0-9]*\).*/\1/p' | tail -1)"
test -n "$JOB_ID"
echo "Job ID: ${JOB_ID}"

if [ "${WAIT_AND_CLEANUP:-1}" != "1" ]; then
  exit 0
fi

STATUS_URI="${PREFIX}/results/status.txt"
RESOURCE_NAME="hk4638-${RUN_ID}_1"
cleanup() {
  set +e
  jobman delete "$JOB_ID" >/dev/null 2>&1
  STATE="$(gcloud compute tpus queued-resources describe "$RESOURCE_NAME" \
    --zone=us-east5-a --format='value(state.state)' 2>/dev/null)"
  if [ "$STATE" = "ACTIVE" ]; then
    gcloud compute tpus tpu-vm delete "$RESOURCE_NAME" --zone=us-east5-a --quiet >/dev/null 2>&1
  fi
  for _ in $(seq 1 120); do
    STATE="$(gcloud compute tpus queued-resources describe "$RESOURCE_NAME" \
      --zone=us-east5-a --format='value(state.state)' 2>/dev/null)"
    if [ -z "$STATE" ] || { [ "$STATE" != "ACTIVE" ] && [ "$STATE" != "SUSPENDING" ]; }; then
      break
    fi
    sleep 5
  done
  gcloud compute tpus queued-resources delete "$RESOURCE_NAME" \
    --zone=us-east5-a --quiet >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM

for _ in $(seq 1 "${POLL_ATTEMPTS:-2160}"); do
  STATUS="$(gcloud storage cat "$STATUS_URI" 2>/dev/null || true)"
  if [ "$STATUS" = "passed" ]; then
    echo "SFT status: passed"
    exit 0
  fi
  if [ "$STATUS" = "failed" ]; then
    echo "SFT status: failed" >&2
    exit 1
  fi
  sleep "${POLL_SECONDS:-20}"
done
echo "Timed out waiting for ${STATUS_URI}" >&2
exit 1
