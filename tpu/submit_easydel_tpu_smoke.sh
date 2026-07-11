#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_ID="${RUN_ID:-easydel-pr-smoke-$(date -u +%Y%m%d-%H%M%S)}"
BUCKET="${TPU_SMOKE_BUCKET:-hk4638-autoresearch-tpu-us-east5}"
PREFIX="gs://${BUCKET}/skyrl-tpu/easydel-smokes/${RUN_ID}"
STAGE="${TMPDIR:-/tmp}/${RUN_ID}"
ARCHIVE="${STAGE}/source.tar.gz"
CONFIG="${STAGE}/jobman.yaml"
LONG_CONTEXT_LENGTHS="${LONG_CONTEXT_LENGTHS:-}"
LONG_CONTEXT_ONLY="${LONG_CONTEXT_ONLY:-0}"
RESUME_RESULT_PREFIX="${RESUME_RESULT_PREFIX:-}"
TPU_PRICING="${TPU_PRICING:-spot}"
TPU_VALID_UNTIL_DURATION="${TPU_VALID_UNTIL_DURATION:-}"
TPU_FLAGS_YAML='[]'
if [ -n "$TPU_VALID_UNTIL_DURATION" ]; then
  TPU_FLAGS_YAML="[\"--valid-until-duration=${TPU_VALID_UNTIL_DURATION}\"]"
fi

command -v gcloud >/dev/null
command -v jobman >/dev/null
mkdir -p "$STAGE"

cd "$REPO_ROOT"
tar -czf "$ARCHIVE" \
  .python-version \
  LICENSE \
  README.md \
  pyproject.toml \
  uv.lock \
  skyrl \
  skyrl-gym \
  tpu
gcloud storage cp "$ARCHIVE" "${PREFIX}/source.tar.gz"

sed \
  -e "s#__SOURCE_ARCHIVE_URI__#${PREFIX}/source.tar.gz#g" \
  -e "s#__RESULT_PREFIX__#${PREFIX}/results#g" \
  -e "s#__RUN_ID__#${RUN_ID}#g" \
  -e "s#__LONG_CONTEXT_LENGTHS__#${LONG_CONTEXT_LENGTHS}#g" \
  -e "s#__LONG_CONTEXT_ONLY__#${LONG_CONTEXT_ONLY}#g" \
  -e "s#__RESUME_RESULT_PREFIX__#${RESUME_RESULT_PREFIX}#g" \
  -e "s#__TPU_PRICING__#${TPU_PRICING}#g" \
  -e "s#__TPU_FLAGS__#${TPU_FLAGS_YAML}#g" \
  "${SCRIPT_DIR}/jobman_easydel_v5p16_smoke.template.yaml" \
  | sed 's/${/\\${/g' > "$CONFIG"

echo "Run ID: ${RUN_ID}"
echo "Result prefix: ${PREFIX}/results"
echo "Rendered config: ${CONFIG}"
CREATE_OUTPUT="$(jobman create "$CONFIG")"
printf '%s\n' "$CREATE_OUTPUT"

JOB_ID="$(printf '%s\n' "$CREATE_OUTPUT" | sed -n 's/.*Created job \([0-9][0-9]*\).*/\1/p' | tail -1)"
if [ -z "$JOB_ID" ]; then
  echo "Could not determine the jobman job ID" >&2
  exit 1
fi
if [ "${WAIT_AND_CLEANUP:-1}" != "1" ]; then
  echo "Job ID: ${JOB_ID}"
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
  for _ in $(seq 1 60); do
    STATE="$(gcloud compute tpus queued-resources describe "$RESOURCE_NAME" \
      --zone=us-east5-a --format='value(state.state)' 2>/dev/null)"
    if [ -z "$STATE" ] || { [ "$STATE" != "ACTIVE" ] && [ "$STATE" != "SUSPENDING" ]; }; then
      break
    fi
    sleep 5
  done
  if [ -n "$STATE" ]; then
    gcloud compute tpus queued-resources delete "$RESOURCE_NAME" \
      --zone=us-east5-a --quiet >/dev/null 2>&1
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

for _ in $(seq 1 "${SMOKE_POLL_ATTEMPTS:-180}"); do
  STATUS="$(gcloud storage cat "$STATUS_URI" 2>/dev/null || true)"
  if [ "$STATUS" = "passed" ] || [ "$STATUS" = "failed" ]; then
    echo "Smoke status: ${STATUS}"
    if [ "$STATUS" = "passed" ]; then
      exit 0
    fi
    exit 1
  fi
  sleep "${SMOKE_POLL_SECONDS:-20}"
done

echo "Timed out waiting for ${STATUS_URI}" >&2
exit 1
