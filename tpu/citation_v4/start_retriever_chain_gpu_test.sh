#!/bin/bash
# Keep a gpu-test arXiv retriever available by chaining short Slurm jobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH_BASE="${SCRATCH_BASE:-/scratch/gpfs/ZHUANGL/hk4638}"
RUN_DIR="${AUTORESEARCH_RUN_DIR:-${SCRATCH_BASE}/autoresearch/runs/retriever-chain-$(date +%Y%m%d-%H%M%S)}"
CURRENT_READY="${AUTORESEARCH_RETRIEVER_CURRENT_READY_FILE:-${RUN_DIR}/retriever_current.env}"
STATUS_FILE="${AUTORESEARCH_RETRIEVER_STATUS_FILE:-${RUN_DIR}/retriever_status.json}"
RETRIEVER_SLURM="${RETRIEVER_SLURM:-${SCRIPT_DIR}/start_retriever_v4_gpu_test.slurm}"
RETRIEVER_TOPK="${AUTORESEARCH_RETRIEVER_SERVER_TOPK:-50}"
RETRIEVER_GRES="${AUTORESEARCH_RETRIEVER_GRES:-gpu:2}"
TOTAL_SECONDS="${AUTORESEARCH_RETRIEVER_CHAIN_SECONDS:-7200}"
SLEEP_SECONDS="${AUTORESEARCH_RETRIEVER_CHAIN_SLEEP_SECONDS:-30}"
START_TIME="$(date +%s)"
ITER=0
ACTIVE_JOB=""

mkdir -p "${RUN_DIR}" "$(dirname "${CURRENT_READY}")"

cleanup() {
    if [ -n "${ACTIVE_JOB}" ]; then
        scancel "${ACTIVE_JOB}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

wait_for_ready() {
    local ready="$1"
    local job_id="$2"
    while [ ! -f "${ready}" ]; do
        if ! job_still_visible "${job_id}"; then
            echo "[$(timestamp)] ERROR: retriever job ${job_id} ended before writing ready file ${ready}" >&2
            return 1
        fi
        sleep 5
    done
}

job_still_visible() {
    local job_id="$1"
    [ -n "$(squeue -j "${job_id}" -h -o '%i' 2>/dev/null)" ]
}

echo "[$(timestamp)] Starting gpu-test retriever chain"
echo "RUN_DIR=${RUN_DIR}"
echo "CURRENT_READY=${CURRENT_READY}"
echo "STATUS_FILE=${STATUS_FILE}"
echo "TOTAL_SECONDS=${TOTAL_SECONDS}"

write_status() {
    local status="$1"
    local job_id="${2:-}"
    local detail="${3:-}"
    python - "$STATUS_FILE" "$status" "$job_id" "$detail" "$ITER" <<'PY'
import json
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "status": sys.argv[2],
    "job_id": sys.argv[3],
    "detail": sys.argv[4],
    "iteration": int(sys.argv[5]),
    "time": int(time.time()),
}
path.write_text(json.dumps(payload, indent=2) + "\n")
PY
}

rm -f "${CURRENT_READY}"
write_status "starting"

while true; do
    NOW="$(date +%s)"
    if [ "$((NOW - START_TIME))" -ge "${TOTAL_SECONDS}" ]; then
        rm -f "${CURRENT_READY}"
        write_status "complete" "" "time_budget_reached"
        echo "[$(timestamp)] Retriever chain reached TOTAL_SECONDS=${TOTAL_SECONDS}"
        exit 0
    fi

    ITER="$((ITER + 1))"
    READY_FILE="${RUN_DIR}/retriever_${ITER}.env"
    rm -f "${READY_FILE}" "${CURRENT_READY}"
    write_status "pending" "" "submitting_retriever"
    echo "[$(timestamp)] Submitting retriever iteration ${ITER}"
    ACTIVE_JOB="$(env -u SBATCH_PARTITION RETRIEVER_READY_FILE="${READY_FILE}" RETRIEVER_TOPK="${RETRIEVER_TOPK}" RETRIEVAL_SERVER="${RETRIEVAL_SERVER:-${SCRIPT_DIR}/retrieval_server.py}" sbatch --parsable --partition=gpu-test --gres="${RETRIEVER_GRES}" --time=00:59:00 "${RETRIEVER_SLURM}")"
    write_status "pending" "${ACTIVE_JOB}" "waiting_for_ready_file"
    echo "[$(timestamp)] Retriever job ${ACTIVE_JOB} submitted"

    if ! wait_for_ready "${READY_FILE}" "${ACTIVE_JOB}"; then
        write_status "pending" "${ACTIVE_JOB}" "retriever_failed_before_ready_retrying"
        echo "[$(timestamp)] Retriever job ${ACTIVE_JOB} failed before ready; retrying in ${SLEEP_SECONDS}s" >&2
        ACTIVE_JOB=""
        sleep "${SLEEP_SECONDS}"
        continue
    fi
    {
        cat "${READY_FILE}"
        echo "RETRIEVER_CHAIN_JOB_ID=${ACTIVE_JOB}"
        echo "RETRIEVER_CHAIN_ITER=${ITER}"
        echo "RETRIEVER_CHAIN_READY_TIME=$(date +%s)"
    } > "${CURRENT_READY}.tmp"
    mv "${CURRENT_READY}.tmp" "${CURRENT_READY}"
    write_status "ready" "${ACTIVE_JOB}" "retriever_ready"
    echo "[$(timestamp)] Current retriever ready file updated: ${CURRENT_READY}"

    while job_still_visible "${ACTIVE_JOB}"; do
        NOW="$(date +%s)"
        if [ "$((NOW - START_TIME))" -ge "${TOTAL_SECONDS}" ]; then
            rm -f "${CURRENT_READY}"
            write_status "complete" "${ACTIVE_JOB}" "time_budget_reached"
            echo "[$(timestamp)] Time budget reached; stopping active retriever ${ACTIVE_JOB}"
            exit 0
        fi
        sleep "${SLEEP_SECONDS}"
    done
    rm -f "${CURRENT_READY}"
    write_status "pending" "" "previous_retriever_ended"
    echo "[$(timestamp)] Retriever job ${ACTIVE_JOB} ended; starting next iteration"
    ACTIVE_JOB=""
done
