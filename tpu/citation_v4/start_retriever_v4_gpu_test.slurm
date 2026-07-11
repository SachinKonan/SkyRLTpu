#!/bin/bash
# ============================================================================
# Start only the Citation Prediction retriever on gpu-test for remote Codex eval.
# The job writes the reachable URL to RETRIEVER_READY_FILE and then stays alive.
# ============================================================================

#SBATCH --job-name=cit-v4-retriever
#SBATCH --output=/scratch/gpfs/ZHUANGL/hk4638/logs/cit-v4-retriever_%A.out
#SBATCH --error=/scratch/gpfs/ZHUANGL/hk4638/logs/cit-v4-retriever_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:59:00
#SBATCH --partition=gpu-test
#SBATCH --constraint=gpu80

set -euo pipefail
export PYTHONUNBUFFERED=1

SCRATCH_BASE="${SCRATCH_BASE:-/scratch/gpfs/ZHUANGL/hk4638}"
RETRIEVAL_SERVER="${RETRIEVAL_SERVER:-/scratch/gpfs/ZHUANGL/hk4638/SkyRLTpu/tpu/citation_v4/retrieval_server.py}"
LOG_DIR="${SCRATCH_BASE}/logs"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-qwen3_4b}"
RETRIEVER_TOPK="${RETRIEVER_TOPK:-10}"
RETRIEVER_HOST="${RETRIEVER_HOST:-0.0.0.0}"
RETRIEVER_READY_FILE="${RETRIEVER_READY_FILE:-${SCRATCH_BASE}/evals/citation_prediction_v4_codex/retriever_${SLURM_JOB_ID:-local}.env}"

pick_free_port() {
    python -c 'import socket; s = socket.socket(); s.bind(("0.0.0.0", 0)); print(s.getsockname()[1]); s.close()'
}
RETRIEVER_PORT="${RETRIEVER_PORT:-$(pick_free_port)}"

mkdir -p "${LOG_DIR}" "$(dirname "${RETRIEVER_READY_FILE}")"

export CONDA_PKGS_DIRS="${SCRATCH_BASE}/conda/pkgs"
export CONDA_ENVS_PATH="${SCRATCH_BASE}/conda/envs"
export XDG_CACHE_HOME="${SCRATCH_BASE}/cache"
export HF_HOME="${SCRATCH_BASE}/huggingface"
export TRANSFORMERS_CACHE="${SCRATCH_BASE}/huggingface/transformers"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

NODE_HOST="$(hostname -f 2>/dev/null || hostname)"
RETRIEVER_URL="http://${NODE_HOST}:${RETRIEVER_PORT}/retrieve"

echo "=== Citation Prediction V4 Retriever Only ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${NODE_HOST}"
echo "Start time: $(date)"
echo "RETRIEVER_HOST=${RETRIEVER_HOST}"
echo "RETRIEVER_PORT=${RETRIEVER_PORT}"
echo "RETRIEVER_URL=${RETRIEVER_URL}"
echo "RETRIEVER_READY_FILE=${RETRIEVER_READY_FILE}"

CORPUS_FILE="${SCRATCH_BASE}/data/citation_prediction/arxiv_wikiformat_with_ids.jsonl"
if [ "$EMBEDDING_MODEL" = "qwen3_4b" ]; then
    RETRIEVER_MODEL_PATH="${SCRATCH_BASE}/huggingface/hub/models--Qwen--Qwen3-Embedding-4B/snapshots/5cf2132abc99cad020ac570b19d031efec650f2b"
    INDEX_FILE="${SCRATCH_BASE}/data/citation_prediction/qwen3_4b_embed/qwen3_Flat.index"
    RETRIEVER_NAME=qwen3
elif [ "$EMBEDDING_MODEL" = "qwen3_06b" ]; then
    RETRIEVER_MODEL_PATH="${SCRATCH_BASE}/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418"
    INDEX_FILE="/home/hk4638/scratch/shared/data/qwen3_06_embed/qwen3_Flat.index"
    RETRIEVER_NAME=qwen3
else
    echo "ERROR: Unknown EMBEDDING_MODEL=${EMBEDDING_MODEL}"
    exit 1
fi

source /usr/licensed/anaconda3/2024.10/etc/profile.d/conda.sh
export CONDA_ENVS_PATH="${SCRATCH_BASE}/conda/envs"
conda activate retriever
export CUDA_VISIBLE_DEVICES=0

"${CONDA_PREFIX}/bin/python" "${RETRIEVAL_SERVER}" \
    --index_path "${INDEX_FILE}" \
    --corpus_path "${CORPUS_FILE}" \
    --topk "${RETRIEVER_TOPK}" \
    --retriever_name "${RETRIEVER_NAME}" \
    --retriever_model "${RETRIEVER_MODEL_PATH}" \
    --host "${RETRIEVER_HOST}" \
    --port "${RETRIEVER_PORT}" &
SERVER_PID="$!"

cleanup() {
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

READY_DEADLINE="$((SECONDS + ${RETRIEVER_STARTUP_TIMEOUT_SECONDS:-300}))"
until curl -fsS -X POST "http://127.0.0.1:${RETRIEVER_PORT}/retrieve" \
    -H "Content-Type: application/json" \
    -d '{"query":"retriever readiness probe","topk":1,"return_scores":true}' >/dev/null; do
    if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
        wait "${SERVER_PID}"
        exit "$?"
    fi
    if [ "${SECONDS}" -ge "${READY_DEADLINE}" ]; then
        echo "ERROR: retriever did not become ready within ${RETRIEVER_STARTUP_TIMEOUT_SECONDS:-300}s" >&2
        exit 1
    fi
    sleep 2
done

cat > "${RETRIEVER_READY_FILE}.tmp" <<EOF
RETRIEVER_JOB_ID=${SLURM_JOB_ID:-local}
RETRIEVER_NODE=${NODE_HOST}
RETRIEVER_PORT=${RETRIEVER_PORT}
RETRIEVER_URL=${RETRIEVER_URL}
EOF
mv "${RETRIEVER_READY_FILE}.tmp" "${RETRIEVER_READY_FILE}"
echo "Retriever is healthy; published ${RETRIEVER_READY_FILE}"

wait "${SERVER_PID}"
