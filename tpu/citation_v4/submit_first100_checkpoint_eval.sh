#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/scratch/gpfs/ZHUANGL/hk4638/SkyRLTpu}"
TOOLS_ROOT="/home/hk4638/SkyRL/.tools"
export PATH="${TOOLS_ROOT}/google-cloud-sdk/bin:${TOOLS_ROOT}/jobman-venv/bin:${PATH}"
export CLOUDSDK_CONFIG="${TOOLS_ROOT}/gcloud-config"

SOURCE_RESULTS="${SOURCE_RESULTS:-gs://hk4638-autoresearch-tpu-us-east5/skyrl-tpu/citation-v4/sft/citation-sft-qwen35-9b-full-20260711-080207/results}"
SOURCE_MODEL_ID="${SOURCE_MODEL_ID:-model_d267c4a3}"
CHECKPOINT_IDS="${CHECKPOINT_IDS:-examples_002000,examples_004000}"
CHECKPOINT_LABELS="${CHECKPOINT_LABELS:-step_2000,step_4000}"
RUN_ID="${RUN_ID:-citation-first100-qwen35-9b-2k-4k-$(date -u +%Y%m%d-%H%M%S)}"
BUCKET="${TPU_BUCKET:-hk4638-autoresearch-tpu-us-east5}"
RESULT_PREFIX="gs://${BUCKET}/skyrl-tpu/citation-v4/evals/${RUN_ID}"
SERVER_PREFIX="${RESULT_PREFIX}/server"
RUN_DIR="${RUN_DIR:-/scratch/gpfs/ZHUANGL/hk4638/tinker_outputs/citation_prediction_v4/${RUN_ID}}"
SERVER_SESSION="${SERVER_SESSION:-${RUN_ID}-server}"
EVAL_SESSION="${EVAL_SESSION:-${RUN_ID}-head}"
TPU_NAME="hk4638-${RUN_ID}_1"

mkdir -p "$RUN_DIR"
cd "$ROOT_DIR"

tmux new-session -d -s "$SERVER_SESSION" \
  "cd '$ROOT_DIR' && MODE=canary WORKLOAD_MODE=server RUN_ID='$RUN_ID' RESULT_PREFIX_OVERRIDE='$SERVER_PREFIX' RESTORE_RESULT_PREFIX='$SOURCE_RESULTS' RESTORE_MODEL_ID='$SOURCE_MODEL_ID' RESTORE_CHECKPOINT_IDS='$CHECKPOINT_IDS' SAMPLE_MAX_MODEL_LEN=131072 SAMPLE_MAX_NUM_SEQUENCES='${SAMPLE_MAX_NUM_SEQUENCES:-1}' SAMPLE_HBM_UTILIZATION='${SAMPLE_HBM_UTILIZATION:-0.1}' bash tpu/citation_v4/submit_citation_sft.sh 2>&1 | tee '$RUN_DIR/server-controller.log'"

# The head handles retriever submission/renewal and retries its TPU tunnel until
# Jobman has allocated the eval server. It writes each completed batch locally.
tmux new-session -d -s "$EVAL_SESSION" \
  "cd '$ROOT_DIR' && RUN_NAME='$RUN_ID' RUN_DIR='$RUN_DIR' TINKER_TPU_NAME='$TPU_NAME' CITATION_HEAD_CLIENT_SCRIPT='$ROOT_DIR/tpu/citation_v4/run_citation_eval_client.sh' TINKER_STATE_PATHS='tinker://$SOURCE_MODEL_ID/weights/examples_002000,tinker://$SOURCE_MODEL_ID/weights/examples_004000' TINKER_CHECKPOINT_LABELS='$CHECKPOINT_LABELS' OUTPUT_DIR='$RUN_DIR/results' RETRIEVER_SERVER_TOPK=50 CITATION_TOP_K=10 CITATION_RERANK_FINAL_TOPK=10 CITATION_MAX_AUTHORS_IN_RESULT=12 EVAL_BATCH_SIZE=1 EVAL_MAX_PARALLEL=1 EVAL_MAX_PROMPT_LENGTH=131072 EVAL_TEMPERATURE=0.6 EVAL_TOP_P=1.0 EVAL_TOP_K=20 bash tpu/citation_v4/run_citation_rl_head.sh 2>&1 | tee '$RUN_DIR/eval-head.log'; status=\${PIPESTATUS[0]}; if [ \$status -eq 0 ]; then gcloud storage cp --recursive '$RUN_DIR/results' '$RESULT_PREFIX/'; printf 'stop\n' > '$RUN_DIR/stop.txt'; gcloud storage cp '$RUN_DIR/stop.txt' '$SERVER_PREFIX/results/stop.txt'; fi; exit \$status"

cat <<EOF
Run ID: $RUN_ID
TPU: $TPU_NAME
Server tmux: $SERVER_SESSION
Eval tmux: $EVAL_SESSION
Local results: $RUN_DIR/results
GCS results: $RESULT_PREFIX/results
EOF
