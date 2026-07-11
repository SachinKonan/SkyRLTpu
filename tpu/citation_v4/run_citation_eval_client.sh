#!/usr/bin/env bash
set -euo pipefail
set -x

ROOT_DIR="${ROOT_DIR:-/scratch/gpfs/ZHUANGL/hk4638/SkyRLTpu}"
AGENT_DIR="${AGENT_DIR:-${ROOT_DIR}/skyrl-agent}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-9B}"
TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-${MODEL_NAME}}"
EVAL_DATASET_FILE="${EVAL_DATASET_FILE:-/scratch/gpfs/ZHUANGL/hk4638/paper_processing_out/parquets_v4/test.parquet}"
TASK_YAML="${TASK_YAML:-${AGENT_DIR}/examples/run_tinker/tinker_citation_prediction_v4.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/gpfs/ZHUANGL/hk4638/tinker_outputs/citation_prediction_v4/checkpoint_eval}"
TINKER_STATE_PATHS="${TINKER_STATE_PATHS:?TINKER_STATE_PATHS is required}"
TINKER_CHECKPOINT_LABELS="${TINKER_CHECKPOINT_LABELS:?TINKER_CHECKPOINT_LABELS is required}"

export CITATION_RETRIEVER_URL="${CITATION_RETRIEVER_URL:?CITATION_RETRIEVER_URL is required}"
export CITATION_TOP_K="${CITATION_TOP_K:-10}"
export CITATION_TIMEOUT="${CITATION_TIMEOUT:-30}"
export CITATION_MAX_QUESTION_SEGMENTS="${CITATION_MAX_QUESTION_SEGMENTS:-4}"
export CITATION_MAX_SEARCHES_PER_QUESTION="${CITATION_MAX_SEARCHES_PER_QUESTION:-4}"
export CITATION_MAX_SEARCHES_TOTAL="${CITATION_MAX_SEARCHES_TOTAL:-16}"
export CITATION_RERANK_RETRIEVAL_TOPK="${CITATION_RERANK_RETRIEVAL_TOPK:-50}"
export CITATION_RERANK_ALPHA="${CITATION_RERANK_ALPHA:-0.1}"
export CITATION_RERANK_FINAL_TOPK="${CITATION_RERANK_FINAL_TOPK:-10}"
export CITATION_RERANK_NORM="${CITATION_RERANK_NORM:-rank}"
export CITATION_COUNT_PATH="${CITATION_COUNT_PATH:-/scratch/gpfs/ZHUANGL/hk4638/SemanticScholar/arxiv_impact_counts_s2.csv}"
export CITATION_METRIC_BETA="${CITATION_METRIC_BETA:-0.5}"
export CITATION_MAX_AUTHORS_IN_RESULT="${CITATION_MAX_AUTHORS_IN_RESULT:-12}"
export CITATION_FORCE_FINAL_TURN_ON_MAX_INPUT_LENGTH="${CITATION_FORCE_FINAL_TURN_ON_MAX_INPUT_LENGTH:-1}"
export CITATION_FINAL_TURN_CONTEXT_RESERVE_TOKENS="${CITATION_FINAL_TURN_CONTEXT_RESERVE_TOKENS:-2048}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT_DIR}/skyrl-agent:${ROOT_DIR}/skyrl-gym:${ROOT_DIR}:${PYTHONPATH:-}"

mkdir -p "$OUTPUT_DIR"
cd "$AGENT_DIR"
UV_ARGS=(--isolated --extra tinker)
if [ -f .env ]; then
  UV_ARGS+=(--env-file .env)
fi

uv run "${UV_ARGS[@]}" -m skyrl_agent.integrations.tinker.tinker_eval \
  model_name="$MODEL_NAME" \
  tokenizer_name_or_path="$TOKENIZER_NAME_OR_PATH" \
  tinker_base_url="${TINKER_BASE_URL:?TINKER_BASE_URL is required}" \
  tinker_api_key="${TINKER_API_KEY:-tml-dummy}" \
  tinker_project_id="${TINKER_PROJECT_ID:-}" \
  lora_rank="${LORA_RANK:-32}" \
  state_paths="$TINKER_STATE_PATHS" \
  checkpoint_labels="$TINKER_CHECKPOINT_LABELS" \
  skyrl_agent_task_yaml="$TASK_YAML" \
  eval_dataset_file="$EVAL_DATASET_FILE" \
  output_dir="$OUTPUT_DIR" \
  max_examples="${EVAL_MAX_EXAMPLES:-100}" \
  batch_size="${EVAL_BATCH_SIZE:-4}" \
  max_parallel="${EVAL_MAX_PARALLEL:-4}" \
  max_prompt_length="${EVAL_MAX_PROMPT_LENGTH:-131072}" \
  temperature="${EVAL_TEMPERATURE:-0.6}" \
  top_p="${EVAL_TOP_P:-1.0}" \
  top_k="${EVAL_TOP_K:-20}" \
  max_tokens="${EVAL_MAX_TOKENS:-4096}"
