#!/usr/bin/env bash
set -euo pipefail
set -x

ROOT_DIR="${ROOT_DIR:-/scratch/gpfs/ZHUANGL/hk4638/SkyRLTpu}"
AGENT_DIR="${AGENT_DIR:-${ROOT_DIR}/skyrl-agent}"

DATA_DIR="${DATA_DIR:-/scratch/gpfs/ZHUANGL/hk4638/data/citation_prediction_v4/rl_exclude_conservative_sft_prompts}"
DATASET_FILE="${DATASET_FILE:-${DATA_DIR}/train.parquet}"
EVAL_DATASET_FILE="${EVAL_DATASET_FILE:-${DATA_DIR}/validation.parquet}"

MODEL_KEY="${MODEL_KEY:-qwen3_4b}"
case "$MODEL_KEY" in
  qwen3_4b) MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}" ;;
  qwen3_8b) MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}" ;;
  qwen35_4b) MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-4B}" ;;
  qwen35_9b) MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-9B}" ;;
  *) MODEL_NAME="${MODEL_NAME:-$MODEL_KEY}" ;;
esac

TINKER_BASE_URL="${TINKER_BASE_URL:-http://127.0.0.1:8000}"
TINKER_API_KEY="${TINKER_API_KEY:-tml-dummy}"
TINKER_PROJECT_ID="${TINKER_PROJECT_ID:-}"
TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-$MODEL_NAME}"

SWEEP_BP="${SWEEP_BP:-40}"
SWEEP_N="${SWEEP_N:-30}"
if [ -z "${SWEEP_LR:-}" ]; then
  SWEEP_LR="3e-6"
fi

LORA_RANK="${LORA_RANK:-32}"
MAX_STEPS="${MAX_STEPS:-100}"
SAVE_EVERY="${SAVE_EVERY:-10}"
EVAL_EVERY="${EVAL_EVERY:-10}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-100}"
LOSS_FN="${LOSS_FN:-cispo}"
GRPO_NORM_BY_STD="${GRPO_NORM_BY_STD:-true}"
TINKER_AGENT_MAX_PARALLEL="${TINKER_AGENT_MAX_PARALLEL:-}"
TINKER_AGENT_MAX_PROMPT_LENGTH="${TINKER_AGENT_MAX_PROMPT_LENGTH:-}"

export CITATION_RETRIEVER_URL="${CITATION_RETRIEVER_URL:-${LOCAL_SEARCH_URL:-}}"
export CITATION_RETRIEVER_READY_FILE="${CITATION_RETRIEVER_READY_FILE:-}"
export CITATION_TOP_K="${CITATION_TOP_K:-5}"
export CITATION_TIMEOUT="${CITATION_TIMEOUT:-30}"
export CITATION_MAX_QUESTION_SEGMENTS="${CITATION_MAX_QUESTION_SEGMENTS:-4}"
export CITATION_MAX_SEARCHES_PER_QUESTION="${CITATION_MAX_SEARCHES_PER_QUESTION:-4}"
export CITATION_MAX_SEARCHES_TOTAL="${CITATION_MAX_SEARCHES_TOTAL:-16}"
export CITATION_RERANK_RETRIEVAL_TOPK="${CITATION_RERANK_RETRIEVAL_TOPK:-}"
export CITATION_RERANK_ALPHA="${CITATION_RERANK_ALPHA:-}"
export CITATION_RERANK_FINAL_TOPK="${CITATION_RERANK_FINAL_TOPK:-}"
export CITATION_RERANK_NORM="${CITATION_RERANK_NORM:-rank}"
export CITATION_COUNT_PATH="${CITATION_COUNT_PATH:-}"
export CITATION_METRIC_BETA="${CITATION_METRIC_BETA:-1.0}"
export CITATION_MAX_AUTHORS_IN_RESULT="${CITATION_MAX_AUTHORS_IN_RESULT:-12}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [ -z "${CITATION_RETRIEVER_URL:-}" ] && [ -z "${CITATION_RETRIEVER_READY_FILE:-}" ]; then
  echo "CITATION_RETRIEVER_URL, LOCAL_SEARCH_URL, or CITATION_RETRIEVER_READY_FILE must point at a running /retrieve endpoint." >&2
  exit 64
fi

export PYTHONPATH="${ROOT_DIR}/skyrl-agent:${ROOT_DIR}/skyrl-gym:${ROOT_DIR}:${PYTHONPATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/gpfs/ZHUANGL/hk4638/tinker_outputs/citation_prediction_v4/${MODEL_KEY}}"
mkdir -p "$OUTPUT_DIR"

WANDB_PROJECT="${WANDB_PROJECT:-tinker-citation-prediction-v4}"
WANDB_NAME="${WANDB_NAME:-cit-v4-tinker-${MODEL_KEY}-bp${SWEEP_BP}-n${SWEEP_N}}"
RESUME_EXP_NAME="${RESUME_EXP_NAME:-}"
INITIAL_STATE_PATH="${TINKER_INITIAL_STATE_PATH:-}"
TASK_YAML="${TASK_YAML:-${AGENT_DIR}/examples/run_tinker/tinker_citation_prediction_v4.yaml}"
EXTRA_TINKER_ARGS=()
if [ -n "${TINKER_AGENT_MAX_PARALLEL:-}" ]; then
  EXTRA_TINKER_ARGS+=(agent_max_parallel="$TINKER_AGENT_MAX_PARALLEL")
fi
if [ -n "${TINKER_AGENT_MAX_PROMPT_LENGTH:-}" ]; then
  EXTRA_TINKER_ARGS+=(agent_max_prompt_length="$TINKER_AGENT_MAX_PROMPT_LENGTH")
fi

echo "================================================"
echo "Tinker Citation v4 RL"
echo "================================================"
echo "Model key/name: ${MODEL_KEY} / ${MODEL_NAME}"
echo "Tinker URL: ${TINKER_BASE_URL}"
echo "Data: ${DATASET_FILE}"
echo "Validation: ${EVAL_DATASET_FILE}"
echo "Retriever: ${CITATION_RETRIEVER_URL}"
echo "Retriever ready file: ${CITATION_RETRIEVER_READY_FILE}"
echo "Batch prompts: ${SWEEP_BP}"
echo "Rollouts per prompt: ${SWEEP_N}"
echo "Learning rate: ${SWEEP_LR}"
echo "LoRA rank: ${LORA_RANK}"
echo "Output: ${OUTPUT_DIR}"
echo "HF offline: HF_HUB_OFFLINE=${HF_HUB_OFFLINE}, TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}, HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE}"
echo "================================================"

cd "$AGENT_DIR"
UV_ARGS=(--isolated --extra tinker)
if [ -f .env ]; then
  UV_ARGS+=(--env-file .env)
fi

uv run "${UV_ARGS[@]}" -m skyrl_agent.integrations.tinker.tinker_train \
  model_name="$MODEL_NAME" \
  tokenizer_name_or_path="$TOKENIZER_NAME_OR_PATH" \
  tinker_base_url="$TINKER_BASE_URL" \
  tinker_api_key="$TINKER_API_KEY" \
  tinker_project_id="$TINKER_PROJECT_ID" \
  skyrl_agent_task_yaml="$TASK_YAML" \
  dataset_file="$DATASET_FILE" \
  eval_dataset_file="$EVAL_DATASET_FILE" \
  batch_size="$SWEEP_BP" \
  eval_batch_size="$EVAL_BATCH_SIZE" \
  learning_rate="$SWEEP_LR" \
  lora_rank="$LORA_RANK" \
  max_steps="$MAX_STEPS" \
  save_every="$SAVE_EVERY" \
  eval_every="$EVAL_EVERY" \
  loss_fn="$LOSS_FN" \
  group_size="$SWEEP_N" \
  grpo_norm_by_std="$GRPO_NORM_BY_STD" \
  cispo_clip_low_threshold="${CISPO_CLIP_LOW_THRESHOLD:-1.0}" \
  cispo_clip_high_threshold="${CISPO_CLIP_HIGH_THRESHOLD:-6.0}" \
  tis_imp_ratio_cap="${TIS_IMP_RATIO_CAP:-2.0}" \
  token_mean=true \
  "${EXTRA_TINKER_ARGS[@]}" \
  wandb_project="$WANDB_PROJECT" \
  wandb_name="$WANDB_NAME" \
  resume_exp_name="$RESUME_EXP_NAME" \
  initial_state_path="$INITIAL_STATE_PATH" \
  log_dir="$OUTPUT_DIR" \
  output_dir="$OUTPUT_DIR" \
  "$@"
