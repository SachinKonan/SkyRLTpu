#!/usr/bin/env bash
# H2 probe: one four-row Qwen3.5-27B update on v6e-16, TP8 x FSDP2.
set -euo pipefail
: "${JOBMAN_WORKER_ID:?}"
: "${JOBMAN_TPU_INTERNAL_IPS:?}"

[ "$JOBMAN_WORKER_ID" = "0" ] || {
  echo "worker $JOBMAN_WORKER_ID: collective process is launched by worker 0"
  exit 0
}

export PATH="$HOME/.local/bin:$PATH"
REPO="${SKYRL_REPO_DIR:-$HOME/SkyRLTpu-v6e}"
KEY="$HOME/.ssh/jobman_tpu_ed25519"
INT="$JOBMAN_TPU_INTERNAL_IPS"
LORA_RANK="${TUNIX_LORA_RANK:-32}"
SMOKE_ROWS="${TUNIX_SMOKE_ROWS:-4}"
SMOKE_REPLAYS="${TUNIX_SMOKE_REPLAYS:-1}"
SMOKE_EXTRA_UPDATES="${TUNIX_SMOKE_EXTRA_UPDATES:-0}"
REQUIRE_SPARSE_EXPERT_GRADIENTS="${TUNIX_REQUIRE_SPARSE_EXPERT_GRADIENTS:-0}"
TRAIN_TP_SIZE="${TRAIN_TP_SIZE:-8}"
TRAIN_FSDP_SIZE="${TRAIN_FSDP_SIZE:-2}"
TUNIX_ROW_SHARD="${TUNIX_ROW_SHARD:-$TRAIN_FSDP_SIZE}"
RESULT_GCS="${SMOKE_RESULT_GCS:-gs://sk7524-tinker-tpu-us-east5/v6e-smoke-results/qwen35-tp8-fsdp2-r${LORA_RANK}-v1.json}"

if [ "$((TRAIN_TP_SIZE * TRAIN_FSDP_SIZE))" -ne 16 ]; then
  echo "v6e-16 trainer mesh must use exactly 16 chips: TP=${TRAIN_TP_SIZE} FSDP=${TRAIN_FSDP_SIZE}" >&2
  exit 2
fi
if [ "$TUNIX_ROW_SHARD" -ne "$TRAIN_FSDP_SIZE" ]; then
  echo "row sharding must match FSDP for this smoke: rows=${TUNIX_ROW_SHARD} FSDP=${TRAIN_FSDP_SIZE}" >&2
  exit 2
fi

if gcloud storage objects describe "$RESULT_GCS" >/dev/null 2>&1; then
  echo "v6e smoke already complete at $RESULT_GCS"
  exit 0
fi
# The same live slice is intentionally reused across model proofs.  Do not let
# the previous model's local JSON look like the current attempt succeeded while
# this one is still loading or compiling; the durable GCS marker above remains
# the source of truth.
rm -f "$HOME/v6e-tunix-smoke.json" "$HOME/tunix-replay-diagnostics-process-0.jsonl"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-27B}"
MAXTEXT_MODEL_NAME="${TUNIX_MAXTEXT_MODEL_NAME:-qwen3.5-27b}"
DEFAULT_MAXTEXT_PIP_SPEC="maxtext @ git+https://github.com/SachinKonan/maxtext.git@0fd409939977ac0ab79a4e64d21730936f253567"
case "$MAXTEXT_MODEL_NAME" in
  gpt-oss-20b)
    # GPT-OSS expert projections stay on MaxText's routed-token MegaBlox/GMM
    # path. Generic Qwix handles attention/router; the pinned MaxText fork
    # installs the six explicit sparse expert factors consumed by that GMM.
    DEFAULT_MAXTEXT_PIP_SPEC="maxtext @ git+https://github.com/SachinKonan/maxtext.git@b77f9f358a1dd9b223fcc16792b7d5c2530d7044"
    DEFAULT_MAXTEXT_KWARGS='{"sparse_matmul":true,"megablox":true,"num_vocab_tiling":64,"remat_policy":"full","allow_split_physical_axes":true}'
    DEFAULT_FLCE_TILE_SIZE=512
    ;;
  gemma4-31b)
    # Gemma-4 has 16 local KV heads (already TP8-divisible) but only four
    # global KV heads.  Pad just the global projections to eight; MaxText's
    # checkpoint aligner repeats them [h0,h0,...,h3,h3], preserving the
    # original eight-query-heads-per-global-KV grouping over 32 Q heads.
    # Long-context training cannot materialize the quadratic dot-product
    # attention matrix. Use Tokamax Splash's fused forward/backward path; this
    # also bypasses the legacy JAX Splash mask-metadata sharding failure on TP8.
    # autoselected keeps the 4-token LoRA-template trace on dot-product (the
    # Tokamax compute block minimum is 128) and selects Splash for real inputs.
    if [ "$TRAIN_TP_SIZE" -eq 4 ]; then
      # Native Gemma-4 topology: all 16 local KV heads and four global KV
      # heads divide TP4, so no logical head padding is needed. This is also
      # the correctness control for the TP8/global-KV-padding replay probe.
      DEFAULT_MAXTEXT_KWARGS='{"num_vocab_tiling":32,"remat_policy":"full","allow_split_physical_axes":true,"attention":"autoselected","use_tokamax_splash":true}'
    else
      DEFAULT_MAXTEXT_KWARGS='{"num_vocab_tiling":32,"remat_policy":"full","allow_split_physical_axes":true,"override_model_config":true,"global_num_kv_heads":8,"attention":"autoselected","use_tokamax_splash":true}'
    fi
    DEFAULT_FLCE_TILE_SIZE=1024
    ;;
  muse-glimmer-30b)
    # This is the parity-proven Muse fork revision. The Qwen/Gemma revision
    # above does not contain Muse's MaxText model or checkpoint mapping.
    # Muse has 32 query heads but only two KV heads, so TP8 cannot partition
    # the physical KV axis.  Instantiate eight logical KV heads and let the
    # same checkpoint-alignment path repeat each physical head four times:
    # [h0,h0,h0,h0,h1,h1,h1,h1]. This preserves Muse's original grouping of
    # 16 query heads per physical KV head. Tokamax Splash supports Muse's
    # alternating sliding/global pattern without materializing a quadratic
    # attention matrix, and avoids the legacy Splash TP8 metadata failure.
    DEFAULT_MAXTEXT_PIP_SPEC="maxtext @ git+https://github.com/SachinKonan/maxtext.git@4f65ba50963bc975e7ad90ebaa1e752d8a9d8c82"
    DEFAULT_MAXTEXT_KWARGS='{"num_vocab_tiling":32,"remat_policy":"full","parameter_memory_host_offload":true,"allow_split_physical_axes":true,"override_model_config":true,"base_num_kv_heads":8,"attention":"autoselected","use_tokamax_splash":true}'
    DEFAULT_FLCE_TILE_SIZE=1024
    ;;
  *)
    # Qwen3.5-27B has four physical KV heads, which cannot be directly
    # partitioned over TP8.  Instantiate eight logical KV heads and let the
    # pinned MaxText checkpoint-alignment path repeat the cached weights as
    # [h0,h0,h1,h1,h2,h2,h3,h3].  With 24 query heads this preserves the
    # original six-query-heads-per-KV-head grouping while making the KV axis
    # divisible by TP8.  override_model_config is required because the model
    # YAML correctly records the unpadded physical count (4).
    # JAX 0.11.1's legacy Pallas Splash mask metadata cannot be manually
    # sharded over this TP8 mesh (it raises before the attention computation).
    # Tokamax Splash uses a different fused forward/backward path and keeps
    # attention memory bounded at the 22K production context length.
    DEFAULT_MAXTEXT_KWARGS='{"num_vocab_tiling":64,"remat_policy":"full","allow_split_physical_axes":true,"override_model_config":true,"base_num_kv_heads":8,"attention":"autoselected","use_tokamax_splash":true}'
    DEFAULT_FLCE_TILE_SIZE=512
    ;;
esac
MAXTEXT_PIP_SPEC="${TUNIX_MAXTEXT_PIP_SPEC:-$DEFAULT_MAXTEXT_PIP_SPEC}"
MAXTEXT_KWARGS="${TUNIX_MAXTEXT_KWARGS:-$DEFAULT_MAXTEXT_KWARGS}"
FLCE_TILE_SIZE="${TUNIX_FLCE_TILE_SIZE:-$DEFAULT_FLCE_TILE_SIZE}"
FREE_BASE_STATE="${TUNIX_FREE_BASE_STATE:-1}"
UNIFORM="${TUNIX_UNIFORM_SEQ_LEN:-4096}"
BUDGET="${TUNIX_TRAIN_TOKEN_BUDGET:-$((4 * UNIFORM))}"
JAX_CACHE_GCS="${TUNIX_JAX_CACHE_GCS:-gs://sk7524-tinker-tpu-us-east5/jax-compile-cache-v6e-qwen35-tp8-fsdp2-r${LORA_RANK}-s${UNIFORM}-v1}"
CKPT_CACHE_GCS="${TUNIX_MAXTEXT_CKPT_CACHE_GCS:-gs://sk7524-tinker-tpu-us-east5/skyrl-maxtext-ckpts}"
CACHE_LOCAL="${TUNIX_JAX_CACHE_LOCAL:-$HOME/jax_cache}"
LOG_BUCKET_ROOT="${RESULT_GCS%%/v6e-smoke-results/*}"
LOG_GCS="${V6E_LOG_GCS:-${LOG_BUCKET_ROOT}/v6e-smoke-logs/${JOBMAN_ATTEMPT_ID:-unknown}}"

# Spot slices have repeatedly disappeared during cold start. Publish the small
# bring-up/API logs while the attempt is alive so the final useful line survives
# VM reclamation. Checkpoints stay in their one-time regional mirror; this only
# transfers diagnostic text.
publish_logs() {
  for path in \
    "$HOME/v6e-tunix-bringup.log" \
    "$HOME/skyrl-logs/tinker-api.log" \
    "$HOME/v6e-tunix-smoke.log" \
    "$HOME/tunix-replay-diagnostics-process-0.jsonl"; do
    if [ -s "$path" ]; then
      gcloud storage cp "$path" "$LOG_GCS/$(basename "$path")" >/dev/null 2>&1 || true
    fi
  done
}

( while true; do
    sleep 30
    publish_logs
  done ) &
LOG_PUBLISHER_PID=$!
CACHE_PUBLISHER_PID=
cleanup() {
  if [ -n "$CACHE_PUBLISHER_PID" ]; then
    kill "$CACHE_PUBLISHER_PID" 2>/dev/null || true
    wait "$CACHE_PUBLISHER_PID" 2>/dev/null || true
  fi
  kill "$LOG_PUBLISHER_PID" 2>/dev/null || true
  wait "$LOG_PUBLISHER_PID" 2>/dev/null || true
  publish_logs
}
trap cleanup EXIT

env \
  TPU_SSH_MODE=direct \
  TPU_EXTERNAL_IPS="$INT" \
  TPU_INTERNAL_IPS="$INT" \
  TPU_NAME="${JOBMAN_TPU_NAME:-sk7524-v6e-qwen-tp8-fsdp2-smoke}" \
  PROJECT="${PROJECT:-vision-mix}" \
  ZONE="${ZONE:-us-east5-b}" \
  REMOTE_USER="${REMOTE_USER:-sk7524_princeton_edu}" \
  SSH_KEY_FILE="$KEY" \
  REMOTE_SKYRL_DIR="$REPO" \
  SYNC_SKYRL=0 \
  START_VLLM=0 \
  START_TINKER=1 \
  TINKER_BACKEND=tunix \
  TRAIN_WORKERS=0,1,2,3 \
  VLLM_WORKERS= \
  VLLM_BASE_URL_OVERRIDE=http://127.0.0.1:1 \
  EXTERNAL_SAMPLING=1 \
  TP_SIZE="$TRAIN_TP_SIZE" \
  FSDP_SIZE="$TRAIN_FSDP_SIZE" \
  TUNIX_ROW_SHARD="$TUNIX_ROW_SHARD" \
  TRAIN_TPU_PROCESS_BOUNDS=2,2,1 \
  TRAIN_TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
  MODEL_NAME="$MODEL_NAME" \
  TUNIX_MAXTEXT_MODEL_NAME="$MAXTEXT_MODEL_NAME" \
  TUNIX_MAXTEXT_PIP_SPEC="$MAXTEXT_PIP_SPEC" \
  TUNIX_MAXTEXT_KWARGS="$MAXTEXT_KWARGS" \
  TUNIX_MAX_TARGET_LENGTH="$UNIFORM" \
  TUNIX_UNIFORM_SEQ_LEN="$UNIFORM" \
  TUNIX_TRAIN_TOKEN_BUDGET="$BUDGET" \
  TUNIX_FLCE_TILE_SIZE="$FLCE_TILE_SIZE" \
  TUNIX_MINIMAL_FB_OUTPUT="${TUNIX_MINIMAL_FB_OUTPUT:-0}" \
  TUNIX_REPLAY_DIAGNOSTICS="${TUNIX_REPLAY_DIAGNOSTICS:-0}" \
  TUNIX_FREE_BASE_STATE="$FREE_BASE_STATE" \
  TRAIN_MICRO_BATCH_SIZE=1 \
  TUNIX_JAX_CACHE_GCS="$JAX_CACHE_GCS" \
  TUNIX_MAXTEXT_CKPT_CACHE_GCS="$CKPT_CACHE_GCS" \
  READY_ATTEMPTS=1200 \
  bash "$REPO/tpu/start_colocated_vllm_tinker.sh" \
  >"$HOME/v6e-tunix-bringup.log" 2>&1

# Preserve a newly completed compile even if spot capacity disappears before
# the transaction finishes. Only process 0 publishes; cache keys are additive.
( for _ in $(seq 1 120); do
    sleep 60
    gcloud storage rsync -r "$CACHE_LOCAL" "$JAX_CACHE_GCS" >/dev/null 2>&1 || true
  done ) &
CACHE_PUBLISHER_PID=$!

SMOKE_ARGS=(
  --base-model "$MODEL_NAME"
  --rank "$LORA_RANK"
  --rows "$SMOKE_ROWS"
  --replays "$SMOKE_REPLAYS"
  --extra-updates "$SMOKE_EXTRA_UPDATES"
  --sequence-length "$UNIFORM"
  --output "$HOME/v6e-tunix-smoke.json"
)
if [ "$REQUIRE_SPARSE_EXPERT_GRADIENTS" = "1" ]; then
  SMOKE_ARGS+=(
    --gradient-diagnostics "$HOME/tunix-replay-diagnostics-process-0.jsonl"
    --require-sparse-expert-gradients
  )
fi

set +e
"$REPO/.venv/bin/python" "$REPO/tpu/v6e_tunix_smoke.py" "${SMOKE_ARGS[@]}" \
  2>&1 | tee "$HOME/v6e-tunix-smoke.log"
SMOKE_RC=${PIPESTATUS[0]}
set -e

if [ "$SMOKE_RC" -ne 0 ]; then
  kill "$CACHE_PUBLISHER_PID" 2>/dev/null || true
  wait "$CACHE_PUBLISHER_PID" 2>/dev/null || true
  CACHE_PUBLISHER_PID=
  gcloud storage rsync -r "$CACHE_LOCAL" "$JAX_CACHE_GCS" || true
  if [ -s "$HOME/v6e-tunix-smoke.json" ]; then
    FAILURE_GCS="${RESULT_GCS%.json}-failed.json"
    gcloud storage cp "$HOME/v6e-tunix-smoke.json" "$FAILURE_GCS" || true
    echo "v6e smoke failed strict acceptance; diagnostics: $FAILURE_GCS" >&2
    if [ "${TUNIX_SMOKE_DIAGNOSTIC_ACCEPT_FAILURE:-0}" = "1" ]; then
      gcloud storage cp "$HOME/v6e-tunix-smoke.json" "$RESULT_GCS"
      publish_logs
      echo "v6e diagnostic completed with expected strict failure: $RESULT_GCS"
      exit 0
    fi
  fi
  exit "$SMOKE_RC"
fi

# MaxText writes process 0's persistent compilation entries here. Publish them
# before the success marker so a replacement VM can restore the one-time v6e
# compile even if the slice is reclaimed immediately after this probe.
kill "$CACHE_PUBLISHER_PID" 2>/dev/null || true
wait "$CACHE_PUBLISHER_PID" 2>/dev/null || true
gcloud storage rsync -r "$CACHE_LOCAL" "$JAX_CACHE_GCS"
gcloud storage cp "$HOME/v6e-tunix-smoke.json" "$RESULT_GCS"
publish_logs
echo "v6e TP8/FSDP2 smoke complete: $RESULT_GCS"
