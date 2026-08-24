#!/usr/bin/env bash
# Max-trainable-sequence-length probe, run as a jobman cell worker (worker 0).
#
# WHY JOBMAN: the same probe hand-rolled on a gcloud-created QR failed five
# times on PROVISIONING gaps, not on the measurement -- worker layout, a vLLM
# readiness wait, a dead env var, spot preemption mid-boot, uv missing from a
# non-login PATH. jobman provisions the user/keys/tools these scripts expect
# and, with loop+resumable, re-creates the slice after preemption instead of
# reporting a corpse as an OOM.
#
# Each candidate needs its OWN boot: TUNIX_UNIFORM_SEQ_LEN pins one compiled
# shape. Per candidate: boot the trainer alone, then fb+optim_step TWICE at
# that length -- cold, then warm. The warm pass is the one that matters; it is
# what demoted qwen from an apparent 24576 to a usable 12288.
#
# Results stream to GCS after EVERY candidate, so a preemption at candidate 3
# still leaves candidates 1-2 banked.
set -uo pipefail

REPO="${REPO:-$HOME/SkyRLTpu-lenprobe}"
MODEL_NAME="${MODEL_NAME:-google/gemma-4-31B-it}"
MAXTEXT_MODEL="${TUNIX_MAXTEXT_MODEL_NAME:-gemma4-31b}"
FLCE_TILE="${FLCE_TILE:-1024}"
VOCAB_TILING="${VOCAB_TILING:-32}"
CANDIDATES="${CANDIDATES:-16384:16384 16384:65536 12288:49152 20480:20480}"
READY_S="${READY_S:-3000}"
RESULTS_GCS="${RESULTS_GCS:-gs://sk7524-tinker-tpu-us-east5/lenprobe/gemma4-31b}"
RESULTS="$HOME/lenprobe-results.txt"
touch "$RESULTS"

# Resume: skip candidates already measured in a previous (preempted) attempt.
gsutil -q cp "${RESULTS_GCS}/results.txt" "$RESULTS" 2>/dev/null || true
echo "=== probe start $(date -u +%H:%M:%S) model=${MODEL_NAME} ==="
cat "$RESULTS"

publish() { gsutil -q cp "$RESULTS" "${RESULTS_GCS}/results.txt" 2>/dev/null || true; }

tinker_up() { curl -fsS -m8 http://127.0.0.1:8000/api/v1/get_server_capabilities >/dev/null 2>&1; }

kill_trainer() {
  tmux kill-session -t skyrl-tinker 2>/dev/null || true
  tmux list-sessions -F '#{session_name}' 2>/dev/null | awk '/^skyrl-tinker-worker-/ {print}' \
    | xargs -r -n1 tmux kill-session -t 2>/dev/null || true
  pkill -TERM -u "$USER" -f '[s]kyrl\.tinker|[s]kyrl\.backends\.jax' 2>/dev/null || true
  sleep 8
  pkill -KILL -u "$USER" -f '[s]kyrl\.tinker|[s]kyrl\.backends\.jax' 2>/dev/null || true
  sleep 4
}

for cand in $CANDIDATES; do
  UNIFORM="${cand%%:*}"; BUDGET="${cand##*:}"
  if grep -q "^uniform=${UNIFORM} budget=${BUDGET}:" "$RESULTS" 2>/dev/null; then
    echo "[skip] ${UNIFORM}:${BUDGET} already measured"; continue
  fi
  echo "=== candidate uniform=${UNIFORM} budget=${BUDGET} $(date -u +%H:%M:%S) ==="
  kill_trainer

  # SYNC_SKYRL=0: the bundle is already on disk (jobman prepare unpacked it),
  # and the sync path wants a worktree checkout this VM does not have.
  # START_VLLM=0 is now honoured -- the vLLM readiness wait is gated on
  # START_VLLM alone (fixed in start_colocated_vllm_tinker.sh).
  env TPU_SSH_MODE=direct TPU_EXTERNAL_IPS=127.0.0.1 TPU_INTERNAL_IPS=127.0.0.1 \
    REMOTE_USER="$USER" SSH_KEY_FILE="$HOME/.ssh/jobman_tpu_ed25519" \
    TINKER_BACKEND=tunix TRAIN_WORKERS=0 VLLM_WORKERS=0 \
    MODEL_NAME="$MODEL_NAME" TUNIX_MAXTEXT_MODEL_NAME="$MAXTEXT_MODEL" \
    TUNIX_MAXTEXT_KWARGS="{\"num_vocab_tiling\": ${VOCAB_TILING}}" \
    TUNIX_MAX_TARGET_LENGTH="$UNIFORM" TUNIX_TRAIN_TOKEN_BUDGET="$BUDGET" \
    TUNIX_FLCE_TILE_SIZE="$FLCE_TILE" TRAIN_MICRO_BATCH_SIZE=1 \
    TUNIX_UNIFORM_SEQ_LEN="$UNIFORM" TUNIX_SEQ_BUCKETS="4096,8192,12288,16384,20480" \
    TUNIX_MINIMAL_FB_OUTPUT=1 SKYRL_SCORE_FIXED_LEN="$UNIFORM" \
    READY_ATTEMPTS=900 SYNC_SKYRL=0 START_VLLM=0 START_TINKER=1 \
    bash "$REPO/tpu/start_colocated_vllm_tinker.sh" > "$HOME/lenprobe-bringup-${UNIFORM}-${BUDGET}.log" 2>&1 \
    || echo "[warn] bring-up rc=$? (health check decides)"

  ready=0; end=$(( $(date +%s) + READY_S ))
  while [ "$(date +%s)" -lt "$end" ]; do
    tinker_up && { ready=1; break; }
    sleep 20
  done

  if [ "$ready" -ne 1 ]; then
    # The trainer failing to COME UP at this length is itself the answer we
    # want (boot-time OOM at compile), but it can also be an infra fault, so
    # record the log tail alongside the verdict rather than a bare "failed".
    {
      echo "uniform=${UNIFORM} budget=${BUDGET}: NOT-READY"
      echo "  bringup: $(tail -3 "$HOME/lenprobe-bringup-${UNIFORM}-${BUDGET}.log" 2>/dev/null | tr '\n' ' ' | cut -c1-300)"
      echo "  tinker : $(tail -3 "$HOME/skyrl-logs/tinker-api.log" 2>/dev/null | tr '\n' ' ' | cut -c1-300)"
    } >> "$RESULTS"
    publish; continue
  fi

  echo "trainer ready $(date -u +%H:%M:%S); probing ${UNIFORM}"
  out=$(cd "$REPO" && TINKER_BASE_URL=http://127.0.0.1:8000 PROBE_BASE_MODEL="$MODEL_NAME" \
        timeout 2400 uv run --extra tpu --extra tinker python "$REPO/tpu/probe_train_len_model.py" "$UNIFORM" 2>&1 | tail -8)
  echo "$out"
  echo "uniform=${UNIFORM} budget=${BUDGET}: $(echo "$out" | grep -E 'PROBE-RESULTS-JSON|cold |warm ' | tr '\n' ' ' | cut -c1-400)" >> "$RESULTS"
  publish
done

kill_trainer
echo "=== PROBE COMPLETE ==="
cat "$RESULTS"
publish
# Signal completion to jobman's probe hook.
touch "$HOME/.lenprobe-done"
