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
# THE MAXTEXT FORK IS LOAD-BEARING, not a preference: the FLCE loss path calls
# model(..., skip_lm_head=True) and uses the returned hidden states. Stock
# PyPI maxtext ignores that kwarg and returns None, so the FIRST
# forward_backward dies with "'NoneType' object has no attribute 'shape'"
# inside _flce_target_logprobs -- while boot, weight load and serving all look
# perfectly healthy. Passing it via the jobman env alone did NOT reach the
# venv build; the league passes it explicitly too (cell_worker.sh).
MAXTEXT_PIP_SPEC="${TUNIX_MAXTEXT_PIP_SPEC:-maxtext @ git+https://github.com/SachinKonan/maxtext.git@skyrl/qwen35-dense}"
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
  # Skip only REAL measurements. An earlier version skipped any recorded
  # line, so an infrastructure fault (NOT-READY from a broken venv, a 400 from
  # a malformed request) permanently poisoned that candidate -- 16384:16384 was
  # never measured because a stale fault from the broken-Python cycle looked
  # like a result.
  if grep "^uniform=${UNIFORM} budget=${BUDGET}:" "$RESULTS" 2>/dev/null \
     | grep -q "PROBE-RESULTS-JSON" \
     && ! grep "^uniform=${UNIFORM} budget=${BUDGET}:" "$RESULTS" 2>/dev/null \
        | grep -qE "status 400|NoneType"; then
    echo "[skip] ${UNIFORM}:${BUDGET} already measured"; continue
  fi
  echo "=== candidate uniform=${UNIFORM} budget=${BUDGET} $(date -u +%H:%M:%S) ==="
  t_cand=$(date +%s)
  kill_trainer

  # SYNC_SKYRL=0: the bundle is already on disk (jobman prepare unpacked it),
  # and the sync path wants a worktree checkout this VM does not have.
  # START_VLLM=0 is now honoured -- the vLLM readiness wait is gated on
  # START_VLLM alone (fixed in start_colocated_vllm_tinker.sh).
  # Exactly the league's addressing: direct ssh over the INTERNAL ips jobman
  # exports. VLLM_WORKERS=1 because TRAIN_WORKERS and VLLM_WORKERS must be
  # disjoint (bring-up asserts it) and worker 1's ip is resolved even though
  # START_VLLM=0 means nothing is ever started there.
  env TPU_SSH_MODE=direct TPU_EXTERNAL_IPS="$JOBMAN_TPU_INTERNAL_IPS" \
    TPU_INTERNAL_IPS="$JOBMAN_TPU_INTERNAL_IPS" \
    REMOTE_USER="$USER" SSH_KEY_FILE="$HOME/.ssh/jobman_tpu_ed25519" \
    REMOTE_SKYRL_DIR="$REPO" TUNIX_MAXTEXT_PIP_SPEC="$MAXTEXT_PIP_SPEC" \
    TINKER_BACKEND=tunix TRAIN_WORKERS=0 VLLM_WORKERS=1 \
    MODEL_NAME="$MODEL_NAME" TUNIX_MAXTEXT_MODEL_NAME="$MAXTEXT_MODEL" \
    TUNIX_MAXTEXT_KWARGS="{\"num_vocab_tiling\": ${VOCAB_TILING}}" \
    TUNIX_MAX_TARGET_LENGTH="$UNIFORM" TUNIX_TRAIN_TOKEN_BUDGET="$BUDGET" \
    TUNIX_FLCE_TILE_SIZE="$FLCE_TILE" TRAIN_MICRO_BATCH_SIZE=1 \
    TUNIX_UNIFORM_SEQ_LEN="$UNIFORM" TUNIX_SEQ_BUCKETS="4096,8192,12288,16384,20480" \
    TUNIX_MINIMAL_FB_OUTPUT=1 SKYRL_SCORE_FIXED_LEN="$UNIFORM" \
    READY_ATTEMPTS=900 SYNC_SKYRL=0 START_VLLM=0 START_TINKER=1 \
    bash "$REPO/tpu/start_colocated_vllm_tinker.sh" > "$HOME/lenprobe-bringup-${UNIFORM}-${BUDGET}.log" 2>&1 \
    || echo "[warn] bring-up rc=$? (health check decides)"

  # A bring-up that dies in seconds is a config fault, not a slow compile:
  # waiting out the full ready window just delays the diagnosis by ~50min.
  if ! tinker_up && [ "$(( $(date +%s) - t_cand ))" -lt 60 ]; then
    echo "[fast-fail] bring-up returned in <60s -- config fault, not a compile"
    { echo "uniform=${UNIFORM} budget=${BUDGET}: BRINGUP-FAULT"
      echo "  $(tail -3 "$HOME/lenprobe-bringup-${UNIFORM}-${BUDGET}.log" 2>/dev/null | tr '\n' ' ' | cut -c1-300)"
    } >> "$RESULTS"
    publish; continue
  fi

  ready=0; end=$(( $(date +%s) + READY_S )); tick=0
  while [ "$(date +%s)" -lt "$end" ]; do
    tinker_up && { ready=1; break; }
    # The launcher runs in tmux and can die seconds in (bad path, bad env)
    # while bring-up already returned 0. Once its session is gone AND the
    # endpoint is down, waiting out the window teaches nothing.
    tick=$(( tick + 1 ))
    if [ "$tick" -ge 6 ] && ! tmux has-session -t skyrl-tinker 2>/dev/null; then
      echo "[fast-fail] tinker tmux session gone"; break
    fi
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
  # A 400 is a request-shape bug on OUR side, and its only real diagnosis is
  # the SERVER traceback -- which lives in a log that the next session
  # truncates and that dies with the slice. Capture it inline: hand-reading it
  # over ssh lost the race with preemption twice.
  if echo "$out" | grep -q "status 400"; then
    {
      echo "  --- server traceback (uniform=${UNIFORM}) ---"
      grep -B25 "has no attribute" "$HOME/skyrl-logs/tinker-api.log" 2>/dev/null \
        | grep -E 'File "|line [0-9]+, in|Error|attribute|raise' | tail -14 | sed 's/^/    /'
    } >> "$RESULTS"
  fi
  publish
done

kill_trainer
echo "=== PROBE COMPLETE ==="
cat "$RESULTS"
publish
# Signal completion to jobman's probe hook.
touch "$HOME/.lenprobe-done"

# BUNDLE CONTENTS THAT ARE LOAD-BEARING (each cost one boot to discover):
#   skyrl-gym       -- pyproject declares it as an editable path dependency
#   .python-version -- pins 3.12; without it uv takes the system 3.11 and
#                      maxtext (>=3.12) becomes unsatisfiable
#   tpu/            -- start_colocated + probe script; REMOTE_SKYRL_DIR must
#                      point at this bundle, the launcher defaults to ~/SkyRLTpu
