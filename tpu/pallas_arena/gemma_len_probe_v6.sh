#!/usr/bin/env bash
# Gemma-4-31B max-trainable-length probe, v6: the start_colocated-from-login
# flow (jobman preps the VM; SYNC_SKYRL rsyncs the REAL worktree; the maxtext
# fork installs through the same venv path that served the qwen35 server for
# weeks). No bundle, no hand-curated manifest -- the entire v1-v5 failure
# class (missing skyrl-gym / .python-version / fork-not-landing) is
# structurally gone. Trainer-only bring-up works now that the vLLM wait is
# gated on START_VLLM (fixed in start_colocated).
#
# Per candidate UNIFORM:BUDGET -- the compiled shape is pinned at boot, so
# each candidate is its own tinker restart -- then fb+optim_step TWICE from
# the login side through a tunnel (cold, then warm: the warm pass demoted
# qwen 24576 -> 12288 and is the only pass that counts).
#
# Run from the login node once tpu/runs/yamls/skyrl_gemma4len_v5p16_spot.yaml
# is READY:  bash tpu/pallas_arena/gemma_len_probe_v6.sh
set -uo pipefail

REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
# SINGLE-INSTANCE LOCK: two concurrent drivers (a stale rerun-chain + a fresh
# slice waiter) interleaved on one VM, each killing the other's tinker server
# between candidates -- 30-second "ready"s against the sibling's server, probes
# cut mid-fb, every verdict empty. Never again.
LOCK=/tmp/gemma-len-probe.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another gemma_len_probe_v6 instance holds $LOCK; exiting"
  exit 0
fi
cd "$REPO"

TPU_NAME="${TPU_NAME:-sk7524-gemma4len-v5p16-east5a_spot}"
PROJECT="${PROJECT:-vision-mix}"
ZONE="${ZONE:-us-east5-a}"
REMOTE_USER="${REMOTE_USER:-sk7524_princeton_edu}"
SSH_KEY_FILE="${SSH_KEY_FILE:-$HOME/.ssh/google_compute_engine}"
LOCAL_PORT="${LOCAL_PORT:-18060}"
CANDIDATES="${CANDIDATES:-16384:16384 16384:65536 12288:49152 20480:20480}"
RESULTS="${REPO}/runs/pallas_arena/gemma-len-v6-results.txt"
PROBE_PY="${REPO}/third_party/discover/.venv-ttd-discover/bin/python"
GCLOUD="${GCLOUD:-/n/fs/vision-mix/sk7524/google-cloud-sdk/bin/gcloud}"
mkdir -p "${REPO}/runs/pallas_arena"
touch "$RESULTS"

tssh() { timeout 900 "$GCLOUD" compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" \
  --project="$PROJECT" --worker=0 --command="$1"; }

tpu_ip() {
  "$GCLOUD" alpha compute tpus tpu-vm describe "$TPU_NAME" --project="$PROJECT" \
    --zone="$ZONE" --format="value(networkEndpoints[0].accessConfig.externalIp)" 2>/dev/null
}

tunnel_pid=""
start_tunnel() {
  [ -n "$tunnel_pid" ] && kill "$tunnel_pid" 2>/dev/null
  ( while true; do
      ip="$(tpu_ip || true)"
      [ -z "$ip" ] && { sleep 30; continue; }
      ssh -i "$SSH_KEY_FILE" -o IdentitiesOnly=yes -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
        -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=30 \
        -N -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:8000" "${REMOTE_USER}@${ip}" || true
      sleep 10
    done ) &
  tunnel_pid=$!
}
trap '[ -n "$tunnel_pid" ] && kill "$tunnel_pid" 2>/dev/null; true' EXIT
start_tunnel

SYNC=1
for cand in $CANDIDATES; do
  U="${cand%%:*}"; B="${cand##*:}"
  if grep -q "^uniform=${U} budget=${B}: PROBE" "$RESULTS" 2>/dev/null; then
    echo "[skip] ${U}:${B} already measured"; continue
  fi
  echo "=== candidate uniform=${U} budget=${B} $(date +%H:%M:%S) ==="

  tssh 'tmux kill-session -t skyrl-tinker 2>/dev/null || true; tmux list-sessions -F "#{session_name}" 2>/dev/null | awk "/^skyrl-tinker-worker-/ {print}" | xargs -r -n1 tmux kill-session -t; pkill -TERM -u "$USER" -f "[s]kyrl\.tinker|[s]kyrl\.backends\.jax" || true; sleep 5; pkill -KILL -u "$USER" -f "[s]kyrl\.tinker|[s]kyrl\.backends\.jax" || true; true' || true

  env TPU_NAME="$TPU_NAME" PROJECT="$PROJECT" ZONE="$ZONE" \
    REMOTE_USER="$REMOTE_USER" SSH_KEY_FILE="$SSH_KEY_FILE" \
    TINKER_BACKEND=tunix TRAIN_WORKERS=0 VLLM_WORKERS=1 \
    START_VLLM=0 START_TINKER=1 \
    MODEL_NAME=google/gemma-4-31B-it TUNIX_MAXTEXT_MODEL_NAME=gemma4-31b \
    TUNIX_MAXTEXT_PIP_SPEC="maxtext @ git+https://github.com/SachinKonan/maxtext.git@skyrl/qwen35-dense" \
    TUNIX_MAXTEXT_KWARGS='{"num_vocab_tiling": 32}' TUNIX_FLCE_TILE_SIZE=1024 \
    TUNIX_MAX_TARGET_LENGTH="$U" TUNIX_UNIFORM_SEQ_LEN="$U" TUNIX_TRAIN_TOKEN_BUDGET="$B" \
    TRAIN_MICRO_BATCH_SIZE=1 TUNIX_MINIMAL_FB_OUTPUT=1 \
    SYNC_SKYRL="$SYNC" \
    bash tpu/start_colocated_vllm_tinker.sh \
    > "${REPO}/runs/pallas_arena/gemma-len-v6-bringup-${U}-${B}.log" 2>&1 \
    || echo "[warn] bring-up rc=$? (health check decides)"
  SYNC=0

  ready=0; end=$(( $(date +%s) + 3600 ))
  while [ "$(date +%s)" -lt "$end" ]; do
    curl -fsS -m8 "http://127.0.0.1:${LOCAL_PORT}/api/v1/get_server_capabilities" >/dev/null 2>&1 && { ready=1; break; }
    sleep 25
  done
  if [ "$ready" -ne 1 ]; then
    {
      echo "uniform=${U} budget=${B}: NOT-READY"
      echo "  bringup: $(tail -3 "${REPO}/runs/pallas_arena/gemma-len-v6-bringup-${U}-${B}.log" 2>/dev/null | tr '\n' ' ' | cut -c1-300)"
      echo "  tinker : $(tssh 'tail -3 ~/skyrl-logs/tinker-api.log 2>/dev/null' 2>/dev/null | tail -3 | tr '\n' ' ' | cut -c1-300)"
    } >> "$RESULTS"
    continue
  fi

  echo "trainer ready $(date +%H:%M:%S); probing ${U}"
  out=$(TINKER_BASE_URL="http://127.0.0.1:${LOCAL_PORT}" PROBE_BASE_MODEL=google/gemma-4-31B-it \
        timeout 3000 "$PROBE_PY" tpu/probe_train_len_model.py "$U" 2>&1 | tail -8)
  echo "$out"
  {
    echo "uniform=${U} budget=${B}: $(echo "$out" | grep -E 'PROBE-RESULTS-JSON|cold |warm ' | tr '\n' ' ' | cut -c1-400)"
    if echo "$out" | grep -q "status 400"; then
      echo "  --- server traceback ---"
      tssh "grep -B25 'has no attribute' ~/skyrl-logs/tinker-api.log 2>/dev/null | grep -E 'File \"|line [0-9]+, in|Error|attribute|raise' | tail -12" 2>/dev/null | tail -12 | sed 's/^/    /'
    fi
  } >> "$RESULTS"
done

tssh 'tmux kill-session -t skyrl-tinker 2>/dev/null || true; true' || true
echo "=== PROBE COMPLETE ==="
cat "$RESULTS"
