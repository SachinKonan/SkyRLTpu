#!/usr/bin/env bash
# Bring up ONE Stage-A cell on a single v5p-32: qwen trainer w0 + vLLM w1,2,3.
#
# One cell per slice (rather than two cells sharing a v5p-64) so a preemption
# costs one experimental condition instead of two, cells do not contend for a
# shared Ray grading cluster or HBM, and the topology exactly matches the one the
# ~2.0-2.4 h/step measurements came from. Chip cost is identical: 6x32 == 3x64.
#
# Engine settings below are copied verbatim from the proven qwen half of
# bringup_2x32_league.sh -- do not "clean them up", they are load-bearing.
#
# Usage: CELL=grpo-n bash tpu/bringup_v5p32_cell.sh
set -uo pipefail
CELL=${CELL:?set CELL, e.g. grpo-n}
REPO=${REPO:-/n/fs/vision-mix/sk7524/SkyRLTpu-league}
MAIN_REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
RUNS=$REPO/runs/stage_a; mkdir -p "$RUNS"
KEY=$HOME/.ssh/google_compute_engine
SSHO="-i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25"
GC=/n/fs/vision-mix/sk7524/google-cloud-sdk/bin/gcloud
Z="--project=vision-mix --zone=us-east5-a"
QR=sk7524-stagea-${CELL}-east5a_spot
PIP="maxtext @ git+https://github.com/SachinKonan/maxtext.git@skyrl/qwen35-dense"
QCACHE=gs://sk7524-tinker-tpu-us-east5/vllm-xla-cache-22k
PROG=$RUNS/${CELL}.progress; : > "$PROG"
log(){ echo "[$(date -u +%H:%M)] $*" | tee -a "$PROG"; }

exec 9>/tmp/sk7524-stagea-$CELL.lock
flock -n 9 || { echo "another bring-up holds the lock for $CELL"; exit 0; }

# shellcheck source=/dev/null
source "$REPO/tpu/stage_a_cells.sh"
CELL_ENV=$(stage_a_env "$CELL") || exit 1
log "cell $CELL -> $(echo "$CELL_ENV" | tr '\n' ' ')"

ssh_ready(){ timeout 15 ssh $SSHO sk7524_princeton_edu@$1 'true' >/dev/null 2>&1; }
qr_state(){ local s; for _ in 1 2 3; do s=$(timeout 45 $GC alpha compute tpus queued-resources describe "$1" $Z --format="value(state.state)" 2>/dev/null | tail -1); [ -n "$s" ] && { echo "$s"; return; }; sleep 4; done; echo ""; }
qr_create(){
  local err rc
  err=$(timeout 120 $GC alpha compute tpus queued-resources create "$1" --node-id="$1" $Z \
    --accelerator-type=v5p-32 --runtime-version=v2-alpha-tpuv5 --provisioning-model=SPOT 2>&1)
  rc=$?
  [ $rc -eq 0 ] && { log "  $1 create submitted"; return 0; }
  log "  $1 create FAILED: $(echo "$err" | grep -oE '"message": "[^"]+"' | head -1 | cut -c1-140)"
  return 1
}

# 1. QR ACTIVE
st=$(qr_state $QR); [ -z "$st" ] && qr_create $QR
log "waiting for $QR ACTIVE..."
for i in $(seq 1 320); do
  st=$(qr_state $QR)
  [ "$st" = ACTIVE ] && break
  if [ "$st" = SUSPENDED ] || [ "$st" = FAILED ]; then
    log "  state=$st -> delete+recreate"
    timeout 90 $GC alpha compute tpus queued-resources delete "$QR" $Z --quiet --force >/dev/null 2>&1
    sleep 5; qr_create $QR
  elif [ -z "$st" ] && [ $((i % 4)) -eq 1 ]; then qr_create $QR; fi
  [ $((i % 8)) -eq 0 ] && log "  waiting (state=${st:-ABSENT}, $((i*45/60))min)"
  sleep 45
done
[ "$(qr_state $QR)" = ACTIVE ] || { log "$QR never ACTIVE"; echo "CELL-NOCAP $CELL"; exit 1; }
log "v5p-32 ACTIVE"

# 2. IPs (4 hosts)
get_ips(){ timeout 40 $GC compute tpus tpu-vm describe "$1" $Z --format="value(networkEndpoints[].$2)" 2>/dev/null | tr ';\t' '\n\n' | grep -E '^[0-9]' | paste -sd, -; }
EXT=$(get_ips $QR "accessConfig.externalIp"); INT=$(get_ips $QR "ipAddress")
W0=$(echo "$EXT"|cut -d, -f1); W0INT=$(echo "$INT"|cut -d, -f1)
log "ext=$EXT"
[ "$(echo $EXT|tr ',' '\n'|wc -l)" -eq 4 ] || { echo "CELL-BADIPS $CELL"; exit 1; }

# 3. keys + provision 4 hosts
log "keys + provision 4 hosts..."
for W in 0 1 2 3; do
  timeout 90 $GC alpha compute tpus tpu-vm ssh sk7524_princeton_edu@"$QR" $Z --worker=$W --ssh-key-file="$KEY" --command 'true' >/dev/null 2>&1 &
done; wait
rdy=1; for w in $(echo "$EXT"|tr ',' ' '); do ok=0; for j in $(seq 1 40); do ssh_ready "$w" && { ok=1; break; }; sleep 15; done; [ "$ok" = 1 ]||rdy=0; done
[ "$rdy" = 1 ] || { log "SSH not ready"; echo "CELL-SSHFAIL $CELL"; exit 1; }
for w in $(echo "$EXT"|tr ',' ' '); do
  timeout 60 scp $SSHO $REPO/tpu/provision_tpu_worker.sh sk7524_princeton_edu@$w:~/ >/dev/null 2>&1
  timeout 600 ssh $SSHO sk7524_princeton_edu@$w 'for i in $(seq 1 60); do sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1||break; sleep 5; done; bash ~/provision_tpu_worker.sh >/tmp/prov.log 2>&1' >/dev/null 2>&1 &
done
wait
log "provision done"

# 4. engines: trainer w0 + vLLM w1,2,3 (verbatim from the proven q32 recipe)
log "engine bring-up: uniform=18432 budget=73728..."
env TPU_SSH_MODE=direct TPU_EXTERNAL_IPS="$EXT" TPU_INTERNAL_IPS="$INT" TPU_NAME="$QR" \
  PROJECT=vision-mix ZONE=us-east5-a REMOTE_USER=sk7524_princeton_edu SSH_KEY_FILE="$KEY" \
  TINKER_BACKEND=tunix TRAIN_WORKERS=0 VLLM_WORKERS=1,2,3 VLLM_RAY_EXECUTOR=0 VLLM_CLIENT_SIDE_ROUND_ROBIN=1 \
  MODEL_NAME=Qwen/Qwen3.5-27B TUNIX_MAXTEXT_MODEL_NAME=qwen3.5-27b TUNIX_MAXTEXT_PIP_SPEC="$PIP" \
  TUNIX_MAXTEXT_KWARGS='{"num_vocab_tiling": 8}' \
  TUNIX_MAX_TARGET_LENGTH=22528 TUNIX_TRAIN_TOKEN_BUDGET=73728 TUNIX_FLCE_TILE_SIZE=2048 TRAIN_MICRO_BATCH_SIZE=1 \
  TUNIX_UNIFORM_SEQ_LEN=18432 TUNIX_SEQ_BUCKETS="4096,8192,12288,16384,20480" TUNIX_MINIMAL_FB_OUTPUT=1 \
  VLLM_MAX_MODEL_LEN=22528 VLLM_MAX_NUM_SEQS=128 VLLM_XLA_CACHE_PATH=/home/sk7524_princeton_edu/vllm-xla-cache-local \
  VLLM_XLA_CACHE_GCS="$QCACHE" HF_CACHE_GCS="gs://sk7524-tinker-tpu-us-east5/hf-cache" \
  VLLM_EXTRA_ARGS="--max-num-batched-tokens 8192 --gpu-memory-utilization 0.85" \
  READY_ATTEMPTS=900 SYNC_SKYRL=1 START_VLLM=1 START_TINKER=1 \
  bash "$REPO/tpu/start_colocated_vllm_tinker.sh" > "$RUNS/${CELL}_engine.log" 2>&1
qk=$(timeout 20 ssh $SSHO sk7524_princeton_edu@$W0 'curl -fsS -m6 http://127.0.0.1:8000/api/v1/get_server_capabilities >/dev/null 2>&1 && echo UP || echo DOWN' 2>/dev/null|grep -v Warning)
log "tinker=$qk"
[ "$qk" = UP ] || { log "engine FAILED -- see ${CELL}_engine.log"; echo "CELL-ENGINEFAIL $CELL"; exit 1; }

# 5. client on w0
log "client provisioning on w0..."
TAR=/tmp/cell-$CELL-$$.tar.gz
cp "$MAIN_REPO/third_party/discover/.env" "$REPO/third_party/discover/.env" 2>/dev/null || true
tar czf "$TAR" -C "$REPO" --exclude=.git --exclude='.venv*' --exclude=runs --exclude='*.log' .
timeout 300 scp $SSHO "$TAR" sk7524_princeton_edu@$W0:/tmp/cell-client.tar.gz >/dev/null 2>&1
rm -f "$TAR"
timeout 120 ssh $SSHO sk7524_princeton_edu@$W0 'rm -rf ~/ttd-client && mkdir -p ~/ttd-client && tar xzf /tmp/cell-client.tar.gz -C ~/ttd-client && ls ~/ttd-client/third_party/discover/.env >/dev/null 2>&1 && echo env-ok || echo env-MISSING' 2>&1 | grep -v Warning | tee -a "$PROG"
timeout 900 ssh $SSHO sk7524_princeton_edu@$W0 'export PATH=$HOME/.local/bin:$PATH; cd ~/ttd-client/third_party/discover && uv sync --extra math --python 3.11 >~/venv-build.log 2>&1; rc=$?; ln -sfn .venv .venv-ttd-discover; .venv-ttd-discover/bin/python -c "import tinker,numpy,wandb;print(\"venv-ok\")" 2>&1|tail -1; echo venv-rc=$rc' 2>&1 | grep -v Warning | tee -a "$PROG"

# 6. ray grading cluster: head on w0 (num-cpus=0), workers w1-3
log "ray grading cluster..."
RAYV=$(timeout 60 ssh $SSHO sk7524_princeton_edu@$W0 '~/ttd-client/third_party/discover/.venv-ttd-discover/bin/python -c "import ray; print(ray.__version__)"' 2>/dev/null | grep -v Warning | tail -1)
log "  ray version pin: ${RAYV:-UNKNOWN}"
if [ -n "$RAYV" ]; then
  timeout 120 ssh $SSHO sk7524_princeton_edu@$W0 '
    pkill -f "ray/core" 2>/dev/null; sleep 2
    ~/ttd-client/third_party/discover/.venv-ttd-discover/bin/ray start --head --port=6379 --num-cpus=0 --disable-usage-stats >/tmp/ray-head.log 2>&1
    echo RAY-HEAD-STARTED' 2>/dev/null | grep -v Warning | tee -a "$PROG"
  for w in $(echo "$EXT"|tr ',' ' '); do
    [ "$w" = "$W0" ] && continue
    timeout 900 ssh $SSHO sk7524_princeton_edu@$w "
      export PATH=\$HOME/.local/bin:\$PATH
      [ -x ~/.venvs/grader/bin/ray ] || {
        uv venv ~/.venvs/grader --python 3.11 >/dev/null 2>&1
        uv pip install --python ~/.venvs/grader/bin/python 'ray==$RAYV' numpy scipy shapely numba scikit-learn psutil >/dev/null 2>&1
      }
      pkill -f 'ray/core' 2>/dev/null; sleep 2
      ~/.venvs/grader/bin/ray start --address=$W0INT:6379 --num-cpus=150 --disable-usage-stats >/tmp/ray-worker.log 2>&1 && echo \"RAY-WORKER \$(hostname)\"
    " 2>/dev/null | grep -v Warning | tee -a "$PROG" &
  done
  wait
  log "  ray: head + $(grep -c 'RAY-WORKER' "$PROG") workers"
fi

log "CELL-ENGINES-UP $CELL w0=$W0"

# 7. launch the client with this cell's knobs
log "launching client..."
timeout 300 ssh $SSHO sk7524_princeton_edu@$W0 \
  "CELL=$CELL $(echo "$CELL_ENV" | tr '\n' ' ') bash ~/ttd-client/tpu/launch_cell.sh" \
  2>&1 | grep -v Warning | tee -a "$PROG"
