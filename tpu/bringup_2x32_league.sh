#!/usr/bin/env bash
# League on TWO v5p-32 slices (capacity race alternative to the single v5p-64):
#   q32 slice: qwen trainer w0 + vLLM w1,2,3       g32 slice: gemma trainer w0 + vLLM w5..7-equivalents
# Client on q32-w0; gemma reached at g32-w0's INTERNAL ip :8000 (same VPC,
# default-allow-internal). Independent preemption domains — losing one slice
# stalls one model, not the experiment.
set -uo pipefail
exec 9>/tmp/sk7524-league32-bringup.lock
flock -n 9 || { echo "another 2x32 bring-up holds the lock; exiting"; exit 0; }
REPO=${REPO:-/n/fs/vision-mix/sk7524/SkyRLTpu-league}
MAIN_REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
RUNS=$REPO/runs/ttd_league; mkdir -p "$RUNS"
KEY=$HOME/.ssh/google_compute_engine
SSHO="-i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25"
GC=/n/fs/vision-mix/sk7524/google-cloud-sdk/bin/gcloud
Z="--project=vision-mix --zone=us-east5-a"
QQR=sk7524-league-q32-east5a_spot
GQR=sk7524-league-g32-east5a_spot
PIP="maxtext @ git+https://github.com/SachinKonan/maxtext.git@skyrl/qwen35-dense"
QCACHE=gs://sk7524-tinker-tpu-us-east5/vllm-xla-cache-22k
GCACHE=gs://sk7524-tinker-tpu-us-east5/vllm-xla-cache-gemma4-31b-16k
PROG=$RUNS/league32.progress; : > "$PROG"
log(){ echo "[$(date -u +%H:%M)] $*" | tee -a "$PROG"; }
ssh_ready(){ timeout 15 ssh $SSHO sk7524_princeton_edu@$1 'true' >/dev/null 2>&1; }
qr_state(){ local s; for _ in 1 2 3; do s=$(timeout 45 $GC alpha compute tpus queued-resources describe "$1" $Z --format="value(state.state)" 2>/dev/null | tail -1); [ -n "$s" ] && { echo "$s"; return; }; sleep 4; done; echo ""; }
qr_create(){
  local name=$1 err rc
  err=$(timeout 120 $GC alpha compute tpus queued-resources create "$name" --node-id="$name" $Z \
    --accelerator-type=v5p-32 --runtime-version=v2-alpha-tpuv5 --provisioning-model=SPOT 2>&1)
  rc=$?
  [ $rc -eq 0 ] && { log "  $name create submitted"; return 0; }
  log "  $name create FAILED: $(echo "$err" | grep -oE '"message": "[^"]+"' | head -1 | cut -c1-140)"
  return 1
}

# 1. both QRs ACTIVE (create/recreate as needed; ~4h budget)
for qr in $QQR $GQR; do st=$(qr_state $qr); [ -z "$st" ] && qr_create $qr; done
log "waiting for q32+g32 ACTIVE..."
for i in $(seq 1 320); do
  qs=$(qr_state $QQR); gs=$(qr_state $GQR)
  [ "$qs" = ACTIVE ] && [ "$gs" = ACTIVE ] && break
  for pair in "$QQR:$qs" "$GQR:$gs"; do
    qr=${pair%%:*}; st=${pair#*:}
    if [ "$st" = SUSPENDED ] || [ "$st" = FAILED ]; then
      log "  $qr state=$st -> delete+recreate"
      timeout 90 $GC alpha compute tpus queued-resources delete "$qr" $Z --quiet --force >/dev/null 2>&1
      sleep 5; qr_create $qr
    elif [ -z "$st" ] && [ $((i % 4)) -eq 1 ]; then qr_create $qr; fi
  done
  [ $((i % 8)) -eq 0 ] && log "  waiting (q32=${qs:-ABSENT} g32=${gs:-ABSENT}, $((i*45/60))min)"
  sleep 45
done
[ "$(qr_state $QQR)" = ACTIVE ] && [ "$(qr_state $GQR)" = ACTIVE ] || { log "2x32 never both ACTIVE"; echo LEAGUE32-NOCAP; exit 1; }
log "both v5p-32 ACTIVE"

# 2. IPs (4 hosts each)
get_ips(){ timeout 40 $GC compute tpus tpu-vm describe "$1" $Z --format="value(networkEndpoints[].$2)" 2>/dev/null | tr ';\t' '\n\n' | grep -E '^[0-9]' | paste -sd, -; }
QEXT=$(get_ips $QQR "accessConfig.externalIp"); QINT=$(get_ips $QQR "ipAddress")
GEXT=$(get_ips $GQR "accessConfig.externalIp"); GINT=$(get_ips $GQR "ipAddress")
QW0=$(echo "$QEXT"|cut -d, -f1); GW0=$(echo "$GEXT"|cut -d, -f1); GW0INT=$(echo "$GINT"|cut -d, -f1)
log "q32 ext=$QEXT"
log "g32 ext=$GEXT (w0 int=$GW0INT)"
[ "$(echo $QEXT|tr ',' '\n'|wc -l)" -eq 4 ] && [ "$(echo $GEXT|tr ',' '\n'|wc -l)" -eq 4 ] || { echo LEAGUE32-BADIPS; exit 1; }
echo "$QEXT" > $RUNS/q32_ext.txt; echo "$QINT" > $RUNS/q32_int.txt
echo "$GEXT" > $RUNS/g32_ext.txt; echo "$GINT" > $RUNS/g32_int.txt

# 3. keys + provision all 8 hosts (both slices in parallel)
log "keys + provision 8 hosts across 2 slices..."
for W in 0 1 2 3; do
  timeout 90 $GC alpha compute tpus tpu-vm ssh sk7524_princeton_edu@"$QQR" $Z --worker=$W --ssh-key-file="$KEY" --command 'true' >/dev/null 2>&1 &
  timeout 90 $GC alpha compute tpus tpu-vm ssh sk7524_princeton_edu@"$GQR" $Z --worker=$W --ssh-key-file="$KEY" --command 'true' >/dev/null 2>&1 &
done; wait
rdy=1; for w in $(echo "$QEXT,$GEXT"|tr ',' ' '); do ok=0; for j in $(seq 1 40); do ssh_ready "$w" && { ok=1; break; }; sleep 15; done; [ "$ok" = 1 ]||rdy=0; done
[ "$rdy" = 1 ] || { log "SSH not ready on all hosts"; echo LEAGUE32-SSHFAIL; exit 1; }
for w in $(echo "$QEXT,$GEXT"|tr ',' ' '); do
  timeout 60 scp $SSHO $REPO/tpu/provision_tpu_worker.sh sk7524_princeton_edu@$w:~/ >/dev/null 2>&1
  timeout 600 ssh $SSHO sk7524_princeton_edu@$w 'for i in $(seq 1 60); do sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1||break; sleep 5; done; bash ~/provision_tpu_worker.sh >/tmp/prov.log 2>&1' >/dev/null 2>&1 &
done
wait
log "provision done"

# 4a. QWEN slice (q32: w0 trainer + w1,2,3 vLLM)
log "qwen slice bring-up (q32): uniform=18432 budget=73728..."
env TPU_SSH_MODE=direct TPU_EXTERNAL_IPS="$QEXT" TPU_INTERNAL_IPS="$QINT" TPU_NAME="$QQR" \
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
  bash "$REPO/tpu/start_colocated_vllm_tinker.sh" > "$RUNS/q32_engine.log" 2>&1
qk=$(timeout 20 ssh $SSHO sk7524_princeton_edu@$QW0 'curl -fsS -m6 http://127.0.0.1:8000/api/v1/get_server_capabilities >/dev/null 2>&1 && echo UP || echo DOWN' 2>/dev/null|grep -v Warning)
log "qwen tinker=$qk"
[ "$qk" = UP ] || { log "qwen slice FAILED -- see q32_engine.log"; echo LEAGUE32-QWENFAIL; exit 1; }

# 4b-pre. gemma GCS cache self-heal (same guard as the v5p-64 script)
GW5=$(echo "$GEXT"|cut -d, -f2)
GREV=842da3794eaa0b77d5f08bae87a17459d91ff475
if ! timeout 60 ssh $SSHO sk7524_princeton_edu@$GW5 "gsutil -q stat gs://sk7524-tinker-tpu-us-east5/hf-cache-gemma4/models--google--gemma-4-31B-it/snapshots/$GREV/model-00001-of-00002.safetensors" >/dev/null 2>&1; then
  log "gemma GCS cache missing shard-1 -> repair on g32-w1..."
  timeout 2400 ssh $SSHO sk7524_princeton_edu@$GW5 "
    SNAP=~/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/$GREV
    rm -f ~/.cache/huggingface/hub/models--google--gemma-4-31B-it/blobs/*.incomplete 2>/dev/null
    mkdir -p \"\$SNAP\"
    curl -fsSL --retry 3 -o \"\$SNAP/model-00001-of-00002.safetensors\" \
      'https://huggingface.co/google/gemma-4-31B-it/resolve/$GREV/model-00001-of-00002.safetensors' \
      >/tmp/shard_fix.log 2>&1 || { tail -2 /tmp/shard_fix.log; exit 1; }
    sz=\$(stat -c%s \"\$SNAP/model-00001-of-00002.safetensors\")
    [ \"\$sz\" -gt 40000000000 ] || { echo \"shard1 too small: \$sz\"; exit 1; }
    gsutil -q rm 'gs://sk7524-tinker-tpu-us-east5/hf-cache-gemma4/models--google--gemma-4-31B-it/blobs/*.incomplete' 2>/dev/null
    gsutil -q cp \"\$SNAP/model-00001-of-00002.safetensors\" 'gs://sk7524-tinker-tpu-us-east5/hf-cache-gemma4/models--google--gemma-4-31B-it/snapshots/$GREV/model-00001-of-00002.safetensors'
    echo SHARD1-REPAIRED
  " 2>/dev/null | grep -v Warning | tee -a "$PROG"
  grep -q SHARD1-REPAIRED "$PROG" || { log "gemma GCS repair FAILED"; echo LEAGUE32-GEMMAREPAIRFAIL; exit 1; }
else
  log "gemma GCS cache complete (shard-1 present)"
fi

# 4b-pre2. Per-worker local weight-cache pre-sync + VERIFY. A worker whose
# restore misfires makes vLLM fall back to a HF-hub xet download that blows the
# disk (seen live on g32-w3). Sync + verify BOTH shards on every gemma host
# up front so the in-script restore is a no-op and the fallback can't trigger.
log "gemma workers: pre-syncing + verifying local HF caches..."
gver=1
for w in $(echo "$GEXT"|cut -d, -f2-|tr ',' ' '); do   # vLLM workers ONLY — the trainer host (w0) loads orbax, and 60G of HF weights there overflows its disk (seen live: uv ENOSPC)
  timeout 1200 ssh $SSHO sk7524_princeton_edu@$w "
    mkdir -p ~/.cache/huggingface/hub
    gsutil -q -m rsync -r gs://sk7524-tinker-tpu-us-east5/hf-cache-gemma4 ~/.cache/huggingface/hub 2>/dev/null
    s1=\$(stat -Lc%s ~/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/$GREV/model-00001-of-00002.safetensors 2>/dev/null || echo 0)
    s2=\$(stat -Lc%s ~/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/$GREV/model-00002-of-00002.safetensors 2>/dev/null || echo 0)
    [ \"\$s1\" -gt 40000000000 ] && [ \"\$s2\" -gt 10000000000 ] && echo \"CACHE-OK \$(hostname)\" || echo \"CACHE-BAD \$(hostname) s1=\$s1 s2=\$s2\"
  " 2>/dev/null | grep -v Warning | tee -a "$PROG" &
done
wait
grep -q "CACHE-BAD" "$PROG" && { log "gemma worker cache verification FAILED"; echo LEAGUE32-GEMMACACHEFAIL; exit 1; }

# 4b. GEMMA slice (g32: w0 trainer + w1,2,3 vLLM)
log "gemma slice bring-up (g32): uniform=12288 + vocab tiling..."
env TPU_SSH_MODE=direct TPU_EXTERNAL_IPS="$GEXT" TPU_INTERNAL_IPS="$GINT" TPU_NAME="$GQR" \
  PROJECT=vision-mix ZONE=us-east5-a REMOTE_USER=sk7524_princeton_edu SSH_KEY_FILE="$KEY" \
  TINKER_BACKEND=tunix TRAIN_WORKERS=0 VLLM_WORKERS=1,2,3 VLLM_RAY_EXECUTOR=0 VLLM_CLIENT_SIDE_ROUND_ROBIN=1 \
  MODEL_NAME=google/gemma-4-31B-it TUNIX_MAXTEXT_MODEL_NAME=gemma4-31b TUNIX_MAXTEXT_PIP_SPEC="$PIP" \
  TUNIX_MAXTEXT_KWARGS='{"num_vocab_tiling": 32}' \
  TUNIX_MAX_TARGET_LENGTH=10240 TUNIX_TRAIN_TOKEN_BUDGET=40960 TUNIX_FLCE_TILE_SIZE=1024 TRAIN_MICRO_BATCH_SIZE=1 \
  TUNIX_UNIFORM_SEQ_LEN=10240 TUNIX_SEQ_BUCKETS="4096,8192,12288,16384,20480" TUNIX_MINIMAL_FB_OUTPUT=1 \
  VLLM_MAX_MODEL_LEN=16384 VLLM_MAX_NUM_SEQS=128 VLLM_XLA_CACHE_PATH=/home/sk7524_princeton_edu/vllm-xla-cache-local \
  VLLM_XLA_CACHE_GCS="$GCACHE" HF_CACHE_GCS="gs://sk7524-tinker-tpu-us-east5/hf-cache-gemma4" \
  VLLM_EXTRA_ARGS="--max-num-batched-tokens 8192 --gpu-memory-utilization 0.85" \
  READY_ATTEMPTS=2000 SYNC_SKYRL=1 START_VLLM=1 START_TINKER=1 \
  bash "$REPO/tpu/start_colocated_vllm_tinker.sh" > "$RUNS/g32_engine.log" 2>&1
gk=$(timeout 25 ssh $SSHO sk7524_princeton_edu@$QW0 "curl -fsS -m8 http://$GW0INT:8000/api/v1/get_server_capabilities >/dev/null 2>&1 && echo UP || echo DOWN" 2>/dev/null|grep -v Warning)
log "gemma tinker (cross-slice from q32-w0 via $GW0INT)=$gk"
if [ "$gk" != UP ]; then
  gk2=$(timeout 25 ssh $SSHO sk7524_princeton_edu@$GW0 'curl -fsS -m8 http://127.0.0.1:8000/api/v1/get_server_capabilities >/dev/null 2>&1 && echo UP || echo DOWN' 2>/dev/null|grep -v Warning)
  log "gemma tinker local-check=$gk2 (cross-slice route blocked if UP here)"
  echo LEAGUE32-GEMMAROUTEFAIL; exit 1
fi

# 5. client on q32-w0
log "client provisioning on q32-w0..."
TAR=/tmp/league32-client-$$.tar.gz
cp "$MAIN_REPO/third_party/discover/.env" "$REPO/third_party/discover/.env" 2>/dev/null || true
tar czf "$TAR" -C "$REPO" --exclude=.git --exclude='.venv*' --exclude=runs --exclude='*.log' .
timeout 300 scp $SSHO "$TAR" sk7524_princeton_edu@$QW0:/tmp/league-client.tar.gz >/dev/null 2>&1
rm -f "$TAR"
timeout 120 ssh $SSHO sk7524_princeton_edu@$QW0 'rm -rf ~/ttd-client && mkdir -p ~/ttd-client && tar xzf /tmp/league-client.tar.gz -C ~/ttd-client && ls ~/ttd-client/third_party/discover/.env >/dev/null 2>&1 && echo env-ok || echo env-MISSING' 2>&1 | grep -v Warning | tee -a "$PROG"
timeout 900 ssh $SSHO sk7524_princeton_edu@$QW0 'export PATH=$HOME/.local/bin:$PATH; cd ~/ttd-client/third_party/discover && uv sync --extra math --python 3.11 >~/venv-build.log 2>&1; rc=$?; ln -sfn .venv .venv-ttd-discover; .venv-ttd-discover/bin/python -c "import tinker,numpy,wandb;print(\"venv-ok\")" 2>&1|tail -1; echo venv-rc=$rc' 2>&1 | grep -v Warning | tee -a "$PROG"

# 6. Ray grading cluster (built + started but INERT until the run flips
#    TTD_EVAL_BACKEND=ray + TTD_RAY_PAYLOAD=1). Head runs from the client venv
#    on q32-w0 (version-matches the client); workers join from a dedicated
#    grader venv on every other host (pinned ray + the math deps the graded
#    programs import via sys.executable).
log "ray grading cluster: grader venvs + head + workers..."
QW0INT=$(echo "$QINT"|cut -d, -f1)
RAYV=$(timeout 60 ssh $SSHO sk7524_princeton_edu@$QW0 '~/ttd-client/third_party/discover/.venv-ttd-discover/bin/python -c "import ray; print(ray.__version__)"' 2>/dev/null | grep -v Warning | tail -1)
log "  ray version pin: ${RAYV:-UNKNOWN}"
if [ -n "$RAYV" ]; then
  timeout 120 ssh $SSHO sk7524_princeton_edu@$QW0 '
    pkill -f "ray/core" 2>/dev/null; sleep 2
    ~/ttd-client/third_party/discover/.venv-ttd-discover/bin/ray start --head --port=6379 --num-cpus=0 --disable-usage-stats >/tmp/ray-head.log 2>&1
    echo RAY-HEAD-STARTED' 2>/dev/null | grep -v Warning | tee -a "$PROG"
  for w in $(echo "$QEXT,$GEXT"|tr ',' ' ' ); do
    [ "$w" = "$QW0" ] && continue
    timeout 900 ssh $SSHO sk7524_princeton_edu@$w "
      export PATH=\$HOME/.local/bin:\$PATH
      [ -x ~/.venvs/grader/bin/ray ] || {
        uv venv ~/.venvs/grader --python 3.11 >/dev/null 2>&1
        uv pip install --python ~/.venvs/grader/bin/python 'ray==$RAYV' numpy scipy shapely numba scikit-learn psutil >/dev/null 2>&1
      }
      pkill -f 'ray/core' 2>/dev/null; sleep 2
      ~/.venvs/grader/bin/ray start --address=$QW0INT:6379 --num-cpus=150 --disable-usage-stats >/tmp/ray-worker.log 2>&1 && echo \"RAY-WORKER \$(hostname)\"
    " 2>/dev/null | grep -v Warning | tee -a "$PROG" &
  done
  wait
  nworkers=$(grep -c "RAY-WORKER" "$PROG")
  log "  ray cluster: head + $nworkers workers"
else
  log "  ray version undetectable; skipping cluster (grading stays local)"
fi

log "LEAGUE32-ENGINES-UP  q32-w0=$QW0  gemma_internal=$GW0INT"
echo "LEAGUE32-ENGINES-UP"
