#!/usr/bin/env bash
# v5p-64 LEAGUE slice: qwen3.5-27b half (w0 trainer + w1,2,3 vLLM) + gemma4-31b
# half (w4 trainer + w5,6,7 vLLM). ONE spot QR; TWO invocations of
# start_colocated_vllm_tinker.sh against the same 8-IP list (the start script
# supports non-zero trainer indices: CLOUD_TPU_TASK_ID is remapped to the
# in-group process index). Client runs on w0: qwen at localhost:8000, gemma at
# <w4-internal>:8000 (tinker binds 0.0.0.0). Both halves use warm GCS caches.
set -uo pipefail
# Singleton: two concurrent bring-ups race on the same hosts (seen live —
# duplicated provision/start invocations wrecked the qwen half). /tmp lock is
# local to the launching login node, which is where every launch happens.
exec 9>/tmp/sk7524-league-bringup-b.lock
flock -n 9 || { echo "another bring-up instance holds the lock; exiting"; exit 0; }
REPO=${REPO:-/n/fs/vision-mix/sk7524/SkyRLTpu-league}
MAIN_REPO=/n/fs/vision-mix/sk7524/SkyRLTpu   # .env source (never printed)
RUNS=$REPO/runs/ttd_league; mkdir -p "$RUNS"
KEY=$HOME/.ssh/google_compute_engine
SSHO="-i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25"
GC=/n/fs/vision-mix/sk7524/google-cloud-sdk/bin/gcloud
Z="--project=vision-mix --zone=us-east5-a"
V64=${V64:-sk7524-league-v5p64b-east5a_spot}
PIP="maxtext @ git+https://github.com/SachinKonan/maxtext.git@skyrl/qwen35-dense"
QCACHE=gs://sk7524-tinker-tpu-us-east5/vllm-xla-cache-22k
GCACHE=gs://sk7524-tinker-tpu-us-east5/vllm-xla-cache-gemma4-31b-16k
PROG=$RUNS/v5p64b_league.progress; : > "$PROG"
log(){ echo "[$(date -u +%H:%M)] $*" | tee -a "$PROG"; }
ssh_ready(){ timeout 15 ssh $SSHO sk7524_princeton_edu@$1 'true' >/dev/null 2>&1; }
qr_state(){ local s; for _ in 1 2 3; do s=$(timeout 45 $GC alpha compute tpus queued-resources describe "$V64" $Z --format="value(state.state)" 2>/dev/null | tail -1); [ -n "$s" ] && { echo "$s"; return; }; sleep 4; done; echo ""; }
qr_create() {
  local err rc
  err=$(timeout 120 $GC alpha compute tpus queued-resources create "$V64" --node-id="$V64" $Z \
    --accelerator-type=v5p-64 --runtime-version=v2-alpha-tpuv5 --provisioning-model=SPOT 2>&1)
  rc=$?
  if [ $rc -eq 0 ]; then log "  QR create submitted"; return 0; fi
  log "  QR create FAILED: $(echo "$err" | grep -oE '"message": "[^"]+"|ERROR:[^<]*' | head -1 | cut -c1-160)"
  return 1
}

# 1. ensure the v5p-64 ACTIVE. Retry creates while ABSENT (zone QR quota 429s
#    resolve when stale QRs get deleted); surface every refusal reason.
log "ensuring v5p-64 ACTIVE ($V64)..."
st=$(qr_state); [ -z "$st" ] && qr_create
active=0
for i in $(seq 1 160); do
  st=$(qr_state)
  [ "$st" = ACTIVE ] && { active=1; break; }
  if [ "$st" = SUSPENDED ] || [ "$st" = FAILED ]; then
    log "  state=$st -> delete+recreate"
    timeout 90 $GC alpha compute tpus queued-resources delete "$V64" $Z --quiet --force >/dev/null 2>&1
    sleep 5; qr_create
  elif [ -z "$st" ] && [ $((i % 4)) -eq 1 ]; then
    qr_create
  else
    [ $((i % 8)) -eq 0 ] && log "  waiting (state=${st:-ABSENT}, $((i*45/60))min)"
  fi
  sleep 45
done
[ "$active" = 1 ] || { log "v5p-64 never went ACTIVE"; echo V5P64-NOCAP; exit 1; }
log "v5p-64 ACTIVE"

# 2. IPs (8 hosts)
EXT=$(timeout 40 $GC compute tpus tpu-vm describe "$V64" $Z --format="value(networkEndpoints[].accessConfig.externalIp)" 2>/dev/null | tr ';\t' '\n\n' | grep -E '^[0-9]' | paste -sd, -)
INT=$(timeout 40 $GC compute tpus tpu-vm describe "$V64" $Z --format="value(networkEndpoints[].ipAddress)" 2>/dev/null | tr ';\t' '\n\n' | grep -E '^[0-9]' | paste -sd, -)
W0=$(echo "$EXT"|cut -d, -f1)
W4INT=$(echo "$INT"|cut -d, -f5)
log "IPs ext=$EXT"
[ "$(echo $EXT|tr ',' '\n'|wc -l)" -eq 8 ] || { log "expected 8 hosts, got: $EXT"; echo V5P64-BADIPS; exit 1; }
echo "$EXT" > $RUNS/v5p64_ext.txt; echo "$INT" > $RUNS/v5p64_int.txt

# 3. keys + provision 8 hosts
log "keys + provision 8 hosts..."
for W in 0 1 2 3 4 5 6 7; do timeout 90 $GC alpha compute tpus tpu-vm ssh sk7524_princeton_edu@"$V64" $Z --worker=$W --ssh-key-file="$KEY" --command 'true' >/dev/null 2>&1 & done; wait
rdy=1; for w in $(echo $EXT|tr ',' ' '); do ok=0; for j in $(seq 1 40); do ssh_ready "$w" && { ok=1; break; }; sleep 15; done; [ "$ok" = 1 ]||rdy=0; done
[ "$rdy" = 1 ] || { log "SSH not ready on all hosts"; echo V5P64-SSHFAIL; exit 1; }
for w in $(echo $EXT|tr ',' ' '); do
  timeout 60 scp $SSHO $REPO/tpu/provision_tpu_worker.sh sk7524_princeton_edu@$w:~/ >/dev/null 2>&1
  timeout 600 ssh $SSHO sk7524_princeton_edu@$w 'for i in $(seq 1 60); do sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1||break; sleep 5; done; bash ~/provision_tpu_worker.sh >/tmp/prov.log 2>&1' >/dev/null 2>&1 &
done
wait
ssh_ready "$W0" || { log "lost after provision"; echo V5P64-LOST; exit 1; }
log "provision done"

# 4a. QWEN half: trainer w0 + vLLM w1,2,3 (proven sweep recipe: uniform 18432)
log "qwen half bring-up (w0 + w1,2,3): uniform=18432 budget=73728 (warm caches)..."
env TPU_SSH_MODE=direct TPU_EXTERNAL_IPS="$EXT" TPU_INTERNAL_IPS="$INT" TPU_NAME="$V64" \
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
  bash "$REPO/tpu/start_colocated_vllm_tinker.sh" > "$RUNS/qwenb_engine.log" 2>&1
qk=$(timeout 20 ssh $SSHO sk7524_princeton_edu@$W0 'curl -fsS -m6 http://127.0.0.1:8000/api/v1/get_server_capabilities >/dev/null 2>&1 && echo UP || echo DOWN' 2>/dev/null|grep -v Warning)
log "qwen tinker=$qk"
[ "$qk" = UP ] || { log "qwen half FAILED -- see qwen_engine.log"; echo V5P64-QWENFAIL; exit 1; }

# 4b-pre. GEMMA GCS cache self-heal: gs://.../hf-cache-gemma4 was seeded from a
# mid-download worker and holds shard-1 only as a poisoned *.incomplete blob
# (missing model-00001-of-00002.safetensors). If the complete shard is absent
# in GCS, download it ONCE on w5 (xet disabled — xet's chunk reconstruction
# doubles disk and crashed repeatedly) and repair the bucket, so the normal
# restore in 4b (and every future bring-up) just works.
W5EXT=$(echo "$EXT"|cut -d, -f6)
GREV=842da3794eaa0b77d5f08bae87a17459d91ff475
if ! timeout 60 ssh $SSHO sk7524_princeton_edu@$W5EXT "gsutil -q stat gs://sk7524-tinker-tpu-us-east5/hf-cache-gemma4/models--google--gemma-4-31B-it/snapshots/$GREV/model-00001-of-00002.safetensors" >/dev/null 2>&1; then
  log "gemma GCS cache missing shard-1 -> one-time repair on w5 (curl download + upload, ~15min)..."
  timeout 2400 ssh $SSHO sk7524_princeton_edu@$W5EXT "
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
  grep -q SHARD1-REPAIRED "$PROG" || { log "gemma GCS repair FAILED"; echo V5P64-GEMMAREPAIRFAIL; exit 1; }
else
  log "gemma GCS cache already complete (shard-1 present)"
fi

# 4b. GEMMA half: trainer w4 + vLLM w5,6,7 (validated uniform-10240 config)
log "gemma half bring-up (w4 + w5,6,7): uniform=10240 tile=1024 nvt=32 (warm caches)..."
env TPU_SSH_MODE=direct TPU_EXTERNAL_IPS="$EXT" TPU_INTERNAL_IPS="$INT" TPU_NAME="$V64" \
  PROJECT=vision-mix ZONE=us-east5-a REMOTE_USER=sk7524_princeton_edu SSH_KEY_FILE="$KEY" \
  TINKER_BACKEND=tunix TRAIN_WORKERS=4 VLLM_WORKERS=5,6,7 VLLM_RAY_EXECUTOR=0 VLLM_CLIENT_SIDE_ROUND_ROBIN=1 \
  MODEL_NAME=google/gemma-4-31B-it TUNIX_MAXTEXT_MODEL_NAME=gemma4-31b TUNIX_MAXTEXT_PIP_SPEC="$PIP" \
  TUNIX_MAXTEXT_KWARGS='{"num_vocab_tiling": 32}' \
  TUNIX_MAX_TARGET_LENGTH=10240 TUNIX_TRAIN_TOKEN_BUDGET=40960 TUNIX_FLCE_TILE_SIZE=1024 TRAIN_MICRO_BATCH_SIZE=1 \
  TUNIX_UNIFORM_SEQ_LEN=10240 TUNIX_SEQ_BUCKETS="4096,8192,12288,16384,20480" TUNIX_MINIMAL_FB_OUTPUT=1 \
  VLLM_MAX_MODEL_LEN=16384 VLLM_MAX_NUM_SEQS=128 VLLM_XLA_CACHE_PATH=/home/sk7524_princeton_edu/vllm-xla-cache-local \
  VLLM_XLA_CACHE_GCS="$GCACHE" HF_CACHE_GCS="gs://sk7524-tinker-tpu-us-east5/hf-cache-gemma4" \
  VLLM_EXTRA_ARGS="--max-num-batched-tokens 8192 --gpu-memory-utilization 0.85" \
  READY_ATTEMPTS=2000 SYNC_SKYRL=1 START_VLLM=1 START_TINKER=1 \
  bash "$REPO/tpu/start_colocated_vllm_tinker.sh" > "$RUNS/gemmab_engine.log" 2>&1
gk=$(timeout 20 ssh $SSHO sk7524_princeton_edu@$W0 "curl -fsS -m6 http://$W4INT:8000/api/v1/get_server_capabilities >/dev/null 2>&1 && echo UP || echo DOWN" 2>/dev/null|grep -v Warning)
log "gemma tinker (from w0 via $W4INT)=$gk"
[ "$gk" = UP ] || { log "gemma half FAILED -- see gemma_engine.log"; echo V5P64-GEMMAFAIL; exit 1; }

# 4z. trainer-state durability: checkpoint tarballs are already durable (the
# server's --checkpoints-base is the gcsfuse bucket mount), but the per-VM
# sqlite registry (models/checkpoints tables) dies with the slice and makes
# create_training_client_from_state 404. Re-register rows for every state the
# run's checkpoints.jsonl mentions whose tarball exists.
GCS_RUN=gs://sk7524-tinker-tpu-us-east5/skyrl-runs/league1b
for spec in "member_qwen:1:Qwen/Qwen3.5-27B" "member_gemma:5:google/gemma-4-31B-it"; do
  md=$(echo "$spec"|cut -d: -f1); f=$(echo "$spec"|cut -d: -f2); bm=$(echo "$spec"|cut -d: -f3)
  h=$(echo "$EXT"|cut -d, -f$f)
  # A FAILED FETCH IS NOT "NO RECORDS". Expired gcloud auth makes this cp fail,
  # and treating that as a fresh lineage silently discards a real weight history
  # (hit the 4-agent 2026-08-08). Distinguish the two: only declare fresh when
  # gsutil says the object does not exist.
  _err=$(gsutil -q cp "$GCS_RUN/tinker_log/*/$md/checkpoints.jsonl" /tmp/rr_b_$md.jsonl 2>&1)
  if [ ! -s /tmp/rr_b_$md.jsonl ]; then
    if printf '%s' "$_err" | grep -qiE "reauthentication|invalid_grant|credentials|401"; then
      log "reregister $md: AUTH-FAIL fetching records -- SKIPPING re-register (NOT declaring fresh)"
    else
      log "reregister $md: no jsonl in GCS (fresh lineage)"
    fi
    continue
  fi
  timeout 60 scp $SSHO "$REPO/tpu/reregister_states.py" sk7524_princeton_edu@$h:~/reregister_states.py 2>/dev/null
  timeout 60 scp $SSHO /tmp/rr_b_$md.jsonl sk7524_princeton_edu@$h:~/rr_$md.jsonl 2>/dev/null
  timeout 90 ssh $SSHO sk7524_princeton_edu@$h "python3 ~/reregister_states.py --base-model '$bm' --jsonl ~/rr_$md.jsonl" 2>/dev/null | grep -v Warning | tee -a "$PROG"
done

# 4y. reclaim vLLM-worker disks: once a worker's server is SERVING, its HF
# weight cache (~60G) is dead weight (weights in HBM; any restart re-restores
# from GCS). Full disks broke LoRA adapter uploads and jammed ray grading.
for w in 2 3 4 6 7 8; do
  h=$(echo "$EXT"|cut -d, -f$w)
  timeout 90 ssh $SSHO sk7524_princeton_edu@$h 'curl -fsS -m5 http://127.0.0.1:8001/v1/models >/dev/null 2>&1 && rm -rf ~/.cache/huggingface && echo "$(hostname) cache-swept"' 2>/dev/null | grep -v Warning | tee -a "$PROG" &
done
wait

# 5. client on w0: league worktree tarball (+ .env from the main tree) + venv
log "client provisioning on w0 (league worktree)..."
TAR=/tmp/league-client-$$.tar.gz
cp "$MAIN_REPO/third_party/discover/.env" "$REPO/third_party/discover/.env" 2>/dev/null || true
tar czf "$TAR" -C "$REPO" --exclude=.git --exclude='.venv*' --exclude=runs --exclude='*.log' .
timeout 300 scp $SSHO "$TAR" sk7524_princeton_edu@$W0:/tmp/league-client.tar.gz >/dev/null 2>&1
rm -f "$TAR"
timeout 120 ssh $SSHO sk7524_princeton_edu@$W0 'rm -rf ~/ttd-client && mkdir -p ~/ttd-client && tar xzf /tmp/league-client.tar.gz -C ~/ttd-client && ls ~/ttd-client/third_party/discover/.env >/dev/null 2>&1 && echo env-ok || echo env-MISSING' 2>&1 | grep -v Warning | tee -a "$PROG"
timeout 900 ssh $SSHO sk7524_princeton_edu@$W0 'export PATH=$HOME/.local/bin:$PATH; cd ~/ttd-client/third_party/discover && uv sync --extra math --python 3.11 >~/venv-build.log 2>&1; rc=$?; ln -sfn .venv .venv-ttd-discover; .venv-ttd-discover/bin/python -c "import tinker,numpy,wandb;print(\"venv-ok\")" 2>&1|tail -1; echo venv-rc=$rc' 2>&1 | grep -v Warning | tee -a "$PROG"

# 6. ray grading cluster: head on w0 (num-cpus=0 -- no grading tasks compete
#    with client/trainer), workers w1-7 in minimal grader venvs (ray pinned to
#    the client venv's version; the payload task ships by-VALUE, see
#    sandbox_reward_evaluator, so graders never import ttt_discover).
log "ray grading cluster: grader venvs + head + workers..."
W0INT=$(echo "$INT"|cut -d, -f1)
RAYV=$(timeout 60 ssh $SSHO sk7524_princeton_edu@$W0 '~/ttd-client/third_party/discover/.venv-ttd-discover/bin/python -c "import ray; print(ray.__version__)"' 2>/dev/null | grep -v Warning | tail -1)
log "  ray version pin: ${RAYV:-UNKNOWN}"
if [ -n "$RAYV" ]; then
  timeout 120 ssh $SSHO sk7524_princeton_edu@$W0 '
    ~/ttd-client/third_party/discover/.venv-ttd-discover/bin/ray stop >/dev/null 2>&1; sleep 2
    ~/ttd-client/third_party/discover/.venv-ttd-discover/bin/ray start --head --port=6379 --num-cpus=0 --disable-usage-stats >/tmp/ray-head.log 2>&1
    echo RAY-HEAD-STARTED' 2>/dev/null | grep -v Warning | tee -a "$PROG"
  for w in $(echo "$EXT"|cut -d, -f2-|tr ',' ' '); do
    timeout 900 ssh $SSHO sk7524_princeton_edu@$w "
      export PATH=\$HOME/.local/bin:\$PATH
      [ -x ~/.venvs/grader/bin/ray ] || {
        uv venv ~/.venvs/grader --python 3.11 >/dev/null 2>&1
        uv pip install --python ~/.venvs/grader/bin/python 'ray==$RAYV' numpy scipy shapely numba scikit-learn psutil >/dev/null 2>&1
      }
      ~/.venvs/grader/bin/ray stop >/dev/null 2>&1; sleep 2
      ~/.venvs/grader/bin/ray start --address=$W0INT:6379 --num-cpus=150 --disable-usage-stats >/tmp/ray-worker.log 2>&1 && echo \"RAY-WORKER \$(hostname)\"
    " 2>/dev/null | grep -v Warning | tee -a "$PROG" &
  done
  wait
  nworkers=$(grep -c "RAY-WORKER" "$PROG")
  log "  ray cluster: head + $nworkers workers"
else
  log "  ray version undetectable; skipping cluster (set TTD_EVAL_BACKEND=local)"
fi

log "LEAGUE-ENGINES-UP  w0(qwen)=$W0  gemma_internal=$W4INT"
echo "LEAGUE-ENGINES-UP"
