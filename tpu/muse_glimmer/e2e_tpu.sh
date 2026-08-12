#!/usr/bin/env bash
# End-to-end Muse-Glimmer-30B (text-only) serving proof on ONE spot v5p-8.
#
# Deliberately standalone rather than a wrapper around
# tpu/start_colocated_vllm_tinker.sh (which is live for the running RL sweep and
# must not be edited). It reuses that script's knobs verbatim -- vllm-tpu wheel,
# the tpu-inference overlay, MODEL_IMPL_TYPE/TPU_BACKEND_TYPE, --download-dir,
# the XLA cache -- but adds the three things this model needs and that script
# has no knob for:
#   1. transformers upgraded inside the vLLM venv (muse_glimmer only exists in
#      transformers >= 5.15.dev; the project pin is <=5.8.0),
#   2. the arch registered with vLLM's OWN ModelRegistry (vLLM 0.23.0 does not
#      know MuseGlimmerForConditionalGeneration; the upstream PR is unmerged),
#   3. a LOCAL tpu-inference overlay (the muse-glimmer branch is not pushed).
#
# QR LIFECYCLE: exactly one QR, deleted on EVERY exit path. `trap ... EXIT` is
# known not to fire on an untrapped SIGTERM here, and a `scancel` has previously
# killed a cleanup mid-delete leaving the QR alive -- so TERM/INT/HUP are
# trapped explicitly and the delete is issued `setsid nohup` (it survives the
# parent dying) BEFORE we wait on it.
set -uo pipefail

REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
GC=/n/fs/vision-mix/sk7524/google-cloud-sdk/bin/gcloud
KEY="${SSH_KEY_FILE:-$HOME/.ssh/google_compute_engine}"
SSHO="-i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25"
USER_R=sk7524_princeton_edu

ZONE="${ZONE:-us-east5-a}"
Z="--project=vision-mix --zone=${ZONE}"
QR="${QR:-sk7524-museglimmer-e2e}"
ACC="${ACC:-v5p-8}"
RUNTIME="${RUNTIME:-v2-alpha-tpuv5}"

GCS_MODEL=gs://sk7524-tinker-tpu-us-east5/hf-cache/models--meta-models--Muse-Glimmer-30B/snapshots/a4e59da52a7bc87ae7251dd5545c0dd437c44b68
REMOTE_MODEL=/home/${USER_R}/muse-glimmer-30b
VLLM_VENV=/home/${USER_R}/.venvs/vllm-mg
VLLM_TPU_VERSION="${VLLM_TPU_VERSION:-0.23.0}"
VLLM_PORT="${VLLM_PORT:-8001}"
TP="${TP:-4}"
MAXLEN="${MAXLEN:-4096}"
MAXSEQS="${MAXSEQS:-16}"

LOGDIR=$REPO/runs/muse_glimmer
mkdir -p "$LOGDIR"
PROG=$LOGDIR/e2e.progress
: > "$PROG"

# Hard cap measured from QR CREATION, not from script start.
CAP_SEC="${CAP_SEC:-10800}"          # 3h
LAND_SEC="${LAND_SEC:-2700}"         # 45min to go ACTIVE
QR_CREATED_AT=""

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$PROG"; }

qr_state() {
  local s
  for _ in 1 2 3; do
    s=$(timeout 45 $GC alpha compute tpus queued-resources describe "$QR" $Z \
          --format="value(state.state)" 2>/dev/null | tail -1)
    [ -n "$s" ] && { echo "$s"; return; }
    sleep 4
  done
  echo ""
}

CLEANED=""
cleanup() {
  local why="${1:-EXIT}"
  [ -n "$CLEANED" ] && return 0
  CLEANED=1
  log "=== CLEANUP (${why}): deleting QR ${QR} in ${ZONE} ==="
  # setsid+nohup: the delete must outlive this shell being killed mid-flight.
  setsid nohup $GC alpha compute tpus queued-resources delete "$QR" $Z --quiet --force \
    >>"$LOGDIR/qr-delete.log" 2>&1 </dev/null &
  disown 2>/dev/null || true
  sleep 10
  local st
  for i in $(seq 1 60); do
    st=$(qr_state)
    if [ -z "$st" ]; then log "CLEANUP: QR ${QR} is GONE (verified)"; return 0; fi
    [ $((i % 4)) -eq 1 ] && log "CLEANUP: QR state=${st}, re-issuing delete and waiting..."
    if [ $((i % 8)) -eq 0 ]; then
      setsid nohup $GC alpha compute tpus queued-resources delete "$QR" $Z --quiet --force \
        >>"$LOGDIR/qr-delete.log" 2>&1 </dev/null &
      disown 2>/dev/null || true
    fi
    sleep 15
  done
  log "CLEANUP: *** WARNING *** QR ${QR} still present after 15min of deletes"
  return 1
}
trap 'cleanup EXIT' EXIT
trap 'cleanup SIGTERM; exit 143' TERM
trap 'cleanup SIGINT;  exit 130' INT
trap 'cleanup SIGHUP;  exit 129' HUP

deadline_check() {
  [ -z "$QR_CREATED_AT" ] && return 0
  local now; now=$(date +%s)
  if [ $((now - QR_CREATED_AT)) -ge "$CAP_SEC" ]; then
    log "HARD CAP ${CAP_SEC}s reached since QR creation -- tearing down"
    exit 9
  fi
}

rsh() { timeout "${2:-120}" ssh $SSHO ${USER_R}@"$HOST" "$1" 2>&1 | grep -v "^Warning: Permanently"; }

# --------------------------------------------------------------- 1. QR ------
existing=$(qr_state)
if [ -n "$existing" ]; then
  log "QR ${QR} already exists (state=${existing}) -- deleting first to guarantee exactly one"
  timeout 300 $GC alpha compute tpus queued-resources delete "$QR" $Z --quiet --force >/dev/null 2>&1
  sleep 20
fi

log "creating spot ${ACC} QR ${QR} in ${ZONE}"
QR_CREATED_AT=$(date +%s)
timeout 180 $GC alpha compute tpus queued-resources create "$QR" --node-id="$QR" $Z \
  --accelerator-type="$ACC" --runtime-version="$RUNTIME" --provisioning-model=SPOT \
  2>&1 | tail -5 | tee -a "$PROG"

active=0
while true; do
  now=$(date +%s); el=$((now - QR_CREATED_AT))
  [ "$el" -ge "$LAND_SEC" ] && { log "NOT LANDED after ${el}s (cap ${LAND_SEC}s) -- giving up"; exit 2; }
  st=$(qr_state)
  log "  QR state=${st:-<empty>} (${el}s)"
  [ "$st" = ACTIVE ] && { active=1; break; }
  if [ "$st" = FAILED ]; then log "QR FAILED"; exit 3; fi
  sleep 30
done
log "QR ACTIVE after $(( $(date +%s) - QR_CREATED_AT ))s"

HOST=$(timeout 60 $GC compute tpus tpu-vm describe "$QR" $Z \
        --format="value(networkEndpoints[].accessConfig.externalIp)" 2>/dev/null \
        | tr ';\t' '\n\n' | grep -E '^[0-9]' | head -1)
[ -n "$HOST" ] || { log "no external IP"; exit 4; }
log "host=${HOST}"

timeout 180 $GC alpha compute tpus tpu-vm ssh ${USER_R}@"$QR" $Z --worker=0 \
  --ssh-key-file="$KEY" --command 'true' >/dev/null 2>&1
ok=0
for i in $(seq 1 40); do timeout 15 ssh $SSHO ${USER_R}@"$HOST" true >/dev/null 2>&1 && { ok=1; break; }; sleep 15; done
[ "$ok" = 1 ] || { log "ssh never came up"; exit 5; }
log "ssh ready"
deadline_check

# ------------------------------------------------------- 2. provision -------
log "provisioning host (apt/uv/gcsfuse)"
timeout 60 scp $SSHO "$REPO/tpu/provision_tpu_worker.sh" ${USER_R}@"$HOST":~/ >/dev/null 2>&1
rsh 'for i in $(seq 1 60); do sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1||break; sleep 5; done; bash ~/provision_tpu_worker.sh' 900 | tail -3 | tee -a "$PROG"
log "disk:"; rsh 'df -h / /dev/shm 2>/dev/null | tail -3' 60 | tee -a "$PROG"
log "chips:"; rsh 'ls /dev/accel* 2>/dev/null | wc -l' 60 | tee -a "$PROG"
deadline_check

# ------------------------------------------- 3. weights + code + venv -------
log "restoring weights from GCS to local SSD (59.5GB, same region)"
# Which CLI is present varies with the TPU runtime image, so try them in order
# of throughput (gcloud storage > gsutil -m) and fall back to the gcsfuse mount
# the provisioner set up.  Weights are the one thing this run cannot proceed
# without, so a single missing binary must not sink a landed slice.
rsh "export PATH=\$HOME/google-cloud-sdk/bin:/usr/lib/google-cloud-sdk/bin:\$PATH
mkdir -p ${REMOTE_MODEL}
echo 'disk before:'; df -BG --output=avail / | tail -1
if command -v gcloud >/dev/null 2>&1; then
  echo 'transfer: gcloud storage rsync'
  time gcloud storage rsync -r ${GCS_MODEL} ${REMOTE_MODEL} 2>&1 | tail -3
elif command -v gsutil >/dev/null 2>&1; then
  echo 'transfer: gsutil -m rsync'
  time gsutil -m -q rsync -r ${GCS_MODEL} ${REMOTE_MODEL} 2>&1 | tail -3
else
  echo 'transfer: cp from the gcsfuse mount (slow fallback)'
  time cp -n \$HOME/gcs/hf-cache/models--meta-models--Muse-Glimmer-30B/snapshots/*/[!.]* ${REMOTE_MODEL}/ 2>&1 | tail -3
fi
du -sh ${REMOTE_MODEL}; ls -la ${REMOTE_MODEL} | head -20" 2400 | tail -30 | tee -a "$PROG"
rsh "test -s ${REMOTE_MODEL}/model-00001-of-00002.safetensors && test -s ${REMOTE_MODEL}/model-00002-of-00002.safetensors && echo WEIGHTS-OK || echo WEIGHTS-MISSING" 60 | tee -a "$PROG"
grep -q WEIGHTS-OK "$PROG" || { log "weights not staged on host"; exit 6; }
deadline_check

log "shipping local tpu-inference (branch agent/muse-glimmer-text, not pushed)"
timeout 900 rsync -az --delete -e "ssh $SSHO" \
  --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' --exclude 'tests' \
  "$REPO/third_party/tpu-inference/" ${USER_R}@"$HOST":/home/${USER_R}/tpu-inference/ \
  >/dev/null 2>&1 || { log "rsync of tpu-inference failed"; exit 7; }
timeout 120 scp $SSHO "$REPO/tpu/muse_glimmer/mg_client.py" \
  ${USER_R}@"$HOST":~/ >/dev/null 2>&1 || { log "scp of client failed"; exit 7; }

log "building vLLM venv (vllm-tpu==${VLLM_TPU_VERSION} + local tpu-inference + transformers@main)"
rsh "export PATH=\$HOME/.local/bin:\$PATH
# See vllm_dryrun.sbatch: any uv config uv happens to discover can carry the
# repo's transformers<=5.8.0 override, which silently defeats the @main install
# below and makes muse_glimmer unparseable. Never let uv read a config here.
export UV_NO_CONFIG=1
set -x
uv venv --python 3.12 ${VLLM_VENV}
uv pip install --python ${VLLM_VENV}/bin/python 'vllm-tpu==${VLLM_TPU_VERSION}' 2>&1 | tail -3
uv pip install --python ${VLLM_VENV}/bin/python --no-deps --force-reinstall /home/${USER_R}/tpu-inference 2>&1 | tail -3
uv pip install --python ${VLLM_VENV}/bin/python --no-deps --force-reinstall 'transformers @ git+https://github.com/huggingface/transformers@main' 2>&1 | tail -3
${VLLM_VENV}/bin/python -c \"
import transformers, vllm
print('transformers', transformers.__version__, '| vllm', vllm.__version__)
from transformers import AutoConfig
c = AutoConfig.from_pretrained('${REMOTE_MODEL}')
print('config parsed:', c.architectures, c.model_type,
      '| text heads', c.get_text_config().num_attention_heads)
from vllm.plugins import load_general_plugins
from vllm.model_executor.models.registry import ModelRegistry
load_general_plugins()
archs = set(ModelRegistry.get_supported_archs())
assert 'MuseGlimmerForConditionalGeneration' in archs, \
    'PLUGIN DID NOT REGISTER THE ARCH -- vllm serve would die in ModelConfig'
print('ARCH-REGISTERED-OK')
\"" 2400 | tail -25 | tee -a "$PROG"
grep -q 'ARCH-REGISTERED-OK' "$PROG" || { log "venv/plugin preflight FAILED"; exit 7; }
deadline_check

# ------------------------------------------------------------- 4. serve -----
log "starting vLLM (TP=${TP}, max-model-len=${MAXLEN}, max-num-seqs=${MAXSEQS})"
rsh "tmux kill-session -t mg 2>/dev/null; pkill -TERM -u \$USER -f '[V]LLM::EngineCore|[v]llm serve|mg_serve' 2>/dev/null; sleep 3; true" 90
rsh "mkdir -p ~/skyrl-logs ~/vllm-xla-cache-mg
cat > ~/run_mg.sh <<'EOS'
#!/usr/bin/env bash
source ${VLLM_VENV}/bin/activate
export MODEL_IMPL_TYPE=flax_nnx
export TPU_BACKEND_TYPE=jax
export SKIP_JAX_PRECOMPILE=1
export HF_HUB_OFFLINE=1
export VLLM_XLA_CACHE_PATH=/home/${USER_R}/vllm-xla-cache-mg
export JAX_COMPILATION_CACHE_DIR=/home/${USER_R}/vllm-xla-cache-mg
# VLLM_PLUGINS deliberately unset: setting it to a subset would skip
# tpu-inference's register_layers entry point, and vLLM swallows plugin errors.
unset VLLM_PLUGINS
vllm serve ${REMOTE_MODEL} \
  --served-model-name muse-glimmer-30b \
  --host 0.0.0.0 --port ${VLLM_PORT} \
  --tensor-parallel-size ${TP} \
  --max-model-len ${MAXLEN} \
  --max-num-seqs ${MAXSEQS} \
  --max-num-batched-tokens ${MAXLEN} \
  --download-dir ${REMOTE_MODEL} \
  2>&1 | tee /home/${USER_R}/skyrl-logs/mg-vllm.log
EOS
chmod +x ~/run_mg.sh
tmux new-session -d -s mg 'bash ~/run_mg.sh'
sleep 5; tmux has-session -t mg && echo tmux-UP || echo tmux-FAIL" 120 | tee -a "$PROG"

log "waiting for the engine to answer a REAL request (not just /health)"
up=0
for i in $(seq 1 180); do
  deadline_check
  r=$(rsh "curl -fsS -m 20 -X POST http://127.0.0.1:${VLLM_PORT}/v1/completions -H 'Content-Type: application/json' -d '{\"model\":\"muse-glimmer-30b\",\"prompt\":\"The capital of France is\",\"max_tokens\":4,\"temperature\":0}' 2>/dev/null | head -c 400" 60)
  if echo "$r" | grep -q '"text"'; then
    log "ENGINE ANSWERED: $r"; up=1; break
  fi
  if [ $((i % 6)) -eq 0 ]; then
    log "  not up yet (${i}0s); tail of vllm log:"
    rsh "tail -n 6 ~/skyrl-logs/mg-vllm.log" 60 | sed 's/^/    /' | tee -a "$PROG"
    rsh "tmux has-session -t mg 2>/dev/null && echo alive || echo DEAD" 30 | tee -a "$PROG"
    grep -q '^DEAD' "$PROG" && { log "engine process died"; rsh "tail -n 80 ~/skyrl-logs/mg-vllm.log" 90 | tee -a "$LOGDIR/mg-vllm-crash.log"; exit 8; }
  fi
  sleep 10
done
[ "$up" = 1 ] || { log "engine never answered"; rsh "tail -n 120 ~/skyrl-logs/mg-vllm.log" 120 | tee -a "$LOGDIR/mg-vllm-crash.log"; exit 8; }
log "=== ENGINE LIVE ==="
deadline_check

# ------------------------------------------------------- 5. the actual test -
log "shipping HF reference dump + running greedy comparison"
timeout 300 scp $SSHO "$LOGDIR/hf_greedy.json" ${USER_R}@"$HOST":~/hf_greedy.json >/dev/null 2>&1 \
  || { log "no hf_greedy.json to compare against"; exit 10; }
rsh "${VLLM_VENV}/bin/python ~/mg_client.py --ref ~/hf_greedy.json --port ${VLLM_PORT} \
      --out ~/mg_results.json --max-model-len ${MAXLEN} --try-prompt-logprobs" 3600 | tee -a "$LOGDIR/mg-client.log"
timeout 120 scp $SSHO ${USER_R}@"$HOST":~/mg_results.json "$LOGDIR/mg_results.json" >/dev/null 2>&1
log "collecting HBM + throughput + engine log"
rsh "tail -n 400 ~/skyrl-logs/mg-vllm.log" 120 > "$LOGDIR/mg-vllm-tail.log" 2>&1
rsh "grep -iE 'HBM|memory|GiB|KV cache|blocks|Graph capt|throughput' ~/skyrl-logs/mg-vllm.log | tail -40" 120 | tee -a "$PROG"

log "=== E2E COMPLETE, tearing down ==="
exit 0
