#!/usr/bin/env bash
# Muse-Glimmer-30B follow-up on ONE spot v5p-8:
#   1. the KV-cache-group change (patched vs reverted, A/B on the same slice)
#   2. context beyond max_model_len=4096
#   3. temperature > 0, including logprobs against HF's own distribution
#   4. a LoRA training smoke on the real weights via tunix + MaxText
#
# Structurally a sibling of e2e_tpu.sh (which stays untouched, it is the
# record of the first run).  The QR lifecycle is copied from it verbatim,
# deliberately: `trap ... EXIT` is known not to fire on an untrapped SIGTERM
# here, a scancel has previously killed a cleanup mid-delete and leaked a QR,
# and the delete has needed re-issuing because the QR still described as ACTIVE
# ~2 min after the first request.  So TERM/INT/HUP are trapped explicitly, the
# delete is issued `setsid nohup` so it outlives this shell, and it is
# re-issued and re-verified until the QR is gone.
#
# Every stage is guarded: one failing stage must not cost the others their
# slice time, because the slice is the scarce thing here.
set -uo pipefail

REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
GC=/n/fs/vision-mix/sk7524/google-cloud-sdk/bin/gcloud
KEY="${SSH_KEY_FILE:-$HOME/.ssh/google_compute_engine}"
SSHO="-i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25"
USER_R=sk7524_princeton_edu

ZONE="${ZONE:-us-east5-a}"
Z="--project=vision-mix --zone=${ZONE}"
QR="${QR:-sk7524-museglimmer-followup}"
ACC="${ACC:-v5p-8}"
RUNTIME="${RUNTIME:-v2-alpha-tpuv5}"

GCS_MODEL=gs://sk7524-tinker-tpu-us-east5/hf-cache/models--meta-models--Muse-Glimmer-30B/snapshots/a4e59da52a7bc87ae7251dd5545c0dd437c44b68
GCS_ORBAX=gs://sk7524-tinker-tpu-us-east5/muse-glimmer/maxtext-orbax
REMOTE_MODEL=/home/${USER_R}/muse-glimmer-30b
REMOTE_ORBAX=/home/${USER_R}/muse-glimmer-orbax
REMOTE_SKYRL=/home/${USER_R}/SkyRLTpu
VLLM_VENV=/home/${USER_R}/.venvs/vllm-mg
VLLM_TPU_VERSION="${VLLM_TPU_VERSION:-0.23.0}"
VLLM_PORT="${VLLM_PORT:-8001}"
TP="${TP:-4}"
MAXSEQS="${MAXSEQS:-16}"
MT_PIP_SPEC="${MT_PIP_SPEC:-maxtext @ git+https://github.com/SachinKonan/maxtext.git@4f65ba509}"

LOGDIR=$REPO/runs/muse_glimmer
mkdir -p "$LOGDIR"
PROG=$LOGDIR/followup.progress
: > "$PROG"

CAP_SEC="${CAP_SEC:-16200}"          # 4h30 of slice, inside the 5h budget
LAND_SEC="${LAND_SEC:-2700}"         # 45min to go ACTIVE, then give up
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

elapsed() { [ -z "$QR_CREATED_AT" ] && echo 0 || echo $(( $(date +%s) - QR_CREATED_AT )); }
deadline_check() {
  [ -z "$QR_CREATED_AT" ] && return 0
  if [ "$(elapsed)" -ge "$CAP_SEC" ]; then
    log "HARD CAP ${CAP_SEC}s reached since QR creation -- tearing down"
    exit 9
  fi
}
# Is there at least N more seconds of budget?  Used to skip a stage rather than
# start one that cannot finish.
have_time() { [ $(( CAP_SEC - $(elapsed) )) -ge "$1" ]; }

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

while true; do
  el=$(elapsed)
  [ "$el" -ge "$LAND_SEC" ] && { log "NOT LANDED after ${el}s (cap ${LAND_SEC}s) -- giving up"; exit 2; }
  st=$(qr_state)
  log "  QR state=${st:-<empty>} (${el}s)"
  [ "$st" = ACTIVE ] && break
  if [ "$st" = FAILED ]; then log "QR FAILED"; exit 3; fi
  sleep 30
done
log "QR ACTIVE after $(elapsed)s"

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
log "provisioning host"
timeout 60 scp $SSHO "$REPO/tpu/provision_tpu_worker.sh" ${USER_R}@"$HOST":~/ >/dev/null 2>&1
rsh 'for i in $(seq 1 60); do sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1||break; sleep 5; done; bash ~/provision_tpu_worker.sh' 900 | tail -3 | tee -a "$PROG"
rsh 'df -BG --output=source,size,avail,target / /dev/shm 2>/dev/null' 60 | tee -a "$PROG"
deadline_check

log "restoring weights from GCS to local SSD"
rsh "export PATH=\$HOME/google-cloud-sdk/bin:/usr/lib/google-cloud-sdk/bin:\$PATH
mkdir -p ${REMOTE_MODEL}
time gcloud storage rsync -r ${GCS_MODEL} ${REMOTE_MODEL} 2>&1 | tail -3
du -sh ${REMOTE_MODEL}" 2400 | tail -8 | tee -a "$PROG"
rsh "test -s ${REMOTE_MODEL}/model-00001-of-00002.safetensors && test -s ${REMOTE_MODEL}/model-00002-of-00002.safetensors && echo WEIGHTS-OK || echo WEIGHTS-MISSING" 60 | tee -a "$PROG"
grep -q WEIGHTS-OK "$PROG" || { log "weights not staged on host"; exit 6; }
deadline_check

log "shipping tpu-inference (patched) + both kv_cache_manager variants + clients"
timeout 900 rsync -az --delete -e "ssh $SSHO" \
  --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' --exclude 'tests' \
  "$REPO/third_party/tpu-inference/" ${USER_R}@"$HOST":/home/${USER_R}/tpu-inference/ \
  >/dev/null 2>&1 || { log "rsync of tpu-inference failed"; exit 7; }
git -C "$REPO/third_party/tpu-inference" show HEAD:tpu_inference/runner/kv_cache_manager.py \
  > "$LOGDIR/kvm_baseline.py" || { log "could not extract the baseline kv_cache_manager"; exit 7; }
cp "$REPO/third_party/tpu-inference/tpu_inference/runner/kv_cache_manager.py" "$LOGDIR/kvm_patched.py"
timeout 120 scp $SSHO "$LOGDIR/kvm_baseline.py" "$LOGDIR/kvm_patched.py" \
  "$REPO/tpu/muse_glimmer/mg_client.py" "$REPO/tpu/muse_glimmer/mg_client2.py" \
  "$REPO/tpu/muse_glimmer/lora_smoke.py" \
  ${USER_R}@"$HOST":~/ >/dev/null 2>&1 || { log "scp of clients failed"; exit 7; }

log "building vLLM venv"
rsh "export PATH=\$HOME/.local/bin:\$PATH
export UV_NO_CONFIG=1
uv venv --python 3.12 ${VLLM_VENV} 2>&1 | tail -2
uv pip install --python ${VLLM_VENV}/bin/python 'vllm-tpu==${VLLM_TPU_VERSION}' 2>&1 | tail -2
uv pip install --python ${VLLM_VENV}/bin/python --no-deps --force-reinstall /home/${USER_R}/tpu-inference 2>&1 | tail -2
uv pip install --python ${VLLM_VENV}/bin/python --no-deps --force-reinstall 'transformers @ git+https://github.com/huggingface/transformers@main' 2>&1 | tail -2
${VLLM_VENV}/bin/python - <<'PY'
import transformers, vllm, tpu_inference, os
print('transformers', transformers.__version__, '| vllm', vllm.__version__)
print('TPU_INFERENCE_SITE', os.path.dirname(tpu_inference.__file__))
from vllm.plugins import load_general_plugins
from vllm.model_executor.models.registry import ModelRegistry
load_general_plugins()
assert 'MuseGlimmerForConditionalGeneration' in set(ModelRegistry.get_supported_archs())
print('ARCH-REGISTERED-OK')
PY" 2400 | tail -12 | tee -a "$PROG"
grep -q 'ARCH-REGISTERED-OK' "$PROG" || { log "venv/plugin preflight FAILED"; exit 7; }
SITE=$(grep -m1 'TPU_INFERENCE_SITE' "$PROG" | awk '{print $2}')
[ -n "$SITE" ] || { log "could not locate the installed tpu_inference"; exit 7; }
KVM="${SITE}/runner/kv_cache_manager.py"
log "installed tpu_inference at ${SITE}"
deadline_check

# ------------------------------------------- 3. training-side prep (bg) -----
# CPU-only work (repo sync + uv sync + MaxText install) that the serving stages
# do not contend with -- vLLM is TPU-bound once loaded.  The orbax checkpoint
# itself is pulled LATER: the boot disk is 97 GB and the 55.5 GiB HF checkpoint
# is already on it, so the ~52 GiB orbax dir only fits once the safetensors are
# gone, which cannot happen while vLLM is serving from them.
log "starting training-side prep in the background"
timeout 1200 rsync -az -e "ssh $SSHO" \
  --exclude '.git' --exclude 'runs' --exclude '.venv' --exclude '__pycache__' \
  --exclude '*.pyc' --exclude 'third_party' --exclude 'docs' --exclude '.claude' \
  "$REPO/" ${USER_R}@"$HOST":${REMOTE_SKYRL}/ >/dev/null 2>&1 \
  && log "repo synced" || log "repo rsync FAILED (item 4 will be skipped)"
rsh "cat > ~/train_prep.sh <<'EOS'
set -x
export PATH=\$HOME/.local/bin:\$PATH
cd ${REMOTE_SKYRL}
python3 - <<'PY'
from pathlib import Path
p = Path('uv.lock')
if p.exists():
    p.write_text(p.read_text().replace('https://download-r2.pytorch.org','https://download.pytorch.org'))
PY
uv sync --extra tpu --extra tinker --extra tunix
cd \$HOME && uv pip install --python ${REMOTE_SKYRL}/.venv/bin/python \\
    '${MT_PIP_SPEC}' aqtp pathwaysutils tokamax tiktoken
${REMOTE_SKYRL}/.venv/bin/python -c 'import maxtext, tunix, qwix, jax; print(\"TRAIN-VENV-OK\", jax.__version__)'
EOS
chmod +x ~/train_prep.sh
tmux kill-session -t prep 2>/dev/null
tmux new-session -d -s prep 'bash ~/train_prep.sh > ~/train_prep.log 2>&1'
echo prep-started" 300 | tee -a "$PROG"

# ------------------------------------------------------------- helpers ------
# Boot vLLM at a given max-model-len with a given kv_cache_manager variant, and
# only report success once it has answered a REAL completion.
VLLM_BOOT_SECONDS=""
start_vllm() {   # $1 = tag  $2 = maxlen  $3 = patched|baseline
  local tag="$1" maxlen="$2" variant="$3"
  log "--- booting vLLM [${tag}] maxlen=${maxlen} kv_cache_manager=${variant} ---"
  rsh "tmux kill-session -t mg 2>/dev/null; pkill -TERM -u \$USER -f '[V]LLM::EngineCore|[v]llm serve' 2>/dev/null; sleep 5; true" 120 >/dev/null
  rsh "cp ~/kvm_${variant}.py ${KVM} && md5sum ${KVM} ~/kvm_${variant}.py | awk '{print \$1}' | uniq | wc -l" 60 | tee -a "$PROG"
  rsh "find ${SITE} -name '__pycache__' -path '*runner*' -exec rm -rf {} + 2>/dev/null; true" 60 >/dev/null
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
unset VLLM_PLUGINS
vllm serve ${REMOTE_MODEL} \
  --served-model-name muse-glimmer-30b \
  --host 0.0.0.0 --port ${VLLM_PORT} \
  --tensor-parallel-size ${TP} \
  --max-model-len ${maxlen} \
  --max-num-seqs ${MAXSEQS} \
  --max-num-batched-tokens ${maxlen} \
  --download-dir ${REMOTE_MODEL} \
  2>&1 | tee /home/${USER_R}/skyrl-logs/mg-${tag}.log
EOS
chmod +x ~/run_mg.sh
tmux new-session -d -s mg 'bash ~/run_mg.sh'
sleep 5; tmux has-session -t mg && echo tmux-UP || echo tmux-FAIL" 180 | tail -2 | tee -a "$PROG"

  local t0; t0=$(date +%s)
  local up=0
  for i in $(seq 1 150); do
    deadline_check
    r=$(rsh "curl -fsS -m 25 -X POST http://127.0.0.1:${VLLM_PORT}/v1/completions -H 'Content-Type: application/json' -d '{\"model\":\"muse-glimmer-30b\",\"prompt\":\"The capital of France is\",\"max_tokens\":4,\"temperature\":0}' 2>/dev/null | head -c 300" 60)
    if echo "$r" | grep -q '"text"'; then
      VLLM_BOOT_SECONDS=$(( $(date +%s) - t0 ))
      log "ENGINE ANSWERED [${tag}] in ${VLLM_BOOT_SECONDS}s: $r"; up=1; break
    fi
    if [ $((i % 6)) -eq 0 ]; then
      rsh "tmux has-session -t mg 2>/dev/null && echo ALIVE || echo DEAD" 30 > /tmp/mgstate.$$ 2>/dev/null
      if grep -q DEAD /tmp/mgstate.$$; then
        log "engine process DIED [${tag}] after $(( $(date +%s) - t0 ))s"
        rsh "tail -n 120 ~/skyrl-logs/mg-${tag}.log" 120 | tee -a "$LOGDIR/crash-${tag}.log" | tail -40 | tee -a "$PROG"
        rm -f /tmp/mgstate.$$; return 1
      fi
      rm -f /tmp/mgstate.$$
      log "  [${tag}] not up yet ($(( $(date +%s) - t0 ))s)"
    fi
    sleep 10
  done
  [ "$up" = 1 ] || { log "engine never answered [${tag}]"
                     rsh "tail -n 120 ~/skyrl-logs/mg-${tag}.log" 120 | tee -a "$LOGDIR/crash-${tag}.log" | tail -30 | tee -a "$PROG"
                     return 1; }
  return 0
}

# Pull the numbers that say whether the KV spec actually took effect.
kv_report() {   # $1 = tag
  local tag="$1"
  log "--- KV / HBM report [${tag}] ---"
  rsh "grep -iE 'Hybrid KV cache layout|Init kv-cache|Memory statistics|KV cache size|maximum concurrency|kv_cache_groups' ~/skyrl-logs/mg-${tag}.log | tail -20" 120 | tee -a "$PROG" | tee "$LOGDIR/kv-${tag}.txt"
  rsh "tail -n 500 ~/skyrl-logs/mg-${tag}.log" 180 > "$LOGDIR/vllm-${tag}-tail.log" 2>&1
}

# The HF-side references are produced by a CPU job running concurrently
# (hf_ext_ref.sbatch), which writes them incrementally -- the cheap
# window-boundary probe set lands within a minute, the 16k teacher-forced dump
# much later.  Ship whatever exists each time, and wait only for the ones a
# stage genuinely cannot run without.
ship_refs() {
  local f
  for f in mg_ext_ref.json hf_ext_ref.json mg_logit_rows.npz hf_greedy.json; do
    [ -s "$LOGDIR/$f" ] || continue
    timeout 300 scp $SSHO "$LOGDIR/$f" ${USER_R}@"$HOST":~/ >/dev/null 2>&1 \
      && log "  shipped $f ($(stat -c%s "$LOGDIR/$f") bytes)"
  done
  # mg_client2 tolerates a missing --hf-ext, but not a missing file argument.
  rsh "test -f ~/hf_ext_ref.json || echo '{}' > ~/hf_ext_ref.json; \
       test -f ~/mg_ext_ref.json || echo '{}' > ~/mg_ext_ref.json" 60 >/dev/null
}
wait_for_ref() {   # $1 = filename  $2 = max seconds to wait  $3 = jq-free key to require
  local f="$LOGDIR/$1" want="${3:-}" i
  for i in $(seq 1 $(( $2 / 20 ))); do
    if [ -s "$f" ]; then
      if [ -z "$want" ] || grep -q "$want" "$f"; then return 0; fi
    fi
    have_time 900 || return 1
    [ $((i % 6)) -eq 1 ] && log "  waiting for $1 ${want:+(needs '$want') }..."
    sleep 20
  done
  return 1
}

STAGES_OK=""
note() { STAGES_OK="${STAGES_OK}\n$*"; log "RESULT: $*"; }

wait_for_ref mg_ext_ref.json 900 '"probes"' \
  && log "boundary reference ready" \
  || log "*** mg_ext_ref.json never appeared; boundary probes will be skipped"
ship_refs

# ============================================ 4. item 1: the KV change =======
PATCHED_BOOT=fail
if start_vllm patched 4096 patched; then
  PATCHED_BOOT=ok
  kv_report patched
  ship_refs
  log "running boundary + decode probes on the PATCHED build"
  rsh "${VLLM_VENV}/bin/python ~/mg_client2.py --ext-ref ~/mg_ext_ref.json --hf-ext ~/hf_ext_ref.json \
        --port ${VLLM_PORT} --out ~/res_patched_probe.json --max-model-len 4096 \
        --phases boundary,decode --label patched-4096" 2400 2>/dev/null | tail -60 | tee -a "$PROG"
  rsh "${VLLM_VENV}/bin/python ~/mg_client.py --ref ~/hf_greedy.json --port ${VLLM_PORT} \
        --out ~/res_patched_e2e.json --max-model-len 4096" 3000 | tail -40 | tee -a "$PROG"
  for f in res_patched_probe res_patched_e2e; do
    timeout 120 scp $SSHO ${USER_R}@"$HOST":~/${f}.json "$LOGDIR/${f}.json" >/dev/null 2>&1
  done
  note "patched build BOOTED at maxlen 4096"
else
  note "patched build FAILED TO SERVE at maxlen 4096 (see crash-patched.log)"
fi
deadline_check

# The reverted build is both the control for the A/B and the code every later
# stage runs on if the patch is not trustworthy.
if start_vllm baseline 4096 baseline; then
  kv_report baseline
  ship_refs
  rsh "${VLLM_VENV}/bin/python ~/mg_client2.py --ext-ref ~/mg_ext_ref.json --hf-ext ~/hf_ext_ref.json \
        --port ${VLLM_PORT} --out ~/res_baseline_probe.json --max-model-len 4096 \
        --phases boundary,decode --label baseline-4096" 2400 2>/dev/null | tail -60 | tee -a "$PROG"
  rsh "${VLLM_VENV}/bin/python ~/mg_client.py --ref ~/hf_greedy.json --port ${VLLM_PORT} \
        --out ~/res_baseline_e2e.json --max-model-len 4096" 3000 | tail -40 | tee -a "$PROG"
  for f in res_baseline_probe res_baseline_e2e; do
    timeout 120 scp $SSHO ${USER_R}@"$HOST":~/${f}.json "$LOGDIR/${f}.json" >/dev/null 2>&1
  done
  note "baseline build served at maxlen 4096"
else
  note "baseline build FAILED TO SERVE at maxlen 4096 -- unexpected, see crash-baseline.log"
fi
deadline_check

# ============================== 5. items 2+3: long context and sampling ======
# The long-context stage is the one that genuinely needs the slow half of the
# HF job (a 16384-token float32 forward).  Wait for it, but not past the budget.
wait_for_ref hf_ext_ref.json 1800 '"8192"' \
  && log "long-context reference ready" \
  || log "*** hf_ext_ref long-context entries missing; long context will be coherence-only"
ship_refs

LONG_OK=""
for ML in ${LONG_LENS:-8192 16384 32768}; do
  have_time 1500 || { log "skipping maxlen ${ML}: not enough budget left"; break; }
  if start_vllm "len${ML}" "$ML" baseline; then
    kv_report "len${ML}"
    LONG_OK="$ML"
    note "served at max_model_len=${ML} (boot ${VLLM_BOOT_SECONDS}s)"
    phases="longctx,longfallback"
    [ "$ML" = "${TEMP_AT:-8192}" ] && phases="longctx,longfallback,temp"
    rsh "${VLLM_VENV}/bin/python ~/mg_client2.py --ext-ref ~/mg_ext_ref.json --hf-ext ~/hf_ext_ref.json \
          --logit-rows ~/mg_logit_rows.npz --port ${VLLM_PORT} --out ~/res_len${ML}.json \
          --max-model-len ${ML} --phases ${phases} --temps 0.7,1.0 --degen-tokens 256 \
          --tokenizer ${REMOTE_MODEL} --fallback-long $(( ML / 2 )),$(( ML - 64 )) \
          --label len${ML}" 5400 2>/dev/null | tail -80 | tee -a "$PROG"
    timeout 120 scp $SSHO ${USER_R}@"$HOST":~/res_len${ML}.json "$LOGDIR/res_len${ML}.json" >/dev/null 2>&1
  else
    note "max_model_len=${ML} DID NOT SERVE -- this is the ceiling (see crash-len${ML}.log)"
    break
  fi
  deadline_check
done

# ============================================ 6. item 4: the LoRA smoke ======
if have_time 3000; then
  log "=== item 4: LoRA training smoke ==="
  rsh "tail -5 ~/train_prep.log; grep -c TRAIN-VENV-OK ~/train_prep.log" 120 | tee -a "$PROG"
  rsh "tmux kill-session -t mg 2>/dev/null; pkill -TERM -u \$USER -f '[V]LLM::EngineCore|[v]llm serve' 2>/dev/null; sleep 20; true" 180 >/dev/null
  log "freeing the boot disk (the safetensors are no longer needed; tokenizer/config stay)"
  rsh "rm -f ${REMOTE_MODEL}/model-0000*-of-00002.safetensors; df -BG --output=avail / | tail -1" 300 | tee -a "$PROG"
  log "pulling the pre-converted orbax checkpoint"
  rsh "export PATH=\$HOME/google-cloud-sdk/bin:/usr/lib/google-cloud-sdk/bin:\$PATH
mkdir -p ${REMOTE_ORBAX}
if gcloud storage ls ${GCS_ORBAX}/0/items >/dev/null 2>&1; then
  time gcloud storage rsync -r ${GCS_ORBAX}/0/items ${REMOTE_ORBAX}/0/items 2>&1 | tail -3
  echo ORBAX-FROM-GCS
else
  echo ORBAX-NOT-IN-GCS
fi
du -sh ${REMOTE_ORBAX} 2>/dev/null; ls ${REMOTE_ORBAX}/0/items 2>/dev/null | head -5" 2400 | tail -12 | tee -a "$PROG"

  # Fallback: the CPU conversion job may not have landed a slot in time.  The
  # safetensors are gone from local disk by now, but the same checkpoint is on
  # the gcsfuse mount, so the converter can read it there and write orbax to
  # the (now free) boot disk.  Slower, but it keeps item 4 off the critical
  # path of someone else's queue.
  if grep -q ORBAX-NOT-IN-GCS "$PROG" && have_time 4200; then
    log "orbax not in GCS -- converting on-host from the gcsfuse mount"
    rsh "ls \$HOME/gcs/hf-cache/models--meta-models--Muse-Glimmer-30B/snapshots/ 2>/dev/null | head -2" 120 | tee -a "$PROG"
    rsh "cd ${REMOTE_SKYRL} && JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1 \
          ${REMOTE_SKYRL}/.venv/bin/python -m maxtext.checkpoint_conversion.to_maxtext \
            model_name=muse-glimmer-30b base_output_directory=${REMOTE_ORBAX} \
            scan_layers=True use_multimodal=false skip_jax_distributed_system=True \
            --lazy_load_tensors=True \
            --hf_model_path \$(ls -d \$HOME/gcs/hf-cache/models--meta-models--Muse-Glimmer-30B/snapshots/*/ | head -1) \
            checkpoint_storage_use_ocdbt=True checkpoint_storage_use_zarr3=True 2>&1 | tail -20" 5400 | tail -25 | tee -a "$PROG"
  fi
  rsh "cd ${REMOTE_SKYRL} && JAX_PLATFORMS=tpu HF_HUB_OFFLINE=1 \
        ${REMOTE_SKYRL}/.venv/bin/python ~/lora_smoke.py \
          --base-model ${REMOTE_MODEL} --orbax ${REMOTE_ORBAX}/0/items \
          --maxtext-model-name muse-glimmer-30b --max-target-length 512 \
          --rank 32 --alpha 64 --steps 3 --out ~/res_lora.json" 5400 2>&1 | tail -70 | tee -a "$PROG" | tee "$LOGDIR/lora-smoke.log"
  timeout 120 scp $SSHO ${USER_R}@"$HOST":~/res_lora.json "$LOGDIR/res_lora.json" >/dev/null 2>&1
  timeout 120 scp $SSHO ${USER_R}@"$HOST":~/train_prep.log "$LOGDIR/train_prep.log" >/dev/null 2>&1
else
  note "item 4 SKIPPED: not enough slice budget left"
fi

log "=== ALL STAGES DONE at $(elapsed)s; tearing down ==="
echo -e "$STAGES_OK" | tee -a "$PROG"
exit 0
