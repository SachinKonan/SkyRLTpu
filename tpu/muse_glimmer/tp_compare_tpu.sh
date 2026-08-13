#!/usr/bin/env bash
# Muse-Glimmer-30B: TP head-to-head on ONE spot v5p-8 (4 chips).
#
#   arm A   one engine,  TP=4, all 4 chips
#   arm B   two engines, TP=2 each, chips {0,1} and {2,3}
#
# Why this is the measurement worth a slice: Muse-Glimmer has
# `num_key_value_heads = 2`, and `tpu_inference/utils.py::get_padded_num_heads`
# pads the KV head count UP to the shard count when heads < shards --
# get_padded_num_heads(2,2)=2 but get_padded_num_heads(2,4)=4.  So per-token KV
# bytes double going from TP=2 to TP=4 while the model's real KV width does not
# change; the KV pool buys half as many tokens.  Pulling the other way, two
# independent engines hold two copies of the 51.9 GiB of weights, so arm B
# starts ~51.9 GiB down on KV before it wins anything back.  Which effect wins
# is an empirical question -- hence this script, rather than an assertion.
#
# v5p-8 is 4 chips (8 TensorCores), so TP=8 does not exist on one host here;
# a v5p-16 is two hosts and TP=8 across them is a multi-host bring-up this
# harness does not do.  4 chips gives the same mechanism one octave down:
# TP=2 is the unpadded config, TP=4 is the 2x-padded one, exactly as TP=4 vs
# TP=8 would be on an 8-chip host.
#
# QR lifecycle is copied VERBATIM from e2e_tpu.sh / followup_tpu.sh: TERM/INT/HUP
# trapped explicitly, delete issued `setsid nohup` so it outlives the shell,
# re-issued and re-verified until the QR is provably gone, in every zone.
set -uo pipefail

REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
GC=/n/fs/vision-mix/sk7524/google-cloud-sdk/bin/gcloud
KEY="${SSH_KEY_FILE:-$HOME/.ssh/google_compute_engine}"
SSHO="-i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 -o ServerAliveInterval=30 -o ServerAliveCountMax=40 -o TCPKeepAlive=yes"
USER_R=sk7524_princeton_edu

ZONE="${ZONE:-us-east5-a}"
Z="--project=vision-mix --zone=${ZONE}"
ZONES="${ZONES:-us-east5-a us-east5-c us-east5-a}"
ZONE_TRY_SEC="${ZONE_TRY_SEC:-900}"
ALL_ZONES="us-east5-a us-east5-b us-east5-c"
QR="${QR:-sk7524-museglimmer-tp}"
ACC="${ACC:-v5p-8}"
RUNTIME="${RUNTIME:-v2-alpha-tpuv5}"

GCS_MODEL=gs://sk7524-tinker-tpu-us-east5/hf-cache/models--meta-models--Muse-Glimmer-30B/snapshots/a4e59da52a7bc87ae7251dd5545c0dd437c44b68
REMOTE_MODEL=/home/${USER_R}/muse-glimmer-30b
VLLM_VENV=/home/${USER_R}/.venvs/vllm-mg
VLLM_TPU_VERSION="${VLLM_TPU_VERSION:-0.23.0}"

# Which kv_cache_manager the whole comparison runs on.  Defaults to the
# reverted/baseline build so a failed item 1 cannot contaminate item 2; set
# KV_VARIANT=patched only if item 1 passed on hardware.
KV_VARIANT="${KV_VARIANT:-baseline}"
# The KV-capacity question only bites at long context; at 4096 this model
# already supports ~739 concurrent sequences, far past max-num-seqs.
BENCH_LENS="${BENCH_LENS:-16384 8192}"
MAXSEQS="${MAXSEQS:-256}"
GEN="${GEN:-128}"
PROMPT_LEN="${PROMPT_LEN:-512}"
CONC="${CONC:-64}"

LOGDIR=$REPO/runs/muse_glimmer
mkdir -p "$LOGDIR"
PROG=$LOGDIR/tpcompare.progress
: > "$PROG"

CAP_SEC="${CAP_SEC:-7200}"           # 2h of slice is plenty for two arms
LAND_SEC="${LAND_SEC:-2700}"         # 45min land-or-abort
QR_CREATED_AT=""

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$PROG"; }

qr_state_z() {
  local s
  for _ in 1 2 3; do
    s=$(timeout 45 $GC alpha compute tpus queued-resources describe "$QR" \
          --project=vision-mix --zone="$1" \
          --format="value(state.state)" 2>/dev/null | tail -1)
    [ -n "$s" ] && { echo "$s"; return; }
    sleep 4
  done
  echo ""
}
qr_state() { qr_state_z "$ZONE"; }

qr_kill_z() {
  local z="$1" rounds="${2:-60}" i st
  setsid nohup $GC alpha compute tpus queued-resources delete "$QR" \
    --project=vision-mix --zone="$z" --quiet --force \
    >>"$LOGDIR/qr-delete.log" 2>&1 </dev/null &
  disown 2>/dev/null || true
  sleep 10
  for i in $(seq 1 "$rounds"); do
    st=$(qr_state_z "$z")
    if [ -z "$st" ]; then log "CLEANUP[${z}]: QR ${QR} is GONE (verified)"; return 0; fi
    [ $((i % 4)) -eq 1 ] && log "CLEANUP[${z}]: QR state=${st}, re-issuing delete and waiting..."
    if [ $((i % 8)) -eq 0 ]; then
      setsid nohup $GC alpha compute tpus queued-resources delete "$QR" \
        --project=vision-mix --zone="$z" --quiet --force \
        >>"$LOGDIR/qr-delete.log" 2>&1 </dev/null &
      disown 2>/dev/null || true
    fi
    sleep 15
  done
  log "CLEANUP[${z}]: *** WARNING *** QR ${QR} still present after $((rounds * 15))s of deletes"
  return 1
}

CLEANED=""
cleanup() {
  local why="${1:-EXIT}" z rc=0
  [ -n "$CLEANED" ] && return 0
  CLEANED=1
  log "=== CLEANUP (${why}): deleting QR ${QR} in ${ALL_ZONES} ==="
  qr_kill_z "$ZONE" 60 || rc=1
  for z in $ALL_ZONES; do
    [ "$z" = "$ZONE" ] && continue
    qr_kill_z "$z" 20 || rc=1
  done
  return $rc
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
have_time() { [ $(( CAP_SEC - $(elapsed) )) -ge "$1" ]; }
rsh() { timeout "${2:-120}" ssh $SSHO ${USER_R}@"$HOST" "$1" 2>&1 | grep -v "^Warning: Permanently"; }

# --------------------------------------------------------------- 1. QR ------
for z in $ALL_ZONES; do
  existing=$(qr_state_z "$z")
  if [ -n "$existing" ]; then
    log "QR ${QR} already exists in ${z} (state=${existing}) -- deleting first"
    qr_kill_z "$z" 20
  fi
done

QR_CREATED_AT=$(date +%s)
LANDED=""
for zone_try in $ZONES; do
  el=$(elapsed)
  [ "$el" -ge "$LAND_SEC" ] && { log "landing budget ${LAND_SEC}s exhausted"; break; }
  ZONE="$zone_try"; Z="--project=vision-mix --zone=${ZONE}"
  zone_deadline=$(( el + ZONE_TRY_SEC )); [ "$zone_deadline" -gt "$LAND_SEC" ] && zone_deadline=$LAND_SEC

  log "creating spot ${ACC} QR ${QR} in ${ZONE} (zone slot until t+${zone_deadline}s)"
  timeout 180 $GC alpha compute tpus queued-resources create "$QR" --node-id="$QR" $Z \
    --accelerator-type="$ACC" --runtime-version="$RUNTIME" --provisioning-model=SPOT \
    2>&1 | tail -12 | tee -a "$PROG"

  created=""
  for _ in 1 2 3; do
    [ -n "$(qr_state)" ] && { created=1; break; }
    sleep 20
  done
  if [ -z "$created" ]; then
    log "CREATE FAILED in ${ZONE}: no QR exists 60s after create -- skipping this zone"
    continue
  fi

  while true; do
    el=$(elapsed)
    st=$(qr_state)
    log "  QR state=${st:-<empty>} zone=${ZONE} (${el}s)"
    if [ "$st" = ACTIVE ]; then LANDED=1; break; fi
    if [ "$st" = FAILED ]; then log "QR FAILED in ${ZONE} -- moving on"; break; fi
    # PROVISIONING means the capacity has already been granted and the VM is
    # being built -- abandoning it for a slot deadline throws away exactly the
    # thing we are hunting for.  This run did that once (granted at 853s, slot
    # expired at 900s) and cost itself ~9 minutes.  Only a QR that is still
    # QUEUED/WAITING_FOR_RESOURCES may be rotated away.
    if [ "$el" -ge "$zone_deadline" ] && [ "$st" != PROVISIONING ]; then
      log "no capacity in ${ZONE} after $((el))s -- deleting and trying the next zone"
      break
    fi
    if [ "$el" -ge "$zone_deadline" ]; then
      log "  slot expired but ${ZONE} is PROVISIONING -- holding, capacity is already granted"
    fi
    sleep 30
  done
  [ -n "$LANDED" ] && break
  if ! qr_kill_z "$ZONE" 20; then
    log "could not confirm deletion in ${ZONE}; retrying rather than creating a second QR"
    qr_kill_z "$ZONE" 40 || {
      log "*** ${QR} still alive in ${ZONE} -- aborting instead of rotating zones"
      exit 8
    }
  fi
done

if [ -z "$LANDED" ]; then
  log "NOT LANDED after $(elapsed)s across zones [${ZONES}] (cap ${LAND_SEC}s) -- giving up"
  exit 2
fi
ACTIVE_AT=$(date +%s)
log "QR ACTIVE in ${ZONE} after $(elapsed)s"

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
rsh 'ls /dev/vfio/ 2>/dev/null | tr "\n" " "; echo; df -BG --output=source,size,avail,target / 2>/dev/null' 60 | tee -a "$PROG"
deadline_check

log "restoring weights from GCS to local SSD"
rsh "export PATH=\$HOME/google-cloud-sdk/bin:/usr/lib/google-cloud-sdk/bin:\$PATH
mkdir -p ${REMOTE_MODEL}
time gcloud storage rsync -r ${GCS_MODEL} ${REMOTE_MODEL} 2>&1 | tail -3
du -sh ${REMOTE_MODEL}" 2400 | tail -8 | tee -a "$PROG"
rsh "test -s ${REMOTE_MODEL}/model-00001-of-00002.safetensors && test -s ${REMOTE_MODEL}/model-00002-of-00002.safetensors && echo WEIGHTS-OK || echo WEIGHTS-MISSING" 60 | tee -a "$PROG"
grep -q WEIGHTS-OK "$PROG" || { log "weights not staged on host"; exit 6; }
deadline_check

log "shipping tpu-inference + both kv_cache_manager variants + the bench"
timeout 900 rsync -az --delete -e "ssh $SSHO" \
  --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' --exclude 'tests' \
  "$REPO/third_party/tpu-inference/" ${USER_R}@"$HOST":/home/${USER_R}/tpu-inference/ \
  >/dev/null 2>&1 || { log "rsync of tpu-inference failed"; exit 7; }
git -C "$REPO/third_party/tpu-inference" show HEAD:tpu_inference/runner/kv_cache_manager.py \
  > "$LOGDIR/kvm_baseline.py" || { log "could not extract the baseline kv_cache_manager"; exit 7; }
cp "$REPO/third_party/tpu-inference/tpu_inference/runner/kv_cache_manager.py" "$LOGDIR/kvm_patched.py"
timeout 120 scp $SSHO "$LOGDIR/kvm_baseline.py" "$LOGDIR/kvm_patched.py" \
  "$REPO/tpu/muse_glimmer/tp_bench.py" "$REPO/tpu/muse_glimmer/mg_client2.py" \
  ${USER_R}@"$HOST":~/ >/dev/null 2>&1 || { log "scp of clients failed"; exit 7; }
# References for the recovered temperature phase (item 4): the follow-up run's
# temp phase died on an unguarded HTTP 400, so it is re-run here on arm A's
# engine, which is already up -- no extra boot.
for f in mg_logit_rows.npz mg_ext_ref.json; do
  [ -s "$LOGDIR/$f" ] && timeout 300 scp $SSHO "$LOGDIR/$f" \
    ${USER_R}@"$HOST":~/ >/dev/null 2>&1 && log "  shipped $f"
done

log "building vLLM venv (detached, polled)"
rsh "cat > ~/build_venv.sh <<'EOS'
set -x
export PATH=\$HOME/.local/bin:\$PATH
export UV_NO_CONFIG=1
uv venv --python 3.12 ${VLLM_VENV}
uv pip install --python ${VLLM_VENV}/bin/python 'vllm-tpu==${VLLM_TPU_VERSION}'
uv pip install --python ${VLLM_VENV}/bin/python --no-deps --force-reinstall /home/${USER_R}/tpu-inference
uv pip install --python ${VLLM_VENV}/bin/python --no-deps --force-reinstall 'transformers @ git+https://github.com/huggingface/transformers@main'
${VLLM_VENV}/bin/python - <<'PY'
import transformers, vllm, tpu_inference, os
print('transformers', transformers.__version__, '| vllm', vllm.__version__)
print('TPU_INFERENCE_SITE', os.path.dirname(tpu_inference.__file__))
from vllm.plugins import load_general_plugins
from vllm.model_executor.models.registry import ModelRegistry
load_general_plugins()
assert 'MuseGlimmerForConditionalGeneration' in set(ModelRegistry.get_supported_archs())
print('ARCH-REGISTERED-OK')
PY
echo VENV-BUILD-EXIT-\$?
EOS
chmod +x ~/build_venv.sh
tmux kill-session -t venv 2>/dev/null
tmux new-session -d -s venv 'bash ~/build_venv.sh > ~/venv_build.log 2>&1'
sleep 3; tmux has-session -t venv && echo venv-build-started" 300 | tail -3 | tee -a "$PROG"

VENV_OK=""
for i in $(seq 1 120); do
  deadline_check
  r=$(rsh "grep -hE 'ARCH-REGISTERED-OK|TPU_INFERENCE_SITE|VENV-BUILD-EXIT' ~/venv_build.log 2>/dev/null; tmux has-session -t venv 2>/dev/null && echo VENV-TMUX-ALIVE || echo VENV-TMUX-GONE" 90)
  echo "$r" | grep -q 'ARCH-REGISTERED-OK' && { VENV_OK=1; echo "$r" | tee -a "$PROG"; break; }
  if echo "$r" | grep -q 'VENV-TMUX-GONE'; then
    log "venv build exited without ARCH-REGISTERED-OK"
    rsh "tail -n 60 ~/venv_build.log" 120 | tail -30 | tee -a "$PROG"
    break
  fi
  [ $((i % 6)) -eq 0 ] && log "  venv build still running (${i}0s * 2)"
  sleep 20
done
[ -n "$VENV_OK" ] || { log "venv/plugin preflight FAILED"; exit 7; }
SITE=$(grep -m1 'TPU_INFERENCE_SITE' "$PROG" | awk '{print $2}')
[ -n "$SITE" ] || { log "could not locate the installed tpu_inference"; exit 7; }
KVM="${SITE}/runner/kv_cache_manager.py"
log "installed tpu_inference at ${SITE}; kv_cache_manager variant=${KV_VARIANT}"
rsh "cp ~/kvm_${KV_VARIANT}.py ${KVM} && find ${SITE} -name '__pycache__' -path '*runner*' -exec rm -rf {} + 2>/dev/null; echo KVM-SET" 120 | tail -1 | tee -a "$PROG"
deadline_check

# ------------------------------------------------------------- helpers ------
# Boot ONE engine on an explicit set of chips.  With PP=1 and multiprocess-DP
# off, tpu_worker leaves these alone, so libtpu sees exactly the chips named
# here -- which is how two engines coexist on one host.  Each gets its own
# libtpu coordination port; the XLA cache is shared deliberately (identical
# shapes, so the second engine's compile is nearly free).
start_engine() {   # $1 tag  $2 maxlen  $3 port  $4 chips ("all" = whole host)  $5 tmux-session
  local tag="$1" maxlen="$2" port="$3" chips="$4" sess="$5"
  local nchips tpuport chipenv
  if [ "$chips" = all ]; then
    # Arm A is the §4 configuration exactly: no TPU_* pinning at all, so the
    # arm we are comparing against is the one already known to serve.  Pinning
    # it "for symmetry" would risk breaking the control.
    nchips=4
    chipenv=""
  else
    nchips=$(awk -F, '{print NF}' <<<"$chips")
    tpuport=$(( 8476 + ${chips%%,*} ))
    chipenv="export TPU_VISIBLE_CHIPS=${chips}
export TPU_CHIPS_PER_PROCESS_BOUNDS=1,${nchips},1
export TPU_PROCESS_BOUNDS=1,1,1
export TPU_PROCESS_PORT=${tpuport}
export TPU_PROCESS_ADDRESSES=localhost:${tpuport}
export CLOUD_TPU_TASK_ID=0"
  fi
  log "--- booting [${tag}] maxlen=${maxlen} port=${port} chips=${chips} TP=${nchips} ---"
  rsh "mkdir -p ~/skyrl-logs ~/vllm-xla-cache-mg
cat > ~/run_${sess}.sh <<'EOS'
#!/usr/bin/env bash
source ${VLLM_VENV}/bin/activate
export MODEL_IMPL_TYPE=flax_nnx
export TPU_BACKEND_TYPE=jax
export SKIP_JAX_PRECOMPILE=1
export HF_HUB_OFFLINE=1
export VLLM_XLA_CACHE_PATH=/home/${USER_R}/vllm-xla-cache-mg
export JAX_COMPILATION_CACHE_DIR=/home/${USER_R}/vllm-xla-cache-mg
${chipenv}
unset VLLM_PLUGINS
vllm serve ${REMOTE_MODEL} \
  --served-model-name muse-glimmer-30b \
  --host 0.0.0.0 --port ${port} \
  --tensor-parallel-size ${nchips} \
  --max-model-len ${maxlen} \
  --max-num-seqs ${MAXSEQS} \
  --max-num-batched-tokens ${maxlen} \
  --download-dir ${REMOTE_MODEL} \
  2>&1 | tee /home/${USER_R}/skyrl-logs/mg-${tag}.log
EOS
chmod +x ~/run_${sess}.sh
tmux kill-session -t ${sess} 2>/dev/null
tmux new-session -d -s ${sess} 'bash ~/run_${sess}.sh'
sleep 5; tmux has-session -t ${sess} && echo tmux-UP || echo tmux-FAIL" 180 | tail -1 | tee -a "$PROG"

  local t0; t0=$(date +%s)
  for i in $(seq 1 150); do
    deadline_check
    r=$(rsh "curl -fsS -m 25 -X POST http://127.0.0.1:${port}/v1/completions -H 'Content-Type: application/json' -d '{\"model\":\"muse-glimmer-30b\",\"prompt\":\"The capital of France is\",\"max_tokens\":4,\"temperature\":0}' 2>/dev/null | head -c 260" 60)
    if echo "$r" | grep -q '"text"'; then
      log "ENGINE ANSWERED [${tag}] in $(( $(date +%s) - t0 ))s: $r"; return 0
    fi
    if [ $((i % 6)) -eq 0 ]; then
      if rsh "tmux has-session -t ${sess} 2>/dev/null && echo ALIVE || echo DEAD" 30 | grep -q DEAD; then
        log "engine process DIED [${tag}] after $(( $(date +%s) - t0 ))s"
        rsh "tail -n 100 ~/skyrl-logs/mg-${tag}.log" 120 | tee -a "$LOGDIR/crash-${tag}.log" | tail -35 | tee -a "$PROG"
        return 1
      fi
      log "  [${tag}] not up yet ($(( $(date +%s) - t0 ))s)"
    fi
    sleep 10
  done
  log "engine never answered [${tag}]"
  rsh "tail -n 100 ~/skyrl-logs/mg-${tag}.log" 120 | tee -a "$LOGDIR/crash-${tag}.log" | tail -30 | tee -a "$PROG"
  return 1
}

kill_engines() {
  rsh "for s in a b solo; do tmux kill-session -t \$s 2>/dev/null; done
       pkill -TERM -u \$USER -f '[V]LLM::EngineCore|[v]llm serve' 2>/dev/null; sleep 25; true" 180 >/dev/null
}

# The capacity half of the comparison comes from the engine's own boot log --
# it is the engine's opinion of its KV pool, not ours.
kv_report() {   # $1 tag
  local tag="$1"
  log "--- KV / HBM report [${tag}] ---"
  rsh "grep -iE 'Hybrid KV cache layout|Available KV cache memory|GPU KV cache size|KV cache size|maximum concurrency|num_kv_cache_groups|Init model \| hbm=|Memory statistics' ~/skyrl-logs/mg-${tag}.log | cut -c1-900 | tail -25" 120 | tee -a "$PROG" | tee "$LOGDIR/tpkv-${tag}.txt"
}

RESULTS=""
note() { RESULTS="${RESULTS}\n$*"; log "RESULT: $*"; }

# ================================================= 3. the two arms ==========
BENCH_LEN=""
for ML in $BENCH_LENS; do
  have_time 2400 || { log "not enough budget for maxlen ${ML}"; break; }

  # ---- arm A: one engine, TP=4, all four chips ----
  kill_engines
  if ! start_engine "tp4-${ML}" "$ML" 8001 all solo; then
    note "arm A (TP=4) FAILED TO SERVE at maxlen ${ML}"
    continue
  fi
  kv_report "tp4-${ML}"
  rsh "${VLLM_VENV}/bin/python ~/tp_bench.py --ports 8001 --tp 4 --label tp4-${ML} \
        --prompt-len ${PROMPT_LEN} --gen ${GEN} --concurrency ${CONC} \
        --out ~/res_tp4_${ML}.json" 2400 2>/dev/null | tail -45 | tee -a "$PROG"
  timeout 120 scp $SSHO ${USER_R}@"$HOST":~/res_tp4_${ML}.json "$LOGDIR/res_tp4_${ML}.json" >/dev/null 2>&1
  note "arm A: TP=4 single engine served at maxlen ${ML}"

  # Item 4 recovery, on the engine that is already live.  Greedy is invariant
  # to the monotone output_multiplier and the T*tanh(logits/T) softcap, so the
  # logprob-vs-HF comparison at matched temperature is the only check that can
  # see a sampling-side rescaling bug.
  if [ -s "$LOGDIR/mg_logit_rows.npz" ]; then
    log "--- item 4: temperature > 0 on arm A ---"
    rsh "${VLLM_VENV}/bin/python ~/mg_client2.py --ext-ref ~/mg_ext_ref.json \
          --logit-rows ~/mg_logit_rows.npz --port 8001 --out ~/res_temp.json \
          --max-model-len ${ML} --phases temp --temps 0.7,1.0 \
          --degen-tokens 256 --label temp-tp4-${ML}" 3000 2>/dev/null | tail -40 | tee -a "$PROG"
    timeout 120 scp $SSHO ${USER_R}@"$HOST":~/res_temp.json "$LOGDIR/res_temp.json" >/dev/null 2>&1
    note "item 4 temperature phase attempted at maxlen ${ML} (see res_temp.json)"
  fi
  deadline_check

  # ---- arm B: two engines, TP=2 each, chips {0,1} and {2,3} ----
  kill_engines
  B_OK=""
  if start_engine "tp2a-${ML}" "$ML" 8001 "0,1" a; then
    if start_engine "tp2b-${ML}" "$ML" 8002 "2,3" b; then
      B_OK=1
    else
      note "arm B: SECOND TP=2 engine failed to boot (chip isolation on the 2nd pair)"
    fi
  else
    note "arm B: FIRST TP=2 engine failed to boot -- chip partitioning unsupported here"
  fi

  if [ -n "$B_OK" ]; then
    kv_report "tp2a-${ML}"
    kv_report "tp2b-${ML}"
    rsh "${VLLM_VENV}/bin/python ~/tp_bench.py --ports 8001,8002 --tp 2 --label tp2x2-${ML} \
          --prompt-len ${PROMPT_LEN} --gen ${GEN} --concurrency ${CONC} \
          --out ~/res_tp2x2_${ML}.json" 2400 2>/dev/null | tail -45 | tee -a "$PROG"
    timeout 120 scp $SSHO ${USER_R}@"$HOST":~/res_tp2x2_${ML}.json "$LOGDIR/res_tp2x2_${ML}.json" >/dev/null 2>&1
    note "arm B: 2 x TP=2 engines served at maxlen ${ML}"
  fi
  BENCH_LEN="$ML"
  deadline_check
  break     # one length is the comparison; a second is only a fallback
done

[ -n "$BENCH_LEN" ] || note "NO arm completed -- see crash-*.log"

# ---------------------------------------------------- 4. the verdict --------
python3 - "$LOGDIR" "${BENCH_LEN:-0}" <<'PY' 2>&1 | tee -a "$PROG" | tee "$LOGDIR/tp_verdict.txt"
import json, os, re, sys
d, ML = sys.argv[1], sys.argv[2]

def J(n):
    try:
        return json.load(open(os.path.join(d, n)))
    except Exception as e:
        return {"_missing": repr(e)}

def logf(tag):
    try:
        return open(os.path.join(d, f"tpkv-{tag}.txt")).read()
    except Exception:
        return ""

def kv_tokens(tag):
    m = re.findall(r"KV cache size:\s*([\d,]+)\s*tokens", logf(tag))
    return int(m[-1].replace(",", "")) if m else None

def kv_gib(tag):
    # `Memory statistics | ... | total_hbm_used_gb=52.22GiB | total_hbm_avail_gb=300.11GiB`
    # avail is what actually becomes the KV pool.
    m = re.findall(r"total_hbm_avail_gb=([\d.]+)GiB", logf(tag))
    return float(m[-1]) if m else None

def weights_gib(tag):
    m = re.findall(r"total_hbm_used_gb=([\d.]+)GiB", logf(tag))
    return float(m[-1]) if m else None

def measured_bytes_per_token(tag):
    g, t = kv_gib(tag), kv_tokens(tag)
    return round(g * (1024 ** 3) / t) if (g and t) else None

def conc(tag):
    m = re.findall(r"aximum concurrency for [\d,]+ tokens per request:\s*([\d.]+)x",
                   logf(tag))
    return float(m[-1]) if m else None

def hbm(tag):
    # `Init model | hbm=[(13.06, 95.74), (13.06, 95.74), ...]GiB` -- per-chip
    # (used, limit) after the weights land.  Take the first chip's pair.
    m = re.findall(r"Init model \| hbm=\[\(([\d.]+),\s*([\d.]+)\)", logf(tag))
    return (float(m[-1][0]), float(m[-1][1])) if m else None

def groups(tag):
    m = re.findall(r"num_kv_cache_groups=(\d+)", logf(tag))
    return int(m[-1]) if m else None

a, b = f"tp4-{ML}", [f"tp2a-{ML}", f"tp2b-{ML}"]
ra, rb = J(f"res_tp4_{ML}.json"), J(f"res_tp2x2_{ML}.json")

print(f"=== TP head-to-head at max_model_len={ML} (v5p-8, 4 chips) ===")
ta, tb = kv_tokens(a), sum(x for x in (kv_tokens(b[0]), kv_tokens(b[1])) if x) or None
ca = conc(a)
cb = [conc(b[0]), conc(b[1])]
cb_tot = sum(x for x in cb if x) if any(cb) else None

print(f"arm A  TP=4 x1 : kv_tokens={ta} kv_pool_GiB={kv_gib(a)} "
      f"weights_hbm_GiB={weights_gib(a)} bytes/token(measured)={measured_bytes_per_token(a)} "
      f"max_conc={ca} hbm_per_chip={hbm(a)} groups={groups(a)}")
print(f"arm B  TP=2 x2 : kv_tokens={tb} (per-engine {kv_tokens(b[0])}, {kv_tokens(b[1])}) "
      f"kv_pool_GiB/engine={kv_gib(b[0])} weights_hbm_GiB/engine={weights_gib(b[0])} "
      f"bytes/token(measured)={measured_bytes_per_token(b[0])} "
      f"max_conc={cb_tot} (per-engine {cb}) hbm_per_chip={hbm(b[0])} groups={groups(b[0])}")
if ta and tb:
    print(f"KV TOKEN RATIO  B/A = {tb/ta:.3f}x")
if ca and cb_tot:
    print(f"MAX CONCURRENCY RATIO B/A = {cb_tot/ca:.3f}x  ({ca:.1f} vs {cb_tot:.1f} seqs)")

for name, r in (("A TP=4x1", ra), ("B TP=2x2", rb)):
    if r.get("_missing"):
        print(f"{name}: bench MISSING {r['_missing']}")
        continue
    an = r.get("analytic") or {}
    b1 = (r.get("batch1") or {}).get("median_over_engines_tok_s")
    sat = r.get("saturation") or {}
    print(f"{name}: per_token_kv_bytes/engine={an.get('per_token_kv_bytes_per_engine')} "
          f"(padded_kv_heads={an.get('padded_kv_heads')}) "
          f"batch1={b1} tok/s  saturation={sat.get('aggregate_tok_s')} tok/s "
          f"@conc={sat.get('concurrency_actual')} p50_lat={sat.get('latency_p50_s')}s")

b1a = ((ra.get("batch1") or {}).get("median_over_engines_tok_s"))
b1b = ((rb.get("batch1") or {}).get("median_over_engines_tok_s"))
sa = (ra.get("saturation") or {}).get("aggregate_tok_s")
sb = (rb.get("saturation") or {}).get("aggregate_tok_s")
if b1a and b1b:
    print(f"BATCH-1 DECODE RATIO A/B = {b1a/b1b:.3f}x "
          "(>1 means the wider TP decodes a single sequence faster)")
if sa and sb:
    print(f"SATURATED THROUGHPUT RATIO B/A = {sb/sa:.3f}x")
PY

log "=== DONE at $(elapsed)s; tearing down ==="
[ -n "${ACTIVE_AT:-}" ] && log "SLICE-ACTIVE-SECONDS $(( $(date +%s) - ACTIVE_AT )) (4 chips, ${ACC}, ${ZONE})"
echo -e "$RESULTS" | tee -a "$PROG"
exit 0
