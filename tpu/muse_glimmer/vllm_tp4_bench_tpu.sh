#!/usr/bin/env bash
# Muse-Glimmer-30B, ONE spot v5p-8, torch/vLLM model, TWO jobs on one slice:
#
#   A. **TP=4 regression on the FIXED submodule** (afe0cb9e9: stock qkv
#      geometry + post-split width assert + LoRA-bypass seam fix; CPU gate
#      ALL-STAGES-GREEN, reproqkv-3714133.log).  Boot TP=4 on all 4 chips,
#      one real completion, the 5-prompt greedy, --enable-lora count (expect
#      260), one adapter round trip.  If the fix were wrong the model's own
#      width assert fires with a clear AssertionError naming the widths --
#      capture it verbatim and STOP (skip B, teardown).
#   B. **1xTP=4 vs 2xTP=2 at max-model-len 22528**, REAL erdos rollout
#      prompts, precompile discipline (every shape warmed untimed before any
#      timed pass -- the prior 6.687x figure is suspected compile-in-window).
#      Per config: boot-log KV pool line, steady-state decode tok/s at
#      concurrency 1/8/32/64/96/128, prefill walls, HBM.  2xTP=2 drives BOTH
#      engines concurrently (round-robined offered load).
#
# QR lifecycle is copied verbatim from followup_tpu.sh, deliberately: `trap ...
# EXIT` is known not to fire on an untrapped SIGTERM here, a scancel has
# previously killed a cleanup mid-delete and leaked a QR, and the delete has
# needed re-issuing because the QR still described as ACTIVE ~2 min after the
# first request.  So TERM/INT/HUP are trapped explicitly, the delete is issued
# `setsid nohup` so it outlives this shell, and it is re-issued and re-verified
# until the QR is gone -- in EVERY zone, because the landing phase rotates.
#
# Every stage is guarded: one failing stage must not cost the others their
# slice time.
set -uo pipefail

REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
GC=/n/fs/vision-mix/sk7524/google-cloud-sdk/bin/gcloud
KEY="${SSH_KEY_FILE:-$HOME/.ssh/google_compute_engine}"
# ServerAlive* is load-bearing: a landed slice was lost when the TPU VM dropped
# an idle SSH channel 2m30s into `uv pip install`.
SSHO="-i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 -o ServerAliveInterval=30 -o ServerAliveCountMax=40 -o TCPKeepAlive=yes"
USER_R=sk7524_princeton_edu

ZONE="${ZONE:-us-east5-a}"
Z="--project=vision-mix --zone=${ZONE}"
# us-east5-b is in the rotation but placed second on purpose: `accelerator-types
# list` advertises v5p-8 there and the create call has previously failed with an
# API error for this project, but a refused create costs ~2.5 min and the zone
# is worth re-probing rather than assumed dead.  A zone that refuses CREATE is
# skipped immediately, so the cost of including it is bounded.
ZONES="${ZONES:-us-east5-a us-east5-b us-east5-c us-east5-a}"
ZONE_TRY_SEC="${ZONE_TRY_SEC:-900}"
# Pause between full rotations when every zone refused the CREATE outright.
ROUND_SLEEP="${ROUND_SLEEP:-120}"
ALL_ZONES="us-east5-a us-east5-b us-east5-c"
QR="${QR:-sk7524-museglimmer-tp4fix}"
ACC="${ACC:-v5p-8}"
RUNTIME="${RUNTIME:-v2-alpha-tpuv5}"

GCS_MODEL=gs://sk7524-tinker-tpu-us-east5/hf-cache/models--meta-models--Muse-Glimmer-30B/snapshots/a4e59da52a7bc87ae7251dd5545c0dd437c44b68
REMOTE_MODEL=/home/${USER_R}/muse-glimmer-30b
VLLM_VENV=/home/${USER_R}/.venvs/vllm-mg
VLLM_TPU_VERSION="${VLLM_TPU_VERSION:-0.23.0}"
VLLM_PORT="${VLLM_PORT:-8001}"
TP="${TP:-4}"
# Phase A runs TP=4 on the whole host (no TPU_* pinning -- the exact config
# that crashed on the pre-fix build).  Phase B's 2xTP=2 arm pins these.
CHIPS="${CHIPS:-0,1}"
CHIPS2="${CHIPS2:-2,3}"
VLLM_PORT2="${VLLM_PORT2:-8002}"
# Ship a clean export of this committed tree, never the working tree: another
# agent may have uncommitted edits in third_party/tpu-inference at any moment.
# afe0cb9e9 = the FIXED build (stock qkv geometry + width assert + LoRA-bypass
# seam fix), CPU-gated ALL-STAGES-GREEN by repro_qkv_width.py.
TPU_INF_SHA="${TPU_INF_SHA:-afe0cb9e9bf259a072242c6f3279d92b702f9f2a}"
# Phase-B benchmark shape: 22528 ctx (prompt + ~11k-token erdos responses,
# cap-censored mean ~10.9k) and enough seq slots for the top bucket.
BENCH_MAXLEN="${BENCH_MAXLEN:-22528}"
BENCH_MAXSEQS="${BENCH_MAXSEQS:-128}"
BENCH_CONCS="${BENCH_CONCS:-1,8,32,64,96,128}"
MAXSEQS="${MAXSEQS:-16}"
MAXLEN="${MAXLEN:-4096}"
LORA_RANK="${LORA_RANK:-8}"
# Adapting every layer of a 30B at rank 8 is ~100M adapter params; that is the
# realistic RL shape and it is what the smoke should carry.
LORA_TARGETS="${LORA_TARGETS:-q_proj,k_proj,v_proj,o_proj,attn_gate_proj,gate_proj,up_proj,down_proj}"

LOGDIR=$REPO/runs/muse_glimmer
mkdir -p "$LOGDIR"
PROG=$LOGDIR/tp4bench.progress
: > "$PROG"

CAP_SEC="${CAP_SEC:-19800}"          # 5h30 since QR creation incl. landing
LAND_SEC="${LAND_SEC:-5400}"         # 90min to go ACTIVE, then give up
QR_CREATED_AT=""
ACTIVE_AT=""

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$PROG"; }

qr_state_z() {   # $1 = zone
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

qr_kill_z() {   # $1 = zone  $2 = rounds
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
  if [ -n "$ACTIVE_AT" ]; then
    log "SLICE-ACTIVE-SECONDS $(( $(date +%s) - ACTIVE_AT ))"
  fi
  log "=== CLEANUP (${why}): deleting QR ${QR} in ${ALL_ZONES} ==="
  qr_kill_z "$ZONE" 60 || rc=1
  for z in $ALL_ZONES; do
    [ "$z" = "$ZONE" ] && continue
    qr_kill_z "$z" 20 || rc=1
  done
  # Final proof for the report. Filter to `muse` on purpose: this project is
  # full of other people's (and other runs') resources -- llamafarm, ladder,
  # rq2, the m-* configs -- and none of them are ours to delete or to report on.
  for z in $ALL_ZONES; do
    log "FINAL[${z}] muse QRs:  '$(timeout 60 $GC alpha compute tpus queued-resources list --project=vision-mix --zone="$z" --format='value(name)' 2>/dev/null | grep -i muse | tr '\n' ' ')'"
    log "FINAL[${z}] muse VMs:  '$(timeout 60 $GC compute tpus tpu-vm list --project=vision-mix --zone="$z" --format='value(name)' 2>/dev/null | grep -i muse | tr '\n' ' ')'"
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

STAGES=""
note() { STAGES="${STAGES}"$'\n'"$*"; log "RESULT: $*"; }

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
# A CREATE refused for quota costs ~40 s and tells us nothing about the next
# minute: the v5p spot pool in us-east5-a is sized by what the rest of the
# project happens to be holding, and QRs there finish and free chips
# continuously.  So the zone rotation is wrapped in a retry loop and the whole
# thing is bounded by LAND_SEC, rather than exiting the moment every zone
# happens to be full.  Waiting is acceptable; a different accelerator is not.
ROUND=0
while [ -z "$LANDED" ] && [ "$(elapsed)" -lt "$LAND_SEC" ]; do
ROUND=$((ROUND + 1))
if [ "$ROUND" -gt 1 ]; then
  log "--- rotation round ${ROUND}: no ${ACC} yet ($(elapsed)s of ${LAND_SEC}s) -- re-probing in ${ROUND_SLEEP}s ---"
  sleep "$ROUND_SLEEP"
  [ "$(elapsed)" -ge "$LAND_SEC" ] && break
fi
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
    # PROVISIONING means capacity is already committed; abandoning it throws
    # away a slice that has been granted (this cost ~9 minutes on the TP run).
    # Only a still-queued QR may be abandoned on the zone deadline.
    #
    # But PROVISIONING is NOT a guarantee: a contended v6e-8 spot pool has been
    # observed cycling (VM reaches CREATING, disappears, is retried) while the
    # QR reports PROVISIONING indefinitely. Without the second test below this
    # loop never exits, and the only thing that ends the run is slurm's
    # pre-timeout signal hours later. LAND_SEC is the hard land-or-abort
    # deadline and applies to PROVISIONING too.
    if [ "$el" -ge "$LAND_SEC" ]; then
      log "land-or-abort deadline ${LAND_SEC}s reached with state=${st} -- giving up on ${ZONE}"
      break
    fi
    if [ "$el" -ge "$zone_deadline" ] && [ "$st" != PROVISIONING ]; then
      log "no capacity in ${ZONE} after ${el}s -- deleting and trying the next zone"
      break
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
done   # rotation retry loop

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
# "a slice that never answers SSH is a lost slice" -- tear down rather than sit.
[ "$ok" = 1 ] || { log "ssh never came up -- abandoning this slice"; exit 5; }
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

# `git archive` of a commit contains only committed files by construction, so
# whatever scratch the other agent has in the shared working tree cannot ship.
STAGE=$(mktemp -d /tmp/tpuinf-stage.XXXXXX)
if ! git -C "$REPO/third_party/tpu-inference" archive "$TPU_INF_SHA" | tar -x -C "$STAGE"; then
  log "git archive of tpu-inference @ ${TPU_INF_SHA} FAILED"; rm -rf "$STAGE"; exit 7
fi
if [ ! -s "$STAGE/tpu_inference/models/vllm/muse_glimmer.py" ]; then
  log "clean export lacks models/vllm/muse_glimmer.py -- wrong commit?"; rm -rf "$STAGE"; exit 7
fi
log "shipping tpu-inference (clean export of ${TPU_INF_SHA}) + clients"
log "  export models/vllm: $(ls "$STAGE/tpu_inference/models/vllm" | tr '\n' ' ')"
timeout 900 rsync -az --delete -e "ssh $SSHO" \
  --exclude '__pycache__' --exclude '*.pyc' --exclude 'tests' \
  "$STAGE/" ${USER_R}@"$HOST":/home/${USER_R}/tpu-inference/ \
  >/dev/null 2>&1 || { log "rsync of tpu-inference failed"; rm -rf "$STAGE"; exit 7; }
rm -rf "$STAGE"
timeout 120 scp $SSHO \
  "$REPO/tpu/muse_glimmer/mg_client.py" \
  "$REPO/tpu/muse_glimmer/make_lora_adapter.py" \
  "$REPO/tpu/muse_glimmer/mg_lora_client.py" \
  "$REPO/tpu/muse_glimmer/mg_bench22k.py" \
  "$LOGDIR/hf_greedy.json" \
  "$LOGDIR/bench22k_prompts.json" \
  ${USER_R}@"$HOST":~/ >/dev/null 2>&1 || { log "scp of clients failed"; exit 7; }

# Built DETACHED, in tmux, and polled: nothing that takes minutes and produces
# no output should ride on a foreground ssh channel.
log "building vLLM venv (detached, polled)"
rsh "cat > ~/build_venv.sh <<'EOS'
set -x
export PATH=\$HOME/.local/bin:\$PATH
# UV_NO_CONFIG is REQUIRED: with cwd inside the repo, uv applies
# override-dependencies (transformers<=5.8.0) even to an explicit git spec with
# --no-deps --force-reinstall, and 5.8.0 cannot parse model_type muse_glimmer.
export UV_NO_CONFIG=1
uv venv --python 3.12 ${VLLM_VENV}
uv pip install --python ${VLLM_VENV}/bin/python 'vllm-tpu==${VLLM_TPU_VERSION}'
uv pip install --python ${VLLM_VENV}/bin/python --no-deps --force-reinstall /home/${USER_R}/tpu-inference
# transformers goes LAST: any resolving install after it re-triggers the override.
# tokenizers is pinned alongside it: @main is a moving target, and on 2026-08-16
# it advanced to 5.16.0.dev0 which raised its floor to tokenizers>=0.23.1 while
# --no-deps left tokenizers at 0.22.2 -- transformers then could not even be
# imported, and the run died on a landed slice. --no-deps is itself required (a
# resolving install re-triggers this repos override-dependencies
# transformers<=5.8.0 pin), so a newer transformers deps are never satisfied
# automatically and must be named here.
# NOTE: this whole block is written through rsh cat > build_venv.sh <<EOS,
# i.e. inside a DOUBLE-QUOTED string. Command substitution expands on the LOCAL
# side and nested quoting breaks the argument -- even inside a comment, since the
# local shell builds the string before anything here is ever parsed as shell.
# Keep every line single-quoted, with no command substitution and no backticks.
uv pip install --python ${VLLM_VENV}/bin/python --no-deps --force-reinstall 'transformers @ git+https://github.com/huggingface/transformers@main' 'tokenizers>=0.23.1,<0.24.0'
${VLLM_VENV}/bin/python - <<'PY'
import transformers, vllm, tpu_inference, os
print('transformers', transformers.__version__, '| vllm', vllm.__version__)
print('TPU_INFERENCE_SITE', os.path.dirname(tpu_inference.__file__))
from vllm.plugins import load_general_plugins
from vllm.model_executor.models.registry import ModelRegistry
load_general_plugins()
archs = set(ModelRegistry.get_supported_archs())
assert 'MuseGlimmerForConditionalGeneration' in archs
# The registered target must now be the TORCH model, not the raising shim.
from tpu_inference.models.common.oot_registration import OUT_OF_TREE_ARCHITECTURES
tgt = OUT_OF_TREE_ARCHITECTURES['MuseGlimmerForConditionalGeneration']
print('REGISTERED-TARGET', tgt)
assert tgt.endswith('MuseGlimmerForCausalLM'), tgt
from tpu_inference.models.vllm.muse_glimmer import MuseGlimmerForCausalLM
from vllm.model_executor.models.interfaces import supports_lora
print('SUPPORTS-LORA', bool(supports_lora(MuseGlimmerForCausalLM)))
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
  r=$(rsh "grep -hE 'ARCH-REGISTERED-OK|TPU_INFERENCE_SITE|REGISTERED-TARGET|SUPPORTS-LORA|VENV-BUILD-EXIT' ~/venv_build.log 2>/dev/null; tmux has-session -t venv 2>/dev/null && echo VENV-TMUX-ALIVE || echo VENV-TMUX-GONE" 90)
  echo "$r" | grep -q 'ARCH-REGISTERED-OK' && { VENV_OK=1; echo "$r" | tee -a "$PROG"; break; }
  if echo "$r" | grep -q 'VENV-TMUX-GONE'; then
    log "venv build exited without ARCH-REGISTERED-OK"
    rsh "tail -n 80 ~/venv_build.log" 120 | tail -40 | tee -a "$PROG"
    break
  fi
  [ $((i % 6)) -eq 0 ] && log "  venv build still running (~$((i * 20))s)"
  sleep 20
done
[ -n "$VENV_OK" ] || { log "venv/plugin preflight FAILED"; exit 7; }
note "venv + registration preflight OK ($(grep -m1 SUPPORTS-LORA "$PROG"))"
deadline_check

# ------------------------------------------------------------- helpers ------
VLLM_BOOT_SECONDS=""
start_vllm() {   # $1 tag  $2 impl  $3 extra flags  $4 chips (""=whole host)
                 # $5 port  $6 tmux session  $7 "keep" = leave other engines up
  local tag="$1" impl="$2" extra="${3:-}"
  local chips="${4:-}" port="${5:-$VLLM_PORT}" sess="${6:-mg}" keep="${7:-}"
  local tpsize chipenv="" nchips tpuport
  if [ -n "$chips" ]; then
    # tp_compare_tpu.sh arm-B pattern: libtpu sees exactly the chips named
    # here, each engine gets its own coordination port, and the XLA cache is
    # shared deliberately (identical shapes -> the second compile is cheap).
    nchips=$(awk -F, '{print NF}' <<<"$chips")
    tpuport=$(( 8476 + ${chips%%,*} ))
    chipenv="export TPU_VISIBLE_CHIPS=${chips}
export TPU_CHIPS_PER_PROCESS_BOUNDS=1,${nchips},1
export TPU_PROCESS_BOUNDS=1,1,1
export TPU_PROCESS_PORT=${tpuport}
export TPU_PROCESS_ADDRESSES=localhost:${tpuport}
export CLOUD_TPU_TASK_ID=0"
    tpsize=$nchips
  else
    tpsize=$TP
  fi
  log "--- booting vLLM [${tag}] MODEL_IMPL_TYPE=${impl} TP=${tpsize} chips='${chips:-all}' port=${port} extra='${extra}' ---"
  if [ -z "$keep" ]; then
    rsh "tmux kill-session -t mg 2>/dev/null; tmux kill-session -t mg2 2>/dev/null; pkill -TERM -u \$USER -f '[V]LLM::EngineCore|[v]llm serve' 2>/dev/null; sleep 8; true" 120 >/dev/null
  else
    rsh "tmux kill-session -t ${sess} 2>/dev/null; sleep 2; true" 60 >/dev/null
  fi
  rsh "mkdir -p ~/skyrl-logs ~/vllm-xla-cache-mg ~/loras
cat > ~/run_${sess}.sh <<'EOS'
#!/usr/bin/env bash
source ${VLLM_VENV}/bin/activate
export MODEL_IMPL_TYPE=${impl}
export VLLM_MODEL_IMPL_TYPE=${impl}
export TPU_BACKEND_TYPE=jax
export SKIP_JAX_PRECOMPILE=1
export HF_HUB_OFFLINE=1
export VLLM_XLA_CACHE_PATH=/home/${USER_R}/vllm-xla-cache-mg
export JAX_COMPILATION_CACHE_DIR=/home/${USER_R}/vllm-xla-cache-mg
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
${chipenv}
unset VLLM_PLUGINS
vllm serve ${REMOTE_MODEL} \
  --served-model-name muse-glimmer-30b \
  --host 0.0.0.0 --port ${port} \
  --tensor-parallel-size ${tpsize} \
  --max-model-len ${MAXLEN} \
  --max-num-seqs ${MAXSEQS} \
  --max-num-batched-tokens ${MAXLEN} \
  --download-dir ${REMOTE_MODEL} ${extra} \
  2>&1 | tee /home/${USER_R}/skyrl-logs/mg-${tag}.log
EOS
chmod +x ~/run_${sess}.sh
tmux new-session -d -s ${sess} 'bash ~/run_${sess}.sh'
sleep 5; tmux has-session -t ${sess} && echo tmux-UP || echo tmux-FAIL" 180 | tail -2 | tee -a "$PROG"

  local t0; t0=$(date +%s)
  local up=0
  for i in $(seq 1 180); do
    deadline_check
    r=$(rsh "curl -fsS -m 25 -X POST http://127.0.0.1:${port}/v1/completions -H 'Content-Type: application/json' -d '{\"model\":\"muse-glimmer-30b\",\"prompt\":\"The capital of France is\",\"max_tokens\":4,\"temperature\":0}' 2>/dev/null | head -c 300" 60)
    if echo "$r" | grep -q '"text"'; then
      VLLM_BOOT_SECONDS=$(( $(date +%s) - t0 ))
      log "ENGINE ANSWERED [${tag}] in ${VLLM_BOOT_SECONDS}s: $r"; up=1; break
    fi
    if [ $((i % 6)) -eq 0 ]; then
      rsh "tmux has-session -t ${sess} 2>/dev/null && echo ALIVE || echo DEAD" 30 > /tmp/mgstate.$$ 2>/dev/null
      if grep -q DEAD /tmp/mgstate.$$; then
        log "engine process DIED [${tag}] after $(( $(date +%s) - t0 ))s"
        rsh "tail -n 150 ~/skyrl-logs/mg-${tag}.log" 150 | tee "$LOGDIR/crash-${tag}.log" | tail -50 | tee -a "$PROG"
        rm -f /tmp/mgstate.$$; return 1
      fi
      rm -f /tmp/mgstate.$$
      log "  [${tag}] not up yet ($(( $(date +%s) - t0 ))s)"
    fi
    sleep 10
  done
  [ "$up" = 1 ] || { log "engine never answered [${tag}]"
                     rsh "tail -n 150 ~/skyrl-logs/mg-${tag}.log" 150 | tee "$LOGDIR/crash-${tag}.log" | tail -50 | tee -a "$PROG"
                     return 1; }
  return 0
}

engine_report() {   # $1 = tag
  local tag="$1"
  log "--- engine report [${tag}] ---"
  rsh "grep -iE 'MODEL_IMPL_TYPE|Loading model with|Registered out-of-tree|kv_cache_groups|KV cache size|maximum concurrency|Init model \| hbm=|LoRA|lora' ~/skyrl-logs/mg-${tag}.log | cut -c1-600 | tail -40" 120 | tee -a "$PROG" | tee "$LOGDIR/report-${tag}.txt"
  rsh "tail -n 600 ~/skyrl-logs/mg-${tag}.log" 180 > "$LOGDIR/vllm-${tag}-tail.log" 2>&1
}

fetch() { timeout 180 scp $SSHO ${USER_R}@"$HOST":~/"$1" "$LOGDIR/$1" >/dev/null 2>&1 && log "  fetched $1"; }

# ============ A. TP=4 regression on the FIXED build (whole host) ============
TP4_BOOT=fail
if start_vllm torch4 vllm "" "" "$VLLM_PORT" mg; then
  TP4_BOOT=ok
  note "FIXED build BOOTED at TP=4 (whole host) in ${VLLM_BOOT_SECONDS}s -- the pre-fix build died here on its first forward"
  engine_report torch4
  log "greedy decode: 5 recorded prompts, token for token"
  rsh "${VLLM_VENV}/bin/python ~/mg_client.py --ref ~/hf_greedy.json --port ${VLLM_PORT} \
        --out ~/res_tp4_e2e.json --max-model-len ${MAXLEN}" 3600 | tail -50 | tee -a "$PROG"
  fetch res_tp4_e2e.json
else
  # The fixed model carries a post-split width assert precisely so a wrong fix
  # fails LOUDLY with the widths in the message instead of a distant reshape
  # crash.  Capture it verbatim; the driver then STOPS (no phase B).
  ASSERT_TXT=$(grep -m1 -B1 -A3 'AssertionError' "$LOGDIR/crash-torch4.log" 2>/dev/null | tr '\n' ' ' | cut -c1-500)
  note "TP=4 FIXED BUILD FAILED TO BOOT -- assert/crash: ${ASSERT_TXT:-no AssertionError found, see crash-torch4.log}"
fi
deadline_check

# ---------------- A2. LoRA on the fixed build at TP=4 ----------------------
if [ "$TP4_BOOT" = ok ] && have_time 3000; then
  log "building two PEFT adapters shaped like RL uploads (rank ${LORA_RANK})"
  rsh "${VLLM_VENV}/bin/python ~/make_lora_adapter.py --model-dir ${REMOTE_MODEL} \
        --out ~/loras/rl-smoke --rank ${LORA_RANK} --seed 17 --targets '${LORA_TARGETS}'
${VLLM_VENV}/bin/python ~/make_lora_adapter.py --model-dir ${REMOTE_MODEL} \
        --out ~/loras/rl-smoke-2 --rank ${LORA_RANK} --seed 4242 --targets '${LORA_TARGETS}'" 900 | tail -6 | tee -a "$PROG"

  if start_vllm lora4 vllm "--enable-lora --max-lora-rank ${LORA_RANK} --max-loras 1" "" "$VLLM_PORT" mg; then
    note "--enable-lora BOOTED at TP=4 on the fixed build (exercises the LoRA-bypass seam fix: wrapped apply must return stock width)"
    engine_report lora4
    rsh "${VLLM_VENV}/bin/python ~/mg_lora_client.py --port ${VLLM_PORT} \
          --base-model muse-glimmer-30b --lora-name rl-smoke \
          --lora-path /home/${USER_R}/loras/rl-smoke \
          --second-lora-path /home/${USER_R}/loras/rl-smoke-2 \
          --out ~/res_lora_tp4.json" 2400 | tail -50 | tee -a "$PROG"
    fetch res_lora_tp4.json
    LW=$(rsh "grep -hoE 'LORA-WRAPPED-MODULES total=[0-9]+ by_class=[^\"]*' ~/skyrl-logs/mg-lora4.log | tail -1" 60 | tr -d '\r')
    note "ADAPTER COUNT at TP=4: ${LW:-LORA-WRAPPED-MODULES line ABSENT}"
  else
    note "--enable-lora FAILED to boot at TP=4 on the fixed build -- see crash-lora4.log"
  fi
else
  log "skipping the TP=4 LoRA stage (boot ${TP4_BOOT}, time left $(( CAP_SEC - $(elapsed) ))s)"
fi
deadline_check

# ============ B. 22k benchmark: 1xTP=4 vs 2xTP=2, real erdos prompts ========
if [ "$TP4_BOOT" = ok ] && have_time 4500; then
  MAXLEN=$BENCH_MAXLEN
  MAXSEQS=$BENCH_MAXSEQS
  # ---- B1: one wide engine ----
  if start_vllm bench4 vllm "" "" "$VLLM_PORT" mg; then
    note "bench engine 1xTP=4 up at maxlen ${MAXLEN} maxseqs ${MAXSEQS} (boot ${VLLM_BOOT_SECONDS}s)"
    engine_report bench4
    rsh "grep -hE 'KV cache size|maximum concurrency|Init model . hbm' ~/skyrl-logs/mg-bench4.log | tail -4" 60 | tee -a "$PROG"
    rsh "${VLLM_VENV}/bin/python ~/mg_bench22k.py --ports ${VLLM_PORT} --label 1xTP4 \
          --prompts ~/bench22k_prompts.json --concurrencies '${BENCH_CONCS}' \
          --out ~/res_bench_1xtp4.json" 6000 | tail -40 | tee -a "$PROG"
    fetch res_bench_1xtp4.json
  else
    note "bench 1xTP=4 engine FAILED to boot at ${MAXLEN} -- see crash-bench4.log"
  fi
  deadline_check

  # ---- B2: two narrow engines, driven concurrently ----
  if have_time 3300; then
    if start_vllm bench2a vllm "" "$CHIPS" "$VLLM_PORT" mg \
       && start_vllm bench2b vllm "" "$CHIPS2" "$VLLM_PORT2" mg2 keep; then
      note "bench engines 2xTP=2 up on {${CHIPS}}/{${CHIPS2}} at maxlen ${MAXLEN}"
      engine_report bench2a
      engine_report bench2b
      rsh "grep -hE 'KV cache size|maximum concurrency|Init model . hbm' ~/skyrl-logs/mg-bench2a.log ~/skyrl-logs/mg-bench2b.log | tail -8" 60 | tee -a "$PROG"
      rsh "${VLLM_VENV}/bin/python ~/mg_bench22k.py --ports ${VLLM_PORT},${VLLM_PORT2} --label 2xTP2 \
            --prompts ~/bench22k_prompts.json --concurrencies '${BENCH_CONCS}' \
            --out ~/res_bench_2xtp2.json" 6000 | tail -40 | tee -a "$PROG"
      fetch res_bench_2xtp2.json
    else
      note "bench 2xTP=2 engines FAILED to boot -- see crash-bench2*.log"
    fi
  else
    log "no time left for the 2xTP=2 bench arm"
  fi
else
  log "skipping the 22k benchmark (TP=4 boot ${TP4_BOOT}, time left $(( CAP_SEC - $(elapsed) ))s)"
fi

log "=== SUMMARY ==="
printf '%b\n' "$STAGES" | tee -a "$PROG"
log "SLICE-ACTIVE-SECONDS-SO-FAR $(( $(date +%s) - ACTIVE_AT ))"
exit 0
