#!/usr/bin/env bash
# Muse-Glimmer-30B reasoning-strength (high vs xhigh) improvement-ability run,
# on ONE spot v5p-8.
#
# WHAT KILLED THE PREVIOUS ROUND, and what this script does about it.
#
#   16/1060 ok=16 err=0 tok=174025 172 tok/s 1012s elapsed
#
# 172 tok/s is four concurrent streams.  At ~10 900 tokens an item, 960 items
# at that rate is ~19 hours -- no spot slice could ever have survived it, and
# the four preemptions that got the blame were a symptom.  Nobody could see it
# until 1012 s of slice time had been spent, because the only progress line
# printed every 16 COMPLETIONS and nothing completes in under four minutes.
#
# So, in order:
#   0. GATE 0 (ii) -- a live smoke that the served model produces BOTH a
#      `to=self` reasoning channel and a `to=user` answer channel through the
#      NEW MuseGlimmerRenderer prompt.  If it does not, everything after is
#      garbage and the run STOPS.
#   1. GATE 1 -- a THROUGHPUT PROBE at the exact concurrency generation will
#      use, on real prompts, warmed once and then measured.  It reports
#      achieved in-flight count and aggregate tok/s in ~3 minutes.  If the
#      measured rate is below MIN_TPS the run STOPS and reports, because a
#      second 19-hour grind is the one outcome that must not happen.
#      TPU page size is the knob it is allowed to try: tpu-inference hardcodes
#      page_size=16 for max_model_len>8192 as "a temporary fix for vmem OOM",
#      and its own comment says small pages spill scalar registers and perform
#      badly.  BLOCK_TRY lets the probe measure 256 against the default.
#   2. generate rollouts from the pre-built manifests (erdos, then jssp, then
#      ac1), flushing every completion to the shared filesystem as it lands.
#
# The QR lifecycle is copied verbatim from tp_compare_tpu.sh, including the
# PROVISIONING fix that run paid ~9 minutes to learn: TERM/INT/HUP trapped
# explicitly, deletes issued `setsid nohup` so they outlive this shell,
# re-issued and re-verified in all three zones.
#
# tpu/start_colocated_vllm_tinker.sh is NOT touched; everything here is env.
set -uo pipefail

REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
GC=/n/fs/vision-mix/sk7524/google-cloud-sdk/bin/gcloud
KEY="${SSH_KEY_FILE:-$HOME/.ssh/google_compute_engine}"
SSHO="-i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 -o ServerAliveInterval=30 -o ServerAliveCountMax=40 -o TCPKeepAlive=yes"
USER_R=sk7524_princeton_edu

ZONE="${ZONE:-us-east5-a}"
Z="--project=vision-mix --zone=${ZONE}"
# us-east5-b and -c have both failed the QR *create* call for this
# project/accelerator in past rounds, but they are cheap to test (the create
# either works or errors inside ~60 s), so they stay in the rotation.
ZONES="${ZONES:-us-east5-a us-east5-b us-east5-c us-east5-a us-east5-a}"
ZONE_TRY_SEC="${ZONE_TRY_SEC:-1200}"
ALL_ZONES="us-east5-a us-east5-b us-east5-c"
QR="${QR:-sk7524-museglimmer-rs2}"
ACC="${ACC:-v5p-8}"
RUNTIME="${RUNTIME:-v2-alpha-tpuv5}"

GCS_MODEL=gs://sk7524-tinker-tpu-us-east5/hf-cache/models--meta-models--Muse-Glimmer-30B/snapshots/a4e59da52a7bc87ae7251dd5545c0dd437c44b68
REMOTE_MODEL=/home/${USER_R}/muse-glimmer-30b
VLLM_VENV=/home/${USER_R}/.venvs/vllm-mg
VLLM_TPU_VERSION="${VLLM_TPU_VERSION:-0.23.0}"
VLLM_PORT="${VLLM_PORT:-8001}"
TP="${TP:-4}"
MAXLEN="${MAXLEN:-16384}"          # round 1's items averaged 11 376 output tokens
# Prompt + reasoning cap. The sweep's value is 13824 against an 18432 window;
# at a 16384 window that would leave only 2494 tokens for the phase-2 answer,
# and round 1's longest answer was 2604. 13312 leaves 3006, which covers every
# answer observed. It costs 512 reasoning tokens that were being truncated
# anyway -- round 1 hit the cap on 49 of 49 rollouts.
PHASE1="${PHASE1:-13312}"
# The measured KV pool holds ~184 concurrent sequences at 16384, so 96 in
# flight has ~1.9x of headroom and >= the 64 the brief requires.
MAXSEQS="${MAXSEQS:-128}"
CONC="${CONC:-96}"
TEMP="${TEMP:-1.0}"
BLOCK_TRY="${BLOCK_TRY:-256 default}"
PROBE_TOKENS="${PROBE_TOKENS:-512}"
MIN_TPS="${MIN_TPS:-600}"          # ~3.5x the 172 tok/s that killed round 1

WORK=/n/fs/vision-mix/sk7524/muse-rs2
MANIFESTS="$WORK/manifests"
GENDIR="$WORK/gen"
mkdir -p "$GENDIR"
LOGDIR=$REPO/runs/muse_glimmer
mkdir -p "$LOGDIR"
PROG=$LOGDIR/rs2.progress
: > "$PROG"

# Hard cap on SLICE time -- measured from the slice going ACTIVE, because
# waiting in WAITING_FOR_RESOURCES costs no chips and should not eat the
# generation budget.  The cap is CUMULATIVE across attempts: the wrapper may
# re-invoke this script after a preemption or a dead host, and 6 h means 6 h of
# chips in total, not 6 h per try.
CAP_SEC="${CAP_SEC:-21600}"        # 6h of live slice, cumulative
LAND_SEC="${LAND_SEC:-2700}"       # 45min per QR to go ACTIVE, then abandon it
HUNT_SEC="${HUNT_SEC:-5400}"       # overall wall clock spent hunting capacity
HOST_DEAD_SEC="${HOST_DEAD_SEC:-600}"   # unreachable this long => lost slice
SLICE_LEDGER="$LOGDIR/rs2.slice_seconds"
PRIOR_SLICE=$(cat "$SLICE_LEDGER" 2>/dev/null || echo 0)
case "$PRIOR_SLICE" in ''|*[!0-9]*) PRIOR_SLICE=0 ;; esac
QR_CREATED_AT=""
HUNT_START=$(date +%s)
ACTIVE_AT=""

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
  if [ -n "$ACTIVE_AT" ]; then
    echo $(( PRIOR_SLICE + $(date +%s) - ACTIVE_AT )) > "$SLICE_LEDGER"
    log "SLICE-ACTIVE-SECONDS $(( $(date +%s) - ACTIVE_AT )) this attempt; cumulative $(cat "$SLICE_LEDGER")"
  fi
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
slice_sec() { [ -z "$ACTIVE_AT" ] && echo "$PRIOR_SLICE" || echo $(( PRIOR_SLICE + $(date +%s) - ACTIVE_AT )); }
deadline_check() {
  [ -z "$ACTIVE_AT" ] && return 0
  if [ "$(slice_sec)" -ge "$CAP_SEC" ]; then
    log "HARD CAP ${CAP_SEC}s of CUMULATIVE LIVE SLICE reached -- tearing down"
    exit 9
  fi
}
remaining() { echo $(( CAP_SEC - $(slice_sec) )); }
rsh() { timeout "${2:-120}" ssh $SSHO ${USER_R}@"$HOST" "$1" 2>&1 | grep -v "^Warning: Permanently"; }

# A landed slice that never answers ssh is a LOST slice, not a slow one: a
# previous round sat polling an unreachable host for 960 s and then died to the
# slurm signal.  Every wait loop funnels its ssh failures through here.
DEAD_SINCE=0
host_watch() {   # $1 = the text rsh returned
  if echo "$1" | grep -qE 'Connection timed out|No route to host|Connection refused|Connection closed|Operation timed out|port 22'; then
    [ "$DEAD_SINCE" -eq 0 ] && DEAD_SINCE=$(date +%s)
    local dead=$(( $(date +%s) - DEAD_SINCE ))
    if [ "$dead" -ge "$HOST_DEAD_SEC" ]; then
      local qs; qs=$(qr_state)
      log "*** host unreachable for ${dead}s; QR state=${qs:-<gone>}"
      if [ "$qs" != ACTIVE ]; then
        log "*** SLICE PREEMPTED -- abandoning"; exit 12
      fi
      log "*** QR says ACTIVE but the host does not answer -- treating as a LOST slice"
      exit 13
    fi
  else
    DEAD_SINCE=0
  fi
}

[ -s "$MANIFESTS/erdos.json" ] || { log "no erdos manifest at $MANIFESTS/erdos.json"; exit 1; }
log "cumulative slice already spent: ${PRIOR_SLICE}s of ${CAP_SEC}s"
[ "$PRIOR_SLICE" -ge "$CAP_SEC" ] && { log "cumulative slice cap already exhausted"; exit 9; }

# --------------------------------------------------------------- 1. QR ------
for z in $ALL_ZONES; do
  existing=$(qr_state_z "$z")
  if [ -n "$existing" ]; then
    log "QR ${QR} already exists in ${z} (state=${existing}) -- deleting first"
    qr_kill_z "$z" 20
  fi
done

LANDED=""
for zone_try in $ZONES; do
  hunted=$(( $(date +%s) - HUNT_START ))
  [ "$hunted" -ge "$HUNT_SEC" ] && { log "capacity hunt budget ${HUNT_SEC}s exhausted"; break; }
  # Each fresh QR restarts its own 45-minute land-or-abandon clock.
  QR_CREATED_AT=$(date +%s)
  el=0
  ZONE="$zone_try"; Z="--project=vision-mix --zone=${ZONE}"
  zone_deadline=$ZONE_TRY_SEC; [ "$zone_deadline" -gt "$LAND_SEC" ] && zone_deadline=$LAND_SEC

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
    # PROVISIONING = capacity already granted; abandoning it for a slot
    # deadline throws away the exact thing we are hunting for.
    if [ "$el" -ge "$zone_deadline" ] && [ "$st" != PROVISIONING ]; then
      log "no capacity in ${ZONE} after ${el}s -- deleting and trying the next zone"
      break
    fi
    if [ "$el" -ge "$LAND_SEC" ]; then
      log "LAND-OR-ABORT: ${LAND_SEC}s from QR creation and still ${st} -- abandoning"
      break
    fi
    [ "$el" -ge "$zone_deadline" ] && log "  slot expired but ${ZONE} is PROVISIONING -- holding"
    sleep 30
  done
  [ -n "$LANDED" ] && break
  if ! qr_kill_z "$ZONE" 20; then
    log "could not confirm deletion in ${ZONE}; retrying rather than creating a second QR"
    qr_kill_z "$ZONE" 40 || { log "*** ${QR} still alive in ${ZONE} -- aborting"; exit 8; }
  fi
done

[ -n "$LANDED" ] || {
  log "NOT LANDED after $(( $(date +%s) - HUNT_START ))s of hunting [${ZONES}] -- giving up"
  exit 2
}
ACTIVE_AT=$(date +%s)
log "QR ACTIVE in ${ZONE} after $(elapsed)s (hunt total $(( ACTIVE_AT - HUNT_START ))s)"

HOST=$(timeout 60 $GC compute tpus tpu-vm describe "$QR" $Z \
        --format="value(networkEndpoints[].accessConfig.externalIp)" 2>/dev/null \
        | tr ';\t' '\n\n' | grep -E '^[0-9]' | head -1)
[ -n "$HOST" ] || { log "no external IP"; exit 4; }
log "host=${HOST}"

timeout 180 $GC alpha compute tpus tpu-vm ssh ${USER_R}@"$QR" $Z --worker=0 \
  --ssh-key-file="$KEY" --command 'true' >/dev/null 2>&1
ok=0
SSH_WAIT_SEC="${SSH_WAIT_SEC:-600}"
t_ssh=$(date +%s)
while [ $(( $(date +%s) - t_ssh )) -lt "$SSH_WAIT_SEC" ]; do
  timeout 15 ssh $SSHO ${USER_R}@"$HOST" true >/dev/null 2>&1 && { ok=1; break; }
  sleep 15
done
if [ "$ok" != 1 ]; then
  qs=$(qr_state)
  log "ssh never came up in ${SSH_WAIT_SEC}s (QR state=${qs:-<gone>}) -- LOST slice, re-hunt"
  [ "$qs" = ACTIVE ] && exit 13 || exit 12
fi
log "ssh ready after $(( $(date +%s) - t_ssh ))s"
deadline_check

# ------------------------------------------------------- 2. provision -------
log "provisioning host"
timeout 60 scp $SSHO "$REPO/tpu/provision_tpu_worker.sh" ${USER_R}@"$HOST":~/ >/dev/null 2>&1
r=$(rsh 'for i in $(seq 1 60); do sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1||break; sleep 5; done; bash ~/provision_tpu_worker.sh' 900)
echo "$r" | tail -3 | tee -a "$PROG"; host_watch "$r"
rsh 'ls /dev/vfio/ 2>/dev/null | tr "\n" " "; echo; df -BG --output=avail / | tail -1' 60 | tee -a "$PROG"
deadline_check

log "restoring weights from GCS to local SSD"
r=$(rsh "export PATH=\$HOME/google-cloud-sdk/bin:/usr/lib/google-cloud-sdk/bin:\$PATH
mkdir -p ${REMOTE_MODEL}
time gcloud storage rsync -r ${GCS_MODEL} ${REMOTE_MODEL} 2>&1 | tail -3
du -sh ${REMOTE_MODEL}" 2400)
echo "$r" | tail -8 | tee -a "$PROG"; host_watch "$r"
rsh "test -s ${REMOTE_MODEL}/model-00001-of-00002.safetensors && test -s ${REMOTE_MODEL}/model-00002-of-00002.safetensors && echo WEIGHTS-OK || echo WEIGHTS-MISSING" 60 | tee -a "$PROG"
grep -q WEIGHTS-OK "$PROG" || { log "weights not staged on host"; exit 6; }
deadline_check

log "shipping tpu-inference + the generation client + manifests"
timeout 900 rsync -az --delete -e "ssh $SSHO" \
  --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' --exclude 'tests' \
  "$REPO/third_party/tpu-inference/" ${USER_R}@"$HOST":/home/${USER_R}/tpu-inference/ \
  >/dev/null 2>&1 || { log "rsync of tpu-inference failed"; exit 7; }
timeout 120 scp $SSHO "$REPO/tpu/muse_glimmer/rs_generate.py" \
  ${USER_R}@"$HOST":~/ >/dev/null 2>&1 || { log "scp of client failed"; exit 7; }
rsh "mkdir -p ~/rs" 60
# erdos is needed early (Gate 0 and the probe read its prompts). jssp and ac1
# are shipped lazily, immediately before their own generation phase, so a
# manifest that is still building on the cluster when the slice lands does not
# silently cost that problem its whole run.
ship_manifest() {
  local m="$1"
  [ -s "$MANIFESTS/$m.json" ] || { log "  manifest $m not built yet"; return 1; }
  timeout 600 scp $SSHO "$MANIFESTS/$m.json" ${USER_R}@"$HOST":~/rs/ >/dev/null 2>&1 \
    && { log "  shipped manifest $m"; return 0; }
  log "  FAILED to ship manifest $m"; return 1
}
ship_manifest erdos || { log "erdos manifest could not be shipped"; exit 7; }

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
  host_watch "$r"
  echo "$r" | grep -q 'ARCH-REGISTERED-OK' && { VENV_OK=1; echo "$r" | tee -a "$PROG"; break; }
  if echo "$r" | grep -q 'VENV-TMUX-GONE'; then
    log "venv build exited without ARCH-REGISTERED-OK"
    rsh "tail -n 60 ~/venv_build.log" 120 | tail -30 | tee -a "$PROG"
    break
  fi
  [ $((i % 6)) -eq 0 ] && log "  venv build still running (~$((i*20))s)"
  sleep 20
done
[ -n "$VENV_OK" ] || { log "venv/plugin preflight FAILED"; exit 7; }
deadline_check

# ------------------------------------------------------------- 3. serve -----
serve_engine() {   # $1 = block size ("default" for tpu-inference's own choice)
  local bs="$1" extra=""
  [ "$bs" != default ] && extra="--block-size ${bs}"
  log "starting vLLM (TP=${TP}, max-model-len=${MAXLEN}, max-num-seqs=${MAXSEQS}, block-size=${bs})"
  # A second engine cannot claim the chips until the first has really let go,
  # so wait for the EngineCore to disappear rather than assuming a fixed sleep.
  rsh "tmux kill-session -t mg 2>/dev/null
pkill -TERM -u \$USER -f '[V]LLM::EngineCore|[v]llm serve' 2>/dev/null
for i in \$(seq 1 30); do pgrep -u \$USER -f '[V]LLM::EngineCore|[v]llm serve' >/dev/null || break; sleep 3; done
pkill -KILL -u \$USER -f '[V]LLM::EngineCore|[v]llm serve' 2>/dev/null
sleep 5; echo ENGINE-SLOT-FREE" 180 | tail -1 | tee -a "$PROG"
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
  --max-model-len ${MAXLEN} \
  --max-num-seqs ${MAXSEQS} ${extra} \
  --download-dir ${REMOTE_MODEL} \
  2>&1 | tee /home/${USER_R}/skyrl-logs/mg-vllm.log
EOS
chmod +x ~/run_mg.sh
tmux new-session -d -s mg 'bash ~/run_mg.sh'
sleep 5; tmux has-session -t mg && echo tmux-UP || echo tmux-FAIL" 120 | tee -a "$PROG"

  log "waiting for the engine to answer a REAL request"
  local up=0 i r alive
  for i in $(seq 1 200); do
    deadline_check
    r=$(rsh "curl -fsS -m 30 -X POST http://127.0.0.1:${VLLM_PORT}/v1/completions -H 'Content-Type: application/json' -d '{\"model\":\"muse-glimmer-30b\",\"prompt\":\"The capital of France is\",\"max_tokens\":4,\"temperature\":0}' 2>/dev/null | head -c 400" 70)
    host_watch "$r"
    if echo "$r" | grep -q '"text"'; then log "ENGINE ANSWERED: $r"; up=1; break; fi
    if [ $((i % 6)) -eq 0 ]; then
      log "  not up yet (~$((i*10))s); vllm log tail:"
      rsh "tail -n 5 ~/skyrl-logs/mg-vllm.log" 60 | sed 's/^/    /' | tee -a "$PROG"
      alive=$(rsh "tmux has-session -t mg 2>/dev/null && echo alive || echo DEAD" 30)
      host_watch "$alive"
      if echo "$alive" | grep -q DEAD; then
        log "engine process died (block-size=${bs})"
        rsh "tail -n 60 ~/skyrl-logs/mg-vllm.log" 90 | tee -a "$LOGDIR/rs2-vllm-crash-${bs}.log"
        return 1
      fi
    fi
    sleep 10
  done
  [ "$up" = 1 ] || { log "engine never answered (block-size=${bs})"; return 1; }
  log "=== ENGINE LIVE (block-size=${bs}) at $(elapsed)s ==="
  rsh "grep -iE 'GPU KV cache size|maximum concurrency|Available KV|block_size|HBM' ~/skyrl-logs/mg-vllm.log | tail -10" 90 | tee -a "$PROG"
  return 0
}

# ------------------------------------------------- 4. GATE 0 (ii): smoke ----
# The renderer and rosetta_stone are new relative to the discover tree. If the
# served model does not produce BOTH channels through the real prompt, the
# system block is wrong and everything after would be garbage.
gate0_smoke() {
  log "=== GATE 0 (ii): live channel smoke on the served model ==="
  rsh "${VLLM_VENV}/bin/python - <<'PY' 2>&1 | tail -40
import json, urllib.request
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('${REMOTE_MODEL}', use_fast=True, local_files_only=True)
man = json.load(open('/home/${USER_R}/rs/erdos.json'))
def single(s):
    ids = tok.encode(s, add_special_tokens=False)
    assert len(ids) == 1, (s, ids)
    return ids[0]
stop = sorted({single('<|eot|>'), single('<|end_of_text|>')})
print('stop_token_ids', stop)
ok = True
for arm in ('high','xhigh'):
    p = man['prompts']['smoke|' + arm]
    print('---', arm, 'prompt_len', p['prompt_len'])
    print('SYSTEM HEAD:', repr(p['text'][:230]))
    body = json.dumps({'model':'muse-glimmer-30b','prompt':p['tokens'],
                       'max_tokens':6000,'temperature':1.0,
                       'stop_token_ids':stop,'return_token_ids':True}).encode()
    req = urllib.request.Request('http://127.0.0.1:${VLLM_PORT}/v1/completions',
                                 data=body, headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=1800) as r:
        res = json.loads(r.read().decode())
    ch = res['choices'][0]
    ids = ch.get('token_ids') or []
    txt = tok.decode(ids) if ids else ch['text']
    fin = ch.get('finish_reason')
    print('gen_tokens', len(ids), 'finish', fin)
    print('FULL:', repr(txt[:900]))
    have_self, have_user = 'to=self' in txt, 'to=user' in txt
    print(arm, 'reasoning_channel(to=self):', have_self,
          '| answer_channel(to=user):', have_user, '| finished:', fin == 'stop')
    ok = ok and have_self and have_user and fin == 'stop'
print('GATE0-SMOKE-' + ('OK' if ok else 'FAIL'))
PY" 2400 | tee -a "$PROG"
  grep -q 'GATE0-SMOKE-OK' "$PROG"
}

# ------------------------------------------------ 5. GATE 1: throughput -----
# Round 1 measured 172 tok/s and only found out 1012 s in.  This measures the
# same quantity in ~3 minutes, at the concurrency generation will actually use,
# on real prompts.  The first burst is thrown away: with SKIP_JAX_PRECOMPILE=1
# the batch shape compiles inside it, and E2E.md 7.2's anomalous "217 tok/s at
# 64 concurrent" is very likely that compile landing inside a single unwarmed
# measurement.
# NB: the result comes back in the PROBE_TPS global rather than on stdout.
# `x=$(probe_tps)` would run it in a subshell, and `host_watch`'s `exit 12`
# would then kill only that subshell -- a lost slice would look like a slow one.
PROBE_TPS=0
probe_tps() {
  local r tps
  r=$(rsh "cd ~ && ${VLLM_VENV}/bin/python ~/rs_generate.py \
        --manifest ~/rs/erdos.json --model-dir ${REMOTE_MODEL} \
        --base http://127.0.0.1:${VLLM_PORT} --probe \
        --probe-conc ${CONC} --probe-tokens ${PROBE_TOKENS} \
        --temperature ${TEMP} 2>&1 | tail -25" 2400)
  echo "$r" | tee -a "$PROG"
  host_watch "$r"
  tps=$(echo "$r" | grep -o 'throughput=[0-9.]*' | tail -1 | cut -d= -f2)
  PROBE_TPS="${tps:-0}"
}

GATE0_DONE=""
BEST_TPS=0; BEST_BLOCK=""; SERVING_BLOCK=""
for bs in $BLOCK_TRY; do
  deadline_check
  serve_engine "$bs" || { log "block-size=${bs} did not serve; next"; continue; }
  SERVING_BLOCK="$bs"
  if [ -z "$GATE0_DONE" ]; then
    if gate0_smoke; then
      log "=== GATE 0 (ii) PASSED ==="; GATE0_DONE=1
    else
      log "*** GATE 0 (ii) FAILED: no reasoning and/or answer channel. STOPPING. ***"
      exit 11
    fi
  fi
  log "=== GATE 1: throughput probe, block-size=${bs}, warm-up burst ==="
  probe_tps; log "  warm-up burst: ${PROBE_TPS} tok/s (discarded)"
  log "=== GATE 1: throughput probe, block-size=${bs}, MEASURED ==="
  probe_tps; tps="$PROBE_TPS"
  log "PROBE block-size=${bs} concurrency=${CONC} -> ${tps} tok/s"
  rsh "grep -o 'Avg generation throughput: [0-9.]* tokens/s, Running: [0-9]* reqs, Waiting: [0-9]* reqs' ~/skyrl-logs/mg-vllm.log | tail -4" 90 | sed 's/^/  ENGINE-SIDE /' | tee -a "$PROG"
  if awk -v a="$tps" -v b="$BEST_TPS" 'BEGIN{exit !(a>b)}'; then
    BEST_TPS="$tps"; BEST_BLOCK="$bs"
  fi
  if awk -v a="$tps" -v m="$MIN_TPS" 'BEGIN{exit !(a>=m)}'; then
    log "block-size=${bs} clears MIN_TPS=${MIN_TPS} -- keeping this engine"
    break
  fi
  log "block-size=${bs} below MIN_TPS=${MIN_TPS}"
done

log "GATE1-BEST block-size=${BEST_BLOCK:-none} throughput=${BEST_TPS} tok/s (min ${MIN_TPS})"
if ! awk -v a="$BEST_TPS" -v m="$MIN_TPS" 'BEGIN{exit !(a>=m)}'; then
  log "*** GATE 1 FAILED: best measured throughput ${BEST_TPS} tok/s < ${MIN_TPS}."
  log "*** 960 items x ~10900 tokens would need $(awk -v a="$BEST_TPS" 'BEGIN{printf "%.1f", 960*10900/(a>0?a:1)/3600}')h. STOPPING rather than grinding."
  exit 14
fi
if [ "$SERVING_BLOCK" != "$BEST_BLOCK" ]; then
  log "re-serving on the winning block-size=${BEST_BLOCK}"
  serve_engine "$BEST_BLOCK" || { log "could not re-serve the winning config"; exit 8; }
  probe_tps; log "  re-warm: ${PROBE_TPS} tok/s"
fi
deadline_check

# ------------------------------------------------------- 6. generation ------
run_problem() {   # $1 problem  $2 seconds to allow
  local prob="$1" budget="$2"
  rsh "test -s ~/rs/${prob}.json && echo has-manifest" 60 | grep -q has-manifest || {
    ship_manifest "$prob" || { log "no manifest for ${prob} -- skipping"; return 1; }
  }
  local dl=$(( $(date +%s) + budget ))
  log "--- generating ${prob} (budget ${budget}s, deadline $(date -u -d @${dl} +%H:%M:%S)) ---"
  # RESUME: ship whatever has already been generated back UP to the host so
  # successive slices accumulate instead of each one restarting from zero.
  if [ -s "$GENDIR/${prob}.jsonl" ]; then
    local have; have=$(wc -l < "$GENDIR/${prob}.jsonl")
    log "  [${prob}] resuming: uploading ${have} existing generations"
    timeout 900 scp $SSHO "$GENDIR/${prob}.jsonl" \
      ${USER_R}@"$HOST":~/rs/${prob}.jsonl >/dev/null 2>&1 \
      || log "  [${prob}] WARNING: resume upload failed, will regenerate"
  fi
  rsh "cd ~ && nohup ${VLLM_VENV}/bin/python ~/rs_generate.py \
        --manifest ~/rs/${prob}.json --out ~/rs/${prob}.jsonl \
        --model-dir ${REMOTE_MODEL} --base http://127.0.0.1:${VLLM_PORT} \
        --phase1-max-tokens ${PHASE1} --context-window ${MAXLEN} \
        --temperature ${TEMP} --concurrency ${CONC} --deadline-epoch ${dl} \
        > ~/rs/${prob}.gen.log 2>&1 &
      sleep 5; echo launched" 120 | tail -1 | tee -a "$PROG"

  # The 2-minute check the brief asks for, from BOTH sides: the client's own
  # in-flight gauge and the engine's scheduler counters.
  sleep 120
  log "  [${prob}] === 2-MINUTE CONCURRENCY CHECK ==="
  rsh "grep -E 'LAUNCHING|HB t=' ~/rs/${prob}.gen.log | tail -4" 90 | sed 's/^/    CLIENT /' | tee -a "$PROG"
  rsh "grep -o 'Avg generation throughput: [0-9.]* tokens/s, Running: [0-9]* reqs, Waiting: [0-9]* reqs' ~/skyrl-logs/mg-vllm.log | tail -3" 90 | sed 's/^/    ENGINE /' | tee -a "$PROG"

  local done_flag="" poll=0
  while true; do
    deadline_check
    sleep 75
    poll=$((poll + 1))
    local s
    # `pgrep -f rs_generate.py` would self-match the remote shell's own argv
    # (a repeat gotcha in this project), so bracket the first character.
    s=$(rsh "tail -n 2 ~/rs/${prob}.gen.log 2>/dev/null; echo ALIVE=\$(pgrep -fc '[r]s_generate.py' || echo 0)" 90)
    log "  [${prob}] $(echo "$s" | tr '\n' ' | ')"
    if [ $((poll % 4)) -eq 0 ]; then
      rsh "grep -o 'Avg generation throughput: [0-9.]* tokens/s, Running: [0-9]* reqs, Waiting: [0-9]* reqs' ~/skyrl-logs/mg-vllm.log | tail -1" 60 | sed 's/^/    ENGINE /' | tee -a "$PROG"
    fi
    host_watch "$s"
    # pull partial results back every poll: the slice can be preempted at any
    # moment and raw generations are the expensive artefact.
    timeout 300 scp $SSHO ${USER_R}@"$HOST":~/rs/${prob}.jsonl "$GENDIR/${prob}.jsonl" >/dev/null 2>&1
    echo "$s" | grep -q '^DONE ' && { done_flag=1; break; }
    if echo "$s" | grep -q '^ALIVE=0'; then
      log "  [${prob}] generator is no longer running"
      sleep 20
      timeout 300 scp $SSHO ${USER_R}@"$HOST":~/rs/${prob}.jsonl "$GENDIR/${prob}.jsonl" >/dev/null 2>&1
      break
    fi
    [ "$(date +%s)" -ge "$dl" ] && { log "  [${prob}] budget reached"; break; }
  done
  timeout 600 scp $SSHO ${USER_R}@"$HOST":~/rs/${prob}.gen.log "$GENDIR/${prob}.gen.log" >/dev/null 2>&1
  local n
  n=$(wc -l < "$GENDIR/${prob}.jsonl" 2>/dev/null || echo 0)
  log "  [${prob}] pulled ${n} raw generations -> $GENDIR/${prob}.jsonl"
  [ -n "$done_flag" ]
}

# Reserve time for the final pull + teardown.
RESERVE=600
for prob in ${PROBLEMS:-erdos jssp ac1}; do
  rem=$(( $(remaining) - RESERVE ))
  if [ "$rem" -lt 600 ]; then log "not enough slice time left for ${prob} (${rem}s)"; break; fi
  run_problem "$prob" "$rem"
done

log "collecting engine stats"
rsh "grep -iE 'GPU KV cache size|maximum concurrency|Available KV|HBM|num_kv_cache_groups|block_size' ~/skyrl-logs/mg-vllm.log | tail -20" 120 | tee -a "$PROG"
rsh "grep -o 'Avg generation throughput: [0-9.]* tokens/s, Running: [0-9]* reqs, Waiting: [0-9]* reqs' ~/skyrl-logs/mg-vllm.log | tail -40" 120 | tee -a "$LOGDIR/rs2-engine-rate.log"
rsh "tail -n 200 ~/skyrl-logs/mg-vllm.log" 120 > "$LOGDIR/rs2-vllm-tail.log" 2>&1

log "=== DONE at $(elapsed)s; tearing down ==="
exit 0
