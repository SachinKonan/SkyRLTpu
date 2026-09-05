#!/usr/bin/env bash
#SBATCH --job-name=arena-coserve
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/pallas_arena/coserve-%j.log
#
# ONE slice runs the generator AND the grader, and rollouts are graded the
# instant they finish.
#
# Copied from the league's current shape (SkyRLTpu-league:tpu/launch_cell.sh):
# one cell per slice, engine on w0, graders on the other hosts, async grading
# (TTD_EVAL_BACKEND=ray). Isolation is deliberate there -- "a preemption costs
# one experimental condition rather than two, and cells do not share a Ray
# grading cluster" -- and it is the fix for what cost this experiment ~40
# hours: the serving slice and the judge slice were acquired SEPARATELY and
# neither would land.
#
#   v5p-16 = 2 hosts x 4 chips
#     w0   vLLM (SERVE_TP=4, the proven *-32k-b16-v5p8 XLA caches)
#     w1   pallas_arena.judge.ray_pool --chips 4 (RAY_CHIPS default)
#   SLURM node   judge/queue.py + gen_smoke client
#
# Grading must NEVER land on w0: timing IS the measurement and vLLM holds w0's
# chips. The league gets that from Ray's `--num-cpus=0` on the head; here it is
# structural, because the pool only ever binds w1's chips.
#
# The queue lives on the SLURM node and w1 reaches it through a REVERSE tunnel,
# exactly as probe/probe.sbatch already does. The client reaches vLLM on w0
# through a forward tunnel. Nothing depends on a shared filesystem between
# hosts -- there isn't one (the league hit this too: sandbox_reward_evaluator
# ships code BY VALUE for the same reason).
set -uo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
cd "$REPO"
# `python -m pallas_arena.judge.queue` needs tpu/ on the path -- the package
# lives at tpu/pallas_arena, not the repo root. rl_judges.sbatch:77 does the
# same. Omitting it killed all four cells instantly on 2026-08-30 with
# ModuleNotFoundError, and lib_splice inside the async submitter needs it too.
export PYTHONPATH="${REPO}/tpu:${PYTHONPATH:-}"
out_dir="$REPO/runs/pallas_arena"
export PATH="/n/fs/vision-mix/sk7524/google-cloud-sdk/bin:$PATH"

CELL=${CELL:?set CELL, e.g. qwen-splash}
TASK=${TASK:?set TASK, e.g. splash_attention}
SEED_FILE=${SEED_FILE:?set SEED_FILE}
SEED_OBS=${SEED_OBS:?set SEED_OBS}
SEED_REWARD=${SEED_REWARD:?set SEED_REWARD}

ACCEL=${ACCEL:-v5p-16}
ZONE=${ZONE:-us-east5-a}
RUNTIME=${RUNTIME:-v2-alpha-tpuv5}          # v5p image; v6e would need -tpuv6e
PROJECT=${PROJECT:-vision-mix}
SERVE_TP=${SERVE_TP:-4}                     # 4 chips per v5p-16 host
# PIN THE BATCH SHAPE. The XLA cache name encodes all three dims it was
# compiled at -- qwen35-*32k*-*b16*-*v5p8* = ctx 32768, max_num_seqs 16, 4
# chips. Any mismatch forces a FULL TPU recompile: measured on arm attempt 3,
# MAX_NUM_SEQS=6 against a b16 cache and the engine never became ready in
# 3600s. This happened to be right only because serve_vllm.sh defaults to 16;
# state it here so the cache contract is visible next to the cache path.
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
RAY_CHIPS=${RAY_CHIPS:-4}
# Standalone-host mesh for w1, derived from the chip count -- the first
# multi-host fix hardcoded the 4-chip v5p values, which would break an 8-chip
# v6e-16 host the same way the missing env broke v5p (wrong process mesh).
case "$RAY_CHIPS" in
  8) W1_MESH=4,2,1; W1_CHIPS=0,1,2,3,4,5,6,7 ;;
  4) W1_MESH=2,2,1; W1_CHIPS=0,1,2,3 ;;
  2) W1_MESH=2,1,1; W1_CHIPS=0,1 ;;
  *) W1_MESH=1,1,1; W1_CHIPS=0 ;;
esac
QR=${QR:-sk7524-coserve-${SLURM_JOB_ID}}
QUEUE_PORT=${QUEUE_PORT:-8811}
VLLM_PORT=${VLLM_PORT:-8001}
# derived from QUEUE_PORT (unique per cell) so two cells scheduled onto the
# same SLURM node cannot collide on the vLLM forward-tunnel port
LOCAL_PORT=${LOCAL_PORT:-$((10000 + QUEUE_PORT))}
LAND_DEADLINE_S=${LAND_DEADLINE_S:-64800}
GROUP_SIZE=${GROUP_SIZE:-32}
CTX=${CTX:-32768}
THINK_BUDGET=${THINK_BUDGET:-12288}
BUCKET=gs://sk7524-tinker-tpu-us-east5
MODEL=${MODEL:-Qwen/Qwen3.5-27B}
HF_GCS=${HF_GCS:-${BUCKET}/hf-cache}
XLA_GCS=${XLA_GCS:-${BUCKET}/vllm-xla-cache-qwen35-32k-b16-v5p8}
CACHE=${CACHE:-gs://sk7524-pallas-arena-us-east5/reward-cache-coserve}
GENS="${out_dir}/coserve-gens-${CELL}-${SLURM_JOB_ID}.jsonl"
GRADED="${out_dir}/coserve-graded-${CELL}-${SLURM_JOB_ID}.json"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

cleanup() {
  log "cleanup: tearing down"
  pkill -f "tpu-vm ssh ${QR}" 2>/dev/null
  kill $(jobs -p) 2>/dev/null                # every background loop, by pid
  timeout 600 gcloud compute tpus queued-resources delete "$QR" \
    --zone="$ZONE" --project="$PROJECT" --force --quiet >/dev/null 2>&1
  for _ in 1 2 3; do
    timeout 200 gcloud compute tpus queued-resources list --zone="$ZONE" --project="$PROJECT" \
      --format='value(name)' 2>/dev/null | grep -qx "$QR" || { log "cleanup: ${QR} gone"; return; }
    sleep 30
    timeout 600 gcloud compute tpus queued-resources delete "$QR" \
      --zone="$ZONE" --project="$PROJECT" --force --quiet >/dev/null 2>&1
  done
  log "cleanup: WARNING ${QR} may still exist"
}
trap cleanup EXIT
trap 'log "caught TERM/INT"; cleanup; exit 143' TERM INT HUP

# CASES ARE MANDATORY, and they are copied VERBATIM from rl_judges.sbatch.
# Two reasons. (1) ray_pool fatals every candidate with "judge_fault: no case
# list configured" when --cases is absent -- the first coserve launch would
# have graded nothing even after its PYTHONPATH fix (found in review,
# 2026-08-31). (2) Reward is a geomean over the scored cases, so any drift in
# this list makes the cell incomparable with every other run -- the same trap
# as the prompt's 7-case 0.262 seed bar. max_tp_width=4 on a 4-chip host
# keeps tp4 coverage identical to the v6e-8 judges (which also capped at 4).
#
# The judge JAX cache points at the v5p-8 prefix, not an ACCEL-keyed v5p-16
# one: w1 is the same 4-chip v5p host shape the phase-2/5 v5p-8 judges used,
# so their compiled artifacts are directly reusable, and a fresh prefix would
# recompile everything from scratch on the first candidate.

# ---------------------------------------------------------------- 1. queue
log "starting queue on $(hostname -s):${QUEUE_PORT}"
uv run --isolated --extra dev --extra jax --with fastapi --with uvicorn --with flatbuffers \
  python -m pallas_arena.judge.queue --port "$QUEUE_PORT" --host 0.0.0.0 \
  > "${out_dir}/coserve-queue-${SLURM_JOB_ID}.log" 2>&1 &
for _ in $(seq 1 300); do
  curl -fsS -m 5 "http://127.0.0.1:${QUEUE_PORT}/status" >/dev/null 2>&1 && break
  sleep 2
done
curl -fsS -m 5 "http://127.0.0.1:${QUEUE_PORT}/status" >/dev/null 2>&1 || {
  log "FATAL: queue never came up"; tail -20 "${out_dir}/coserve-queue-${SLURM_JOB_ID}.log"; exit 1; }
log "queue up"

# ------------------------------------------------------------------- 2. QR
# create_qr VERIFIES the request actually exists afterward, because
# `queued-resources create` can print a JSON error body + "network
# connectivity" noise and still exit 0 -- the v6e-16 create for cell
# gemma-rglru did exactly that on 2026-09-01, and describe then returned
# NOT_FOUND while the old poll loop treated the empty state as "keep waiting"
# and hung for its full 18h deadline holding a slurm allocation. Permission
# denial is terminal; anything else is retried.
create_qr() {
  local out
  out=$(timeout 600 gcloud compute tpus queued-resources create "$QR" --node-id="$QR" \
    --accelerator-type="$ACCEL" --runtime-version="$RUNTIME" --spot \
    --zone="$ZONE" --project="$PROJECT" 2>&1)
  echo "$out" | grep -iE "error|exhaust|quota|code|reservation|permission" | head -3 || echo "$out" | tail -3
  echo "$out" | grep -qE "does not have permission|PERMISSION_DENIED|\"code\": 7" && {
    log "FATAL: ${ACCEL} not permitted in ${ZONE}"; exit 2; }
  # proof of existence, not the create's exit code
  timeout 120 gcloud compute tpus queued-resources describe "$QR" --zone="$ZONE" \
    --project="$PROJECT" --format='value(name)' >/dev/null 2>&1
}
log "creating spot QR ${QR} (${ACCEL}, ${ZONE}, runtime ${RUNTIME})"
for attempt in 1 2 3 4 5; do
  create_qr && { log "QR request exists (attempt ${attempt})"; break; }
  log "create attempt ${attempt} left no QR; retrying in 60s"
  timeout 120 gcloud compute tpus queued-resources delete "$QR" --zone="$ZONE" \
    --project="$PROJECT" --force --quiet >/dev/null 2>&1
  sleep 60
  [ "$attempt" = 5 ] && { log "FATAL: could not create ${QR} after 5 attempts"; exit 1; }
done

deadline=$(( $(date +%s) + LAND_DEADLINE_S )); st=""; gone=0
while [ "$(date +%s)" -lt "$deadline" ]; do
  st=$(timeout 300 gcloud compute tpus queued-resources describe "$QR" --zone="$ZONE" \
        --project="$PROJECT" --format='value(state.state)' 2>/dev/null)
  [ "$st" = "ACTIVE" ] && break
  case "$st" in
    FAILED|SUSPENDED|SUSPENDING) log "QR state=$st; giving up"; exit 1 ;;
    "" )  # empty = describe found nothing; two in a row means the QR vanished
      gone=$(( gone + 1 ))
      if [ "$gone" -ge 2 ]; then
        log "QR is GONE (NOT_FOUND x2); re-creating"
        create_qr || true; gone=0
      fi ;;
    * ) gone=0 ;;
  esac
  sleep 60
done
[ "$st" = "ACTIVE" ] || { log "FATAL: not ACTIVE (state=${st:-gone})"; exit 1; }
log "QR ACTIVE"

# --------------------------------------------------- 3. per-host bringup
# w0 and w1 in PARALLEL: the installs and weight pulls are the long poles and
# they are independent (the league does the same).
for w in 0 1; do
  for _ in 1 2 3 4 5 6; do
    timeout 900 gcloud compute tpus tpu-vm ssh "$QR" --zone="$ZONE" --project="$PROJECT" \
      --worker="$w" --command="echo ssh-ok w${w}" && break
    sleep 20
  done
done

timeout 900 gcloud compute tpus tpu-vm scp "${REPO}/tpu/pallas_arena/probe/serve_vllm.sh" \
  "${QR}:~/serve_vllm.sh" --zone="$ZONE" --project="$PROJECT" --worker=0
timeout 900 gcloud compute tpus tpu-vm scp --recurse "${REPO}/tpu/pallas_arena" \
  "${QR}:~/arena-pallas_arena" --zone="$ZONE" --project="$PROJECT" --worker=1

( timeout 5400 gcloud compute tpus tpu-vm ssh "$QR" --zone="$ZONE" --project="$PROJECT" --worker=0 \
    --command="MODEL='${MODEL}' PORT=${VLLM_PORT} MAX_MODEL_LEN=${CTX} SERVE_TP=${SERVE_TP} \
      MAX_NUM_SEQS=${MAX_NUM_SEQS} \
      HF_CACHE_GCS='${HF_GCS}' XLA_CACHE_GCS='${XLA_GCS}' ${EXTRA_PIP:+EXTRA_PIP='${EXTRA_PIP}'} \
      bash ~/serve_vllm.sh" > "${out_dir}/coserve-w0-serve-${SLURM_JOB_ID}.log" 2>&1
  log "w0 serve launcher rc=$?" ) &
p_serve=$!

# STANDALONE-HOST TPU ENV, the multi-host lesson (2026-08-31): a v5p-16 is ONE
# 2-host slice, and default libtpu init waits for every host to join a single
# coordinated program -- w1's lone provision self-check therefore died with
# "Worker failed to join a slice within 15m", on all four cells. vLLM on w0
# only survives because serve_vllm.sh sets exactly these vars. The grading
# tasks are already safe (_pin_chips sets the same trio per task, per width,
# before importing jax); the provision sanity check was the ONE bare TPU init.
( timeout 5400 gcloud compute tpus tpu-vm ssh "$QR" --zone="$ZONE" --project="$PROJECT" --worker=1 \
    --command="rm -rf ~/arena && mkdir -p ~/arena && mv ~/arena-pallas_arena ~/arena/pallas_arena && \
      TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=${W1_MESH} \
      TPU_VISIBLE_CHIPS=${W1_CHIPS} CLOUD_TPU_TASK_ID=0 \
      ARENA_CACHE_PREFIX='${CACHE}' bash ~/arena/pallas_arena/phase2/provision_judge.sh" \
    > "${out_dir}/coserve-w1-prov-${SLURM_JOB_ID}.log" 2>&1
  log "w1 provision rc=$?" ) &
p_prov=$!
# EXPLICIT PIDS: bare `wait` also waits on the queue server, which never
# exits -- all four cells hung here for 90 minutes with idle ACTIVE slices
# (probe.sbatch waits on "$p0" "$p1" for the same reason).
wait "$p_serve" "$p_prov"
log "both hosts provisioned"

# ------------------------------------------------- 4. tunnels + judge loop
# forward: client -> w0 vLLM ; reverse: w1 -> our queue
( while true; do
    timeout 43200 gcloud compute tpus tpu-vm ssh "$QR" --zone="$ZONE" --project="$PROJECT" \
      --worker=0 -- -L "${LOCAL_PORT}:localhost:${VLLM_PORT}" -N \
      -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=4 \
      >> "${out_dir}/coserve-tunnel-${SLURM_JOB_ID}.log" 2>&1
    sleep 5
  done ) &

( while true; do
    timeout 43200 gcloud compute tpus tpu-vm ssh "$QR" --zone="$ZONE" --project="$PROJECT" --worker=1 \
      --command="set -o pipefail; cd ~/arena && RAY_CHIPS=${RAY_CHIPS} \
        bash ~/arena/pallas_arena/phase2/ray_start_tpu.sh; \
        PYTHONPATH=~/arena JAX_COMPILATION_CACHE_DIR=\$HOME/jax-compile-cache \
        ~/arena-venv/bin/python -m pallas_arena.judge.ray_pool \
        --queue http://127.0.0.1:${QUEUE_PORT} --problems ${TASK} \
        --cases 'rg_lru=probe-4x2048x2560,probe-2x1024x2560,probe-8x512x2560,probe-2x4096x2560,probe-4x2048x1024,probe-holdout-2x1500x2560,tp4-4x2048x2560,tp4-holdout-2x1500x2560' \
        --cases 'splash_attention=probe-h8-s4096,probe-h4-s2048,probe-h16-s1024,probe-h8-s4096-d64,mixtral-8x7b-gqa32x8-s4096,mqa-h32kv1-s4096,deepseek2-16b-s1024-d192-dv128,probe-holdout-h4-s2049,mixtral-holdout-gqa32x8-s2049,tp4-h32-s4096,tp4-gqa32x8-s4096,tp4-mqa-h32kv1-s4096,tp4-holdout-h32-s2049' \
        --chips ${RAY_CHIPS} --max-tp-width 4 --timing-pairs 20 \
        --cache '${CACHE}' --poll-s 1 \
        --jax-cache-gcs '${JAX_CACHE_GCS:-gs://sk7524-pallas-arena-us-east5/judge-jax-cache-v5p-8}' \
        2>&1 | tee -a \$HOME/pool.log" \
      -- -R "${QUEUE_PORT}:localhost:${QUEUE_PORT}" -o ExitOnForwardFailure=yes \
         -o ServerAliveInterval=30 -o ServerAliveCountMax=4 \
      >> "${out_dir}/coserve-w1-pool-${SLURM_JOB_ID}.log" 2>&1
    log "w1 pool session ended rc=$?; restarting"
    sleep 10
  done ) &

# ------------------------------------------------------ 5. wait for engine
log "waiting for the engine on 127.0.0.1:${LOCAL_PORT}"
for _ in $(seq 1 180); do
  curl -fsS -m 5 "http://127.0.0.1:${LOCAL_PORT}/v1/models" >/dev/null 2>&1 && break
  sleep 30
done
curl -fsS -m 5 "http://127.0.0.1:${LOCAL_PORT}/v1/models" >/dev/null 2>&1 || {
  log "FATAL: engine never became ready"; exit 1; }
log "=== engine ready ==="
for _ in $(seq 1 60); do
  n=$(curl -fsS -m 5 "http://127.0.0.1:${QUEUE_PORT}/status" 2>/dev/null | python3 -c \
      "import json,sys; d=json.load(sys.stdin); print(len([w for w,a in (d.get('workers_seen') or {}).items() if a<300]))" 2>/dev/null)
  [ "${n:-0}" -ge 1 ] && { log "judge attached (${n} worker)"; break; }
  sleep 20
done

# ---------------------------------------------- 6. generate + grade ASYNC
log "generating ${GROUP_SIZE} rollouts, grading each on completion"
uv run --isolated --extra jax --with fastapi python tpu/pallas_arena/probe/gen_smoke.py \
  --server "http://127.0.0.1:${LOCAL_PORT}" --model "$MODEL" \
  --out "$GENS" --group-size "$GROUP_SIZE" --concurrency 6 \
  --cells "${TASK}:rf3s" --ctx "$CTX" --think-budget "$THINK_BUDGET" \
  --max-tokens 10240 --answer-cap 15360 \
  --seed-file "$SEED_FILE" --seed-observation "$SEED_OBS" --seed-reward "$SEED_REWARD" \
  --lib-imports ${ENABLE_THINKING_KWARG:+--enable-thinking-kwarg} \
  --grade-queue "http://127.0.0.1:${QUEUE_PORT}" --grade-seed-file "$SEED_FILE" \
  --grade-out "$GRADED" --grade-wait-s 14400
rc=$?
log "generate+grade rc=${rc}; gens=${GENS} graded=${GRADED}"
exit $rc
