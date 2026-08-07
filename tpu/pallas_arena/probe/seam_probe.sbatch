#!/bin/bash
# SEAM PROBE for the Pallas kernel arena. NO RL: generation + grading only.
#
# 2 models x 2 variants (reference | seam) x 5 tasks = 20 configurations,
# round-robin by round so every cell has the SAME number of complete groups
# whenever the wall clock stops.
#
# Two spot QRs, distinctly named, and never more than two alive:
#   sk7524-seam-serve  v5p-16 us-east5-a  -- worker 0 Qwen3.5-27B (thinking
#                                            DISABLED), worker 1 gemma-4-31B-it,
#                                            each vLLM TP=4
#   sk7524-seam-judge  v6e-1  us-east5-b  -- the arena judge, ALL FIVE problems
#
# HARD RESOURCE RULES, in the code rather than in a comment:
#   * both QRs deleted on EVERY exit path -- success, failure, exception,
#     TERM/INT/HUP (slurm timeout, scancel, node drain) and the hard cap.
#     Phase 4 found `trap cleanup EXIT` does NOT fire on an untrapped
#     SIGTERM, which is exactly what a slurm timeout sends;
#   * a 3 h hard cap measured from the FIRST QR create;
#   * if the v5p-16 has not landed in 45 min we report and STOP. No falling
#     back to a bigger slice: capacity is contended;
#   * the end state is verified empty in BOTH zones, for BOTH names.
#
#SBATCH -p all
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 4:30:00
#SBATCH --job-name=seam-probe
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/pallas_arena/seam-probe-%j.log

set -uo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: run under SLURM (sbatch); no compute on the login node." >&2
  exit 1
fi

PROJECT=${PROJECT:-vision-mix}
SERVE_QR=${SERVE_QR:-sk7524-seam-serve}
SERVE_ZONE=${SERVE_ZONE:-us-east5-a}
SERVE_ACCEL=${SERVE_ACCEL:-v5p-16}
SERVE_RUNTIME=${SERVE_RUNTIME:-v2-alpha-tpuv5}
JUDGE_QR=${JUDGE_QR:-sk7524-seam-judge}
JUDGE_ZONE=${JUDGE_ZONE:-us-east5-b}
JUDGE_ACCEL=${JUDGE_ACCEL:-v6e-1}
JUDGE_RUNTIME=${JUDGE_RUNTIME:-v2-alpha-tpuv6e}

QUEUE_PORT=${QUEUE_PORT:-8793}
VLLM_PORT=${VLLM_PORT:-8001}
QWEN_LOCAL_PORT=${QWEN_LOCAL_PORT:-8111}
GEMMA_LOCAL_PORT=${GEMMA_LOCAL_PORT:-8112}

LAND_DEADLINE_S=${LAND_DEADLINE_S:-2700}   # 45 min, per the hard rules
HARD_CAP_S=${HARD_CAP_S:-10800}            # 3 h from the first QR create
SERVE_READY_S=${SERVE_READY_S:-3600}
LEASE_TIMEOUT_S=${LEASE_TIMEOUT_S:-600}
GRADE_BUDGET_S=${GRADE_BUDGET_S:-300}
COMPILE_BUDGET_S=${COMPILE_BUDGET_S:-90}
TIMING_PAIRS=${TIMING_PAIRS:-6}
# 20 configurations x 16 = 320 generations a round, and ONE v6e-1 grades every
# survivor. The headline counts only COMPLETE groups, so a round whose
# candidates are still in the judge queue when the clock stops contributes
# nothing -- better to finish 2-3 rounds across all 20 cells than to start 5
# and complete none. ROUND_RESERVE buys a long pure-draining tail.
ROUNDS=${ROUNDS:-3}
ROUND_RESERVE_S=${ROUND_RESERVE_S:-1800}
GROUP_SIZE=${GROUP_SIZE:-16}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-12000}
CACHE=${CACHE:-gs://sk7524-pallas-arena-us-east5/reward-cache-seam}

QWEN_MODEL=${QWEN_MODEL:-Qwen/Qwen3.5-27B}
GEMMA_MODEL=${GEMMA_MODEL:-google/gemma-4-31B-it}
QWEN_MAXLEN=${QWEN_MAXLEN:-22528}          # matches vllm-xla-cache-22k
GEMMA_MAXLEN=${GEMMA_MAXLEN:-16384}        # matches vllm-xla-cache-gemma4-31b-16k
BUCKET=gs://sk7524-tinker-tpu-us-east5
QWEN_HF_GCS=${QWEN_HF_GCS:-${BUCKET}/hf-cache}
GEMMA_HF_GCS=${GEMMA_HF_GCS:-${BUCKET}/hf-cache-gemma4}
QWEN_XLA_GCS=${QWEN_XLA_GCS:-${BUCKET}/vllm-xla-cache-22k}
GEMMA_XLA_GCS=${GEMMA_XLA_GCS:-${BUCKET}/vllm-xla-cache-gemma4-31b-16k}

# All five problems on one chip. NO SPACES anywhere in this string: the
# worker's parser strips only the outer whitespace of a spec, so `a : b`
# becomes the problem name `'a '` and a KeyError that kills the whole worker
# before any problem boots. Mirrors probe/configs.py:TASK_CASES exactly.
PROBLEM_ARG=${PROBLEM_ARG:-'splash_attention:probe-h8-s4096,probe-h4-s2048,probe-holdout-h4-s2049;flce:probe-4096x2880x151936,probe-2048x2880x151936,probe-holdout-3000x2880x151936;ragged_paged_attention:probe-b16-len1024,probe-b8-len512,probe-holdout-b17-len512;megablox_gmm:probe-m4096-e4-uniform,probe-m2048-e4-zipf,probe-holdout-m3000-e4-zipf;rg_lru:probe-4x2048x2560,probe-2x1024x2560,probe-holdout-2x1500x2560'}

repo_root=/n/fs/vision-mix/sk7524/SkyRLTpu
out_dir="${repo_root}/runs/pallas_arena"
work_dir="${out_dir}/seam-work-${SLURM_JOB_ID}"
boot_report="${out_dir}/seam-judge-boot-${SLURM_JOB_ID}.json"
mkdir -p "$out_dir" "$work_dir"
cd "$repo_root"
export PYTHONPATH="${repo_root}/tpu:${PYTHONPATH:-}"

queue_pid=""; watchdog_pid=""; judge_pid=""; serve_pid=""; bootfetch_pid=""
tunnel_pids=()
t_first_qr=""

# ------------------------------------------------------------------ cleanup
cleaned=""
cleanup() {
  [ -n "$cleaned" ] && return 0
  cleaned=1
  echo "[cleanup] $(date +%H:%M:%S) tearing down (always-delete, both QRs)"
  for p in "${tunnel_pids[@]:-}" "$judge_pid" "$serve_pid" "$watchdog_pid" "$queue_pid" "$bootfetch_pid"; do
    [ -n "${p:-}" ] && kill "$p" 2>/dev/null
  done
  # the supervisors' gcloud-ssh grandchildren hold the tunnels and would
  # outlive their parents; scoped to our own QR names
  pkill -P $$ 2>/dev/null
  pkill -f "tpu-vm ssh ${SERVE_QR}" 2>/dev/null
  pkill -f "tpu-vm ssh ${JUDGE_QR}" 2>/dev/null
  sleep 1
  del_pids=()
  ( timeout 600 gcloud compute tpus queued-resources delete "$SERVE_QR" --zone="$SERVE_ZONE" \
      --project="$PROJECT" --force --quiet >/dev/null 2>&1; echo "[cleanup] delete issued: $SERVE_QR" ) &
  del_pids+=($!)
  ( timeout 600 gcloud compute tpus queued-resources delete "$JUDGE_QR" --zone="$JUDGE_ZONE" \
      --project="$PROJECT" --force --quiet >/dev/null 2>&1; echo "[cleanup] delete issued: $JUDGE_QR" ) &
  del_pids+=($!)
  # NEVER a bare `wait` here: the queue server is a background child that by
  # design never exits (phase-5 attempt 1 wedged on exactly that, with five
  # QRs alive and nobody watching). Always wait on explicit PIDs.
  wait "${del_pids[@]}" 2>/dev/null

  # "Issued" is not "gone". The only acceptable end state is an empty list of
  # BOTH names in BOTH zones -- and a QR delete is supposed to take its node
  # with it, but "supposed to" is not "did", so check nodes too.
  for z in "$SERVE_ZONE" "$JUDGE_ZONE"; do
    left=$(timeout 300 gcloud compute tpus queued-resources list --zone="$z" --project="$PROJECT" \
             --format='value(name)' 2>/dev/null | grep -E "^(${SERVE_QR}|${JUDGE_QR})$" || true)
    if [ -n "$left" ]; then
      echo "[cleanup] SURVIVORS in $z: $left -- deleting again"
      for name in $left; do
        timeout 600 gcloud compute tpus queued-resources delete "$name" --zone="$z" \
          --project="$PROJECT" --force --quiet >/dev/null 2>&1
      done
      sleep 25
      timeout 300 gcloud compute tpus queued-resources list --zone="$z" --project="$PROJECT" \
        --format='value(name,state.state)' 2>/dev/null | grep -E "^(${SERVE_QR}|${JUDGE_QR})\b" \
        && echo "[cleanup] QRs STILL PRESENT in $z -- MANUAL ACTION NEEDED" \
        || echo "[cleanup] verified empty of seam QRs in $z"
    else
      echo "[cleanup] verified empty of seam QRs in $z"
    fi
    nodes=$(timeout 300 gcloud compute tpus tpu-vm list --zone="$z" --project="$PROJECT" \
              --format='value(name)' 2>/dev/null | grep -E "^(${SERVE_QR}|${JUDGE_QR})$" || true)
    if [ -n "$nodes" ]; then
      echo "[cleanup] ORPHAN NODES in $z: $nodes -- deleting"
      for n in $nodes; do
        timeout 600 gcloud compute tpus tpu-vm delete "$n" --zone="$z" --project="$PROJECT" --quiet >/dev/null 2>&1
      done
    else
      echo "[cleanup] verified empty of seam nodes in $z"
    fi
  done
  [ -n "$t_first_qr" ] && echo "[cleanup] QR lifetime $(( $(date +%s) - t_first_qr ))s"
}
trap cleanup EXIT
trap 'echo "[trap] caught TERM/INT/HUP"; cleanup; exit 143' TERM INT HUP

# -------------------------------------------------------------------- queue
echo "=== seam: queue on $(hostname):${QUEUE_PORT} $(date +%H:%M:%S) ==="
uv run --isolated --extra dev --extra jax --with fastapi --with uvicorn --with flatbuffers \
  python -m pallas_arena.judge.queue --port "$QUEUE_PORT" --host 127.0.0.1 \
  --lease-timeout "$LEASE_TIMEOUT_S" > "${out_dir}/seam-queue-${SLURM_JOB_ID}.log" 2>&1 &
queue_pid=$!
for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:${QUEUE_PORT}/status" >/dev/null 2>&1 && break
  sleep 2
done
curl -sf "http://127.0.0.1:${QUEUE_PORT}/status" >/dev/null || { echo "queue failed to start"; exit 1; }

# --------------------------------------------------------------- create QRs
echo "=== creating 2 spot QRs $(date +%H:%M:%S) ==="
create_pids=()
( timeout 600 gcloud compute tpus queued-resources create "$SERVE_QR" --node-id="$SERVE_QR" \
    --accelerator-type="$SERVE_ACCEL" --runtime-version="$SERVE_RUNTIME" --spot \
    --zone="$SERVE_ZONE" --project="$PROJECT" 2>&1 | tail -1 ) & create_pids+=($!)
( timeout 600 gcloud compute tpus queued-resources create "$JUDGE_QR" --node-id="$JUDGE_QR" \
    --accelerator-type="$JUDGE_ACCEL" --runtime-version="$JUDGE_RUNTIME" --spot \
    --zone="$JUDGE_ZONE" --project="$PROJECT" 2>&1 | tail -1 ) & create_pids+=($!)
wait "${create_pids[@]}" 2>/dev/null
t_first_qr=$(date +%s)
echo "=== QRs created at $(date +%H:%M:%S); hard cap at $(date -d "@$(( t_first_qr + HARD_CAP_S ))" +%H:%M:%S) ==="

( trap - EXIT TERM INT HUP
  sleep "$HARD_CAP_S"
  echo "[watchdog] $(date +%H:%M:%S) HARD CAP ${HARD_CAP_S}s since the first QR create -- tearing down"
  kill -TERM $$ ) &
watchdog_pid=$!

wait_active() {  # name zone -> 0 if ACTIVE within LAND_DEADLINE_S
  local name=$1 zone=$2 st="" deadline=$(( t_first_qr + LAND_DEADLINE_S ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    st=$(timeout 300 gcloud compute tpus queued-resources describe "$name" --zone="$zone" \
           --project="$PROJECT" --format='value(state.state)' 2>/dev/null)
    [ "$st" = "ACTIVE" ] && { echo "[$name] ACTIVE at $(date +%H:%M:%S)"; return 0; }
    case "$st" in FAILED) echo "[$name] state=FAILED"; return 1 ;; esac
    sleep 25
  done
  echo "[$name] not ACTIVE within ${LAND_DEADLINE_S}s (state=${st:-gone})"
  return 1
}

# ---------------------------------------------------------- judge supervisor
judge_log="${out_dir}/seam-judge-${SLURM_JOB_ID}.log"
judge_supervisor() {
  trap - EXIT TERM INT HUP
  wait_active "$JUDGE_QR" "$JUDGE_ZONE" >> "$judge_log" 2>&1 || return 1
  local ssh="timeout 900 gcloud compute tpus tpu-vm ssh $JUDGE_QR --zone=$JUDGE_ZONE --project=$PROJECT --worker=0"
  for _ in 1 2 3 4 5 6; do $ssh --command="echo ssh-ok" >> "$judge_log" 2>&1 && break; sleep 20; done
  timeout 900 gcloud compute tpus tpu-vm scp --recurse "${repo_root}/tpu/pallas_arena" \
    "${JUDGE_QR}:~/arena-pallas_arena" --zone="$JUDGE_ZONE" --project="$PROJECT" --worker=0 >> "$judge_log" 2>&1
  $ssh --command="rm -rf ~/arena && mkdir -p ~/arena && mv ~/arena-pallas_arena ~/arena/pallas_arena" >> "$judge_log" 2>&1
  $ssh --command="ARENA_CACHE_PREFIX='${CACHE}' bash ~/arena/pallas_arena/phase2/provision_judge.sh" >> "$judge_log" 2>&1 || {
    echo "[judge] provision failed" >> "$judge_log"; return 1; }
  touch "${work_dir}/judge-ready"
  echo "[judge] provisioned at $(date +%H:%M:%S)" >> "$judge_log"
  local fails=0 t_s dur rc
  while true; do
    t_s=$(date +%s)
    timeout 14400 gcloud compute tpus tpu-vm ssh "$JUDGE_QR" --zone="$JUDGE_ZONE" --project="$PROJECT" --worker=0 \
      --command="cd ~/arena && ARENA_CHILD_JAX_PLATFORMS=cpu ARENA_RLIMIT_GB=48 PYTHONPATH=~/arena \
        ~/arena-venv/bin/python -m pallas_arena.judge.worker \
        --problem '${PROBLEM_ARG}' \
        --queue http://127.0.0.1:${QUEUE_PORT} --sim-mode real --no-adversarial \
        --timing-pairs ${TIMING_PAIRS} --worker-id ${JUDGE_QR} \
        --grade-budget-s ${GRADE_BUDGET_S} --compile-budget-s ${COMPILE_BUDGET_S} \
        --cache '${CACHE}' --boot-report \$HOME/boot-report.json --poll-s 2 \
        2>&1 | tee -a \$HOME/worker-progress.log" \
      -- -R "${QUEUE_PORT}:localhost:${QUEUE_PORT}" -o ExitOnForwardFailure=yes \
         -o ServerAliveInterval=30 -o ServerAliveCountMax=4 >> "$judge_log" 2>&1
    rc=$?; dur=$(( $(date +%s) - t_s ))
    echo "[judge] worker session ended rc=${rc} after ${dur}s $(date +%H:%M:%S)" >> "$judge_log"
    if [ "$dur" -lt 60 ]; then fails=$(( fails + 1 )); else fails=0; fi
    [ "$fails" -ge 4 ] && { echo "[judge] 4 short sessions; giving up" >> "$judge_log"; return 1; }
    sleep 10
  done
}

# The headline metric is "within-group spread ABOVE THE JUDGE'S OWN NOISE
# FLOOR", and the floor lives in the boot report on the judge. Pull it as soon
# as it exists and keep it fresh, so the driver's running snapshots can compute
# the real headline instead of a spread with nothing to compare it to.
bootfetch_supervisor() {
  trap - EXIT TERM INT HUP
  while true; do
    timeout 180 gcloud compute tpus tpu-vm scp "${JUDGE_QR}:~/boot-report.json" "$boot_report" \
      --zone="$JUDGE_ZONE" --project="$PROJECT" --worker=0 >/dev/null 2>&1 \
      && echo "[boot] fetched judge boot report $(date +%H:%M:%S)"
    sleep 120
  done
}

# ---------------------------------------------------------- serve supervisor
serve_log="${out_dir}/seam-serve-${SLURM_JOB_ID}.log"
serve_supervisor() {
  trap - EXIT TERM INT HUP
  wait_active "$SERVE_QR" "$SERVE_ZONE" >> "$serve_log" 2>&1 || { touch "${work_dir}/serve-failed"; return 1; }
  touch "${work_dir}/serve-active"
  local w
  for w in 0 1; do
    for _ in 1 2 3 4 5 6; do
      timeout 900 gcloud compute tpus tpu-vm ssh "$SERVE_QR" --zone="$SERVE_ZONE" --project="$PROJECT" \
        --worker="$w" --command="echo ssh-ok w${w}" >> "$serve_log" 2>&1 && break
      sleep 20
    done
    timeout 900 gcloud compute tpus tpu-vm scp "${repo_root}/tpu/pallas_arena/probe/serve_vllm.sh" \
      "${SERVE_QR}:~/serve_vllm.sh" --zone="$SERVE_ZONE" --project="$PROJECT" --worker="$w" >> "$serve_log" 2>&1
  done
  # both hosts bring up in PARALLEL: the install and the weight pull are the
  # long poles and they are independent
  ( timeout 3600 gcloud compute tpus tpu-vm ssh "$SERVE_QR" --zone="$SERVE_ZONE" --project="$PROJECT" --worker=0 \
      --command="MODEL='${QWEN_MODEL}' PORT=${VLLM_PORT} MAX_MODEL_LEN=${QWEN_MAXLEN} \
        HF_CACHE_GCS='${QWEN_HF_GCS}' XLA_CACHE_GCS='${QWEN_XLA_GCS}' bash ~/serve_vllm.sh" \
      >> "$serve_log" 2>&1; echo "[serve] w0 launcher rc=$?" >> "$serve_log" ) &
  local p0=$!
  ( timeout 3600 gcloud compute tpus tpu-vm ssh "$SERVE_QR" --zone="$SERVE_ZONE" --project="$PROJECT" --worker=1 \
      --command="MODEL='${GEMMA_MODEL}' PORT=${VLLM_PORT} MAX_MODEL_LEN=${GEMMA_MAXLEN} \
        HF_CACHE_GCS='${GEMMA_HF_GCS}' XLA_CACHE_GCS='${GEMMA_XLA_GCS}' bash ~/serve_vllm.sh" \
      >> "$serve_log" 2>&1; echo "[serve] w1 launcher rc=$?" >> "$serve_log" ) &
  local p1=$!
  wait "$p0" "$p1" 2>/dev/null
  touch "${work_dir}/serve-launched"
  echo "[serve] both launchers returned at $(date +%H:%M:%S)" >> "$serve_log"
}

judge_supervisor & judge_pid=$!
serve_supervisor & serve_pid=$!
bootfetch_supervisor & bootfetch_pid=$!

# ------------------------------------------------------ wait for the serving
# slice, which is the one with the 45 min landing rule
echo "=== waiting up to ${LAND_DEADLINE_S}s for ${SERVE_QR} to land $(date +%H:%M:%S) ==="
while [ "$(date +%s)" -lt "$(( t_first_qr + LAND_DEADLINE_S + 60 ))" ]; do
  [ -e "${work_dir}/serve-active" ] && break
  [ -e "${work_dir}/serve-failed" ] && break
  sleep 20
done
if [ ! -e "${work_dir}/serve-active" ]; then
  echo "=== FATAL: ${SERVE_ACCEL} did not land within ${LAND_DEADLINE_S}s. Reporting and stopping."
  echo "=== (no fallback to a larger slice: capacity in ${SERVE_ZONE} is contended)"
  exit 2
fi

# ---------------------------------------------------- tunnels to the servers
echo "=== opening tunnels $(date +%H:%M:%S) ==="
open_tunnel() {  # worker local_port
  local w=$1 lp=$2
  while true; do
    timeout 14400 gcloud compute tpus tpu-vm ssh "$SERVE_QR" --zone="$SERVE_ZONE" --project="$PROJECT" \
      --worker="$w" -- -L "${lp}:localhost:${VLLM_PORT}" -N \
      -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=4 \
      >> "${out_dir}/seam-tunnel-w${w}-${SLURM_JOB_ID}.log" 2>&1
    sleep 5
  done
}
( trap - EXIT TERM INT HUP; open_tunnel 0 "$QWEN_LOCAL_PORT" ) & tunnel_pids+=($!)
( trap - EXIT TERM INT HUP; open_tunnel 1 "$GEMMA_LOCAL_PORT" ) & tunnel_pids+=($!)

# -------------------------------------------------- wait for the two engines
echo "=== waiting up to ${SERVE_READY_S}s for both vLLM engines $(date +%H:%M:%S) ==="
ready_q=0; ready_g=0
end=$(( $(date +%s) + SERVE_READY_S ))
while [ "$(date +%s)" -lt "$end" ]; do
  [ "$ready_q" -eq 0 ] && curl -sf "http://127.0.0.1:${QWEN_LOCAL_PORT}/v1/models" >/dev/null 2>&1 \
    && { ready_q=1; echo "  qwen ready $(date +%H:%M:%S)"; }
  [ "$ready_g" -eq 0 ] && curl -sf "http://127.0.0.1:${GEMMA_LOCAL_PORT}/v1/models" >/dev/null 2>&1 \
    && { ready_g=1; echo "  gemma ready $(date +%H:%M:%S)"; }
  [ "$ready_q" -eq 1 ] && [ "$ready_g" -eq 1 ] && break
  # never let the engine wait eat the hard cap
  [ "$(date +%s)" -gt "$(( t_first_qr + HARD_CAP_S - 2700 ))" ] && { echo "  giving up on stragglers"; break; }
  sleep 20
done
echo "=== engines: qwen=${ready_q} gemma=${ready_g} $(date +%H:%M:%S) ==="

servers=""
[ "$ready_q" -eq 1 ] && servers="qwen35-27b=http://127.0.0.1:${QWEN_LOCAL_PORT}"
[ "$ready_g" -eq 1 ] && servers="${servers:+${servers},}gemma4-31b=http://127.0.0.1:${GEMMA_LOCAL_PORT}"
if [ -z "$servers" ]; then
  echo "FATAL: no vLLM engine came up"; exit 3
fi

# ------------------------------------------------- verify the chat rendering
# The qwen renderer CHANGED this run (thinking disabled). Prove it against the
# server's own template before a single sample is spent.
echo "=== verifying chat rendering $(date +%H:%M:%S) ==="
uv run --isolated --extra dev python tpu/pallas_arena/probe/verify_render.py \
  --servers "$servers" --out "${out_dir}/seam-render-${SLURM_JOB_ID}.json" 2>&1 | tail -40

# -------------------------------------------------------- judge readiness
if [ -e "${work_dir}/judge-ready" ]; then
  echo "=== judge is up; grading enabled ==="
  queue_arg="--queue http://127.0.0.1:${QUEUE_PORT}"
else
  echo "=== judge NOT up; running pre-gate only (gates ast/exec/export still measured) ==="
  queue_arg=""
fi

# --------------------------------------------------------------------- probe
remain=$(( t_first_qr + HARD_CAP_S - $(date +%s) - 900 ))
[ "$remain" -lt 600 ] && remain=600
echo "=== running the seam probe $(date +%H:%M:%S) (wall ${remain}s) ==="
uv run --isolated --extra dev --extra jax --with fastapi --with uvicorn --with flatbuffers \
  python tpu/pallas_arena/probe/probe_driver.py \
  --servers "$servers" $queue_arg \
  --out "${out_dir}/seam-results-${SLURM_JOB_ID}.json" \
  --boot-report "$boot_report" \
  --rounds "$ROUNDS" --group-size "$GROUP_SIZE" --max-new-tokens "$MAX_NEW_TOKENS" \
  --pregate-workers 14 --wall-s "$remain" --round-reserve-s "$ROUND_RESERVE_S"
rc=$?

# ---------------------------------------------------------------- artifacts
echo "=== fetching artifacts $(date +%H:%M:%S) ==="
timeout 300 gcloud compute tpus tpu-vm scp "${JUDGE_QR}:~/boot-report.json" "$boot_report" \
  --zone="$JUDGE_ZONE" --project="$PROJECT" --worker=0 >/dev/null 2>&1 || true
timeout 300 gcloud compute tpus tpu-vm scp "${JUDGE_QR}:~/worker-progress.log" \
  "${out_dir}/seam-worker-progress-${SLURM_JOB_ID}.log" --zone="$JUDGE_ZONE" --project="$PROJECT" --worker=0 >/dev/null 2>&1 || true
for w in 0 1; do
  timeout 300 gcloud compute tpus tpu-vm scp "${SERVE_QR}:~/vllm-probe.log" \
    "${out_dir}/seam-vllm-w${w}-${SLURM_JOB_ID}.log" --zone="$SERVE_ZONE" --project="$PROJECT" --worker="$w" >/dev/null 2>&1 || true
done
curl -sf "http://127.0.0.1:${QUEUE_PORT}/status" > "${out_dir}/seam-queue-status-${SLURM_JOB_ID}.json" 2>/dev/null || true

echo "=== seam probe done rc=${rc} $(date +%H:%M:%S) ==="
exit "$rc"
