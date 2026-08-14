#!/bin/bash
# SEAM + DIALECT probe. NO RL: generation + grading only.
#
# The prompt ladder measured splash_attention, ragged_paged_attention and
# megablox_gmm at 0 PASS out of 96 candidates EACH, at every rung, all dying at
# `aot_export` on the pallas_call PLUMBING that no whole-program rung supplies.
# The only variant that has ever exported splash is the SEAM (harness owns the
# pallas_call; model writes one strategy function), and the only addition that
# ever cleanly removed a failure class is the P1 DIALECT list. So:
#
#   3 dead kernels x { sd1 = seam+dialect, sd2 = seam+dialect+typed skeleton }
#   + ONE control cell, rg_lru at p1 -- the ladder's best cell (25/32 PASS). If
#     the control does not reproduce, the harness is broken and no negative
#     result in this run means anything.
#   = 7 cells, 2 rounds of 16 = 32 samples per cell, round-robin by round so
#     every cell has the SAME number of complete groups whenever the clock
#     stops.
#
# Two spot QRs, distinctly named, never more than two alive:
#   sk7524-sd-serve  v5p-8  us-east5-a  -- gemma-4-31B-it, vLLM TP=4, 16384
#                                          context against the EXISTING
#                                          vllm-xla-cache-gemma4-31b-16k
#   sk7524-sd-judge  v6e-1  us-east5-b  -- the arena judge, FOUR problems
#                                          (flce is not in this grid)
#
# A v5p-8 is a SINGLE host, so the two-host mesh trap does not apply; the
# standalone-runtime exports in serve_vllm.sh describe exactly the 4 local
# chips and are harmless here.
#
# HARD RESOURCE RULES, in the code rather than in a comment:
#   * both QRs deleted on EVERY exit path -- success, failure, exception,
#     TERM/INT/HUP (slurm timeout, scancel, node drain) and the hard cap.
#     `trap cleanup EXIT` does NOT fire on an untrapped SIGTERM, which is
#     exactly what a slurm timeout sends;
#   * the deletes are issued `setsid nohup` FIRST, because scancel kills the
#     trap ~30 s in while one queued-resource delete takes ~90 s to return --
#     measured on job 3650988, where both QRs survived their own teardown;
#   * a 4 h hard cap measured from the FIRST QR create;
#   * if the v5p-8 has not landed in 45 min we report and STOP;
#   * the end state is verified empty in BOTH zones, for BOTH names.
#
#SBATCH -p all
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH -t 3:20:00
#SBATCH --job-name=repair-probe
#SBATCH --signal=B:TERM@180
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/pallas_arena/repair-probe-%j.log

set -uo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: run under SLURM (sbatch); no compute on the login node." >&2
  exit 1
fi

PROJECT=${PROJECT:-vision-mix}
# job-unique: a fixed name races the previous job's detached teardown delete
# (measured, job 3691912); the cleanup survivor sweep greps the prefix.
SERVE_QR=${SERVE_QR:-sk7524-sd-serve-${SLURM_JOB_ID}}
SERVE_ZONE=${SERVE_ZONE:-us-east5-a}
SERVE_ACCEL=${SERVE_ACCEL:-v5p-8}
SERVE_RUNTIME=${SERVE_RUNTIME:-v2-alpha-tpuv5}
JUDGE_QR=${JUDGE_QR:-sk7524-sd-judge-${SLURM_JOB_ID}}
JUDGE_ZONE=${JUDGE_ZONE:-us-east5-b}
JUDGE_ACCEL=${JUDGE_ACCEL:-v6e-1}
JUDGE_RUNTIME=${JUDGE_RUNTIME:-v2-alpha-tpuv6e}

QUEUE_PORT=${QUEUE_PORT:-8794}
VLLM_PORT=${VLLM_PORT:-8001}
GEMMA_LOCAL_PORT=${GEMMA_LOCAL_PORT:-8113}

LAND_DEADLINE_S=${LAND_DEADLINE_S:-2700}   # 45 min, per the hard rules
HARD_CAP_S=${HARD_CAP_S:-10800}            # 3 h from the first QR create (cap is 4 h)
SERVE_READY_S=${SERVE_READY_S:-3600}
LEASE_TIMEOUT_S=${LEASE_TIMEOUT_S:-600}
GRADE_BUDGET_S=${GRADE_BUDGET_S:-300}
COMPILE_BUDGET_S=${COMPILE_BUDGET_S:-90}
TIMING_PAIRS=${TIMING_PAIRS:-6}
ROUNDS=${ROUNDS:-2}                        # 2 x 16 = the 32 samples per cell
ROUND_RESERVE_S=${ROUND_RESERVE_S:-1200}
GROUP_SIZE=${GROUP_SIZE:-16}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-12000}
MODELS=${MODELS:-gemma4-31b}
# The grid is NOT a cross product: six arm cells plus one control cell on a
# different kernel at a different variant. `--cells` is the only way to say
# that without spending chip time on five cells nobody asked for.
CELLS=${CELLS:-'gemma4-31b|sd1|splash_attention,gemma4-31b|sd2|splash_attention,gemma4-31b|sd1|ragged_paged_attention,gemma4-31b|sd2|ragged_paged_attention,gemma4-31b|sd1|megablox_gmm,gemma4-31b|sd2|megablox_gmm,gemma4-31b|p1|rg_lru'}
CACHE=${CACHE:-gs://sk7524-pallas-arena-us-east5/reward-cache-seamdialect}

GEMMA_MODEL=${GEMMA_MODEL:-google/gemma-4-31B-it}
GEMMA_MAXLEN=${GEMMA_MAXLEN:-16384}        # matches vllm-xla-cache-gemma4-31b-16k
BUCKET=gs://sk7524-tinker-tpu-us-east5
GEMMA_HF_GCS=${GEMMA_HF_GCS:-${BUCKET}/hf-cache-gemma4}
GEMMA_XLA_GCS=${GEMMA_XLA_GCS:-${BUCKET}/vllm-xla-cache-gemma4-31b-16k}
# transformers 5.15.0 cannot load gemma-4-31B (AmbiguousGlobalPerLayerAttributeError:
# 'head_dim' is per-layer). The llama-farm pins 5.14.0 for gemma workers only.
GEMMA_EXTRA_PIP=${GEMMA_EXTRA_PIP:-transformers==5.14.0}

# The FOUR problems this grid needs, on one chip. flce is deliberately absent:
# no cell grades it, and its 151936-wide reference is the most expensive thing
# the judge boots. NO SPACES anywhere in this string: the worker's parser
# strips only the outer whitespace of a spec, so `a : b` becomes the problem
# name `'a '` and a KeyError that kills the whole worker before any problem
# boots. Mirrors probe/configs.py:TASK_CASES exactly.
PROBLEM_ARG=${PROBLEM_ARG:-'splash_attention:probe-h8-s4096,probe-h4-s2048,probe-holdout-h4-s2049;ragged_paged_attention:probe-b16-len1024,probe-b8-len512,probe-holdout-b17-len512;megablox_gmm:probe-m4096-e4-uniform,probe-m2048-e4-zipf,probe-holdout-m3000-e4-zipf;rg_lru:probe-4x2048x2560,probe-2x1024x2560,probe-holdout-2x1500x2560'}

repo_root=/n/fs/vision-mix/sk7524/SkyRLTpu
out_dir="${repo_root}/runs/pallas_arena"
work_dir="${out_dir}/sd-work-${SLURM_JOB_ID}"
boot_report="${out_dir}/sd-judge-boot-${SLURM_JOB_ID}.json"
mkdir -p "$out_dir" "$work_dir"
cd "$repo_root"
export PYTHONPATH="${repo_root}/tpu:${PYTHONPATH:-}"

queue_pid=""; watchdog_pid=""; judge_pid=""; serve_pid=""; bootfetch_pid=""
tunnel_pids=()
t_first_qr=""

# --------------------------------------------------------------- uv, retried
# MEASURED (job 3686851): the grid never ran. Both slices were up, both engines
# healthy, the judge had booted all five problems -- and `uv run --isolated`
# died in 13 seconds resolving `causal-conv1d` from a GitHub release URL:
#   http2 error: stream error received: refused stream ...
# One transient HTTP fetch cost 31 minutes of bring-up on two spot slices and
# produced zero candidates. The same flake had already hit the CPU control job
# once. So: every uv invocation retries, and the driver's environment is BUILT
# AND PROVEN BEFORE A SINGLE QR IS CREATED, where a failure costs nothing.
UV_ENV_ARGS=(--isolated --extra dev --extra jax --with fastapi --with uvicorn --with flatbuffers)
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-120}

uv_retry() {  # uv_retry <label> -- <uv args...>
  local label=$1; shift; [ "${1:-}" = "--" ] && shift
  local n=${UV_RETRIES:-4} i rc
  for i in $(seq 1 "$n"); do
    uv "$@"
    rc=$?
    [ "$rc" -eq 0 ] && return 0
    echo "[uv] ${label}: attempt ${i}/${n} failed rc=${rc}; retrying in $(( 10 * i ))s"
    sleep $(( 10 * i ))
  done
  # OFFLINE FALLBACK. The failing dependency is `causal-conv1d`, pulled from a
  # GitHub *release URL* by the `dev` extra -- nothing this run uses, and a host
  # this run cannot influence (measured: http2 refused-stream, then a sustained
  # 503). The local uv cache already holds it from the environments built
  # earlier in this same job, so resolving from cache is both correct and the
  # only thing that does not depend on github.com being healthy.
  echo "[uv] ${label}: all ${n} attempts failed; retrying OFFLINE from the local uv cache"
  UV_OFFLINE=1 uv "$@"
  rc=$?
  [ "$rc" -eq 0 ] && { echo "[uv] ${label}: OK offline"; return 0; }
  echo "[uv] ${label}: FAILED after ${n} attempts and an offline retry"
  return 1
}

# ------------------------------------------------------------------ cleanup
cleaned=""
cleanup() {
  [ -n "$cleaned" ] && return 0
  cleaned=1
  echo "[cleanup] $(date +%H:%M:%S) tearing down (always-delete, both QRs)"
  for p in "${tunnel_pids[@]:-}" "$judge_pid" "$serve_pid" "$watchdog_pid" "$queue_pid" "$bootfetch_pid"; do
    [ -n "${p:-}" ] && kill "$p" 2>/dev/null
  done
  pkill -P $$ 2>/dev/null
  pkill -f "tpu-vm ssh ${SERVE_QR}" 2>/dev/null
  pkill -f "tpu-vm ssh ${JUDGE_QR}" 2>/dev/null
  sleep 1
  # DETACHED deletes FIRST -- they must outlive this shell being killed.
  setsid nohup timeout 900 gcloud compute tpus queued-resources delete "$SERVE_QR" \
    --zone="$SERVE_ZONE" --project="$PROJECT" --force --quiet \
    > "${out_dir}/sd-del-serve-${SLURM_JOB_ID}.log" 2>&1 < /dev/null &
  setsid nohup timeout 900 gcloud compute tpus queued-resources delete "$JUDGE_QR" \
    --zone="$JUDGE_ZONE" --project="$PROJECT" --force --quiet \
    > "${out_dir}/sd-del-judge-${SLURM_JOB_ID}.log" 2>&1 < /dev/null &
  echo "[cleanup] detached deletes issued for $SERVE_QR and $JUDGE_QR"
  del_pids=()
  ( timeout 600 gcloud compute tpus queued-resources delete "$SERVE_QR" --zone="$SERVE_ZONE" \
      --project="$PROJECT" --force --quiet >/dev/null 2>&1; echo "[cleanup] delete confirmed: $SERVE_QR" ) &
  del_pids+=($!)
  ( timeout 600 gcloud compute tpus queued-resources delete "$JUDGE_QR" --zone="$JUDGE_ZONE" \
      --project="$PROJECT" --force --quiet >/dev/null 2>&1; echo "[cleanup] delete confirmed: $JUDGE_QR" ) &
  del_pids+=($!)
  # NEVER a bare `wait`: the queue server is a background child that by design
  # never exits. Always wait on explicit PIDs.
  wait "${del_pids[@]}" 2>/dev/null

  # "Issued" is not "gone". The only acceptable end state is an empty list of
  # BOTH names in BOTH zones -- and nodes too, because a QR delete is supposed
  # to take its node with it and "supposed to" is not "did".
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
        || echo "[cleanup] verified empty of sd QRs in $z"
    else
      echo "[cleanup] verified empty of sd QRs in $z"
    fi
    nodes=$(timeout 300 gcloud compute tpus tpu-vm list --zone="$z" --project="$PROJECT" \
              --format='value(name)' 2>/dev/null | grep -E "^(${SERVE_QR}|${JUDGE_QR})$" || true)
    if [ -n "$nodes" ]; then
      echo "[cleanup] ORPHAN NODES in $z: $nodes -- deleting"
      for n in $nodes; do
        timeout 600 gcloud compute tpus tpu-vm delete "$n" --zone="$z" --project="$PROJECT" --quiet >/dev/null 2>&1
      done
    else
      echo "[cleanup] verified empty of sd nodes in $z"
    fi
  done
  [ -n "$t_first_qr" ] && echo "[cleanup] QR lifetime $(( $(date +%s) - t_first_qr ))s"
}
trap cleanup EXIT
trap 'echo "[trap] caught TERM/INT/HUP"; cleanup; exit 143' TERM INT HUP

# ------------------------------------------------------- PRE-WARM (no chips)
# Build and PROVE the exact environment the grid will run in, and import the
# prompt/ladder modules, before anything is provisioned. A dependency-resolution
# flake, a syntax error in a prompt module or a missing name is then a 60-second
# failure at zero chip cost instead of a dead 31-minute bring-up.
echo "=== pre-warming the driver environment $(date +%H:%M:%S) ==="
uv_retry prewarm -- run "${UV_ENV_ARGS[@]}" python -c "
import sys; sys.path.insert(0, '${repo_root}/tpu')
from pallas_arena.probe.prompt_seam_dialect import SEAM_DIALECT_PROMPTS, RUNGS
from pallas_arena.probe.prompts import get_prompt
from pallas_arena.probe.seam import SEAMS, compose, extract_fill
from pallas_arena.probe import configs as C
cfgs = C.parse_cells('''${CELLS}''')
assert len(cfgs) == 7, cfgs
for c in cfgs:
    p = get_prompt(c.task, c.variant)
    assert len(p) > 4000, (c.name, len(p))
    print('[prewarm]', c.name, len(p), 'chars ~', len(p)//3, 'tok(pessimistic)')
# the compose path the SEAM cells depend on, end to end, with a decoy block
seam = SEAMS['megablox_gmm']
resp = '\`\`\`python\ndef choose_tiles(*a):\n    raise NotImplementedError\n\`\`\`\n\n\`\`\`python\ndef choose_tiles(m,k,n,g):\n    return 512,1024,512\n\ndef gmm_tile(l,r,o,*,bm,bk,bn,k):\n    o[...] = 0.0\n\`\`\`'
fill, how, missing = extract_fill(resp, seam.required)
assert not missing and 'NotImplementedError' not in fill, (how, missing)
assert 'def kernel(' in compose('megablox_gmm', fill)
print('[prewarm] OK', len(cfgs), 'cells; extract+compose green (extraction=' + how + ')')
" || { echo "=== FATAL: could not build the driver environment; no QR was created. ==="; exit 6; }

# -------------------------------------------------------------------- queue
echo "=== sd: queue on $(hostname):${QUEUE_PORT} $(date +%H:%M:%S) ==="
uv run "${UV_ENV_ARGS[@]}" \
  python -m pallas_arena.judge.queue --port "$QUEUE_PORT" --host 127.0.0.1 \
  --lease-timeout "$LEASE_TIMEOUT_S" > "${out_dir}/sd-queue-${SLURM_JOB_ID}.log" 2>&1 &
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
judge_log="${out_dir}/sd-judge-${SLURM_JOB_ID}.log"
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
    timeout 18000 gcloud compute tpus tpu-vm ssh "$JUDGE_QR" --zone="$JUDGE_ZONE" --project="$PROJECT" --worker=0 \
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
# FLOOR", and the floor lives in the boot report on the judge.
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
serve_log="${out_dir}/sd-serve-${SLURM_JOB_ID}.log"
serve_supervisor() {
  trap - EXIT TERM INT HUP
  wait_active "$SERVE_QR" "$SERVE_ZONE" >> "$serve_log" 2>&1 || { touch "${work_dir}/serve-failed"; return 1; }
  touch "${work_dir}/serve-active"
  for _ in 1 2 3 4 5 6; do
    timeout 900 gcloud compute tpus tpu-vm ssh "$SERVE_QR" --zone="$SERVE_ZONE" --project="$PROJECT" \
      --worker=0 --command="echo ssh-ok" >> "$serve_log" 2>&1 && break
    sleep 20
  done
  timeout 900 gcloud compute tpus tpu-vm scp "${repo_root}/tpu/pallas_arena/probe/serve_vllm.sh" \
    "${SERVE_QR}:~/serve_vllm.sh" --zone="$SERVE_ZONE" --project="$PROJECT" --worker=0 >> "$serve_log" 2>&1
  timeout 3600 gcloud compute tpus tpu-vm ssh "$SERVE_QR" --zone="$SERVE_ZONE" --project="$PROJECT" --worker=0 \
    --command="MODEL='${GEMMA_MODEL}' PORT=${VLLM_PORT} MAX_MODEL_LEN=${GEMMA_MAXLEN} \
      MAX_NUM_SEQS=${GROUP_SIZE} EXTRA_PIP='${GEMMA_EXTRA_PIP}' \
      HF_CACHE_GCS='${GEMMA_HF_GCS}' XLA_CACHE_GCS='${GEMMA_XLA_GCS}' bash ~/serve_vllm.sh" \
    >> "$serve_log" 2>&1
  touch "${work_dir}/serve-launched"
  echo "[serve] launcher returned rc=$? at $(date +%H:%M:%S)" >> "$serve_log"
}

judge_supervisor & judge_pid=$!
serve_supervisor & serve_pid=$!
bootfetch_supervisor & bootfetch_pid=$!

# ------------------------------------------------------ wait for the serving
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

# ----------------------------------------------------- tunnel to the server
echo "=== opening tunnel $(date +%H:%M:%S) ==="
open_tunnel() {
  while true; do
    timeout 18000 gcloud compute tpus tpu-vm ssh "$SERVE_QR" --zone="$SERVE_ZONE" --project="$PROJECT" \
      --worker=0 -- -L "${GEMMA_LOCAL_PORT}:localhost:${VLLM_PORT}" -N \
      -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=4 \
      >> "${out_dir}/sd-tunnel-${SLURM_JOB_ID}.log" 2>&1
    sleep 5
  done
}
( trap - EXIT TERM INT HUP; open_tunnel ) & tunnel_pids+=($!)

# --------------------------------------------------------- wait for vLLM
echo "=== waiting up to ${SERVE_READY_S}s for the vLLM engine $(date +%H:%M:%S) ==="
ready_g=0
end=$(( $(date +%s) + SERVE_READY_S ))
while [ "$(date +%s)" -lt "$end" ]; do
  curl -sf "http://127.0.0.1:${GEMMA_LOCAL_PORT}/v1/models" >/dev/null 2>&1 \
    && { ready_g=1; echo "  gemma ready $(date +%H:%M:%S)"; break; }
  [ "$(date +%s)" -gt "$(( t_first_qr + HARD_CAP_S - 5400 ))" ] && { echo "  giving up waiting"; break; }
  sleep 20
done
if [ "$ready_g" -ne 1 ]; then echo "FATAL: the vLLM engine never came up"; exit 3; fi
servers="gemma4-31b=http://127.0.0.1:${GEMMA_LOCAL_PORT}"

# ------------------------------------------------- verify the chat rendering
echo "=== verifying chat rendering $(date +%H:%M:%S) ==="
uv_retry render -- run --isolated --extra dev python tpu/pallas_arena/probe/verify_render.py \
  --servers "$servers" --out "${out_dir}/sd-render-${SLURM_JOB_ID}.json" 2>&1 | tail -30

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
half=$(( remain / 2 ))
echo "=== repair probe: IMPROVE mode $(date +%H:%M:%S) (wall ${half}s) ==="
uv_retry improver -- run "${UV_ENV_ARGS[@]}" \
  python tpu/pallas_arena/probe/repair_driver.py \
  --servers "$servers" $queue_arg --model gemma4-31b \
  --mode improve --seeds tpu/pallas_arena/probe/repair_seeds.json \
  --cells "splash_attention:sd1;ragged_paged_attention:sd2;megablox_gmm:sd2;rg_lru:p1" \
  --candidates "${IMPROVE_CANDIDATES:-4}" --turns "${TURNS:-3}" --parallel-chains 4 \
  --out "${out_dir}/repair-improve-${SLURM_JOB_ID}.jsonl" \
  --pregate-workers 10 --wall-s "$half"
rc_improve=$?
remain2=$(( t_first_qr + HARD_CAP_S - $(date +%s) - 900 ))
if [ "$remain2" -gt 900 ]; then
echo "=== repair probe: COLD mode on the weak cells $(date +%H:%M:%S) (wall ${remain2}s) ==="
uv_retry colder -- run "${UV_ENV_ARGS[@]}" \
  python tpu/pallas_arena/probe/repair_driver.py \
  --servers "$servers" $queue_arg --model gemma4-31b \
  --mode cold \
  --cells "splash_attention:sd1;ragged_paged_attention:sd2" \
  --candidates "${COLD_CANDIDATES:-8}" --turns "${TURNS:-3}" --parallel-chains 4 \
  --out "${out_dir}/repair-cold-${SLURM_JOB_ID}.jsonl" \
  --pregate-workers 10 --wall-s "$remain2"
rc_cold=$?
else
  echo "=== skipping cold mode: only ${remain2}s left ==="
  rc_cold=0
fi
rc=$(( rc_improve > rc_cold ? rc_improve : rc_cold ))

# ---------------------------------------------------------------- artifacts
echo "=== fetching artifacts $(date +%H:%M:%S) ==="
timeout 300 gcloud compute tpus tpu-vm scp "${JUDGE_QR}:~/boot-report.json" "$boot_report" \
  --zone="$JUDGE_ZONE" --project="$PROJECT" --worker=0 >/dev/null 2>&1 || true
timeout 300 gcloud compute tpus tpu-vm scp "${JUDGE_QR}:~/worker-progress.log" \
  "${out_dir}/sd-worker-progress-${SLURM_JOB_ID}.log" --zone="$JUDGE_ZONE" --project="$PROJECT" --worker=0 >/dev/null 2>&1 || true
timeout 300 gcloud compute tpus tpu-vm scp "${SERVE_QR}:~/vllm-probe.log" \
  "${out_dir}/sd-vllm-${SLURM_JOB_ID}.log" --zone="$SERVE_ZONE" --project="$PROJECT" --worker=0 >/dev/null 2>&1 || true
curl -sf "http://127.0.0.1:${QUEUE_PORT}/status" > "${out_dir}/sd-queue-status-${SLURM_JOB_ID}.json" 2>/dev/null || true

echo "=== seam+dialect probe done rc=${rc} $(date +%H:%M:%S) ==="
exit "$rc"
