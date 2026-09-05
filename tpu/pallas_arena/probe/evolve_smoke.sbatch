#!/bin/bash
#SBATCH -p all
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 6:00:00
#SBATCH --job-name=evolve-smoke
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/pallas_arena/evolve-smoke-%j.log

# rf3 prompt smoke: ONE v5p-8 spot QR serving qwen3.5-27B, generate
# GROUP_SIZE completions per (task, variant) cell at the RL sampling
# distribution, dump raw JSONL to NFS. Grading is OFFLINE on CPU
# (grade_smoke.py) -- this job never touches a judge chip.
#
# Lifecycle discipline inherited from baselines_tpu.sbatch: job-unique QR
# name, retry-on-FAILED inside the landing window, hard-cap watchdog,
# teardown verified in both zones on every exit path.

set -uo pipefail
[ -z "${SLURM_JOB_ID:-}" ] && { echo "run under sbatch"; exit 1; }

repo_root=/n/fs/vision-mix/sk7524/SkyRLTpu
cd "$repo_root"
out_dir=${repo_root}/runs/pallas_arena

PROJECT=vision-mix
ZONE=${ZONE:-us-east5-a}
OTHER_ZONE=${OTHER_ZONE:-us-east5-b}
ACCEL=${ACCEL:-v5p-8}
RUNTIME=${RUNTIME:-v2-alpha-tpuv5}
LAND_DEADLINE_S=${LAND_DEADLINE_S:-10800}
HARD_CAP_S=${HARD_CAP_S:-14400}
SERVE_READY_S=${SERVE_READY_S:-5400}   # fresh 32k compile on first boot
GROUP_SIZE=${GROUP_SIZE:-32}
# Two-phase: MAX_NEW_TOKENS is the PHASE-1 (thinking) cap; when it
# truncates, the client forces the answer with ANSWER_CAP more tokens
# (rq2 pattern). Single-phase caps measured 53-97% truncation at any size.
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-10240}
# 15360: 45% of splash answers still clipped at 12288; thinking saturated
# 13824 with null quality return (muse study), so the budget moved
# think->answer. 2.7k prompt + 10.2k think + 15.4k answer ~= 28.3k < 32k;
# gen_smoke sizes phase 2 from actual usage so bigger prompts self-clamp.
ANSWER_CAP=${ANSWER_CAP:-15360}
# RESERVE-FIRST BUDGET (ttt_discover TwoPhaseTokenCompleter semantics). When
# set, this is the total context phase 1 may occupy INCLUDING the prompt, so
# the answer is GUARANTEED ctx - PHASE1_TOTAL - 64 tokens however much the
# model thinks. Measured need: with a fixed think cap and a leftover answer,
# 7 of 32 qwen rg_lru candidates finished normally having produced only
# reasoning and no program (2026-08-27).
PHASE1_TOTAL=${PHASE1_TOTAL:-0}
# THINK_BUDGET is the preferred knob: tokens of THINKING allowed, with the
# answer guaranteed ctx - prompt - THINK_BUDGET - 64. It self-adjusts to the
# measured prompt, which differs by model and task (qwen 3067/5823 tokens for
# rg_lru/splash, gemma 3262/6416). 12288 was chosen from behaviour: qwen
# SATURATED a 10240 cap in 18 of 32 completions while gemma's whole
# completions ran 6769-9062 tokens, and it still leaves every cell an answer
# of 2.5x-7.6x its seed program.
THINK_BUDGET=${THINK_BUDGET:-0}
QWEN_MODEL=${QWEN_MODEL:-Qwen/Qwen3.5-27B}
# 32k context, like the llamafarm serving: thinking stays ON and the
# completion budget absorbs it instead of truncating. First boot at 32k
# compiles fresh (the 22k cache cannot match) and saves back via
# serve_vllm.sh; later boots restore.
QWEN_MAXLEN=${QWEN_MAXLEN:-32768}
BUCKET=gs://sk7524-tinker-tpu-us-east5
QWEN_HF_GCS=${QWEN_HF_GCS:-${BUCKET}/hf-cache}
# cache prefix encodes the FULL config (RQ2 convention: model-ctx-bN),
# plus hardware: XLA caches are keyed by topology too, so a v6e-8 b64
# cache and this v5p-8 b16 cache can never be shared. The bare -32k
# name would have collided with RQ2\x27s (never-populated) -32k-b64.
QWEN_XLA_GCS=${QWEN_XLA_GCS:-${BUCKET}/vllm-xla-cache-qwen35-32k-b16-v5p8}
VLLM_PORT=8001
# Job-unique: two smoke jobs coscheduled on ONE slurm node collided on a
# shared 18001 -- the second job's readiness probe answered through the
# FIRST job's tunnel (measured: gemma 'ready' 17s after ACTIVE, 192
# instant 404s against the wrong engine).
LOCAL_PORT=$(( 15000 + SLURM_JOB_ID % 20000 ))

QR="sk7524-evsmoke-${SLURM_JOB_ID}"
GENS_OUT="${out_dir}/evolve-smoke-gens-${SLURM_JOB_ID}.jsonl"

cleanup() {
  # engine post-mortem BEFORE the VM disappears: three attempts have died
  # with the serve log unread because it only lived on the VM
  timeout 240 gcloud compute tpus tpu-vm scp "${QR}:~/vllm-probe.log" \
    "${out_dir}/evolve-smoke-engine-${SLURM_JOB_ID}.log" \
    --zone="$ZONE" --project="$PROJECT" --worker=0 2>/dev/null \
    && echo "[cleanup] engine log captured" || echo "[cleanup] engine log unreachable"
  echo "[cleanup] $(date +%H:%M:%S) tearing down ${QR}"
  timeout 600 gcloud compute tpus queued-resources delete "$QR" \
    --zone="$ZONE" --project="$PROJECT" --force --quiet 2>/dev/null || true
  ( timeout 600 gcloud compute tpus queued-resources delete "$QR" --zone="$ZONE" \
      --project="$PROJECT" --force --quiet 2>/dev/null || true ) &
  # VERIFY FOR REAL, AND RETRY. This block used to print "verified gone"
  # whenever the grep failed to match -- which is also what happens when the
  # list call itself fails -- so a QR that survived teardown was reported as
  # removed. Three FAILED evsmoke QRs accumulated that way on 2026-08-27 and
  # silently consumed the v6e-8 quota until new creates stopped being
  # granted, which stalled both splash arms for over an hour with no error
  # anywhere in their logs.
  for z in "$ZONE" "$OTHER_ZONE"; do
    for attempt in 1 2 3; do
      lst=$(timeout 300 gcloud compute tpus queued-resources list --zone="$z" \
              --project="$PROJECT" --format='value(name)' 2>/dev/null)
      rc=$?
      if [ $rc -ne 0 ]; then
        echo "[cleanup] WARNING: could not list ${z} (rc=$rc) -- cannot verify ${QR}"
        break
      fi
      if ! echo "$lst" | grep -qx "$QR"; then
        echo "[cleanup] verified ${QR} gone from ${z}"
        break
      fi
      echo "[cleanup] ${QR} STILL PRESENT in ${z} (attempt ${attempt}/3) -- deleting again"
      timeout 600 gcloud compute tpus queued-resources delete "$QR" --zone="$z" \
        --project="$PROJECT" --force --quiet 2>/dev/null || true
      sleep 30
      [ "$attempt" = "3" ] && echo "[cleanup] ALARM: ${QR} survived teardown in ${z} -- it will consume quota"
    done
    nodes=$(timeout 300 gcloud compute tpus tpu-vm list --zone="$z" --project="$PROJECT" \
              --format='value(name)' 2>/dev/null)
    echo "$nodes" | grep -qx "$QR" \
      && echo "[cleanup] ALARM: node ${QR} still in ${z}" \
      || echo "[cleanup] verified ${QR} node gone from ${z}"
  done
}
trap 'cleanup' EXIT
trap 'echo "[trap] TERM/INT/HUP"; cleanup; exit 143' TERM INT HUP

echo "=== creating spot QR ${QR} (${ACCEL}, ${ZONE}) $(date +%H:%M:%S) ==="
# `| tail -1` used to be all that was logged, and gcloud's failures are
# multi-line JSON, so the log recorded a bare "}" and nothing else. Four p2
# arms then spun for 50 minutes re-creating a QR that could never exist,
# because the one line that said why was thrown away:
#   code 7: "User does not have permission to submit requests into this queue
#            for accelerator type v6e-8 in location us-east5-a"
# Keep the whole message, and make a permission/quota refusal TERMINAL for
# this zone -- retrying it is guaranteed to fail again, and the supervisor's
# next attempt rotates to a zone where the request is allowed.
create_qr() {
  local out rc
  out=$(timeout 600 gcloud compute tpus queued-resources create "$QR" --node-id="$QR" \
        --accelerator-type="$ACCEL" --runtime-version="$RUNTIME" --spot \
        --zone="$ZONE" --project="$PROJECT" 2>&1); rc=$?
  echo "$out" | tail -12
  if echo "$out" | grep -qE "does not have permission|PERMISSION_DENIED|\"code\": 7"; then
    echo "FATAL: ${ACCEL} is not permitted in ${ZONE} -- not retrying this zone"
    return 2
  fi
  return $rc
}
create_qr; create_rc=$?
[ "$create_rc" -eq 2 ] && exit 2
t_qr=$(date +%s)

( trap - EXIT TERM INT HUP; sleep "$HARD_CAP_S"
  echo "[watchdog] HARD CAP ${HARD_CAP_S}s -- tearing down"; kill -TERM $$ ) &

st=""; deadline=$(( t_qr + LAND_DEADLINE_S )); misses=0
# HEALTHY STATES ARE NOT MISSES, AND ONE FAILED READ IS NOT A DEATH.
# An empty describe usually means the API call failed, not that the QR
# vanished: gcloud prints "network connectivity issues" for transient errors
# AND for 429 rate-limiting. Deleting on a single empty read turned all four
# arms into a delete/re-create loop on 2026-08-27 -- every ~2.3 min, which
# guarantees the QR never lands and adds API load that provokes more 429s.
# Same debounce the judge fleet's supervisor v3 already uses: three
# consecutive misses AND a confirming list call before destroying anything.
while [ "$(date +%s)" -lt "$deadline" ]; do
  st=$(timeout 300 gcloud compute tpus queued-resources describe "$QR" --zone="$ZONE" \
         --project="$PROJECT" --format='value(state.state)' 2>/dev/null)
  [ "$st" = "ACTIVE" ] && break
  case "$st" in
    WAITING_FOR_RESOURCES|PROVISIONING|ACCEPTED|CREATING|RESUMING|WAITING)
      misses=0; sleep 25; continue ;;
  esac
  if [ "$st" = "FAILED" ] || [ "$st" = "SUSPENDED" ] || [ -z "$st" ]; then
    misses=$(( misses + 1 ))
    echo "[$QR] state=${st:-unreadable} miss ${misses}/3 at $(date +%H:%M:%S)"
    if [ "$misses" -lt 3 ]; then sleep 45; continue; fi
    # Confirm with an independent LIST before destroying: if the QR is listed,
    # the describes were lying and we keep waiting.
    if timeout 300 gcloud compute tpus queued-resources list --zone="$ZONE" \
         --project="$PROJECT" --format='value(name)' 2>/dev/null | grep -qx "$QR"; then
      echo "[$QR] list still shows it -- describe was unreliable; not deleting"
      misses=0; sleep 60; continue
    fi
    echo "[$QR] confirmed gone at $(date +%H:%M:%S) -- re-creating"
    timeout 600 gcloud compute tpus queued-resources delete "$QR" \
      --zone="$ZONE" --project="$PROJECT" --force --quiet 2>/dev/null || true
    sleep 60
    create_qr; create_rc=$?
    [ "$create_rc" -eq 2 ] && exit 2
    # ONLY A SUCCESSFUL CREATE EARNS A FRESH CLOCK. Resetting the window after
    # a FAILED create makes the deadline unreachable: the p2 arms re-created
    # every ~3.5 min against a zone that refused them, reset the 10800s window
    # each time, and so could never time out and let the supervisor rotate
    # zones. They span for 50 minutes and would have span until the hard cap.
    if [ "$create_rc" -ne 0 ]; then
      echo "[$QR] create FAILED (rc=$create_rc); leaving the landing window running"
      misses=0
      sleep 60
      continue
    fi
    misses=0
    # A FRESH REQUEST DESERVES A FRESH CLOCK. The deadline was absolute from
    # the FIRST create, so a re-create late in the window had no time to land
    # and died seconds later: measured 2026-08-27, a splash arm re-created at
    # 09:07 and hit "FATAL: not ACTIVE (state=gone)" at 09:13. The hard-cap
    # watchdog still bounds total runtime.
    deadline=$(( $(date +%s) + LAND_DEADLINE_S ))
    echo "[$QR] re-created; landing window reset to ${LAND_DEADLINE_S}s"
  fi
  sleep 25
done
# DON'T ABANDON A SLICE THAT IS ACTIVELY BEING ALLOCATED. PROVISIONING means
# the request was granted and the node is being built -- minutes away -- while
# WAITING_FOR_RESOURCES means we are still in the queue. Aborting on the
# former throws away a slice about to land: measured 2026-08-27, both splash
# arms hit the deadline mid-PROVISIONING and were torn down. Grant a bounded
# grace only for the in-flight states.
grace=0
while [ "$st" != "ACTIVE" ] && [ "$grace" -lt "${LAND_GRACE_TRIES:-6}" ]; do
  case "$st" in
    PROVISIONING|CREATING|ACCEPTED|RESUMING) : ;;
    *) break ;;
  esac
  grace=$(( grace + 1 ))
  echo "[$QR] deadline reached but state=${st} (allocating) -- grace ${grace}/${LAND_GRACE_TRIES:-6}"
  sleep 300
  st=$(timeout 300 gcloud compute tpus queued-resources describe "$QR" --zone="$ZONE" \
         --project="$PROJECT" --format='value(state.state)' 2>/dev/null)
done
[ "$st" = "ACTIVE" ] || { echo "FATAL: not ACTIVE (state=${st:-gone})"; exit 1; }
echo "[$QR] ACTIVE at $(date +%H:%M:%S)"

for _ in 1 2 3 4 5 6; do
  timeout 900 gcloud compute tpus tpu-vm ssh "$QR" --zone="$ZONE" --project="$PROJECT" \
    --worker=0 --command="echo ssh-ok" && break
  sleep 20
done
# DOES THIS NODE ACTUALLY HAVE TPUs?
#
# Count the accelerator PCI functions. NOT /dev/accel*: an earlier version of
# this check tested that path and was WRONG. v6e exposes chips through VFIO
# (/dev/vfio/N, driver vfio_pci), so /dev/accel* is absent on every HEALTHY
# v6e node -- verified 2026-08-29 against a v6e-8 that had been serving
# continuously for three days:
#     ls /dev/accel*  -> No such file or directory
#     ls /dev/vfio    -> 0 1 2 3 4 ...
#     lspci           -> "Processing accelerators: Google, Inc. Device 006f" x N
# That bad check ran for ~6h and rejected healthy slices as dead, which is
# strictly worse than the 90-minute boot it was meant to save. lspci is used
# here because it reports the hardware itself rather than whichever driver
# convention this TPU generation happens to use.
#
# Exit non-zero so the supervisor resubmits onto a different node.
n_tpu=0
for _try in 1 2; do
  n_tpu=$(timeout 300 gcloud compute tpus tpu-vm ssh "$QR" --zone="$ZONE" --project="$PROJECT" \
            --worker=0 --command="lspci 2>/dev/null | grep -ci 'processing accelerators'" 2>/dev/null | tr -dc '0-9')
  [ "${n_tpu:-0}" -ge 1 ] && break
  echo "[$QR] tpu probe ${_try}/2 saw '${n_tpu:-<empty>}' accelerator PCI functions"
  sleep 20
done
if [ "${n_tpu:-0}" -lt 1 ]; then
  echo "WARNING: ${QR} reports no accelerator PCI functions; continuing anyway"
  echo "         (the probe is advisory -- a false negative must never discard a good slice)"
else
  echo "[$QR] ${n_tpu} accelerator PCI function(s) present"
fi

timeout 900 gcloud compute tpus tpu-vm scp "${repo_root}/tpu/pallas_arena/probe/serve_vllm.sh" \
  "${QR}:~/serve_vllm.sh" --zone="$ZONE" --project="$PROJECT" --worker=0

# Engine config must MATCH THE XLA CACHE exactly: attempt 3 set
# MAX_NUM_SEQS=6 and the engine never became ready in 3600s -- the 22k cache
# was compiled at max_num_seqs=16, so the changed batch config forced a full
# TPU recompile. Keep the engine at its proven cached config; the crash
# mitigation moves ENTIRELY client-side (concurrency 6): if the engine still
# dies with only 6 requests in flight, concurrency was never the killer.
( timeout 7200 gcloud compute tpus tpu-vm ssh "$QR" --zone="$ZONE" --project="$PROJECT" --worker=0 \
    --command="MODEL='${QWEN_MODEL}' PORT=${VLLM_PORT} MAX_MODEL_LEN=${QWEN_MAXLEN} \
      HF_CACHE_GCS='${QWEN_HF_GCS}' XLA_CACHE_GCS='${QWEN_XLA_GCS}' \
      ${SERVE_VISIBLE_CHIPS:+SERVE_VISIBLE_CHIPS='${SERVE_VISIBLE_CHIPS}'} \
      ${SERVE_TP:+SERVE_TP='${SERVE_TP}'} \
      ${EXTRA_PIP:+EXTRA_PIP='${EXTRA_PIP}'} bash ~/serve_vllm.sh" \
    ; echo "[serve] launcher rc=$?" ) &

( trap - EXIT TERM INT HUP
  while true; do
    timeout 14400 gcloud compute tpus tpu-vm ssh "$QR" --zone="$ZONE" --project="$PROJECT" \
      --worker=0 -- -L "${LOCAL_PORT}:localhost:${VLLM_PORT}" -N \
      -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=4 \
      >> "${out_dir}/evolve-smoke-tunnel-${SLURM_JOB_ID}.log" 2>&1
    sleep 5
  done ) &

echo "=== waiting up to ${SERVE_READY_S}s for the engine $(date +%H:%M:%S) ==="
ready=0; end=$(( $(date +%s) + SERVE_READY_S )); hb=0
while [ "$(date +%s)" -lt "$end" ]; do
  curl -sf "http://127.0.0.1:${LOCAL_PORT}/v1/models" >/dev/null 2>&1 && { ready=1; break; }
  # DON'T WAIT OUT A DEAD SLICE. A preempted QR can never serve, and this
  # loop would sit on it for the full SERVE_READY_S (measured 2026-08-27:
  # gemma's slice was reclaimed mid-boot and the job would have burned two
  # hours waiting on an engine that no longer had a machine). Check slice
  # health every ~5 min and bail immediately so the supervisor re-rolls.
  hb=$(( hb + 1 ))
  if [ $(( hb % 15 )) -eq 0 ]; then
    qst=$(timeout 120 gcloud compute tpus queued-resources describe "$QR" --zone="$ZONE" \
            --project="$PROJECT" --format='value(state.state)' 2>/dev/null)
    case "$qst" in
      PREEMPTED|SUSPENDED|SUSPENDING|FAILED)
        echo "FATAL: slice went ${qst} while waiting for the engine"; exit 1 ;;
    esac
  fi
  sleep 20
done
[ "$ready" -eq 1 ] || { echo "FATAL: engine never became ready"; exit 1; }
echo "=== engine ready $(date +%H:%M:%S) ==="
# "the boot-log line is truth": pull the engine's KV-pool/max_num_seqs boot
# lines to the login-side log NOW -- the serving-facts memory says to verify
# the pool against max_num_seqs from this line, and the VM log dies with the
# VM (two preempted/crashed attempts never surfaced it).
timeout 300 gcloud compute tpus tpu-vm ssh "$QR" --zone="$ZONE" --project="$PROJECT" --worker=0   --command="grep -iE 'kv cache|num_seqs|blocks|memory' ~/vllm-probe.log | head -20" || true

# SLICE HEALTH DURING GENERATION. The engine wait checks slice state, but
# generation had no such check: when the slice was preempted mid-run the
# client just burned its connection retries and wrote an error row per
# sample. Measured 2026-08-27 on the first clean qwen cell -- 6 rows, all
# 'Connection refused', while ssh to the VM had been timing out for minutes
# and the QR was already SUSPENDED. A cell of error rows is worse than no
# cell: it looks like data. Kill the job instead so the supervisor re-rolls.
( trap - EXIT TERM INT HUP
  while true; do
    sleep 120
    qs=$(timeout 120 gcloud compute tpus queued-resources describe "$QR" --zone="$ZONE" \
           --project="$PROJECT" --format='value(state.state)' 2>/dev/null)
    case "$qs" in
      ACTIVE|"") : ;;
      *) echo "[watchdog] slice went ${qs} DURING generation -- aborting so the supervisor re-rolls"
         kill -TERM $$ ; exit 0 ;;
    esac
  done ) &

# Engine-log snapshots to NFS while generating: 3 of the last 4 VM deaths
# struck as SPLASH requests began, and every post-mortem was blind because
# the engine log dies with the VM. A 5-min snapshot loop leaves the crash
# trace on NFS.
( while true; do
    timeout 120 gcloud compute tpus tpu-vm ssh "$QR" --zone="$ZONE" --project="$PROJECT" \
      --worker=0 --command="tail -c 200000 ~/vllm-probe.log" \
      > "/n/fs/vision-mix/sk7524/SkyRLTpu/runs/pallas_arena/engine-snap-${SLURM_JOB_ID}.log" 2>/dev/null || true
    sleep 300
  done ) &
# No trap here: the EXIT trap at the top owns teardown (re-trapping EXIT
# would REPLACE it and leak the QR). Slurm reaps this loop at job end.
ENGINE_SNAP_PID=$!

uv run --isolated --extra jax --with fastapi python tpu/pallas_arena/probe/gen_smoke.py \
  --server "http://127.0.0.1:${LOCAL_PORT}" --model "$QWEN_MODEL" \
  --out "$GENS_OUT" --group-size "$GROUP_SIZE" --max-tokens "$MAX_NEW_TOKENS" --concurrency 6 \
  --answer-cap "$ANSWER_CAP" --cells "${SMOKE_CELLS:-}" --ctx "$QWEN_MAXLEN" \
  ${PHASE1_TOTAL:+--phase1-total "$PHASE1_TOTAL"} \
  ${THINK_BUDGET:+--think-budget "$THINK_BUDGET"} \
  ${REPAIR_FROM:+--repair-from "$REPAIR_FROM" --gens-from "$GENS_FROM"} \
  ${ENABLE_THINKING_KWARG:+--enable-thinking-kwarg} \
  ${NO_THINK:+--no-think} \
  ${SEED_FILE:+--seed-file "$SEED_FILE"} \
  ${SEED_OBS:+--seed-observation "$SEED_OBS"} \
  ${LIB_IMPORTS:+--lib-imports} \
  ${SEED_REWARD:+--seed-reward "$SEED_REWARD"}
rc=$?
echo "=== generation rc=${rc}; gens at ${GENS_OUT} ==="
exit "$rc"
