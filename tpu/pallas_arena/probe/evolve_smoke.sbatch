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
LOCAL_PORT=18001

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
  for z in "$ZONE" "$OTHER_ZONE"; do
    timeout 300 gcloud compute tpus queued-resources list --zone="$z" --project="$PROJECT" 2>/dev/null \
      | grep -q "^${QR}\b" || echo "[cleanup] verified ${QR} gone from ${z}"
    timeout 300 gcloud compute tpus tpu-vm list --zone="$z" --project="$PROJECT" 2>/dev/null \
      | grep -q "^${QR}\b" || echo "[cleanup] verified ${QR} node gone from ${z}"
  done
}
trap 'cleanup' EXIT
trap 'echo "[trap] TERM/INT/HUP"; cleanup; exit 143' TERM INT HUP

echo "=== creating spot QR ${QR} (${ACCEL}, ${ZONE}) $(date +%H:%M:%S) ==="
timeout 600 gcloud compute tpus queued-resources create "$QR" --node-id="$QR" \
  --accelerator-type="$ACCEL" --runtime-version="$RUNTIME" --spot \
  --zone="$ZONE" --project="$PROJECT" 2>&1 | tail -1
t_qr=$(date +%s)

( trap - EXIT TERM INT HUP; sleep "$HARD_CAP_S"
  echo "[watchdog] HARD CAP ${HARD_CAP_S}s -- tearing down"; kill -TERM $$ ) &

st=""; deadline=$(( t_qr + LAND_DEADLINE_S ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  st=$(timeout 300 gcloud compute tpus queued-resources describe "$QR" --zone="$ZONE" \
         --project="$PROJECT" --format='value(state.state)' 2>/dev/null)
  [ "$st" = "ACTIVE" ] && break
  if [ "$st" = "FAILED" ] || [ -z "$st" ]; then
    # empty state == the QR vanished (failed-and-purged between polls, job
    # 3740157 idled 3h on it); treat exactly like FAILED
    echo "[$QR] state=${st:-gone} at $(date +%H:%M:%S) -- deleting and re-creating"
    timeout 600 gcloud compute tpus queued-resources delete "$QR" \
      --zone="$ZONE" --project="$PROJECT" --force --quiet 2>/dev/null || true
    sleep 60
    timeout 600 gcloud compute tpus queued-resources create "$QR" --node-id="$QR" \
      --accelerator-type="$ACCEL" --runtime-version="$RUNTIME" --spot \
      --zone="$ZONE" --project="$PROJECT" 2>&1 | tail -1
  fi
  sleep 25
done
[ "$st" = "ACTIVE" ] || { echo "FATAL: not ACTIVE (state=${st:-gone})"; exit 1; }
echo "[$QR] ACTIVE at $(date +%H:%M:%S)"

for _ in 1 2 3 4 5 6; do
  timeout 900 gcloud compute tpus tpu-vm ssh "$QR" --zone="$ZONE" --project="$PROJECT" \
    --worker=0 --command="echo ssh-ok" && break
  sleep 20
done
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
      HF_CACHE_GCS='${QWEN_HF_GCS}' XLA_CACHE_GCS='${QWEN_XLA_GCS}' bash ~/serve_vllm.sh" \
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
ready=0; end=$(( $(date +%s) + SERVE_READY_S ))
while [ "$(date +%s)" -lt "$end" ]; do
  curl -sf "http://127.0.0.1:${LOCAL_PORT}/v1/models" >/dev/null 2>&1 && { ready=1; break; }
  sleep 20
done
[ "$ready" -eq 1 ] || { echo "FATAL: engine never became ready"; exit 1; }
echo "=== engine ready $(date +%H:%M:%S) ==="
# "the boot-log line is truth": pull the engine's KV-pool/max_num_seqs boot
# lines to the login-side log NOW -- the serving-facts memory says to verify
# the pool against max_num_seqs from this line, and the VM log dies with the
# VM (two preempted/crashed attempts never surfaced it).
timeout 300 gcloud compute tpus tpu-vm ssh "$QR" --zone="$ZONE" --project="$PROJECT" --worker=0   --command="grep -iE 'kv cache|num_seqs|blocks|memory' ~/vllm-probe.log | head -20" || true

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
  ${REPAIR_FROM:+--repair-from "$REPAIR_FROM" --gens-from "$GENS_FROM"} \
  ${ENABLE_THINKING_KWARG:+--enable-thinking-kwarg}
rc=$?
echo "=== generation rc=${rc}; gens at ${GENS_OUT} ==="
exit "$rc"
