#!/bin/bash
# rf3c generation smoke against the ARENA CELL's own vLLM engine (worker 1 of
# the v5p-32) -- no QR of its own, no serving bring-up: the cell is already
# serving for RL, and 64 smoke generations before step 0 cost it nothing.
#
# Submit:  sbatch tpu/pallas_arena/probe/smoke_on_cell.sbatch
#
#SBATCH -p all
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH --job-name=rf3c-smoke
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/pallas_arena/rf3c-smoke-%j.log
set -uo pipefail
[ -z "${SLURM_JOB_ID:-}" ] && { echo "run under sbatch"; exit 1; }

REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
cd "$REPO"
out_dir=$REPO/runs/pallas_arena

TPU_NAME="${TPU_NAME:-sk7524-qwen35arena-v5p32-east5a_spot}"
PROJECT=vision-mix
ZONE=us-east5-a
WORKER="${WORKER:-1}"            # the cell's first vLLM host
REMOTE_PORT="${REMOTE_PORT:-8001}"
LOCAL_PORT=$(( 19000 + SLURM_JOB_ID % 1000 ))
SSH_KEY="${SSH_KEY:-$HOME/.ssh/google_compute_engine}"
GCLOUD=/n/fs/vision-mix/sk7524/google-cloud-sdk/bin/gcloud
CELLS="${SMOKE_CELLS:-rg_lru:rf3c,splash_attention:rf3c}"
GENS_OUT="${out_dir}/rf3c-smoke-gens-${SLURM_JOB_ID}.jsonl"

ip=$("$GCLOUD" alpha compute tpus tpu-vm describe "$TPU_NAME" --project=$PROJECT --zone=$ZONE \
      --format="value(networkEndpoints[$WORKER].accessConfig.externalIp)" 2>/dev/null)
[ -n "$ip" ] || { echo "FATAL: cannot resolve worker $WORKER ip"; exit 1; }

( while true; do
    ssh -i "$SSH_KEY" -o IdentitiesOnly=yes -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
      -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=30 \
      -N -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" "sk7524_princeton_edu@${ip}" || true
    sleep 10
  done ) &
TUN=$!
trap 'kill $TUN 2>/dev/null || true' EXIT

echo "=== waiting for the cell engine (w${WORKER}:${REMOTE_PORT}) ==="
for i in $(seq 1 120); do
  curl -fsS -m5 "http://127.0.0.1:${LOCAL_PORT}/v1/models" >/dev/null 2>&1 && { echo "engine reachable"; break; }
  [ "$i" = 120 ] && { echo "FATAL: engine unreachable after 60min"; exit 1; }
  sleep 30
done

# RL-matched budgets: two-phase think 13824-equivalent is prompt-inclusive in
# RL; the smoke client's caps are think-only -- 10240 think + answer 6144
# approximates the RL answer tail while the serving ctx (18432) bounds totals.
uv run --isolated --extra jax --with fastapi python tpu/pallas_arena/probe/gen_smoke.py \
  --server "http://127.0.0.1:${LOCAL_PORT}" --model "Qwen/Qwen3.5-27B" \
  --out "$GENS_OUT" --group-size "${GROUP_SIZE:-32}" \
  --max-tokens "${MAX_NEW_TOKENS:-10240}" --answer-cap "${ANSWER_CAP:-6144}" \
  --ctx "${CTX:-18432}" --concurrency 6 --cells "$CELLS"
rc=$?
echo "=== generation rc=${rc}; gens at ${GENS_OUT} ==="
exit "$rc"
