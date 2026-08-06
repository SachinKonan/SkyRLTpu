#!/bin/bash
# T2 farm smoke/run from a compute node: open SSH tunnel to the TPU's vLLM, wait for health
# (first XLA compile can take ~55 min), then sample.
#   sbatch t2_smoke.sh <tpu_name> <model_name> <problem> <out_dir> [n] [concurrency] [cell]
#SBATCH --job-name=rq1_t2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --exclude=neu301
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu-rq1/runs/rq1/logs/%x_%j.log
set -uo pipefail
TPU_NAME="$1"; MODEL="$2"; PROBLEM="$3"; OUT="$4"; N="${5:-5}"; CONC="${6:-4}"; CELL="${7:-C}"
RQ1=/n/fs/vision-mix/sk7524/SkyRLTpu-rq1/tpu/rq1
PORT=$((18000 + RANDOM % 1000))

echo "[t2smoke] tunnel localhost:$PORT -> $TPU_NAME worker1:8001"
gcloud alpha compute tpus tpu-vm ssh "sk7524_princeton_edu@$TPU_NAME" \
  --project=vision-mix --zone=us-east5-a --worker=1 \
  --ssh-key-file="$HOME/.ssh/jobman_tpu_ed25519" \
  -- -L "$PORT:localhost:8001" -N -o ServerAliveInterval=60 -o ExitOnForwardFailure=yes &
TUN=$!
trap 'kill $TUN 2>/dev/null' EXIT

echo "[t2smoke] waiting for vLLM health (up to 75 min: model load + XLA compile)..."
for i in $(seq 1 150); do
  if curl -sf -m 10 "http://127.0.0.1:$PORT/v1/models" > /tmp/models_$$.json 2>/dev/null; then
    echo "[t2smoke] vLLM healthy after ~$((i*30))s:"; cat /tmp/models_$$.json; echo
    break
  fi
  kill -0 $TUN 2>/dev/null || { echo "[t2smoke] tunnel died; TPU unreachable (preempted?)" >&2; exit 1; }
  [ "$i" -eq 150 ] && { echo "[t2smoke] vLLM never became healthy" >&2; exit 1; }
  sleep 30
done

cd "$RQ1/client"
uv run collect_t2.py --problem "$PROBLEM" --n "$N" --concurrency "$CONC" \
  --farm-url "http://127.0.0.1:$PORT" --model "$MODEL" --out "$OUT" --cell "$CELL" --resume
