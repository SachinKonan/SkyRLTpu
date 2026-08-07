#!/bin/bash
# T2 farm smoke/run from a compute node: open SSH tunnel to the TPU's vLLM, wait for health
# (first XLA compile can take ~55 min), then sample.
#   sbatch t2_smoke.sh <tpu_name> <model_name> <problem> <out_dir> [n] [concurrency] [cell] [extra_body_json]
# gemma4 thinking REQUIRES extra_body_json='{"chat_template_kwargs": {"enable_thinking": true}}'
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
EXTRA_BODY="${8:-}"
RQ1=/n/fs/vision-mix/sk7524/SkyRLTpu-rq1/tpu/rq1
PORT=$((18000 + RANDOM % 1000))

echo "[t2smoke] tunnel localhost:$PORT -> $TPU_NAME worker1:8001"
# Self-healing tunnel. gcloud's ssh wrapper drops with rc=255 on transient faults and its
# internal retry leaves the forward DOWN for longer than a client retry window -- measured:
# health check passed, then 5/5 requests died with "All connection attempts failed" while the
# TPU itself stayed ACTIVE. So supervise it and reconnect immediately, for the whole run.
tunnel_loop() {
  while :; do
    gcloud alpha compute tpus tpu-vm ssh "sk7524_princeton_edu@$TPU_NAME" \
      --project=vision-mix --zone=us-east5-a --worker=1 \
      --ssh-key-file="$HOME/.ssh/jobman_tpu_ed25519" \
      -- -L "$PORT:localhost:8001" -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
         -o ExitOnForwardFailure=yes -o StrictHostKeyChecking=no >/dev/null 2>&1
    echo "[t2smoke] tunnel dropped (rc=$?); reconnecting in 10s"
    sleep 10
  done
}
tunnel_loop &
TUN=$!
# Kill the loop AND its live gcloud child -- but never the process group ($$ included), which
# would signal this script and turn a clean 200/200 collection into a FAILED job state.
cleanup() { RC=$?; pkill -P "$TUN" 2>/dev/null; kill "$TUN" 2>/dev/null; exit $RC; }
trap cleanup EXIT

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
EXTRA_ARGS=()
[ -n "$EXTRA_BODY" ] && EXTRA_ARGS=(--extra-body "$EXTRA_BODY")
uv run collect_t2.py --problem "$PROBLEM" --n "$N" --concurrency "$CONC" \
  --farm-url "http://127.0.0.1:$PORT" --model "$MODEL" --out "$OUT" --cell "$CELL" --resume \
  "${EXTRA_ARGS[@]}"
