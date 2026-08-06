#!/usr/bin/env bash
# RQ1 farm bring-up: ONE v5p-16 spot slice serving vLLM ONLY (no tinker/trainer) for
# high-volume sampling. Control-plane orchestration only (gcloud + ssh) -- run from a tmux on
# the login node or any host with gcloud auth + the jobman ssh key.
#
#   ./farm_up.sh qwen35   # Qwen/Qwen3.5-27B  on sk7524-tunix-qwen35-v5p16-r1-east5a_spot
#   ./farm_up.sh gemma4   # google/gemma-4-31B-it on sk7524-tunix-gemma4-v5p16-r1-east5a_spot
#
# Reuses the PROVEN per-model env files + colocated launcher from the MAIN checkout, with
# sampling overrides: START_TINKER=0 (worker 0 idles -- proven config, zero new topology risk),
# 32k context, thinking left to the model's own chat template (we bypass tinker renderers).
# Access: SSH tunnel only, e.g.
#   gcloud alpha compute tpus tpu-vm ssh sk7524_princeton_edu@<TPU_NAME> --project=vision-mix \
#     --zone=us-east5-a --worker=1 --ssh-key-file=$HOME/.ssh/jobman_tpu_ed25519 \
#     -- -L 18001:localhost:8001 -N     # then --farm-url http://127.0.0.1:18001
# On preemption: just re-run this script (QR ensure is idempotent); collect_t2 --resume
# tolerates the gap.
set -euo pipefail

MAIN=/n/fs/vision-mix/sk7524/SkyRLTpu
ZONE=us-east5-a
PROJECT=vision-mix

case "${1:?usage: farm_up.sh qwen35|gemma4}" in
  qwen35) ENVF="$MAIN/tpu/runs/qwen35-27b.env" ;;
  gemma4) ENVF="$MAIN/tpu/runs/gemma4-31b.env" ;;
  *) echo "unknown model $1" >&2; exit 2 ;;
esac

set -a; source "$ENVF"; set +a
YAML="$MAIN/$JOBMAN_YAML"
[[ -f "$YAML" ]] || { echo "missing $YAML" >&2; exit 2; }

state() {
  gcloud compute tpus queued-resources describe "$TPU_NAME" \
    --zone=$ZONE --project=$PROJECT --format='value(state.state)' 2>/dev/null || true
}

st=$(state)
echo "[farm] $TPU_NAME state=${st:-ABSENT}"
if [[ "$st" == "SUSPENDED" || "$st" == "FAILED" ]]; then
  echo "[farm] deleting dead QR..."
  gcloud compute tpus queued-resources delete "$TPU_NAME" --zone=$ZONE --project=$PROJECT --force --quiet
  while [[ -n "$(state)" ]]; do sleep 20; done
fi
# jobman notes (all earned today): the CLI has no __main__ guard and its .venv never had the
# package installed, so the click group must be called directly from source; and
# `jobman create` launches its worker as `jobman run <id>` inside a detached tmux -- which
# dies instantly for the same missing-entrypoint reason. So: register the job with create
# (or reuse an existing snapshot for this TPU), then run the worker OURSELVES in the
# foreground -- it requests the QR, waits for capacity, and does full node setup (ssh,
# gcsfuse, code bundle) before returning.
JM="$MAIN/third_party/jobman"
jobman_cli() { (cd "$JM" && PYTHONPATH=. .venv/bin/python -c "from jobman.cli import cli; cli()" "$@"); }

jid=$(grep -l "name: ${TPU_NAME}" "$JM"/jobs/sk7524/*/config.yaml 2>/dev/null | tail -1 \
      | sed -E 's|.*/([0-9]+)/config.yaml|\1|')
if [[ -z "$jid" ]]; then
  echo "[farm] jobman create $YAML"
  jid=$(jobman_cli create "$YAML" 2>&1 | tee /dev/stderr | grep -oE 'Created job [0-9]+' | grep -oE '[0-9]+' | head -1)
  [[ -n "$jid" ]] || { echo "[farm] create failed" >&2; exit 1; }
fi
tmux kill-session -t "job_${jid}" 2>/dev/null || true   # the broken detached worker, if any
echo "[farm] jobman run $jid (foreground: QR request + wait + node setup)"
jobman_cli run "$jid"

st=$(state)
echo "[farm] after jobman run: state=$st"
[[ "$st" == "ACTIVE" ]] || { echo "[farm] QR not ACTIVE after jobman run" >&2; exit 1; }

echo "[farm] bring-up: vLLM only, 32k ctx"
START_TINKER=0 START_VLLM=1 \
VLLM_MAX_MODEL_LEN=32768 VLLM_MAX_NUM_SEQS=64 VLLM_MAX_CONCURRENT_REQUESTS=64 \
  "$MAIN/tpu/start_colocated_vllm_tinker.sh"

cat <<EOF
[farm] $1 up. Tunnel from the machine running collect_t2:
  gcloud alpha compute tpus tpu-vm ssh sk7524_princeton_edu@$TPU_NAME --project=$PROJECT \\
    --zone=$ZONE --worker=1 --ssh-key-file=\$HOME/.ssh/jobman_tpu_ed25519 -- -L 18001:localhost:8001 -N
  probe:  curl -s http://127.0.0.1:18001/v1/models
  sample: uv run collect_t2.py --problem fc46 --n 200 --farm-url http://127.0.0.1:18001 \\
            --model $MODEL_NAME --out ../../runs/rq1/fc46_C --resume
EOF
