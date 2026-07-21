#!/bin/bash
# ttt-discover Erdos client on ONE neuronic compute node, pointed at the
# self-hosted SkyRL Tinker server (Qwen3.5-27B) on the qwen35 v5p-16 TPU
# (PR #4 infrastructure) instead of the hosted Thinking Machines API.
#
# The job opens its own SSH tunnel to TPU worker 0 (the Tinker API host) and
# keeps it alive for the whole run; the external IP is re-resolved on every
# reconnect, so a spot preemption+recreate only needs the server restarted
# (tpu/start_colocated_vllm_tinker.sh), not this job -- until then requests
# fail and discover's per-step retries ride it out or the job dies and is
# resubmitted.
#
# Arms (objective A/B, mirroring the gpt-oss 16x32 objective runs):
#   sbatch --export=ALL,EXPERIMENT_NAME=erdos-qwen35-obj-ttt,TTD_RUN_DIR=...,TTD_ADV_ESTIMATOR=entropic_adaptive_beta tpu/run_ttd_qwen35_neuronic.sbatch
#   sbatch --export=ALL,EXPERIMENT_NAME=erdos-qwen35-obj-grpo,TTD_RUN_DIR=...,TTD_ADV_ESTIMATOR=mean_baseline tpu/run_ttd_qwen35_neuronic.sbatch
#
#SBATCH -p all
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH -t 3-00:00:00
#SBATCH -J ttd-qwen35
#SBATCH -o /n/fs/vision-mix/sk7524/SkyRLTpu/runs/ttd_qwen35/slurm_%j.out
#SBATCH -e /n/fs/vision-mix/sk7524/SkyRLTpu/runs/ttd_qwen35/slurm_%j.out
set -euo pipefail

REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
mkdir -p "${REPO}/runs/ttd_qwen35"

# --- TPU tinker server (see tpu/runs/qwen35-27b.env for the server side) ---
TPU_NAME="${TPU_NAME:-sk7524-tunix-qwen35-v5p16-r1-east5a_spot}"
PROJECT="${PROJECT:-vision-mix}"
ZONE="${ZONE:-us-east5-a}"
TPU_REMOTE_USER="${TPU_REMOTE_USER:-sk7524_princeton_edu}"
TPU_SSH_KEY="${TPU_SSH_KEY:-$HOME/.ssh/google_compute_engine}"
LOCAL_PORT="${LOCAL_PORT:-18040}"
GCLOUD="${GCLOUD:-/n/fs/vision-mix/sk7524/google-cloud-sdk/bin/gcloud}"

resolve_tpu_ip() {
  # TPU_IP pins the address when gcloud auth is unavailable (expired token on
  # the cluster). Note: a spot preemption+recreate changes the IP, so a pinned
  # IP job cannot self-heal across recreation — restore gcloud auth for that.
  if [[ -n "${TPU_IP:-}" ]]; then
    echo "$TPU_IP"
    return 0
  fi
  "$GCLOUD" alpha compute tpus tpu-vm describe "$TPU_NAME" \
    --project="$PROJECT" --zone="$ZONE" \
    --format="value(networkEndpoints[0].accessConfig.externalIp)" 2>/dev/null
}

# Self-healing tunnel: re-resolve the IP on every (re)connect.
tunnel_loop() {
  while true; do
    ip="$(resolve_tpu_ip || true)"
    if [[ -z "$ip" ]]; then
      echo "[tunnel] could not resolve ${TPU_NAME} external IP; retrying in 60s" >&2
      sleep 60
      continue
    fi
    echo "[tunnel] connecting 127.0.0.1:${LOCAL_PORT} -> ${ip}:8000" >&2
    ssh -i "$TPU_SSH_KEY" \
      -o IdentitiesOnly=yes -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
      -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -o ConnectTimeout=30 \
      -N -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:8000" \
      "${TPU_REMOTE_USER}@${ip}" || true
    echo "[tunnel] dropped; reconnecting in 15s" >&2
    sleep 15
  done
}
tunnel_loop &
TUNNEL_PID=$!
trap 'kill "$TUNNEL_PID" 2>/dev/null || true' EXIT

# Wait for the server through the tunnel before starting the client.
for i in $(seq 1 60); do
  if curl -fsS -m 5 "http://127.0.0.1:${LOCAL_PORT}/api/v1/get_server_capabilities" >/dev/null 2>&1; then
    echo "[tunnel] tinker server reachable"
    break
  fi
  [[ "$i" == 60 ]] && { echo "[tunnel] server unreachable after 5min" >&2; exit 1; }
  sleep 5
done

CORES="${SLURM_CPUS_PER_TASK:-${TTD_GRADE_CORES:-64}}"
echo "[neuronic] node=$(hostname) job=${SLURM_JOB_ID:-none} cores=${CORES} arm=${TTD_ADV_ESTIMATOR:-entropic_adaptive_beta}"

# --- client -> self-hosted server ---
export TINKER_BASE_URL="http://127.0.0.1:${LOCAL_PORT}"
export TINKER_API_KEY="${TINKER_API_KEY:-tml-dummy}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-27B}"
export RENDERER_NAME="${RENDERER_NAME:-qwen3}"

# --- discover knobs (defaults mirror the gpt-oss objective A/B: 16x32, top-2
# elite, no distill, 30 steps) ---
export GROUP_SIZE="${GROUP_SIZE:-16}"
export GROUPS_PER_BATCH="${GROUPS_PER_BATCH:-32}"
export NUM_EPOCHS="${NUM_EPOCHS:-30}"
export TTD_ELITE_SLOTS="${TTD_ELITE_SLOTS:-2}"
export SAVE_EVERY="${SAVE_EVERY:-10}"
export CONTEXT_WINDOW="${CONTEXT_WINDOW:-32768}"
# Qwen path ignores PHASE1_MAX_TOKENS (no two-phase). Generation is capped so
# prompt+completion stays under the trainer's REAL ceiling: the MaxText 27B fb
# graph on one v5p-8 host trains 24576-token sequences fresh off boot
# (probe_train_len.py), but same-boot churn (loaded program arenas) eats ~28G
# of HBM, after which the ~24k-bucket arena (55.3G) no longer fits (54G free
# observed at the first real train step). 12288 keeps the worst-case bucket
# (~17.5k, arena ~40G) inside the churned budget. Median qwen generation is
# ~3.6k; only the tail clips.
export COMPLETION_MAX_TOKENS="${COMPLETION_MAX_TOKENS:-12288}"
# KL-to-base at the canonical 0.1: compute_logprobs is served natively on the
# TRAINER chips (api routes prompt_logprobs sample requests to the engine;
# tunix backend scores them with an on-device gather). The vLLM path never
# sees prompt_logprobs (it kills that backend's EngineCore).
export KL_PENALTY_COEF="${KL_PENALTY_COEF:-0.1}"
export TTD_ADV_ESTIMATOR="${TTD_ADV_ESTIMATOR:-entropic_adaptive_beta}"

export TTD_EVAL_BACKEND=local
export TTD_SAFE_GRADE_MAX_WORKERS="${TTD_SAFE_GRADE_MAX_WORKERS:-${CORES}}"
export NUM_CPUS_PER_TASK="${NUM_CPUS_PER_TASK:-1}"
export EVAL_TIMEOUT="${EVAL_TIMEOUT:-1100}"
export TTD_DISCOVER_SYNC="${TTD_DISCOVER_SYNC:-0}"

bash "${REPO}/tpu/run_ttd_gptoss20b.sh"
