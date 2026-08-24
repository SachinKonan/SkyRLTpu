#!/bin/bash
#SBATCH -p all
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 8:00:00
#SBATCH --job-name=gemma-len-probe
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/pallas_arena/gemma-len-probe-%j.log

# Can gemma-4-31B train ABOVE its 10240 ceiling?
#
# WHY THIS EXISTS: 10240 was never measured as gemma's limit -- it was chosen
# because Erdos generations were short and everything fit (cell_worker.sh:75).
# The kernel arena needs improvement turns, whose prompts carry a parent
# program: measured 6-6.5k prompt + ~4.4k answer does NOT fit 10240 with any
# thinking at all, so the arm is blocked on this number being a leftover
# rather than a wall.
#
# METHOD: one v5p-8 spot slice = the trainer host shape (the qwen ceiling came
# off the same shape). Per candidate (UNIFORM:BUDGET), boot ONLY the tinker
# server -- TUNIX_UNIFORM_SEQ_LEN pins one compiled shape, so each candidate
# needs its own boot -- then fb+optim_step TWICE at that length (cold, then
# warm: the warm pass is what demoted qwen from 24576 to 12288).
#
# Tiling is NOT a variable here: FLCE 1024 / vocab 32 are gemma's validated
# values and the tile trap (smaller tiles shrink fb but GROW the scorer, ~100G
# and uncompilable at 512) means retuning them is its own project.
#
# Submit:  sbatch tpu/probe_gemma_train_len.sbatch
#          CANDIDATES="16384:16384 12288:49152" sbatch tpu/probe_gemma_train_len.sbatch

set -uo pipefail
[ -z "${SLURM_JOB_ID:-}" ] && { echo "run under sbatch"; exit 1; }

repo_root=/n/fs/vision-mix/sk7524/SkyRLTpu
cd "$repo_root"
out_dir=${repo_root}/runs/pallas_arena

PROJECT=vision-mix
ZONE=${ZONE:-us-east5-a}
OTHER_ZONE=${OTHER_ZONE:-us-east5-b}
# v5p-16, not v5p-8: start_colocated_vllm_tinker.sh resolves a vLLM
# COORDINATOR ip unconditionally (line ~383) even with START_VLLM=0, so a
# single-host slice dies with 'TPU worker 1 is out of range for 1
# endpoints' before anything is measured (job 3748201). The trainer still
# runs alone on worker 0 -- same per-chip HBM, same single-host mesh as
# production -- worker 1 just idles. Patching the shared bring-up script
# mid-league-campaign is the riskier fix.
ACCEL=${ACCEL:-v5p-16}
RUNTIME=${RUNTIME:-v2-alpha-tpuv5}
LAND_DEADLINE_S=${LAND_DEADLINE_S:-10800}
HARD_CAP_S=${HARD_CAP_S:-25200}
TINKER_READY_S=${TINKER_READY_S:-4500}

# "UNIFORM:BUDGET" pairs, most valuable first. 16384 = gemma's SERVING length
# (the natural target: generation never exceeds what the trainer can hold).
# The 1x-budget entry isolates pure sequence length; the 4x entry is the
# production shape gemma runs today (40960 = 4 x 10240).
CANDIDATES=${CANDIDATES:-"16384:16384 16384:65536 12288:49152"}

MODEL_NAME=${MODEL_NAME:-google/gemma-4-31B-it}
MAXTEXT_MODEL=${MAXTEXT_MODEL:-gemma4-31b}
FLCE_TILE=${FLCE_TILE:-1024}
VOCAB_TILING=${VOCAB_TILING:-32}
HF_GCS=${HF_GCS:-gs://sk7524-tinker-tpu-us-east5/hf-cache-gemma4}
JAX_CACHE_GCS=${JAX_CACHE_GCS:-gs://sk7524-tinker-tpu-us-east5/jax-compile-cache-gemma4-lenprobe}
# TRAINER ONLY: no vLLM. The shared bring-up used to wait for vLLM whenever
# START_TINKER=1 -- fixed here in main (start_colocated ~line 502) so the wait
# is gated on START_VLLM alone. This matters for survival, not just tidiness:
# with serving in the loop, bring-up ran ~90min and spot preemption killed the
# VM mid-boot (job 3748521, VM gone, its IP reassigned to a stranger's machine
# -> host-key error). Trainer-only roughly halves the exposure window. It also
# drops the whole serving failure surface (transformers pin, XLA cache, engine
# compile), none of which this measurement needs.
# VLLM_WORKERS=1 is still set: worker-1's ip is resolved unconditionally at
# bring-up (~line 383), which is why the slice is v5p-16 rather than v5p-8.
MAXTEXT_PIP=${MAXTEXT_PIP:-"maxtext @ git+https://github.com/SachinKonan/maxtext.git@skyrl/qwen35-dense"}

QR="sk7524-gemmalen-${SLURM_JOB_ID}"
RESULTS="${out_dir}/gemma-len-probe-results-${SLURM_JOB_ID}.txt"

cleanup() {
  timeout 240 gcloud compute tpus tpu-vm scp "${QR}:~/skyrl-logs/tinker-api.log" \
    "${out_dir}/gemma-len-probe-tinker-${SLURM_JOB_ID}.log" \
    --zone="$ZONE" --project="$PROJECT" --worker=0 2>/dev/null \
    && echo "[cleanup] tinker log captured" || echo "[cleanup] tinker log unreachable"
  echo "[cleanup] $(date +%H:%M:%S) tearing down ${QR}"
  timeout 600 gcloud compute tpus queued-resources delete "$QR" \
    --zone="$ZONE" --project="$PROJECT" --force --quiet 2>/dev/null || true
  for z in "$ZONE" "$OTHER_ZONE"; do
    timeout 300 gcloud compute tpus queued-resources list --zone="$z" --project="$PROJECT" 2>/dev/null \
      | grep -q "^${QR}\b" || echo "[cleanup] verified ${QR} gone from ${z}"
  done
  echo "=== RESULTS ==="; cat "$RESULTS" 2>/dev/null || echo "(none)"
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
    echo "[$QR] state=${st:-gone} -- deleting and re-creating"
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

# The trainer launcher calls bare `uv`, but the tmux shell it runs under does
# not source the profile that puts ~/.local/bin on PATH -- "uv: command not
# found" (job 3748560). The vLLM bootstrap used to paper over this by
# installing uv system-wide, so dropping serving from the probe exposed it.
timeout 300 gcloud compute tpus tpu-vm ssh "$QR" --zone="$ZONE" --project="$PROJECT" --worker=0 \
  --command="sudo -u sk7524_princeton_edu bash -lc 'command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh)' >/dev/null 2>&1; \
             sudo ln -sf /home/sk7524_princeton_edu/.local/bin/uv /usr/local/bin/uv; \
             sudo -u sk7524_princeton_edu bash -c 'command -v uv'" 2>&1 | tail -2

: > "$RESULTS"
SYNC=1
for cand in $CANDIDATES; do
  UNIFORM=${cand%%:*}; BUDGET=${cand##*:}
  echo "=== candidate uniform=${UNIFORM} budget=${BUDGET} $(date +%H:%M:%S) ==="

  # Fresh trainer per candidate: the compiled shape is pinned at boot.
  timeout 300 gcloud compute tpus tpu-vm ssh "$QR" --zone="$ZONE" --project="$PROJECT" --worker=0 \
    --command='tmux kill-session -t skyrl-tinker 2>/dev/null || true; tmux list-sessions -F "#{session_name}" 2>/dev/null | awk "/^skyrl-tinker-worker-/ {print}" | xargs -r -n1 tmux kill-session -t; pkill -TERM -u "$USER" -f "[s]kyrl\.tinker|[s]kyrl\.backends\.jax" || true; sleep 5; pkill -KILL -u "$USER" -f "[s]kyrl\.tinker|[s]kyrl\.backends\.jax" || true; true' 2>/dev/null || true

  env TPU_NAME="$QR" PROJECT="$PROJECT" ZONE="$ZONE" \
    TINKER_BACKEND=tunix TRAIN_WORKERS=0 \
    MODEL_NAME="$MODEL_NAME" TUNIX_MAXTEXT_MODEL_NAME="$MAXTEXT_MODEL" \
    TUNIX_MAXTEXT_KWARGS="{\"num_vocab_tiling\": ${VOCAB_TILING}}" \
    TUNIX_MAX_TARGET_LENGTH="$UNIFORM" TUNIX_TRAIN_TOKEN_BUDGET="$BUDGET" \
    TUNIX_FLCE_TILE_SIZE="$FLCE_TILE" TRAIN_MICRO_BATCH_SIZE=1 \
    TUNIX_UNIFORM_SEQ_LEN="$UNIFORM" TUNIX_SEQ_BUCKETS="4096,8192,12288,16384,20480" \
    TUNIX_MINIMAL_FB_OUTPUT=1 SKYRL_SCORE_FIXED_LEN="$UNIFORM" \
    HF_CACHE_GCS="$HF_GCS" JAX_CACHE_GCS="$JAX_CACHE_GCS" \
    TUNIX_MAXTEXT_PIP_SPEC="$MAXTEXT_PIP" \
    VLLM_WORKERS=1 \
    READY_ATTEMPTS=900 SYNC_SKYRL="$SYNC" START_VLLM=0 START_TINKER=1 \
    bash "${repo_root}/tpu/start_colocated_vllm_tinker.sh" \
      > "${out_dir}/gemma-len-bringup-${SLURM_JOB_ID}-${UNIFORM}-${BUDGET}.log" 2>&1 \
    || echo "[warn] bring-up returned nonzero (health check decides)"
  SYNC=0

  ready=0; gone=0; end=$(( $(date +%s) + TINKER_READY_S )); poll=0
  while [ "$(date +%s)" -lt "$end" ]; do
    if timeout 60 gcloud compute tpus tpu-vm ssh "$QR" --zone="$ZONE" --project="$PROJECT" --worker=0 \
         --command="curl -fsS -m8 http://127.0.0.1:8000/api/v1/get_server_capabilities >/dev/null" 2>/dev/null; then
      ready=1; break
    fi
    # Every ~5min, check the SLICE is still alive. Spot preemption during a
    # long bring-up otherwise burns the entire ready window and then reports
    # the same "NEVER READY" as a real OOM -- and the next candidate boots
    # against a corpse (job 3748521).
    poll=$(( poll + 1 ))
    if [ $(( poll % 10 )) -eq 0 ]; then
      if ! timeout 120 gcloud compute tpus tpu-vm list --zone="$ZONE" --project="$PROJECT" 2>/dev/null | grep -q "^${QR}\b"; then
        echo "[preempt] slice vanished during bring-up"; gone=1; break
      fi
    fi
    sleep 30
  done
  if [ "$gone" -eq 1 ]; then
    echo "uniform=${UNIFORM} budget=${BUDGET}: SLICE PREEMPTED during bring-up (no measurement)" | tee -a "$RESULTS"
    break
  fi
  if [ "$ready" -ne 1 ]; then
    echo "uniform=${UNIFORM} budget=${BUDGET}: TRAINER NEVER READY -- see bringup/tinker tails below" | tee -a "$RESULTS"
    # BOTH logs: "never ready" conflates boot-time OOM, compile failure, and
    # plain infrastructure faults, and the bring-up log is where the last one
    # shows up (job 3748201 lost an hour to a single-host worker-list error
    # that the tinker log never saw).
    echo "--- bringup tail ---" | tee -a "$RESULTS"
    tail -12 "${out_dir}/gemma-len-bringup-${SLURM_JOB_ID}-${UNIFORM}-${BUDGET}.log" 2>/dev/null | tee -a "$RESULTS"
    echo "--- tinker tail ---" | tee -a "$RESULTS"
    timeout 120 gcloud compute tpus tpu-vm ssh "$QR" --zone="$ZONE" --project="$PROJECT" --worker=0 \
      --command="tail -40 ~/skyrl-logs/tinker-api.log 2>/dev/null" 2>/dev/null | tail -20 | tee -a "$RESULTS"
    continue
  fi
  echo "trainer ready $(date +%H:%M:%S); probing"

  timeout 900 gcloud compute tpus tpu-vm scp "${repo_root}/tpu/probe_train_len_model.py" \
    "${QR}:~/probe_train_len_model.py" --zone="$ZONE" --project="$PROJECT" --worker=0

  out=$(timeout 2400 gcloud compute tpus tpu-vm ssh "$QR" --zone="$ZONE" --project="$PROJECT" --worker=0 \
    --command="cd ~/SkyRLTpu && TINKER_BASE_URL=http://127.0.0.1:8000 PROBE_BASE_MODEL='${MODEL_NAME}' \
      uv run --extra tpu --extra tinker python ~/probe_train_len_model.py ${UNIFORM} 2>&1 | tail -12" 2>&1)
  echo "$out"
  echo "uniform=${UNIFORM} budget=${BUDGET}: $(echo "$out" | grep -E 'PROBE-RESULTS-JSON|cold |warm ' | tr '\n' ' ')" | tee -a "$RESULTS"

  # A failed 1x-budget run means the same length with a BIGGER budget is dead
  # too -- skip it rather than burn a 20-minute boot proving the obvious.
  if echo "$out" | grep -q '"usable": false' && [ "$UNIFORM" = "$BUDGET" ]; then
    echo "[skip] ${UNIFORM} unusable at 1x budget; larger budgets at this length cannot help"
    CANDIDATES=$(echo "$CANDIDATES" | tr ' ' '\n' | grep -v "^${UNIFORM}:" | tr '\n' ' ')
  fi
done

echo "=== probe complete $(date +%H:%M:%S) ==="
