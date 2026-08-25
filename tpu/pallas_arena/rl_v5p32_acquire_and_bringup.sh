#!/usr/bin/env bash
# Acquire the arena RL v5p-32 through the project spot-quota crunch, prep it,
# and run the cell bring-up. One chain, login-run, background-safe.
#
# WHY NOT JOBMAN FOR THE RETRY: `jobman create` reports "finished
# successfully" even when the allocation 429s on quota
# (TPUV5PPreemptiblePerProjectPerZoneForTPUAPI: 1536 chips/zone, lab-shared),
# so a retry needs to parse its logs anyway. gcloud QR create is retried
# directly; the prep afterwards replicates the yaml's prepare block (apt, uv)
# on ALL FOUR workers -- start_colocated's remote scripts assume uv exists.
set -uo pipefail

REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
cd "$REPO"

# Single instance only (same lesson as the gemma probe: two supervisors
# fighting over one slice is worse than none).
LOCK=/tmp/arena-v5p32-lifecycle.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another lifecycle instance holds $LOCK; exiting"
  exit 0
fi

QR="${QR:-sk7524-qwen35arena-v5p32-east5a_spot}"
PROJECT="${PROJECT:-vision-mix}"
ZONE="${ZONE:-us-east5-a}"
RETRY_S="${RETRY_S:-480}"
GCLOUD="${GCLOUD:-/n/fs/vision-mix/sk7524/google-cloud-sdk/bin/gcloud}"

SMOKE_ON_UP="${SMOKE_ON_UP:-1}"

# OUTER LIFECYCLE LOOP: spot preemption killed the first cell ~40min into
# bring-up. Acquire -> prep -> bring up -> watch the slice; when it dies,
# re-acquire and do it all again. Every successful bring-up (re)submits the
# rf3c smoke so generation evidence accumulates whenever the cell is alive.
while true; do

echo "=== acquiring ${QR} (retry every ${RETRY_S}s through quota 429s) ==="
while true; do
  st=$(timeout 300 "$GCLOUD" compute tpus queued-resources describe "$QR" --zone="$ZONE" \
         --project="$PROJECT" --format='value(state.state)' 2>/dev/null || true)
  case "$st" in
    ACTIVE) echo "[acquire] ACTIVE $(date +%H:%M:%S)"; break ;;
    WAITING_FOR_RESOURCES|PROVISIONING|CREATING|ACCEPTED)
      echo "[acquire] state=$st $(date +%H:%M:%S)"; sleep 60; continue ;;
    FAILED|SUSPENDED|SUSPENDING)
      echo "[acquire] state=$st -- deleting and recreating"
      timeout 600 "$GCLOUD" compute tpus queued-resources delete "$QR" \
        --zone="$ZONE" --project="$PROJECT" --force --quiet 2>/dev/null || true
      sleep 60 ;;
  esac
  out=$(timeout 600 "$GCLOUD" compute tpus queued-resources create "$QR" --node-id="$QR" \
    --accelerator-type=v5p-32 --runtime-version=v2-alpha-tpuv5 --spot \
    --zone="$ZONE" --project="$PROJECT" 2>&1 | tail -2)
  if echo "$out" | grep -q "RESOURCE_EXHAUSTED\|429"; then
    echo "[acquire] quota full $(date +%H:%M:%S); retry in ${RETRY_S}s"
    sleep "$RETRY_S"
  else
    echo "[acquire] create issued: $(echo "$out" | tail -1 | cut -c1-120)"
    sleep 45
  fi
done

echo "=== prep (apt/uv) on all 4 workers ==="
for w in 0 1 2 3; do
  timeout 900 "$GCLOUD" compute tpus tpu-vm ssh "$QR" --zone="$ZONE" --project="$PROJECT" --worker="$w" \
    --command='sudo apt-get update -y >/dev/null 2>&1 || true
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y git curl tmux python3 python3-venv python3-pip rsync >/dev/null 2>&1 || true
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
sudo ln -sf "$HOME/.local/bin/uv" /usr/local/bin/uv 2>/dev/null || true
echo "worker prep ok: $(hostname)"' &
done
wait
echo "=== prep done; cell bring-up ==="
TPU_NAME="$QR" bash tpu/pallas_arena/rl_v5p32_bringup.sh
brc=$?
echo "=== bring-up rc=${brc} $(date +%H:%M:%S) ==="
if [ "$brc" -eq 0 ] && [ "$SMOKE_ON_UP" = "1" ]; then
  sbatch tpu/pallas_arena/probe/smoke_on_cell.sbatch 2>&1 | tail -1
fi

# Watch the slice; on death, loop back to acquisition. A single failed gcloud
# call (timeout, auth blip, API 5xx) must NOT count as slice loss -- the exit
# path DELETES the QR, so a false positive destroys a healthy cell. Require 3
# consecutive misses, and distinguish "list call failed" from "listed not-READY".
miss=0
while true; do
  listing=$(timeout 120 "$GCLOUD" compute tpus tpu-vm list --zone="$ZONE" --project="$PROJECT" 2>/dev/null)
  if [ -z "$listing" ]; then
    miss=$((miss+1)); echo "[watch] list call failed (${miss}/3) $(date +%H:%M:%S)"
  elif echo "$listing" | grep -q "^${QR}\b.*READY"; then
    miss=0
  else
    miss=$((miss+1)); echo "[watch] slice not READY (${miss}/3) $(date +%H:%M:%S)"
  fi
  [ "$miss" -ge 3 ] && break
  sleep 180
done
echo "=== slice lost $(date +%H:%M:%S); re-acquiring ==="
timeout 600 "$GCLOUD" compute tpus queued-resources delete "$QR"   --zone="$ZONE" --project="$PROJECT" --force --quiet 2>/dev/null || true
sleep 60
done  # outer lifecycle loop
