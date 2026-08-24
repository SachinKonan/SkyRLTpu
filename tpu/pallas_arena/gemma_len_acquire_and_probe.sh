#!/usr/bin/env bash
# Re-acquire the gemma length-probe v5p-16 through spot churn, prep both
# workers, and run the (flock-guarded) probe driver once. Mirrors the arena
# acquire loop; jobman is skipped because it reports quota 429s as success.
set -uo pipefail

REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
cd "$REPO"

QR="${QR:-sk7524-gemma4len-v5p16-east5a_spot}"
PROJECT=vision-mix
ZONE=us-east5-a
RETRY_S="${RETRY_S:-480}"
GCLOUD=/n/fs/vision-mix/sk7524/google-cloud-sdk/bin/gcloud

echo "=== acquiring ${QR} ==="
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
    --accelerator-type=v5p-16 --runtime-version=v2-alpha-tpuv5 --spot \
    --zone="$ZONE" --project="$PROJECT" 2>&1 | tail -2)
  if echo "$out" | grep -q "RESOURCE_EXHAUSTED\|429"; then
    echo "[acquire] quota full $(date +%H:%M:%S); retry in ${RETRY_S}s"
    sleep "$RETRY_S"
  else
    echo "[acquire] create issued: $(echo "$out" | tail -1 | cut -c1-120)"
    sleep 45
  fi
done

echo "=== prep (apt/uv) on both workers ==="
for w in 0 1; do
  timeout 900 "$GCLOUD" compute tpus tpu-vm ssh "$QR" --zone="$ZONE" --project="$PROJECT" --worker="$w" \
    --command='sudo apt-get update -y >/dev/null 2>&1 || true
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y git curl tmux python3 python3-venv python3-pip rsync >/dev/null 2>&1 || true
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
sudo ln -sf "$HOME/.local/bin/uv" /usr/local/bin/uv 2>/dev/null || true
echo "worker prep ok: $(hostname)"' &
done
wait
echo "=== prep done; running probe driver ==="
bash tpu/pallas_arena/gemma_len_probe_v6.sh
echo "=== driver rc=$? $(date +%H:%M:%S) ==="
grep -E "^uniform" runs/pallas_arena/gemma-len-v6-results.txt || true
