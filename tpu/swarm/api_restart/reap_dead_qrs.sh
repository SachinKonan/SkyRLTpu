#!/usr/bin/env bash
# Delete ONLY our own terminal (SUSPENDED/FAILED) TPU queued resources in one zone.
# Ownership = name prefix tpuswarm- AND our SkyPilot client hash 7bfcb694.
# ACTIVE / WAITING_FOR_RESOURCES / PROVISIONING (real reservations) are never matched.
set -uo pipefail
Z="${1:?zone}"; P=vision-mix
export CLOUDSDK_CONFIG=/home/sk7524/.config/gcloud-tpuswarm-compute-sa-v6e32 GOOGLE_APPLICATION_CREDENTIALS=/home/sk7524/.config/gcloud/vision-mix-compute-sa-key.json
mkdir -p /scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/api-restart; LOG=/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/api-restart/reap_${Z}.log
gcloud compute tpus queued-resources list --zone "$Z" --project "$P" \
  --filter='name~tpuswarm-.*-7bfcb694- AND (state.state=SUSPENDED OR state.state=FAILED)' \
  --format='value(name,state.state)' > "$LOG.targets" 2>/dev/null
n=$(wc -l < "$LOG.targets"); echo "$Z: $n dead owned requests" | tee -a "$LOG"
while read -r name state; do
  [ -n "$name" ] || continue
  case "$state" in SUSPENDED|FAILED) ;; *) echo "SKIP $name ($state)" | tee -a "$LOG"; continue;; esac
  case "$name" in tpuswarm-*-7bfcb694-*) ;; *) echo "SKIP $name (not ours)" | tee -a "$LOG"; continue;; esac
  if gcloud compute tpus queued-resources delete "$name" --zone "$Z" --project "$P" --force --async --quiet >/dev/null 2>>"$LOG"; then
    echo "delete-requested $state $name" | tee -a "$LOG"
  else echo "FAILED-to-delete $state $name" | tee -a "$LOG"; fi
done < "$LOG.targets"
