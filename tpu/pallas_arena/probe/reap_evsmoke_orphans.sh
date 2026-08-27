#!/usr/bin/env bash
# Reap ARM queued-resources whose slurm job is gone.
#
# GCP will not delete a QR that is mid-PROVISIONING, so an arm that times out
# in that state leaves its request behind holding quota -- measured
# 2026-08-27, three accumulated FAILED/PROVISIONING corpses starved both
# splash arms for over an hour with no error in any log. The job's own
# teardown now retries and ALARMs, but it cannot outlast the provisioning
# window, so this catches what it cannot.
#
# SCOPE: only names matching sk7524-evsmoke-<jobid>, and only when no slurm
# job with that id is running. It can never touch the judge slice or another
# user's resources.
G=/n/fs/vision-mix/sk7524/google-cloud-sdk/bin/gcloud
ZONE=${ZONE:-us-east5-b}
while true; do
  live=$(squeue -u sk7524 -h -o "%i" 2>/dev/null | tr '\n' ' ')
  while read -r name state; do
    [ -z "$name" ] && continue
    case "$name" in sk7524-evsmoke-*) : ;; *) continue ;; esac
    jid=${name##*-}
    if echo " $live " | grep -q " $jid "; then continue; fi
    if timeout 200 "$G" compute tpus queued-resources delete "$name" --zone="$ZONE" \
         --project=vision-mix --force --quiet >/dev/null 2>&1; then
      echo "$(date +%H:%M:%S) reaped $name (was $state)"
    fi
  done <<< "$(timeout 150 "$G" compute tpus queued-resources list --zone="$ZONE" \
               --project=vision-mix --format='value(name,state.state)' 2>/dev/null)"
  sleep 120
done
