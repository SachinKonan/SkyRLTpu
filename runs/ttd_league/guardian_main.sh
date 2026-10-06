#!/bin/bash
# Full-cycle guardian for the 4-AGENT arm on v5p64c: recycle the QR on
# preemption, rerun bringup_v5p64_league.sh, and on LEAGUE-ENGINES-UP launch
# the 4-agent client with the fresh IPs parsed from the sentinel line. Loops
# forever (24h horizon) so the arm survives repeated spot churn unattended.
exec 8>/tmp/sk7524-league-main-guardian-v2.lock
flock -n 8 || exit 0
cd /n/fs/vision-mix/sk7524/SkyRLTpu-league || exit 1
RUNS=runs/ttd_league
PROG=$RUNS/v5p64_league.progress
GC=/n/fs/vision-mix/sk7524/google-cloud-sdk/bin/gcloud
LOCK=/tmp/sk7524-league-bringup.lock
QR=sk7524-league-v5p64-east5a_spot
Z="--project=vision-mix --zone=us-east5-a"
KEY=$HOME/.ssh/google_compute_engine
SSHO="-i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20"
END=$(( $(date +%s) + 604800 )); n=0; miss=0; emiss=0

state_of() {
  # Must distinguish "QR genuinely absent" from "gcloud cannot authenticate".
  # Auth expires ~daily; when it does, describe returns nothing on stdout and the
  # empty result used to trip the missing-QR self-heal on EVERY arm at once.
  local out
  out=$(timeout 60 $GC alpha compute tpus queued-resources describe "$QR" $Z \
        --format="value(state.state)" 2>&1)
  if printf '%s' "$out" | grep -qiE "reauthentication|invalid_grant|refreshing your current auth|credentials"; then
    printf 'AUTHFAIL'; return 0
  fi
  printf '%s' "$out" | grep -oE '^[A-Z_]+$' | tail -1
}
clear_lock_holders() {
  for p in $(lsof -t "$LOCK" 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
  sleep 2
}

while [ "$(date +%s)" -lt "$END" ]; do
  st=$(state_of)

  if [ "$st" = "AUTHFAIL" ]; then
    [ "${authmsg:-0}" = "0" ] && echo "[guardian_main] gcloud auth expired -- parking until re-auth (running slices unaffected)"
    authmsg=1; sleep 300; continue
  fi
  authmsg=0

  # a failed create during recycle (name still releasing) leaves NO QR at all:
  # every branch below tests a state string, so an empty state deadlocks the
  # guardian forever. Recreate and continue.
  if [ -z "$st" ]; then
    echo "[guardian_main] QR missing (recycle create failed) -> recreating"
    timeout 200 $GC alpha compute tpus queued-resources create "$QR" --node-id="$QR" $Z \
      --accelerator-type=v5p-64 --runtime-version=v2-alpha-tpuv5 --provisioning-model=SPOT >/dev/null 2>&1
    sleep 60; continue
  fi
  running=$(pgrep -f "^bash tpu/bringup_v5p64_league.sh" 2>/dev/null | wc -l)

  if [ "$st" = "SUSPENDED" ] || [ "$st" = "SUSPENDING" ]; then
    if [ "$running" -ge 1 ]; then
      echo "[guardian_main] slice $st mid-attempt -> kill attempt + lock holders"
      pkill -f "^bash tpu/bringup_v5p64_league.sh" 2>/dev/null
      sleep 3; clear_lock_holders
    fi
    echo "[guardian_main] recycling $QR"
    timeout 300 $GC alpha compute tpus queued-resources delete "$QR" $Z --force --quiet >/dev/null 2>&1
    for _c in 1 2 3; do
      timeout 200 $GC alpha compute tpus queued-resources create "$QR" --node-id="$QR" $Z \
        --accelerator-type=v5p-64 --runtime-version=v2-alpha-tpuv5 --provisioning-model=SPOT >/dev/null 2>&1 && break
      sleep 30
    done
    # stale sentinel/IPs would deadlock the bring-up branch AND aim client
    # launches at the dead slice -- clear it (bring-up rewrites it fresh)
    : > "$PROG"
    sleep 120; continue
  fi

  # engines up (on the CURRENT, ACTIVE slice) but client dead -> launch client
  if [ "$st" = "ACTIVE" ] && grep -q "LEAGUE-ENGINES-UP" "$PROG" 2>/dev/null; then
    fails=0
    line=$(grep "LEAGUE-ENGINES-UP" "$PROG" | tail -1)
    W0=$(echo "$line" | grep -oE "w0\(qwen\)=[0-9.]+" | cut -d= -f2)
    GI=$(echo "$line" | grep -oE "gemma_internal=[0-9.]+" | cut -d= -f2)
    engines_ok=$(timeout 90 ssh $SSHO sk7524_princeton_edu@$W0 "ok=1; for ip in 127.0.0.1 $GI; do curl -fsS -m6 http://\$ip:8000/api/v1/get_server_capabilities >/dev/null 2>&1 || ok=0; done; echo \$ok" 2>/dev/null | tail -1)
    if [ "$engines_ok" = "1" ]; then emiss=0; else emiss=$((emiss+1)); fi
    if [ "$emiss" -ge 2 ]; then
      echo "[guardian_main] engines unreachable twice (partial host failure?) -> recycle"
      pkill -f "^bash tpu/bringup_" 2>/dev/null; sleep 3; clear_lock_holders
      timeout 300 $GC alpha compute tpus queued-resources delete "$QR" $Z --force --quiet >/dev/null 2>&1
      timeout 120 $GC alpha compute tpus queued-resources create "$QR" --node-id="$QR" $Z --accelerator-type=v5p-64 --runtime-version=v2-alpha-tpuv5 --provisioning-model=SPOT >/dev/null 2>&1
      : > "$PROG"; emiss=0; sleep 120; continue
    fi
    # A failed ssh must NOT read as "client dead": these hosts are probed by the
    # guardian, the run monitor and the sidecar healer at once, so refused/slow
    # connections are routine. Without the PROBE-OK sentinel an ssh hiccup twice
    # in a row killed a HEALTHY client -- fc46 was relaunched every 10min for an
    # hour, losing its in-flight step each time.
    probe=$(timeout 40 ssh $SSHO sk7524_princeton_edu@$W0 \
      'tmux has-session -t league 2>/dev/null && pgrep -f "[r]un_ttd_ensemble" >/dev/null && echo YES; echo PROBE-OK' 2>/dev/null)
    if ! printf '%s' "$probe" | grep -q PROBE-OK; then
      :                                   # could not check -- leave miss unchanged
    elif printf '%s' "$probe" | grep -q YES; then
      miss=0
    else
      miss=$((miss+1))
    fi
    if [ "$miss" -ge 2 ] && [ -n "$W0" ] && [ -n "$GI" ]; then
      # Do not resurrect a FINISHED run: the client exits cleanly once it has
      # done NUM_EPOCHS, the probe correctly sees no client, and the guardian
      # would relaunch it forever -- fc46 re-ran its final step for an hour on a
      # live v5p-64. Treat "loop returned" as terminal and leave the arm idle.
      # Emit an explicit DONE=<n>: a missing console log (fresh host) makes grep
      # print NOTHING, and the old `head -1` then read the PROBE-OK sentinel as
      # the count -- which blocked client launches on every rebuilt arm.
      done=$(timeout 40 ssh $SSHO sk7524_princeton_edu@$W0 \
        'n=$(grep -ac "done — loop returned" ~/skyrl-runs/league1-qwen-gemma-erdos.console.log 2>/dev/null); echo "DONE=${n:-0}"; echo PROBE-OK' 2>/dev/null)
      dn=$(printf '%s' "$done" | grep -oE 'DONE=[0-9]+' | cut -d= -f2)
      if printf '%s' "$done" | grep -q PROBE-OK && [ -n "$dn" ] && [ "$dn" -ge 1 ] 2>/dev/null; then
        [ "${donemsg:-0}" = "0" ] && echo "[guardian_main] RUN-COMPLETE (loop returned) -- not relaunching; slice idle"
        donemsg=1; miss=0; sleep 300; continue
      fi
      echo "[guardian_main] launching@$(date -u +%H:%M) client on $W0 (gemma=$GI)"
      timeout 240 ssh $SSHO sk7524_princeton_edu@$W0 \
        "bash ~/ttd-client/tpu/launch_league_run.sh $GI" 2>/dev/null | grep -v Warning
      miss=0
    fi
    sleep 300; continue
  fi

  # no sentinel yet: run a bring-up attempt if none is running and slice exists
  if [ "$running" -eq 0 ] && { [ "$st" = "ACTIVE" ] || [ "$st" = "PROVISIONING" ] || [ "$st" = "WAITING_FOR_RESOURCES" ]; }; then
    # sick-slice guard: the QR can stay ACTIVE while a worker host is dead, so
    # every attempt dies the same way and no SUSPEND branch ever fires. After
    # 2 consecutive failed attempts on one slice, force a recycle instead of
    # retrying the same broken hardware forever. (fails resets on ENGINES-UP.)
    if [ "$st" = "ACTIVE" ] && [ "${fails:-0}" -ge 2 ]; then
      echo "[guardian_main] $fails consecutive failed attempts on an ACTIVE slice -> force recycle"
      timeout 300 $GC alpha compute tpus queued-resources delete "$QR" $Z --force --quiet >/dev/null 2>&1
      for _c in 1 2 3; do
        timeout 200 $GC alpha compute tpus queued-resources create "$QR" --node-id="$QR" $Z \
          --accelerator-type=v5p-64 --runtime-version=v2-alpha-tpuv5 --provisioning-model=SPOT >/dev/null 2>&1 && break
        sleep 30
      done
      : > "$PROG"; fails=0; sleep 120; continue
    fi
    fails=$(( ${fails:-0} + 1 ))
    flock -n -x "$LOCK" -c true 2>/dev/null || clear_lock_holders
    n=$((n+1))
    echo "[guardian_main] bring-up attempt $n at $(date -u +%H:%M) (qr=$st)"
    setsid bash tpu/bringup_v5p64_league.sh 8>&- > "$RUNS/bringup_v64main_attempt_$n.log" 2>&1 < /dev/null &
    sleep 60
  fi
  sleep 150
done
echo "[guardian_main] 24h horizon expired"
