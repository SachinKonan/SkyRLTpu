#!/bin/bash
# Full-cycle guardian for the 4-AGENT arm on v5p64c: recycle the QR on
# preemption, rerun bringup_v5p64d_league.sh, and on LEAGUE-ENGINES-UP launch
# the 4-agent client with the fresh IPs parsed from the sentinel line. Loops
# forever (24h horizon) so the arm survives repeated spot churn unattended.
exec 8>/tmp/sk7524-league-d-guardian.lock
flock -n 8 || exit 0
cd /n/fs/vision-mix/sk7524/SkyRLTpu-league || exit 1
RUNS=runs/ttd_league
PROG=$RUNS/v5p64d_league.progress
GC=/n/fs/vision-mix/sk7524/google-cloud-sdk/bin/gcloud
LOCK=/tmp/sk7524-league-bringup-d.lock
QR=sk7524-league-v5p64d-east5a_spot
Z="--project=vision-mix --zone=us-east5-a"
KEY=$HOME/.ssh/google_compute_engine
SSHO="-i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20"
END=$(( $(date +%s) + 86400 )); n=0; miss=0; emiss=0

state_of() {
  timeout 60 $GC alpha compute tpus queued-resources describe "$QR" $Z \
    --format="value(state.state)" 2>/dev/null | tail -1
}
clear_lock_holders() {
  for p in $(lsof -t "$LOCK" 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
  sleep 2
}

while [ "$(date +%s)" -lt "$END" ]; do
  st=$(state_of)
  running=$(pgrep -f "^bash tpu/bringup_v5p64d_league.sh" 2>/dev/null | wc -l)

  if [ "$st" = "SUSPENDED" ] || [ "$st" = "SUSPENDING" ]; then
    if [ "$running" -ge 1 ]; then
      echo "[guardian_d] slice $st mid-attempt -> kill attempt + lock holders"
      pkill -f "^bash tpu/bringup_v5p64d_league.sh" 2>/dev/null
      sleep 3; clear_lock_holders
    fi
    echo "[guardian_d] recycling $QR"
    timeout 300 $GC alpha compute tpus queued-resources delete "$QR" $Z --force --quiet >/dev/null 2>&1
    timeout 120 $GC alpha compute tpus queued-resources create "$QR" --node-id="$QR" $Z \
      --accelerator-type=v5p-64 --runtime-version=v2-alpha-tpuv5 --provisioning-model=SPOT >/dev/null 2>&1
    # stale sentinel/IPs would deadlock the bring-up branch AND aim client
    # launches at the dead slice -- clear it (bring-up rewrites it fresh)
    : > "$PROG"
    sleep 120; continue
  fi

  # engines up (on the CURRENT, ACTIVE slice) but client dead -> launch client
  if [ "$st" = "ACTIVE" ] && grep -q "LEAGUE-ENGINES-UP" "$PROG" 2>/dev/null; then
    line=$(grep "LEAGUE-ENGINES-UP" "$PROG" | tail -1)
    W0=$(echo "$line" | grep -oE "w0\(qwen\)=[0-9.]+" | cut -d= -f2)
    GI=$(echo "$line" | grep -oE "gemma_internal=[0-9.]+" | cut -d= -f2)
    engines_ok=$(timeout 90 ssh $SSHO sk7524_princeton_edu@$W0 "ok=1; for ip in 127.0.0.1 $GI; do curl -fsS -m6 http://\$ip:8000/api/v1/get_server_capabilities >/dev/null 2>&1 || ok=0; done; echo \$ok" 2>/dev/null | tail -1)
    if [ "$engines_ok" = "1" ]; then emiss=0; else emiss=$((emiss+1)); fi
    if [ "$emiss" -ge 2 ]; then
      echo "[guardian_d] engines unreachable twice (partial host failure?) -> recycle"
      pkill -f "^bash tpu/bringup_" 2>/dev/null; sleep 3; clear_lock_holders
      timeout 300 $GC alpha compute tpus queued-resources delete "$QR" $Z --force --quiet >/dev/null 2>&1
      timeout 120 $GC alpha compute tpus queued-resources create "$QR" --node-id="$QR" $Z --accelerator-type=v5p-64 --runtime-version=v2-alpha-tpuv5 --provisioning-model=SPOT >/dev/null 2>&1
      : > "$PROG"; emiss=0; sleep 120; continue
    fi
    alive=$(timeout 40 ssh $SSHO sk7524_princeton_edu@$W0 \
      'tmux has-session -t league 2>/dev/null && pgrep -f "[r]un_ttd_ensemble" >/dev/null && echo YES' 2>/dev/null | grep -c YES)
    if [ "$alive" -ge 1 ]; then miss=0; else miss=$((miss+1)); fi
    if [ "$miss" -ge 2 ] && [ -n "$W0" ] && [ -n "$GI" ]; then
      echo "[guardian_d] launching@$(date -u +%H:%M) client on $W0 (gemma=$GI)"
      timeout 240 ssh $SSHO sk7524_princeton_edu@$W0 \
        "bash ~/ttd-client/tpu/launch_league_frontier.sh $GI" 2>/dev/null | grep -v Warning
      miss=0
    fi
    sleep 300; continue
  fi

  # no sentinel yet: run a bring-up attempt if none is running and slice exists
  if [ "$running" -eq 0 ] && { [ "$st" = "ACTIVE" ] || [ "$st" = "PROVISIONING" ] || [ "$st" = "WAITING_FOR_RESOURCES" ]; }; then
    flock -n -x "$LOCK" -c true 2>/dev/null || clear_lock_holders
    n=$((n+1))
    echo "[guardian_d] bring-up attempt $n at $(date -u +%H:%M) (qr=$st)"
    setsid bash tpu/bringup_v5p64d_league.sh > "$RUNS/bringup_v64d_attempt_$n.log" 2>&1 < /dev/null &
    sleep 60
  fi
  sleep 150
done
echo "[guardian_d] 24h horizon expired"
