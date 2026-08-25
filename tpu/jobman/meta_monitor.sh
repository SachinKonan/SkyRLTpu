#!/usr/bin/env bash
# jobman monitor (workers: 0): supervise three member clients, each living on
# its own trainer host. exit 0 = generation complete; exit 1 = a member died or
# its trainer is unhealthy (jobman loops: engines re-ensured, clients
# relaunched idempotently by meta_launch).
set -euo pipefail
: "${ARM:?}"; : "${GEN:?}"; : "${JOBMAN_TPU_INTERNAL_IPS:?}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INT="$JOBMAN_TPU_INTERNAL_IPS"
ip_at() { echo "$INT" | cut -d, -f"$1"; }
KEY="$HOME/.ssh/jobman_tpu_ed25519"
SSHO="-i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15"
declare -A TRAINER=( [qwen]="$(ip_at ${T_QWEN:-1})" [gemma]="$(ip_at ${T_GEMMA:-5})" [muse]="$(ip_at ${T_MUSE:-9})" )
MONITOR_INTERVAL="${MONITOR_INTERVAL_SECONDS:-30}"
SYNC_EVERY="${SYNC_EVERY_SECONDS:-300}"
STEPS="${NUM_EPOCHS:-15}"
declare -A hfail=( [qwen]=0 [gemma]=0 [muse]=0 )

rex() {  # $1 tag  $2 cmd  [$3 timeout]
  local t="${TRAINER[$1]}"
  if [ "$t" = "$(ip_at 1)" ]; then bash -c "$2"
  else timeout "${3:-60}" ssh $SSHO sk7524_princeton_edu@"$t" "$2"; fi
}

bash "${SCRIPT_DIR}/meta_launch.sh" || true
last_sync=0
while true; do
  if bash "${SCRIPT_DIR}/meta_probe.sh"; then
    echo "generation complete -- final sync"
    bash "${SCRIPT_DIR}/meta_sync.sh" || true
    exit 0
  fi
  dead=0
  for tag in qwen gemma muse; do
    run="${ARM}-g${GEN}-${tag}"
    rex "$tag" "RUN='$run' TARGET='$STEPS' bash \$HOME/SkyRLTpu-league/tpu/jobman/member_done.sh" >/dev/null 2>&1 && continue
    if ! rex "$tag" "tmux has-session -t 'cell-$tag' 2>/dev/null"; then
      echo "member $tag client gone before completion" >&2
      # per-member sick marker lives on the member's own host, next to its tinker
      if rex "$tag" "[ -f \$HOME/ENGINE-SICK-$tag ]"; then
        echo "ENGINE-SICK-$tag -- recycling $tag tinker" >&2
        rex "$tag" "tmux kill-session -t skyrl-tinker 2>/dev/null; rm -f \$HOME/ENGINE-SICK-$tag" || true
      fi
      dead=1
    fi
    if rex "$tag" "curl -fsS --max-time 6 http://127.0.0.1:8000/api/v1/get_server_capabilities >/dev/null 2>&1"; then
      hfail[$tag]=0
    else
      hfail[$tag]=$(( ${hfail[$tag]} + 1 ))
      echo "member $tag tinker health fail (${hfail[$tag]}/4)" >&2
      [ "${hfail[$tag]}" -ge 4 ] && dead=1
    fi
  done
  if [ "$dead" = 1 ]; then
    bash "${SCRIPT_DIR}/meta_sync.sh" || true
    exit 1
  fi
  now=$(date +%s)
  if (( now - last_sync >= SYNC_EVERY )); then
    bash "${SCRIPT_DIR}/meta_sync.sh" || true
    last_sync=$now
  fi
  sleep "$MONITOR_INTERVAL"
done
