#!/usr/bin/env bash
# jobman monitor (workers: 0): ensure all three member clients run; supervise.
#   exit 0 -> generation complete (probe green)
#   exit 1 -> a member died / trainer unhealthy -> jobman loops (re-ensures
#             engines via meta_worker, relaunches missing clients here)
# Per-member ENGINE-SICK-<tag> markers recycle ONLY that member's tinker.
set -euo pipefail
: "${ARM:?}"; : "${GEN:?}"; : "${JOBMAN_TPU_INTERNAL_IPS:?}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INT="$JOBMAN_TPU_INTERNAL_IPS"
ip_at() { echo "$INT" | cut -d, -f"$1"; }
KEY="$HOME/.ssh/jobman_tpu_ed25519"
SSHO="-i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20"
declare -A TRAINER=( [qwen]="$(ip_at 1)" [gemma]="$(ip_at 5)" [muse]="$(ip_at 9)" )
MONITOR_INTERVAL="${MONITOR_INTERVAL_SECONDS:-30}"
SYNC_EVERY="${SYNC_EVERY_SECONDS:-300}"
declare -A hfail=( [qwen]=0 [gemma]=0 [muse]=0 )

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
    # member complete? then a missing tmux session is fine
    if ARM="$ARM" GEN="$GEN" NUM_EPOCHS="${NUM_EPOCHS:-15}" TAG_ONLY="$tag" python3 - <<'PY'
import glob, json, os, sys
arm, gen, target, tag = os.environ["ARM"], os.environ["GEN"], int(os.environ["NUM_EPOCHS"]), os.environ["TAG_ONLY"]
run = f"{arm}-g{gen}-{tag}"
if glob.glob(os.path.expanduser(f"~/skyrl-runs/{run}/tinker_log/*/CONVERGED")): sys.exit(0)
latest, final = None, False
for p in glob.glob(os.path.expanduser(f"~/skyrl-runs/{run}/tinker_log/*/member_*/checkpoints.jsonl")):
    for line in open(p):
        line=line.strip()
        if not line: continue
        try: row=json.loads(line)
        except ValueError: continue
        b=row.get("batch")
        if isinstance(b,int): latest=b if latest is None else max(latest,b)
        if row.get("name")=="final": final=True
sys.exit(0 if (final or (latest is not None and latest>=target)) else 1)
PY
    then continue; fi
    if ! tmux has-session -t "cell-$tag" 2>/dev/null; then
      echo "member $tag client gone before completion" >&2
      if [ -f "$HOME/ENGINE-SICK-$tag" ]; then
        echo "ENGINE-SICK-$tag ($(head -1 "$HOME/ENGINE-SICK-$tag" 2>/dev/null)) -- recycling $tag tinker" >&2
        t="${TRAINER[$tag]}"
        if [ "$t" = "$(ip_at 1)" ]; then tmux kill-session -t skyrl-tinker 2>/dev/null || true
        else timeout 60 ssh $SSHO sk7524_princeton_edu@"$t" "tmux kill-session -t skyrl-tinker 2>/dev/null" || true; fi
        rm -f "$HOME/ENGINE-SICK-$tag"
      fi
      dead=1
    fi
    # trainer health per member
    if curl -fsS --max-time 6 "http://${TRAINER[$tag]}:8000/api/v1/get_server_capabilities" >/dev/null 2>&1; then
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
