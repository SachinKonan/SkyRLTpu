#!/bin/bash
# restart_at_step.sh <run>...: kill each run's client right after its global step advances, so the scheduler
# restarts it at a step boundary with the current scheduler settings (ingress-only overrides).
L=/scratch/gpfs/ZHUANGL/sk7524/rq4-local
step() { python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["global_step"])' "$L/runs/$1/runs/$1/client/tinker_log/$1/global_step.json" 2>/dev/null || echo 0; }
declare -A start; for r in "$@"; do start[$r]=$(step $r); echo "$(date -u +%TZ) $r waiting past step ${start[$r]}"; done
left=("$@")
while [ ${#left[@]} -gt 0 ]; do
  sleep 30; next=()
  for r in "${left[@]}"; do
    if [ "$(step $r)" -gt "${start[$r]}" ]; then
      for p in $(pgrep -f "[r]un_ttd_ensemble.py"); do
        grep -qz "^EXPERIMENT_NAME=$r$" /proc/$p/environ 2>/dev/null || continue
        [ "$(ps -o pgid= -p $p | tr -d ' ')" = "$p" ] && kill -KILL -- -$p && echo "$(date -u +%TZ) $r at step $(step $r): client $p killed"
      done
    else next+=("$r"); fi
  done
  left=("${next[@]}")
done
