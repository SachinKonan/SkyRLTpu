#!/usr/bin/env bash
# Supervised Inkling Tinker runs on della-vis1 (systemd --user transient units; linger is on).
#   run_service.sh run NAME STEPS SEED [VAR=VALUE ...]   start run unit tinker-run-NAME (restart on failure, resumes)
#   run_service.sh stop NAME                             stop it (no restart)
#   run_service.sh tunnel                                start tinker-tunnel (SSH forwards to gradefarm, always restarted)
#   run_service.sh watch [VAR=VALUE ...]                 start tinker-watch (hang watchdog, capacity + failure email)
#   run_service.sh grid GRID [VAR=VALUE ...]             start tinker-grid-GRID (grid scheduler loop, grid.py)
#   run_service.sh status                                list the units
# After a vis1 reboot: run `tunnel`, `watch`, then `run ...` again for each unfinished run (they resume from disk).
set -euo pipefail
D=/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling/ops
RUNS=/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling-runs
cmd=${1:-status}; shift || true

start_unit() {  # unit, restart policy, then setenv args and command
  local unit=$1 restart=$2; shift 2
  systemctl --user stop "$unit" 2>/dev/null || true
  systemctl --user reset-failed "$unit" 2>/dev/null || true
  systemd-run --user --unit="$unit" --collect -p Restart="$restart" -p RestartSec=120 \
      -p StartLimitIntervalSec=0 "$@"
}

case "$cmd" in
  run)
    NAME=$1; STEPS=$2; SEED=$3; shift 3
    SETENV=(); for kv in "$@"; do SETENV+=(--setenv="$kv"); done
    mkdir -p "$RUNS"
    start_unit "tinker-run-$NAME" on-failure "${SETENV[@]}" \
      /bin/bash -c "exec /bin/bash $D/launch_run.sh $NAME $STEPS $SEED >> $RUNS/$NAME.log 2>&1"
    echo "started tinker-run-$NAME (log $RUNS/$NAME.log)";;
  stop)
    systemctl --user stop "tinker-run-$1"; echo "stopped tinker-run-$1";;
  tunnel)
    pkill -u "$(id -u)" -f "$D/tunnel_loop.sh" 2>/dev/null || true  # an unsupervised loop would hold the ports
    sleep 2
    start_unit tinker-tunnel always /bin/bash -c "exec /bin/bash $D/tunnel_loop.sh >> $D/tunnel_loop.log 2>&1"
    echo "started tinker-tunnel";;
  watch)
    SETENV=(); for kv in "$@"; do SETENV+=(--setenv="$kv"); done
    start_unit tinker-watch always "${SETENV[@]}" /usr/bin/python3 -u "$D/tinker_watch.py"
    echo "started tinker-watch";;
  grid)    # run_service.sh grid GRID_NAME [VAR=VALUE ...]  -> unit tinker-grid-GRID_NAME (scheduler loop)
    GN=$1; shift
    SETENV=(--setenv=GRID="$GN"); for kv in "$@"; do SETENV+=(--setenv="$kv"); done
    start_unit "tinker-grid-$GN" always "${SETENV[@]}" /usr/bin/python3 -u "$D/grid.py" loop
    echo "started tinker-grid-$GN";;
  status)
    systemctl --user list-units --all --plain --no-legend 'tinker-*' || true;;
  *) echo "unknown command $cmd"; exit 2;;
esac
