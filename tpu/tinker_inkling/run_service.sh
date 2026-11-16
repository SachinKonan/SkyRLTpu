#!/usr/bin/env bash
# Supervised Inkling Tinker runs on della-vis1 (systemd --user transient units; linger is on).
#   run_service.sh run NAME STEPS SEED [VAR=VALUE ...]   start run unit tinker-run-NAME (restart on failure, resumes)
#   run_service.sh stop NAME                             stop it (no restart)
#   run_service.sh tunnel                                start tinker-tunnel (SSH forwards to gradefarm, always restarted)
#   run_service.sh watch [VAR=VALUE ...]                 start tinker-watch (hang watchdog, capacity + failure email)
#   run_service.sh grid GRID [VAR=VALUE ...]             start tinker-grid-GRID (grid scheduler loop, grid.py)
#   run_service.sh grid-stop GRID                        stop it AND keep it stopped across reboots
#   run_service.sh resume-all                            what the boot unit runs: tunnel, watch and every grid not
#                                                        grid-stopped, with the arguments they were last started with
#   run_service.sh install-boot                          install + enable tinker-boot.service (runs resume-all at boot;
#                                                        ConditionHost keeps it from starting on other della nodes)
#   run_service.sh status                                list the units
# Reboot: transient units vanish; tinker-boot.service brings back tunnel, watch and grids, and the grids restart their
# runs (each resumes from its checkpoint and tree). Runs started by hand with `run` are NOT restarted (email lists them).
set -euo pipefail
D=/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling/ops
RUNS=/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling-runs
TO=sk7524@princeton.edu
cmd=${1:-status}; shift || true

start_unit() {  # unit, restart policy, then setenv args and command
  local unit=$1 restart=$2; shift 2
  systemctl --user stop "$unit" 2>/dev/null || true
  systemctl --user reset-failed "$unit" 2>/dev/null || true
  systemd-run --user --unit="$unit" --collect -p Restart="$restart" -p RestartSec=120 \
      -p StartLimitIntervalSec=0 "$@"
}

save_args() {  # file, then the VAR=VALUE args, one per line (atomic)
  local f=$1; shift
  mkdir -p "$(dirname "$f")"
  if (($#)); then printf '%s\n' "$@" > "$f.tmp"; else : > "$f.tmp"; fi
  mv "$f.tmp" "$f"
}

case "$cmd" in
  run)
    NAME=$1; STEPS=$2; SEED=$3; shift 3
    if [[ -e $RUNS/COST_STOP ]]; then echo "refused: $RUNS/COST_STOP exists (spend hard stop)"; exit 3; fi
    if [[ -e $RUNS/$NAME/BLOCKED ]]; then echo "refused: $RUNS/$NAME/BLOCKED exists (crash loop); delete it to allow"; exit 3; fi
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
    save_args "$D/watch.args" "$@"
    SETENV=(); for kv in "$@"; do SETENV+=(--setenv="$kv"); done
    start_unit tinker-watch always "${SETENV[@]}" /usr/bin/python3 -u "$D/tinker_watch.py"
    echo "started tinker-watch";;
  grid)    # run_service.sh grid GRID_NAME [VAR=VALUE ...]  -> unit tinker-grid-GRID_NAME (scheduler loop)
    GN=$1; shift
    save_args "$RUNS/grid-$GN/service.args" "$@"
    rm -f "$RUNS/grid-$GN/STOPPED"
    SETENV=(--setenv=GRID="$GN"); for kv in "$@"; do SETENV+=(--setenv="$kv"); done
    start_unit "tinker-grid-$GN" always "${SETENV[@]}" /usr/bin/python3 -u "$D/grid.py" loop
    echo "started tinker-grid-$GN";;
  grid-stop)
    GN=$1
    mkdir -p "$RUNS/grid-$GN"; date -u +%FT%TZ > "$RUNS/grid-$GN/STOPPED"
    systemctl --user stop "tinker-grid-$GN" 2>/dev/null || true
    echo "stopped tinker-grid-$GN (stays stopped across reboots; its runs keep going until they finish or you stop them)";;
  resume-all)
    started=()
    bash "$0" tunnel && started+=(tunnel)
    if [[ -f $D/watch.args ]]; then
      mapfile -t A < "$D/watch.args"; bash "$0" watch "${A[@]}" && started+=(watch)
    fi
    for f in "$RUNS"/grid-*/service.args; do
      [[ -f $f ]] || continue
      g=$(basename "$(dirname "$f")"); g=${g#grid-}
      [[ -f $RUNS/grid-$g/STOPPED ]] && continue
      mapfile -t A < "$f"; bash "$0" grid "$g" "${A[@]}" && started+=("grid $g")
    done
    printf 'To: %s\nSubject: [tinker-inkling] %s: units restored at boot\n\nresume-all started: %s\nGrids restart their runs within %s s; runs started by hand were not restarted.\n' \
      "$TO" "$(hostname -s)" "${started[*]}" "${GRID_LOOP_S:-120}" | /usr/sbin/sendmail -t || true
    echo "resume-all: ${started[*]}";;
  install-boot)
    U=$HOME/.config/systemd/user/tinker-boot.service
    mkdir -p "$(dirname "$U")"
    cat > "$U.tmp" <<EOF
[Unit]
Description=Inkling Tinker runs: restore tunnel, watchdog and grid schedulers after a della-vis1 reboot
# The home directory is shared by all della nodes: this unit must only ever act on della-vis1.
ConditionHost=della-vis1*

[Service]
Type=oneshot
ExecStartPre=/bin/sleep 60
ExecStart=/bin/bash $D/run_service.sh resume-all

[Install]
WantedBy=default.target
EOF
    mv "$U.tmp" "$U"
    systemctl --user daemon-reload
    systemctl --user enable tinker-boot.service
    echo "installed $U (enabled; runs resume-all at boot on della-vis1 only)";;
  status)
    systemctl --user list-units --all --plain --no-legend 'tinker-*' || true;;
  *) echo "unknown command $cmd"; exit 2;;
esac
