#!/bin/bash
# launch_host.sh "<chip>|<name>|<args>" ... : start detached per-chip trainings unless already running.
for spec in "$@"; do
  IFS='|' read -r chip name args <<< "$spec"
  if pgrep -f "runs/$name " >/dev/null || pgrep -f "out /dev/shm/fair/runs/$name\b" >/dev/null; then echo "skip $name (running)"; continue; fi
  setsid nohup bash /dev/shm/fair/code/run_chip.sh $chip $name $args > /dev/null 2>&1 < /dev/null &
  echo "started chip$chip $name"
done
