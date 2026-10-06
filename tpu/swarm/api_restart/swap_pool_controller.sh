#!/usr/bin/env bash
# Replace ONE pool controller with the API server's own HA-recovery respawn:
#   kill -9 the controller's process group (controller + its 2 multiprocessing
#   helpers; SIGKILL runs no cleanup, so no replica is torn down by the exit),
#   then wait for sky/server/daemons.py's 20 s pool HA-recovery loop to respawn
#   it from the stored recovery script (venv python => picks up the edited
#   replica_managers.py; inherits the bounded API env and CPUs 0-3).
set -uo pipefail
SVC="${1:?service name}"; SKYHOME=/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/sky-home-v6e32
PY=/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-multihost/third_party/TPUSwarm/.venv/bin/python
dbpid() { $PY - "$SVC" <<'PY'
import sqlite3,sys
s=sqlite3.connect("file:/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/sky-home-v6e32/.sky/serve/services.db?mode=ro",uri=True)
r=s.execute("select controller_pid from services where name=?",(sys.argv[1],)).fetchone(); print(r[0] if r else "")
PY
}
old=$(pgrep -f -- "sky.serve.service --service-name $SVC " | head -1)
[ -n "$old" ] || { echo "$SVC: no running controller found"; exit 2; }
pg=$(ps -o pgid= -p "$old" | tr -d ' ')
echo "$(date -u +%H:%M:%SZ) $SVC: killing controller pid=$old pgid=$pg (children: $(ps -o pid= --ppid $old | tr -s ' \n' ' '))"
kill -9 -- "-$pg" 2>/dev/null; sleep 2
pgrep -f -- "sky.serve.service --service-name $SVC " >/dev/null && { echo "still alive after SIGKILL?"; exit 3; }
echo "$(date -u +%H:%M:%SZ) $SVC: dead; waiting for HA-recovery respawn (<=180 s)"
for i in $(seq 1 36); do
  sleep 5
  new=$(pgrep -f -- "python -u -m sky.serve.service --service-name $SVC " | head -1)
  if [ -n "$new" ] && [ "$new" != "$old" ]; then
    echo "$(date -u +%H:%M:%SZ) $SVC: respawned pid=$new affinity=$(taskset -cp $new | grep -oE '[0-9-]+$') db_pid=$(dbpid) log=$(readlink /proc/$new/fd/1 2>/dev/null)"
    exit 0
  fi
done
echo "$(date -u +%H:%M:%SZ) $SVC: NOT respawned within 180 s; check /tmp/pool_ha_recovery.log"; exit 4
