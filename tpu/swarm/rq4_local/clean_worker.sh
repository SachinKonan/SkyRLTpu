#!/bin/bash
# clean_worker.sh <cluster> [--check]: on every host of a pool worker, force-stop leftover farm runtimes and grading
# units, kill private Ray clusters, then verify no runtime/grading unit, private Ray or TPU owner remains.
# --check only reports. Run right after `sky jobs cancel` so the worker is clean before the pool reuses it.
set -u
c=$1; mode=${2:-}
D=/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/sky-home-v6e32/.sky/generated/ssh
read -r -d '' HOST <<'EOF'
units() { systemctl list-units --no-legend --plain --all 'skyrl-runtime-*' 'math-grade-*' 'science-grade-*' 'placement-grade-*' 'ac2-grade-*' 2>/dev/null | awk '$3!="inactive" || $4!="dead" {print $1}'; }
if [ "$MODE" != --check ]; then
  for u in $(units); do sudo -n systemctl kill -s SIGKILL "$u" 2>/dev/null; done
  for u in $(units); do sudo -n systemctl stop "$u" 2>/dev/null; sudo -n systemctl reset-failed "$u" 2>/dev/null; done
  pkill -9 -u "$(id -u)" -f '/\.rtr-' 2>/dev/null
  # Idle workers of the SkyPilot runtime Ray pile up on long-lived pool workers (204 = 18 GB seen on
  # erdos-3225 host 0) and starve the next farm's RAM-cache admission check. Ray respawns them on demand.
  pkill -u "$(id -u)" -f '^ray::IDLE' 2>/dev/null
fi
for i in $(seq 1 20); do
  left=$(units | tr '\n' ' '); owners=$(sudo -n fuser /dev/vfio/[0-9]* /dev/accel* 2>/dev/null | tr -s ' '); rtr=$(pgrep -u "$(id -u)" -f '/\.rtr-' | tr '\n' ' ')
  [ -z "$left$owners$rtr" ] && break; [ "$MODE" = --check ] && break; sleep 1
done
state=clean; [ -n "$left$owners$rtr" ] && state=DIRTY
echo "$(hostname) $state units=[${left}] tpu_owners=[${owners}] private_ray=[${rtr}] mem_avail_gib=$(awk '/MemAvailable/{printf "%.0f", $2/1048576}' /proc/meminfo)"
EOF
timeout 150 ssh -F "$D/$c" -o BatchMode=yes -o ConnectTimeout=10 "$c" "MODE='$mode' bash -s" <<OUTER 2>&1 | grep -v 'Permanently added' | sed "s/^/$c: /"
cat > /tmp/clean_host.sh <<'INNER'
$HOST
INNER
P=\$(python3 -c 'import json;print(json.load(open("/home/gcpuser/.sky/ray_port.json"))["ray_port"])' 2>/dev/null)
ips=\$(~/skypilot-runtime/bin/python -c "
import ray; ray.init(address='127.0.0.1:\$P', logging_level='ERROR')
print(' '.join(sorted({n['NodeManagerAddress'] for n in ray.nodes() if n['Alive']})))" 2>/dev/null | tail -1)
[ -z "\$ips" ] && { echo "could not list worker hosts; head only"; ips=\$(hostname -I | awk '{print \$1}'); }
me=\$(hostname -I)
for ip in \$ips; do
  case " \$me " in *" \$ip "*) MODE="\$MODE" bash /tmp/clean_host.sh & ;;
  *) ssh -F /dev/null -i ~/.ssh/sky-cluster-key -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=10 \$ip "MODE='\$MODE' bash -s" < /tmp/clean_host.sh 2>/dev/null & ;;
  esac
done; wait
OUTER
