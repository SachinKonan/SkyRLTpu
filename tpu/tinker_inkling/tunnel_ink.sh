#!/usr/bin/env bash
# SSH forwards to the dedicated Inkling v6e-128 grading farms (train-faster built them; list in ink-farms.json).
# Forward 127.0.0.1:(27870+i) -> farm i's internal URL, farms sorted by name (the same order launch_run.sh's GRADE_FARMS=ink
# registry uses). Jump = an Inkling farm host's external IP (NOT train-faster's gen host). Restarts ssh whenever it exits
# and re-reads the farm list each time. Run on della-vis1 (supervised: run_service.sh tunnel-ink).
INK="${INK_FARMS:-/scratch/gpfs/ZHUANGL/sk7524/rq4-local/tvg/inkling/ink-farms.json}"
BASE="${PORT_BASE:-27870}"   # circuit farms: INK_FARMS=ink-circuit-farms*.json PORT_BASE=27880
K=/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/sky-home-v6e32/.sky/clients/7bfcb694/ssh/sky-key
while true; do
  read -r JUMP L < <(python3 - "$INK" "$BASE" <<'PY'
import json, sys
from urllib.parse import urlparse
d = json.load(open(sys.argv[1])); farms = d.get('farms', d) if isinstance(d, dict) else d
farms = sorted(farms, key=lambda f: f['name'])
ext = [f.get('external_ip') or f.get('external') for f in farms if f.get('external_ip') or f.get('external')]
fw = []
for i, f in enumerate(farms, start=1):
    u = urlparse(f.get('internal_url') or f['url'])
    fw.append(f'-L 127.0.0.1:{int(sys.argv[2]) + i}:{u.hostname}:{u.port or 24800}')
import os
print(os.environ.get('JUMP_IP') or (ext[0] if ext else 'NONE'), ' '.join(fw))
PY
)
  if [[ "$JUMP" == NONE || -z "$L" ]]; then echo "$(date -u +%FT%TZ) no farms/jump in $INK; retry in 60 s"; sleep 60; continue; fi
  echo "$(date -u +%FT%TZ) starting ssh via $JUMP: $L"
  # shellcheck disable=SC2086
  ssh -i "$K" -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -o ExitOnForwardFailure=yes -N $L "gcpuser@$JUMP"
  echo "$(date -u +%FT%TZ) ssh exited rc=$?; restarting in 10 s"
  sleep 10
done
