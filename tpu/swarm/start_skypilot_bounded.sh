#!/usr/bin/env bash
# Restart the durable viz-node API without host-sized worker pools.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATE_ROOT=/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state
export HOME="$STATE_ROOT/sky-home-v6e32"
export SKYPILOT_CONFIG="$STATE_ROOT/skypilot-config.yaml"
export CLOUDSDK_CONFIG=/home/sk7524/.config/gcloud-tpuswarm-compute-sa-v6e32
export GOOGLE_APPLICATION_CREDENTIALS=/home/sk7524/.config/gcloud/vision-mix-compute-sa-key.json
export SKYPILOT_API_SERVER_ENDPOINT=http://127.0.0.1:46580
export SKYPILOT_DISABLE_LOCAL_API_SERVER=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# Logical sizing for I/O concurrency; actual CPU affinity remains four CPUs.
export SKYPILOT_POD_CPU_CORE_LIMIT=8 SKYPILOT_POD_MEMORY_GB_LIMIT=25
export SKYPILOT_MEMORY_AWARE_WORKER_SIZING=true
export SKYPILOT_CONTROLLER_BOOT_TIMEOUT_SECONDS=1800
export SKYPILOT_POOL_MAX_INFLIGHT_LAUNCHES=160
PYTHON="$REPO_ROOT/third_party/TPUSwarm/.venv/bin/python"
MODE="${1:---check}"
if [[ "$MODE" != --check && "$MODE" != --start ]]; then
    printf 'Usage: bash %s [--check|--start]\n' "$0" >&2
    exit 2
fi
cd "$REPO_ROOT"

# Affinity is inherited by API workers and recovery controllers. It is a CPU
# safeguard, not a user-wide process limit or an OS memory limit.
CPU_SET="$(/usr/bin/python3 -c 'import os; print(",".join(str(c) for c in sorted(os.sched_getaffinity(0))[:4]))')"
taskset -c "$CPU_SET" "$PYTHON" - <<'PY'
import dataclasses
import json
from sky.server.config import compute_server_config
from sky.utils import controller_utils

reserved = controller_utils.compute_memory_reserved_for_controllers(True)
config = compute_server_config(True, reserved_memory_mb=reserved, quiet=True)
controllers = controller_utils.get_number_of_jobs_controllers()
pools = controller_utils._get_number_of_services(True)
assert config.num_server_workers == 8, config
assert config.long_worker_config.garanteed_parallelism == 16, config
assert config.short_worker_config.garanteed_parallelism == 14, config
assert config.long_worker_config.burstable_parallelism == 0, config
assert config.short_worker_config.burstable_parallelism == 0, config
assert controllers <= 8, controllers
assert pools >= 6, pools
print(json.dumps(dict(server=dataclasses.asdict(config),
                      managed_job_controllers=controllers, pool_slots=pools),
                 default=str))
PY
printf 'CPU affinity: %s\n' "$CPU_SET"
[[ "$MODE" == --start ]] || exit 0

# Do not use `sky check` here: it can implicitly start an unbounded local API.
ACTIVE_ACCOUNT="$(gcloud auth list --filter=status:ACTIVE --format='value(account)')"
[[ "$ACTIVE_ACCOUNT" == 289186856710-compute@developer.gserviceaccount.com ]]
"$PYTHON" - <<'PY'
import google.auth
from google.auth.transport.requests import Request
import socket

with socket.socket() as sock:
    sock.settimeout(1)
    if sock.connect_ex(('127.0.0.1', 46580)) == 0:
        raise SystemExit('API already listening; refusing a duplicate start')
credentials, project = google.auth.default(
    scopes=['https://www.googleapis.com/auth/cloud-platform'])
assert credentials.service_account_email == '289186856710-compute@developer.gserviceaccount.com'
assert project == 'vision-mix'
credentials.refresh(Request())
print('gcloud and ADC use the same verified service account')
PY

LOG="$STATE_ROOT/sky-api-bounded-$(date -u +%Y%m%dT%H%M%SZ).log"
/usr/bin/python3 - "$CPU_SET" "$PYTHON" "$LOG" <<'PY'
import subprocess
import sys

cpu_set, python, log_path = sys.argv[1:]
with open(log_path, 'ab') as log:
    import os
    entrypoint = ['-m', 'sky.server.server']
    if os.environ.get('SKYPILOT_RESTART_MANIFEST'):
        entrypoint = ['tpu/swarm/skypilot_preserving_restart.py']
    process = subprocess.Popen(
        ['taskset', '-c', cpu_set, python, *entrypoint,
         '--deploy', '--host=127.0.0.1', '--port=46580'],
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True)
print(f'SkyPilot PID: {process.pid}\nStartup log: {log_path}')
PY
