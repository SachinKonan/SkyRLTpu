#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-vision-mix}"
ZONE="${ZONE:-us-east5-a}"
TPU_NAME="${TPU_NAME:-sk7524-tinker-v5p64-east5a_spot}"
REMOTE_USER="${REMOTE_USER:-sk7524_princeton_edu}"
SSH_KEY_FILE="${SSH_KEY_FILE:-$HOME/.ssh/jobman_tpu_ed25519}"
REMOTE_SKYRL_DIR="${REMOTE_SKYRL_DIR:-/home/${REMOTE_USER}/SkyRLTpu}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
TPU_WORKER_IDS="${TPU_WORKER_IDS:-}"
TRAIN_PROCESSES="${TRAIN_PROCESSES:-6}"
COORD_PORT="${COORD_PORT:-12355}"
TPU_PROCESS_PORT="${TPU_PROCESS_PORT:-8476}"
TPU_PROCESS_BOUNDS="${TPU_PROCESS_BOUNDS:-1,2,4}"
TPU_CHIPS_PER_PROCESS_BOUNDS="${TPU_CHIPS_PER_PROCESS_BOUNDS:-2,2,1}"
PROBE_TIMEOUT_SEC="${PROBE_TIMEOUT_SEC:-480}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
probe_script="${script_dir}/probe_subset_meshes.py"
if [[ ! -f "${probe_script}" ]]; then
  echo "Missing ${probe_script}" >&2
  exit 1
fi

mapfile -t endpoint_ips < <(
  gcloud alpha compute tpus tpu-vm describe "${TPU_NAME}" \
    --project="${PROJECT}" \
    --zone="${ZONE}" \
    --format='value(networkEndpoints.ipAddress)' | tr ';' '\n' | awk 'NF'
)
if (( ${#endpoint_ips[@]} < NUM_PROCESSES )); then
  echo "Need ${NUM_PROCESSES} TPU endpoints, found ${#endpoint_ips[@]}" >&2
  exit 1
fi

mapfile -t tpu_worker_ids < <(
  python3 - <<PY
raw = "${TPU_WORKER_IDS}".strip()
num_processes = int("${NUM_PROCESSES}")
workers = [int(x.strip()) for x in raw.split(",") if x.strip()] if raw else list(range(num_processes))
if len(workers) != num_processes:
    raise SystemExit(f"TPU_WORKER_IDS has {len(workers)} entries but NUM_PROCESSES={num_processes}")
if len(set(workers)) != len(workers):
    raise SystemExit(f"TPU_WORKER_IDS contains duplicates: {workers}")
max_worker = int("${#endpoint_ips[@]}") - 1
for worker in workers:
    if worker < 0 or worker > max_worker:
        raise SystemExit(f"TPU worker {worker} is out of range 0..{max_worker}")
    print(worker)
PY
)

coord_worker="${tpu_worker_ids[0]}"
coord_address="${endpoint_ips[$coord_worker]}:${COORD_PORT}"
process_addresses=""
for worker in "${tpu_worker_ids[@]}"; do
  if [[ -n "${process_addresses}" ]]; then
    process_addresses+=","
  fi
  process_addresses+="${endpoint_ips[$worker]}:${TPU_PROCESS_PORT}"
done

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_dir="${repo_root}/benchmark_artifacts/subset_mesh_probe/${timestamp}"
mkdir -p "${log_dir}"
remote_probe="/tmp/skyrl_subset_mesh_probe.py"

for worker in "${tpu_worker_ids[@]}"; do
  gcloud alpha compute tpus tpu-vm scp "${probe_script}" "${REMOTE_USER}@${TPU_NAME}:${remote_probe}" \
    --project="${PROJECT}" --zone="${ZONE}" --worker="${worker}" --ssh-key-file="${SSH_KEY_FILE}" --quiet
done

pids=()
for ((process_id = 0; process_id < NUM_PROCESSES; process_id++)); do
  worker="${tpu_worker_ids[$process_id]}"
  remote_cmd="
set -euo pipefail
export PATH=\"\$HOME/.local/bin:\$PATH\"
export TPU_PROCESS_BOUNDS='${TPU_PROCESS_BOUNDS}'
export TPU_CHIPS_PER_PROCESS_BOUNDS='${TPU_CHIPS_PER_PROCESS_BOUNDS}'
export TPU_PROCESS_ADDRESSES='${process_addresses}'
export TPU_PROCESS_PORT='${TPU_PROCESS_PORT}'
unset TPU_VISIBLE_CHIPS
cd '${REMOTE_SKYRL_DIR}'
timeout '${PROBE_TIMEOUT_SEC}s' uv run --extra tpu --extra tinker --extra jax python3 '${remote_probe}' \
  --coordinator-address '${coord_address}' \
  --num-processes '${NUM_PROCESSES}' \
  --process-id '${process_id}' \
  --train-processes '${TRAIN_PROCESSES}'
"
  gcloud alpha compute tpus tpu-vm ssh "${REMOTE_USER}@${TPU_NAME}" \
    --project="${PROJECT}" --zone="${ZONE}" --worker="${worker}" --ssh-key-file="${SSH_KEY_FILE}" --quiet \
    --command "${remote_cmd}" >"${log_dir}/worker-${worker}-process-${process_id}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

cat "${log_dir}"/worker-*.log
echo "Subset mesh probe logs: ${log_dir}"
exit "${status}"
