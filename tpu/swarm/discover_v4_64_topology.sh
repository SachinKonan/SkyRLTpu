#!/usr/bin/env bash
# Discover and validate a 4-host training row inside one v4-64 TPU slice.
set -euo pipefail

: "${JOBMAN_TPU_INTERNAL_IPS:?}"
: "${SSH_KEY_FILE:?}"
: "${OUTPUT_ENV_FILE:?}"

REMOTE_USER="${REMOTE_USER:-gcpuser}"
REPO="${SKYRL_REPO_DIR:-$HOME/SkyRLTpu-tpuswarm}"
PROBE_PYTHON="${V4_64_TOPOLOGY_PROBE_PYTHON:-$HOME/.venvs/tpu-topology/bin/python}"
PROBE_TIMEOUT_SEC="${V4_64_TOPOLOGY_PROBE_TIMEOUT_SEC:-180}"
FULL_COORD_PORT="${V4_64_TOPOLOGY_FULL_COORD_PORT:-7788}"
SUBSET_COORD_PORT="${V4_64_TOPOLOGY_SUBSET_COORD_PORT:-7789}"
SUBSET_TPU_PORT="${V4_64_TOPOLOGY_SUBSET_TPU_PORT:-8488}"

IFS=',' read -r -a worker_ips <<< "$JOBMAN_TPU_INTERNAL_IPS"
if (( ${#worker_ips[@]} != 8 )); then
  echo "v4-64 topology discovery requires eight Sky workers" >&2
  exit 2
fi

SSHO=(
  -F /dev/null
  -i "$SSH_KEY_FILE"
  -o IdentitiesOnly=yes
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o ConnectTimeout=30
)
tmpdir=$(mktemp -d)
trap 'rm -rf -- "$tmpdir"' EXIT

run_probe() {
  local rank="$1" process_id="$2" num_processes="$3" coordinator="$4"
  local process_bounds="${5:-}" process_addresses="${6:-}"
  local remote_env="unset TPU_PROCESS_BOUNDS TPU_CHIPS_PER_PROCESS_BOUNDS TPU_PROCESS_ADDRESSES TPU_PROCESS_PORT CLOUD_TPU_TASK_ID TPU_VISIBLE_CHIPS;"
  if [[ -n "$process_bounds" ]]; then
    remote_env+=" export TPU_PROCESS_BOUNDS='$process_bounds';"
    remote_env+=" export TPU_CHIPS_PER_PROCESS_BOUNDS='2,2,1';"
    remote_env+=" export TPU_PROCESS_ADDRESSES='$process_addresses';"
    remote_env+=" export TPU_PROCESS_PORT='$SUBSET_TPU_PORT';"
    remote_env+=" export CLOUD_TPU_TASK_ID='$process_id';"
  fi
  ssh "${SSHO[@]}" "$REMOTE_USER@${worker_ips[$rank]}" \
    "set -euo pipefail; export JAX_PLATFORMS=tpu; $remote_env timeout '${PROBE_TIMEOUT_SEC}s' '$PROBE_PYTHON' '$REPO/tpu/probe_topology.py' '$process_id' '$coordinator' '$num_processes'"
}

full_coordinator="${worker_ips[0]}:${FULL_COORD_PORT}"
pids=()
for rank in "${!worker_ips[@]}"; do
  run_probe "$rank" "$rank" 8 "$full_coordinator" >"$tmpdir/full-$rank.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if (( failed )); then
  echo "full-slice v4-64 topology probe failed" >&2
  tail -20 "$tmpdir"/full-*.log >&2 || true
  exit 1
fi

records="$tmpdir/topology.jsonl"
for rank in "${!worker_ips[@]}"; do
  result=$(grep '^PROBE_RESULT ' "$tmpdir/full-$rank.log" | tail -1 | sed 's/^PROBE_RESULT //')
  if [[ -z "$result" ]]; then
    echo "full-slice probe omitted rank $rank result" >&2
    exit 1
  fi
  printf '%s\n' "$result" >> "$records"
done

selection="$tmpdir/selection.env"
python3 "$REPO/tpu/swarm/select_v4_64_topology.py" < "$records" > "$selection"
# shellcheck disable=SC1090
source "$selection"
: "${TRAIN_WORKERS:?topology selector omitted TRAIN_WORKERS}"
: "${VLLM_WORKERS:?topology selector omitted VLLM_WORKERS}"
echo "v4-64 topology selected TRAIN_WORKERS=$TRAIN_WORKERS VLLM_WORKERS=$VLLM_WORKERS"

IFS=',' read -r -a train_workers <<< "$TRAIN_WORKERS"
if (( ${#train_workers[@]} != 4 )) || [[ "${train_workers[0]}" != "0" ]]; then
  echo "topology selector must return four trainer ranks beginning with rank 0" >&2
  exit 1
fi
subset_addresses=""
for rank in "${train_workers[@]}"; do
  if [[ -n "$subset_addresses" ]]; then subset_addresses+=","; fi
  subset_addresses+="${worker_ips[$rank]}:${SUBSET_TPU_PORT}"
done

sleep 2
subset_coordinator="${worker_ips[0]}:${SUBSET_COORD_PORT}"
pids=()
for process_id in "${!train_workers[@]}"; do
  rank="${train_workers[$process_id]}"
  run_probe "$rank" "$process_id" 4 "$subset_coordinator" "1,1,4" "$subset_addresses" \
    >"$tmpdir/subset-$process_id.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if (( failed )); then
  echo "selected v4-64 trainer submesh failed validation" >&2
  tail -20 "$tmpdir"/subset-*.log >&2 || true
  exit 1
fi
for process_id in "${!train_workers[@]}"; do
  grep -q '^PROBE_RESULT ' "$tmpdir/subset-$process_id.log" || {
    echo "trainer submesh probe omitted process $process_id result" >&2
    exit 1
  }
done
echo "v4-64 trainer submesh validation passed"

{
  printf 'TRAIN_WORKERS=%q\n' "$TRAIN_WORKERS"
  printf 'VLLM_WORKERS=%q\n' "$VLLM_WORKERS"
} > "$OUTPUT_ENV_FILE"
