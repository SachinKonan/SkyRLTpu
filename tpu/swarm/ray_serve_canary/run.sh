#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
: "${SKYPILOT_NODE_IPS:?}"
: "${SKYPILOT_NODE_RANK:?}"
mapfile -t ips < <(printf '%s\n' "$SKYPILOT_NODE_IPS" | awk 'NF')
export CANARY_HOST_COUNT="${CANARY_HOST_COUNT:-4}"
[[ ${#ips[@]} == "$CANARY_HOST_COUNT" ]] || { echo "canary requires $CANARY_HOST_COUNT hosts"; exit 2; }
export CANARY_HEAD_IP="${ips[0]}"
export CANARY_ROOT="${CANARY_ROOT:-$HOME/ray-serve-canary-v1}"
export RAY_ADDRESS="$CANARY_HEAD_IP:16379"
export RAY_TMPDIR="${CANARY_RAY_TMPDIR:-$CANARY_ROOT/ray-tmp}"
export PYTHONPATH="$CANARY_ROOT/code"
export JAX_PLATFORMS=cpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export CLOUDSDK_STORAGE_SLICED_OBJECT_DOWNLOAD_THRESHOLD="${CANARY_GCLOUD_SLICE_THRESHOLD:-0}"
export CLOUDSDK_STORAGE_SLICED_OBJECT_DOWNLOAD_MAX_COMPONENTS="${CANARY_GCLOUD_SLICE_COMPONENTS:-8}"
export CLOUDSDK_STORAGE_PROCESS_COUNT="${CANARY_GCLOUD_PROCESSES:-4}"
export CLOUDSDK_STORAGE_THREAD_COUNT="${CANARY_GCLOUD_THREADS:-8}"
export RAY_USAGE_STATS_ENABLED=0
# A managed job can end while a detached training process remains. Refuse
# such a worker; never stop another experiment to make this canary fit.
if ps -eo args= | grep -Eq '[s]kyrl\.tinker\.(api|engine)|[t]mux.*skyrl-tinker'; then
  echo 'refusing canary on a worker with a detached trainer' >&2
  exit 2
fi
mkdir -p "$CANARY_ROOT"
if [[ ! -f "$CANARY_ROOT/source-ready" ]]; then
  mkdir -p "$CANARY_ROOT/source"
  gcloud storage cp "$BASE_BUNDLE" "$CANARY_ROOT/source.tar.gz"
  tar -xzf "$CANARY_ROOT/source.tar.gz" -C "$CANARY_ROOT/source"
  rm "$CANARY_ROOT/source.tar.gz"
  touch "$CANARY_ROOT/source-ready"
fi
if [[ ! -f "$CANARY_ROOT/env-ready" ]]; then
  uv venv --python 3.12 "$CANARY_ROOT/venv"
  uv pip install --python "$CANARY_ROOT/venv/bin/python" \
    'vllm-tpu==0.23.0' 'ray[serve]==2.58.0' 'transformers==5.8.0' 'jax==0.10.1' httpx psutil
  uv pip install --python "$CANARY_ROOT/venv/bin/python" --no-deps \
    "$CANARY_ROOT/source/third_party/tpu-inference"
  touch "$CANARY_ROOT/env-ready"
fi
export PATH="$CANARY_ROOT/venv/bin:$PATH"
cleanup() {
  CANARY_RAY_ADDRESS="$RAY_ADDRESS" python "$CANARY_ROOT/code/stop_ray.py" || true
}
trap cleanup EXIT
cleanup
rm -f "$CANARY_ROOT/done" "$CANARY_ROOT/result-exit-code"
resources='{"TPU":0,"canary_grade":16}'
mode=worker
if [[ "$SKYPILOT_NODE_RANK" == 0 ]]; then
  resources='{"TPU":0,"canary_grade":16,"canary_head":1}'
  mode=head
elif [[ "$SKYPILOT_NODE_RANK" == 1 || "$SKYPILOT_NODE_RANK" == 2 ]]; then
  export TPU_VISIBLE_CHIPS=0,1,2,3
  resources='{"TPU":4,"canary_grade":16,"canary_infer":1}'
  if [[ "${CANARY_RAM_CACHE_GIB:-0}" != 0 ]]; then
    python "$CANARY_ROOT/code/ram_cache.py"
  fi
  python "$CANARY_ROOT/code/prepare_cache.py"
  mkdir -p "$CANARY_ROOT/xla"
  if [[ ! -f "$CANARY_ROOT/xla-ready" ]]; then
    python "$CANARY_ROOT/code/prepare_xla.py"
    touch "$CANARY_ROOT/xla-ready"
  fi
fi
common=(--node-ip-address="${ips[$SKYPILOT_NODE_RANK]}" --num-cpus=32 \
  --resources="$resources" --object-store-memory=1073741824 \
  --object-manager-port=18379 --node-manager-port=18380 \
  --dashboard-agent-listen-port=18370 --dashboard-agent-grpc-port=18371 \
  --runtime-env-agent-port=18372 --metrics-export-port=18373 \
  --min-worker-port=41000 --max-worker-port=41999 --disable-usage-stats)
if [[ "$mode" == head ]]; then
  ray start --head --port=16379 --dashboard-port=18267 --ray-client-server-port=18303 \
    --temp-dir="$RAY_TMPDIR" "${common[@]}"
  python - <<'PY'
import os,time,ray
ray.init(address=os.environ['RAY_ADDRESS'])
for _ in range(720):
    nodes=[n for n in ray.nodes() if n['Alive']]
    print('CANARY nodes joined',len(nodes),flush=True)
    if len(nodes)==int(os.environ['CANARY_HOST_COUNT']): break
    time.sleep(10)
else: raise RuntimeError('workers did not finish setup in two hours')
ray.shutdown()
PY
  python -u "$CANARY_ROOT/code/service.py" > "$CANARY_ROOT/serve.log" 2>&1 &
  server_pid=$!
  rc=0
  python -u "$CANARY_ROOT/code/client.py" || rc=$?
  printf '%s\n' "$rc" > "$CANARY_ROOT/result-exit-code"
  gcloud storage cp "$CANARY_ROOT/events.jsonl" "$RESULT_GCS/events.jsonl" || true
  gcloud storage cp "$CANARY_ROOT/serve.log" "$RESULT_GCS/serve.log" || true
  key="$HOME/ray_bootstrap_key.pem"
  for ip in "${ips[@]:1}"; do
    ssh -n -F /dev/null -i "$key" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -o ConnectTimeout=10 "gcpuser@$ip" "touch '$CANARY_ROOT/done'" || true
  done
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  exit "$rc"
else
  for _ in $(seq 1 720); do
    if ray health-check --address="$RAY_ADDRESS" >/dev/null 2>&1; then break; fi
    sleep 10
  done
  ray start --address="$RAY_ADDRESS" "${common[@]}"
  for _ in $(seq 1 2160); do
    [[ -e "$CANARY_ROOT/done" ]] && exit 0
    sleep 10
  done
  echo 'canary exceeded six-hour worker lifetime' >&2
  exit 1
fi
