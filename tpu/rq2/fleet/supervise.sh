#!/usr/bin/env bash
# llama-farm fleet supervisor: keep 3x v5p-64 alive and serving, and publish a health-gated
# registry of endpoints for the clients.
#
#   sbatch supervise.sh              # runs until cancelled
#
# Per slice: 8 workers (1 host, 4 chips each). Workers 0-3 serve Qwen3.5-27B, workers 4-7 serve
# gemma-4-31B, so a slice loss costs both models equally and the 12/12 fleet split matches the
# campaign's average demand (1/3 qwen-only, 1/3 gemma-only, 1/3 mixed).
#
# Everything here was learned the hard way during RQ1 and is deliberate:
#   * jobman's CLI has no console entrypoint and its .venv never installed the package, so the
#     click group is invoked from source;
#   * `jobman create` launches its worker in a detached tmux that dies for that same reason,
#     so the worker runs in the foreground here;
#   * `jobman run` is NON-BLOCKING and does NOT re-request a DELETED queued resource -- it
#     considers the request already placed. That hole burned 10 dead supervisor rounds, so an
#     absent QR always triggers a fresh `create`;
#   * jobman's setup pass hangs forever on "SSH: Attempting to connect to worker 0" when a
#     worker is unreachable, so every remote-touching step is wrapped in `timeout`.
#
# Serving uses the probe's stripped per-host launcher (sampling-only, TP=4, no LoRA) pointed at
# the ALREADY-COMPILED cache prefixes, so a recovered slice restores in ~1 min instead of
# recompiling for ~55.
#
#SBATCH --job-name=llamafarm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --exclude=neu301
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu-rq1/runs/rq2/logs/%x_%j.log
set -uo pipefail

MAIN=/n/fs/vision-mix/sk7524/SkyRLTpu
RQ2=/n/fs/vision-mix/sk7524/SkyRLTpu-rq1/tpu/rq2
RUNS=/n/fs/vision-mix/sk7524/SkyRLTpu-rq1/runs/rq2
REGISTRY="${REGISTRY:-$RUNS/fleet/registry.json}"
ZONE=us-east5-a
PROJECT=vision-mix
USER_AT=sk7524_princeton_edu
KEY="$HOME/.ssh/jobman_tpu_ed25519"
JM="$MAIN/third_party/jobman"
SERVE_SH="$MAIN/tpu/pallas_arena/probe/serve_vllm.sh"
PY=/n/fs/vision-mix/sk7524/SkyRLTpu/third_party/discover/.venv-ttd-discover/bin/python

SLICES="${SLICES:-sk7524-llamafarm-a-v5p64-east5a sk7524-llamafarm-b-v5p64-east5a sk7524-llamafarm-c-v5p64-east5a}"
PORT=8001
# cached shapes -- do NOT change these without accepting a ~55 min cold compile per model
BUCKET=gs://sk7524-tinker-tpu-us-east5
QWEN_MODEL=Qwen/Qwen3.5-27B;        QWEN_MAXLEN=22528
QWEN_HF=$BUCKET/hf-cache;           QWEN_XLA=$BUCKET/vllm-xla-cache-22k
GEMMA_MODEL=google/gemma-4-31B-it;  GEMMA_MAXLEN=16384
GEMMA_HF=$BUCKET/hf-cache-gemma4;   GEMMA_XLA=$BUCKET/vllm-xla-cache-gemma4-31b-16k
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
QWEN_WORKERS="${QWEN_WORKERS:-0 1 2 3}"
GEMMA_WORKERS="${GEMMA_WORKERS:-4 5 6 7}"
PERIOD="${PERIOD:-180}"
LAUNCH_GRACE="${LAUNCH_GRACE:-1500}"   # 25 min: first install is venv build + weight restore

mkdir -p "$RUNS/fleet" "$RUNS/logs"
# NB: path is hardcoded, not "$JM" -- this function is shipped into a `timeout bash -c` subshell
# via `declare -f`, which carries the function body but NOT the enclosing variables. With $JM
# empty the `cd` lands nowhere, PYTHONPATH=. resolves wrong, and the import fails silently.
jobman_cli() { (cd "/n/fs/vision-mix/sk7524/SkyRLTpu/third_party/jobman" \
  && PYTHONPATH=. .venv/bin/python -c "from jobman.cli import cli; cli()" "$@"); }
qr_state() { gcloud compute tpus queued-resources describe "$1" --zone=$ZONE --project=$PROJECT \
               --format='value(state.state)' 2>/dev/null || true; }
log() { echo "[$(date '+%F %T')] $*"; }

# ---- credential preflight. Expired ADC is silent and lethal: it does not stop the QR API but
# it DOES break `gcloud storage`, which jobman calls for its bucket/region check -- so every
# bring-up fails with no obvious cause. RQ1 lost hours to exactly this.
creds_ok() {
  # Retry: the FIRST gcloud call on a fresh node is slow (service-account credentials load from
  # NFS-backed ~/.config/gcloud) and a 60s timeout tripped on it, stalling the whole fleet on a
  # false "bad credentials". Only a repeated failure is real.
  for try in 1 2 3; do
    timeout 180 gcloud storage buckets describe "$BUCKET" --format="value(location)" >/dev/null 2>&1 \
      && return 0
    sleep 15
  done
  return 1
}

ensure_slice() {           # $1 = slice name (QR is "<name>_spot")
  local name="$1" qr="${1}_spot" st
  st=$(qr_state "$qr")
  if [[ "$st" == "SUSPENDED" || "$st" == "FAILED" ]]; then
    log "$name: QR $st -> deleting"
    timeout 600 gcloud compute tpus queued-resources delete "$qr" --zone=$ZONE --project=$PROJECT --force --quiet >/dev/null 2>&1
    for _ in $(seq 1 30); do [[ -z "$(qr_state "$qr")" ]] && break; sleep 20; done
    st=""
  fi
  if [[ -z "$st" ]]; then
    # Reuse an existing job record if there is one -- `create` every cycle just piles up
    # duplicates (measured: 3 records, still no QR).
    local jid
    jid=$(grep -l "name: ${name}" "$JM"/jobs/sk7524/*/config.yaml 2>/dev/null | tail -1 \
          | sed -E 's|.*/([0-9]+)/config.yaml|\1|')
    if [[ -z "$jid" ]]; then
      log "$name: no QR, no job record -> jobman create"
      jid=$(timeout 900 bash -c "$(declare -f jobman_cli); jobman_cli create '$RQ2/fleet/yamls/${name}.yaml'" 2>&1 \
            | grep -viE "userwarning|warnings.warn" | tee >(sed "s|^|    [jobman] |" >&2) \
            | grep -oE 'Created job [0-9]+' | grep -oE '[0-9]+' | head -1)
    fi
    [[ -n "$jid" ]] || { log "$name: create failed"; return 1; }
    # THE REQUEST IS PLACED HERE, not by create. `create` spawns its worker as `jobman run <id>`
    # in a detached tmux which dies instantly (no console entrypoint), so create alone leaves no
    # queued resource at all -- the loop then re-creates forever. Run the worker ourselves.
    log "$name: jobman run $jid (places the QR request)"
    timeout 900 bash -c "$(declare -f jobman_cli); jobman_cli run '$jid'" 2>&1 \
      | grep -viE "userwarning|warnings.warn" | sed "s|^|    [jobman] |"
    sleep 60
    return 1
  fi
  [[ "$st" == "ACTIVE" ]] || { log "$name: $st (waiting)"; return 1; }
  return 0
}

# jobman names the TPU VM "<slice>_spot", same as the queued resource (verified: the VM is
# sk7524-llamafarm-a-v5p64-east5a_spot). Every node-level gcloud call must use the suffixed name.
vm_name() { echo "${1}_spot"; }

worker_ips() {             # $1 = slice -> lines "<worker> <ip>"
  timeout 120 gcloud compute tpus tpu-vm describe "$(vm_name "$1")" --zone=$ZONE --project=$PROJECT \
    --format="json(networkEndpoints)" 2>/dev/null \
  | $PY -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: raise SystemExit
for i,e in enumerate(d.get('networkEndpoints',[])):
    ip=(e.get('accessConfig') or {}).get('externalIp')
    if ip: print(i, ip)"
}

serve_worker() {           # $1=slice $2=worker $3=model-key
  local slice="$1" w="$2" key="$3" model maxlen hf xla
  if [[ "$key" == "qwen35" ]]; then model=$QWEN_MODEL; maxlen=$QWEN_MAXLEN; hf=$QWEN_HF; xla=$QWEN_XLA
  else model=$GEMMA_MODEL; maxlen=$GEMMA_MAXLEN; hf=$GEMMA_HF; xla=$GEMMA_XLA; fi
  # The USER matters: scp without one uses gcloud's default, which is NOT sk7524_princeton_edu
  # now that we authenticate as the service account -- the file landed in /home/sk7524 while ssh
  # ran as sk7524_princeton_edu and saw nothing. Keep the log; a silent scp failure looks exactly
  # like a slow bring-up.
  timeout 300 gcloud compute tpus tpu-vm scp "$SERVE_SH" \
    "${USER_AT}@$(vm_name "$slice"):~/serve_vllm.sh" \
    --zone=$ZONE --project=$PROJECT --worker="$w" \
    >>"$RUNS/logs/serve_${slice}_w${w}.log" 2>&1 || { echo "scp failed w$w"; return 1; }
  timeout 3600 gcloud alpha compute tpus tpu-vm ssh "${USER_AT}@$(vm_name "$slice")" --zone=$ZONE \
    --project=$PROJECT --worker="$w" --ssh-key-file="$KEY" \
    --command="MODEL='${model}' PORT=${PORT} MAX_MODEL_LEN=${maxlen} MAX_NUM_SEQS=${MAX_NUM_SEQS} \
      BIND_HOST=0.0.0.0 HF_CACHE_GCS='${hf}' XLA_CACHE_GCS='${xla}' bash ~/serve_vllm.sh" \
    >>"$RUNS/logs/serve_${slice}_w${w}.log" 2>&1
}

log "llama-farm supervisor up. slices: $SLICES"
log "registry: $REGISTRY"
while :; do
  if ! creds_ok; then
    log "FATAL-ISH: gcloud credentials are not usable (gcloud storage failed). Run 'gcloud auth login'."
    log "  holding; every bring-up would fail silently until this is fixed."
    sleep "$PERIOD"; continue
  fi

  ENTRIES="[]"
  for slice in $SLICES; do
    if ! ensure_slice "$slice"; then continue; fi
    mapfile -t IPS < <(worker_ips "$slice")
    [[ ${#IPS[@]} -eq 0 ]] && { log "$slice: ACTIVE but no worker IPs yet"; continue; }

    # bring up anything not already answering, all workers in parallel
    pids=()
    for line in "${IPS[@]}"; do
      w="${line%% *}"; ip="${line##* }"
      key="gemma4"; for q in $QWEN_WORKERS; do [[ "$w" == "$q" ]] && key="qwen35"; done
      if timeout 8 curl -sf "http://${ip}:${PORT}/v1/models" >/dev/null 2>&1; then
        rm -f "$RUNS/fleet/.launched_${slice}_w${w}"; continue
      fi
      # serve_vllm.sh starts vLLM in a tmux and RETURNS IMMEDIATELY, so a launch looks finished
      # while the server still has ~10-20 min of venv build + weight restore ahead of it. Without
      # a grace period every cycle re-launches every worker, stacking duplicate installs on the
      # same host.
      mark="$RUNS/fleet/.launched_${slice}_w${w}"
      if [[ -f "$mark" ]]; then
        age=$(( $(date +%s) - $(stat -c %Y "$mark") ))
        if (( age < LAUNCH_GRACE )); then
          log "$slice/w$w ($key): launched ${age}s ago, still coming up (grace ${LAUNCH_GRACE}s)"
          continue
        fi
        log "$slice/w$w ($key): still dead after ${age}s -> relaunching"
      fi
      touch "$mark"
      log "$slice/w$w ($key): not serving -> launching"
      ( serve_worker "$slice" "$w" "$key" ) & pids+=($!)
    done
    [[ ${#pids[@]} -gt 0 ]] && wait "${pids[@]}" 2>/dev/null

    for line in "${IPS[@]}"; do
      w="${line%% *}"; ip="${line##* }"
      key="gemma4"; for q in $QWEN_WORKERS; do [[ "$w" == "$q" ]] && key="qwen35"; done
      ENTRIES=$($PY -c "
import json,sys
e=json.loads(sys.argv[1]); e.append({'slice':sys.argv[2],'worker':int(sys.argv[3]),
  'model':sys.argv[4],'ip':sys.argv[5],'port':int(sys.argv[6]),
  'url':'http://%s:%s'%(sys.argv[5],sys.argv[6]),'healthy':False,'served_model':None,
  'last_ok':None,'checked':None}); print(json.dumps(e))" "$ENTRIES" "$slice" "$w" "$key" "$ip" "$PORT")
    done
  done

  # publish, then health-gate: a slice reading READY does not mean its workers serve
  $PY -c "
import json,sys; sys.path.insert(0,'$RQ2/fleet')
import registry as R
R.write('$REGISTRY', json.loads(sys.argv[1]))" "$ENTRIES"
  read -r H N < <($PY -c "
import sys; sys.path.insert(0,'$RQ2/fleet')
import registry as R
h,n = R.reprobe('$REGISTRY'); print(h,n)")
  log "registry: ${H}/${N} healthy endpoints"
  sleep "$PERIOD"
done
