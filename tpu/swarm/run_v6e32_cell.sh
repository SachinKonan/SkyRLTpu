#!/usr/bin/env bash
# Runs ONE proven stage/meta Erdős cell on one eight-VM tpu-v6e-32 pool worker
# in the 2+6 layout: VMs 0-1 (one physical host, 8 chips) = trainer + RL client
# + grader Ray head on VM 0; VMs 2-7 = six TP=4 vLLM engines.
#
# The cell itself (cell_worker.sh -> start_colocated_vllm_tinker.sh ->
# cell_monitor.sh) is the unchanged jobman unit; like run_v5p32_cell.sh this
# wrapper only adapts SkyPilot's multi-node environment to it. Differences from
# the v5p-32 wrapper: 8 VMs, TRAIN_WORKERS spans two VMs, so every trainer VM
# stages the orbax checkpoint, and the IP list is NOT rotated (TRAIN_WORKERS
# indexes SKYPILOT_NODE_IPS positions and VMs 0,1 must stay physically adjacent
# for the 2,1,1 process grid); if this VM is rank 0 but not first in the list
# we exit 33 (recover_on_exit_codes) instead of guessing a topology.
#
# Layout knobs come from the task yaml: TRAIN_WORKERS=0,1 VLLM_WORKERS=2..7
# TRAIN_TP_SIZE=8 TRAIN_FSDP_SIZE=1 TUNIX_ROW_SHARD=1
# TRAIN_TPU_PROCESS_BOUNDS=2,1,1 TRAIN_TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1.
# The v6e chip has 32 GB HBM (v5p: 95 GB): TP 8 over 8 chips keeps the
# per-chip weight footprint under the v5p TP-4 value and the yaml halves the
# token budget so one fb call holds two rows, as on v5p.
set -euo pipefail
: "${SKYPILOT_NODE_RANK:=0}"
: "${SKYPILOT_NODE_IPS:?SkyPilot must provide all v6e-32 TPU VM IPs}"
: "${CELL:?CELL selects the model and cell knobs, e.g. g-v32-ttd-n}"

export JOBMAN_WORKER_ID="$SKYPILOT_NODE_RANK"
ips=$(printf '%s\n' "$SKYPILOT_NODE_IPS" | awk 'NF' | paste -sd, -)
node_count=$(awk -F, '{print NF}' <<<"$ips")
if [ "$node_count" -ne 8 ]; then
  echo "a v6e-32 cell requires 8 TPU VMs; SkyPilot supplied $node_count" >&2
  exit 2
fi
export JOBMAN_TPU_INTERNAL_IPS="$ips"
export TRAIN_WORKERS="${TRAIN_WORKERS:-0,1}"
export VLLM_WORKERS="${VLLM_WORKERS:-2,3,4,5,6,7}"

export REMOTE_USER="${REMOTE_USER:-$(id -un)}"
export SSH_KEY_FILE="${SSH_KEY_FILE:-$HOME/ray_bootstrap_key.pem}"
mkdir -p "$HOME/.ssh" "$HOME/skyrl-runs" "$HOME/skyrl-logs"
chmod 700 "$HOME/.ssh"
REPO="${SKYRL_REPO_DIR:-$PWD}"

# Pool workers are reused across jobs and models: drop the other models' caches
# and superseded bundle generations on EVERY VM first (best effort).
bash "$REPO/tpu/swarm/reconcile_v5p32_worker.sh" \
  || echo "worker reconcile failed (continuing)" >&2

# SkyPilot invokes the run command on every VM. The cell is head-driven and
# reaches the other ranks over the internal network, so only rank 0 continues.
# Non-zero ranks must NOT stage the orbax checkpoint here: which VM partners
# the trainer is decided below from physical topology, and a 40 GB orbax copy
# on a VM that ends up serving filled its 150 GB disk to zero next to the HF
# weights (jobs 635/636, rank 1). Rank 0 stages its own copy and the chosen
# partner's over ssh after the selection.
if [ "$JOBMAN_WORKER_ID" != "0" ]; then
  echo "rank $JOBMAN_WORKER_ID ready; rank 0 owns cell $CELL"
  exit 0
fi
TRAIN_WORKERS=0 bash "$REPO/tpu/jobman/ensure_orbax_ckpt.sh"
if [ ! -f "$SSH_KEY_FILE" ]; then
  echo "SkyPilot TPU pod key is missing: $SSH_KEY_FILE" >&2
  exit 2
fi
chmod 600 "$SSH_KEY_FILE"
ln -sfn "$SSH_KEY_FILE" "$HOME/.ssh/jobman_tpu_ed25519"

# cell_worker treats the FIRST address as the trainer/client host and
# TRAIN_WORKERS as positions in that list. Rank 0 must therefore be first.
first_ip=$(cut -d, -f1 <<<"$ips")
if ! hostname -I 2>/dev/null | tr ' ' '\n' | grep -qxF "$first_ip"; then
  echo "rank 0 ($(hostname -I)) is not first in SKYPILOT_NODE_IPS ($ips); recovering onto another worker" >&2
  exit 33
fi

# SkyPilot's rank order is NOT the physical host order (job 631: ranks 0,1 =
# physical workers 0,3, a diagonal pair; job 632: 0,7). A two-VM trainer must
# be an ICI-adjacent pair declared to libtpu as a 2,1,1 or 1,2,1 process grid
# with task 0 at the lower coordinate, or libtpu aborts with "Mesh build was
# incomplete". v6e-32 HOST_BOUNDS is 2,4,1 and GCE's agent-worker-number w
# is the physical host index, so host (x, y) = (w % 2, w / 2). Rank 0 (the
# API/client host) is always task 0, so its partner must sit at x+1 (2,1,1)
# or, failing that, at y+1 (1,2,1); the far corner host has neither -> 33.
SSHO=(-F /dev/null -i "$SSH_KEY_FILE" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o LogLevel=ERROR)
meta_url=http://metadata.google.internal/computeMetadata/v1/instance/attributes/agent-worker-number
declare -a phys=()
rank=0
for ip in ${ips//,/ }; do
  if [ "$rank" -eq 0 ]; then
    w=$(curl -sf -H 'Metadata-Flavor: Google' "$meta_url" || true)
  else
    w=$(timeout 40 ssh "${SSHO[@]}" "$REMOTE_USER@$ip" "curl -sf -H 'Metadata-Flavor: Google' $meta_url" 2>/dev/null || true)
  fi
  if ! [[ "$w" =~ ^[0-7]$ ]]; then
    echo "topology: could not read agent-worker-number of rank $rank ($ip): '$w'; recovering" >&2
    exit 33
  fi
  phys[$rank]=$w
  rank=$((rank + 1))
done
w0=${phys[0]}; x0=$((w0 % 2)); y0=$((w0 / 2))
# Pair orientation. The two 2x2-chip hosts form either a 4x2 chip block
# (partner at x+1, process grid 2,1,1) or a 2x4 block (partner at y+1, grid
# 1,2,1). 2x4 is the shape of a real v6e-8 slice, so it is the default; the
# 4x2 attempt on job 643 died in libtpu's START_SESSION, though that worker's
# fabric later failed a whole-slice init too, so the verdict is still open.
# V6E_PAIR_AXIS=x forces the 4x2 orientation.
pick_partner() {
  case "$1" in
    y) [ "$y0" -lt 3 ] && { partner_w=$((w0 + 2)); bounds="1,2,1"; return 0; } ;;
    x) [ "$x0" -eq 0 ] && { partner_w=$((w0 + 1)); bounds="2,1,1"; return 0; } ;;
  esac
  return 1
}
partner_w=""; bounds=""
case "${V6E_PAIR_AXIS:-y}" in
  x) pick_partner x || pick_partner y ;;
  *) pick_partner y || pick_partner x ;;
esac
if [ -z "$partner_w" ]; then
  echo "topology: rank 0 is physical worker $w0 (far corner); no partner with a higher coordinate; recovering" >&2
  exit 33
fi
partner_rank=""
for r in "${!phys[@]}"; do [ "${phys[$r]}" = "$partner_w" ] && partner_rank=$r; done
if [ -z "$partner_rank" ]; then
  echo "topology: physical worker $partner_w not found among ranks (${phys[*]}); recovering" >&2
  exit 33
fi
export TRAIN_WORKERS="0,$partner_rank"
export TRAIN_TPU_PROCESS_BOUNDS="$bounds"
export VLLM_WORKERS
# `[ ] && echo` as the loop's last command returns 1 when rank 7 is the
# partner, and under set -e that killed the head with a silent exit 1 (jobs
# 638/639 on worker 1610, whose physical host 1 is Sky rank 7). Use if/then.
VLLM_WORKERS=$(for r in 1 2 3 4 5 6 7; do if [ "$r" -ne "$partner_rank" ]; then echo "$r"; fi; done | paste -sd, -)
echo "topology: sky rank -> physical worker: $(for r in "${!phys[@]}"; do printf '%s->%s ' "$r" "${phys[$r]}"; done)"
echo "cell $CELL on v6e-32 (2+6): trainer ranks $TRAIN_WORKERS (physical $w0,$partner_w; process grid $bounds; rank 0 = client/API); engine ranks $VLLM_WORKERS"
# Rank 0 already staged the orbax checkpoint above; the chosen partner must
# hold it too (it may not be rank 1, which is the one the yaml default staged).
timeout 3600 ssh "${SSHO[@]}" "$REMOTE_USER@$(cut -d, -f$((partner_rank + 1)) <<<"$ips")" \
  "CELL='$CELL' TRAIN_WORKERS='$TRAIN_WORKERS' JOBMAN_WORKER_ID='$partner_rank' SKYRL_REPO_DIR='$REPO' TUNIX_MAXTEXT_CKPT_CACHE_GCS='${TUNIX_MAXTEXT_CKPT_CACHE_GCS:-}' bash '$REPO/tpu/jobman/ensure_orbax_ckpt.sh'" 2>&1 | sed "s/^/[partner] /" \
  || { echo "topology: partner rank $partner_rank could not stage the orbax checkpoint" >&2; exit 33; }

# Whatever way the cell dies, leave all eight VMs clean for the next job (the
# pool fork never re-places a failed job). cleanup_v5p32_worker.sh iterates
# over JOBMAN_TPU_INTERNAL_IPS, so it covers eight hosts unchanged.
cleanup_on_failure() {
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    bash "$REPO/tpu/swarm/cleanup_v5p32_worker.sh" "$rc" 2>&1 \
      || echo "cleanup_v5p32_worker.sh itself failed (rc=$?)" >&2
  fi
  exit "$rc"
}
trap cleanup_on_failure EXIT

bash "$REPO/tpu/jobman/cell_worker.sh"
monitor_rc=0
bash "$REPO/tpu/jobman/cell_monitor.sh" || monitor_rc=$?
exit "$monitor_rc"
