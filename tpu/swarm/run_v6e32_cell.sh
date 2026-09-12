#!/usr/bin/env bash
# Runs ONE proven stage/meta Erdős cell on one eight-VM tpu-v6e-32 pool worker
# in the 2+6 layout: VMs 0-1 (one physical host, 8 chips) = trainer + RL client
# + grader Ray head on VM 0; VMs 2-7 = six TP=4 vLLM engines.
#
# The cell itself (cell_worker.sh -> start_colocated_vllm_tinker.sh ->
# cell_monitor.sh) is the unchanged jobman unit; like run_v5p32_cell.sh this
# wrapper only adapts SkyPilot's multi-node environment to it. Differences from
# the v5p-32 wrapper: 8 VMs, the trainer spans several of them so every trainer
# VM stages the orbax checkpoint and the resume archives, and the IP list is
# NOT rotated (TRAIN_WORKERS indexes SKYPILOT_NODE_IPS positions). The trainer
# VMs are chosen below from a chip-coordinate probe, not from the yaml; a
# placement this wrapper cannot use exits 33 (recover_on_exit_codes) so the
# pool re-places the job rather than building a broken slice.
#
# V6E_TRAINER_VMS picks the width: 2 (default, qwen) or 4 (gemma, which needs
# 16 chips). TRAIN_TP_SIZE / TRAIN_FSDP_SIZE / TUNIX_ROW_SHARD still come from
# the yaml; TRAIN_WORKERS, VLLM_WORKERS and TRAIN_TPU_PROCESS_BOUNDS do not.
# The v6e chip has 32 GB HBM (v5p: 95 GB): TP 8 over 8 chips keeps the
# per-chip weight footprint under the v5p TP-4 value, and the yaml sets the
# token budget to ONE sequence per forward/backward call (v5p packed four).
# Two rows overflowed: job 650 reached CELL-UP and then died loading jit_scan
# while building its LoRA template ("reserve 6.87G, 2.68G free"). Inference,
# not training, is the bottleneck in these cells, so paying for the smaller
# call is much cheaper than taking two more VMs away from the engines.
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

# vLLM's engine subprocess renames itself "VLLM::EngineCore" and outlives its
# parent; the v20/v21 bundles' cleanup_v5p32_worker.sh predates the fix that
# kills it, so every cell on a reused worker inherited the previous cell's
# engine cores holding the chips (job 648's trainer partner died with "TPU is
# already in use by process 17223", a 643 engine core). Nothing of ours runs
# on this VM yet, so any survivor is an orphan.
orphans=$(pgrep -u "$(id -un)" -fc 'VLLM::EngineCor[e]' || true)
if [ "${orphans:-0}" -gt 0 ]; then
  echo "rank $JOBMAN_WORKER_ID: killing $orphans orphaned VLLM::EngineCore process(es)"
  pkill -9 -u "$(id -un)" -f 'VLLM::EngineCor[e]' 2>/dev/null || true
  sleep 2
fi

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

# SkyPilot's rank order is not the physical host order, and NEITHER IS THE
# GCE agent-worker-number: a probe of an idle v6e-32 (worker 1862, 2026-09-11)
# read worker 0 at grid (1,3), 1 at (1,0), 2 at (0,1), 3 at (1,2), 4 at (1,1),
# 5 at (0,2), 6 at (0,0), 7 at (0,3) -- scrambled, and the scramble differs per
# slice. Selecting a trainer block from worker numbers therefore picks
# non-contiguous hosts, which libtpu kills with SLICE_FAILURE_CHIP_DRIVER_ERROR
# (jobs 671 and 672 on two different workers) or refuses in START_SESSION
# (job 643). The only reliable source is the chips themselves, so run
# tpu/probe_topology.py as an 8-process job and read each host's global chip
# coordinates. A v6e-32 is a 4x8 chip grid = 2x4 hosts of 2x2 chips, so host
# position = (min_x / 2, min_y / 2).
SSHO=(-F /dev/null -i "$SSH_KEY_FILE" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o LogLevel=ERROR)
probe_py="$REPO/tpu/probe_topology.py"
if [ ! -r "$probe_py" ]; then
  echo "topology: $probe_py missing from the bundle; recovering" >&2
  exit 33
fi
# jax + libtpu only, in its own venv: the bundle's .venv is built later by
# cell_worker, and this must run before the split is known. Cached per VM.
topo_setup='export PATH="$HOME/.local/bin:$PATH"
  if [ ! -x "$HOME/topo-venv/bin/python" ]; then
    uv venv --python 3.12 "$HOME/topo-venv" >/dev/null 2>&1 \
      && uv pip install --python "$HOME/topo-venv/bin/python" -q jax==0.11.1 jaxlib==0.11.1 libtpu==0.0.46 requests==2.32.5 >/dev/null 2>&1
  fi
  "$HOME/topo-venv/bin/python" -c "import jax" 2>/dev/null && echo TOPO_VENV_OK || echo TOPO_VENV_FAIL'
probe_dir=$(mktemp -d)
rank=0
for ip in ${ips//,/ }; do
  ( if [ "$rank" -eq 0 ]; then out=$(bash -c "$topo_setup" 2>&1 | tail -1)
    else out=$(timeout 600 ssh "${SSHO[@]}" "$REMOTE_USER@$ip" "$topo_setup" 2>&1 | tail -1); fi
    echo "$out" > "$probe_dir/venv.$rank" ) &
  rank=$((rank + 1))
done
wait
for r in $(seq 0 7); do
  if ! grep -q TOPO_VENV_OK "$probe_dir/venv.$r" 2>/dev/null; then
    echo "topology: rank $r could not build the probe venv; recovering" >&2
    rm -rf "$probe_dir"; exit 33
  fi
done
probe_cmd="env -u TPU_PROCESS_BOUNDS -u TPU_CHIPS_PER_PROCESS_BOUNDS -u TPU_PROCESS_ADDRESSES -u CLOUD_TPU_TASK_ID -u TPU_VISIBLE_CHIPS JAX_PLATFORMS=tpu \$HOME/topo-venv/bin/python $probe_py"
rank=0
for ip in ${ips//,/ }; do
  ( if [ "$rank" -eq 0 ]; then out=$(bash -c "$probe_cmd $rank $(cut -d, -f1 <<<"$ips"):9917 8" 2>&1 | grep PROBE_RESULT || true)
    else out=$(timeout 600 ssh "${SSHO[@]}" "$REMOTE_USER@$ip" "$probe_cmd $rank $(cut -d, -f1 <<<"$ips"):9917 8" 2>&1 | grep PROBE_RESULT || true); fi
    echo "$out" > "$probe_dir/probe.$rank" ) &
  rank=$((rank + 1))
done
wait
# The probe holds the chips until it exits; make sure none linger.
rank=0
for ip in ${ips//,/ }; do
  if [ "$rank" -eq 0 ]; then pkill -9 -u "$(id -un)" -f '[p]robe_topology' 2>/dev/null || true
  else timeout 40 ssh "${SSHO[@]}" "$REMOTE_USER@$ip" "pkill -9 -u \$(id -un) -f '[p]robe_topology' 2>/dev/null; true" >/dev/null 2>&1 || true; fi
  rank=$((rank + 1))
done
split=$(cat "$probe_dir"/probe.* 2>/dev/null | sed 's/^PROBE_RESULT //' | python3 -c '
import json, sys
# host grid position from the chip coordinates a host reports
pos = {}
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    d = json.loads(line)
    c = d["coords"]
    pos[int(d["process_id"])] = (min(p[0] for p in c) // 2, min(p[1] for p in c) // 2)
if len(pos) != 8:
    sys.exit("probe returned %d of 8 hosts" % len(pos))
want = int(sys.argv[1]) if len(sys.argv) > 1 else 2
x0, y0 = pos[0]
at = {v: k for k, v in pos.items()}
if len(at) != 8:
    sys.exit("two hosts reported the same grid position")
if want == 4:
    # 2x2 block containing rank 0, with rank 0 at its low corner so that it is
    # task 0 (CLOUD_TPU_TASK_ID is x-fastest: local_x + 2 * local_y).
    # The block must also be ALIGNED to its own height: every 4-host block
    # tried at y0=1 (chip rows 2-5) died in libtpu init with
    # SLICE_FAILURE_SW_INJECT_ERROR (677 x3, 678 x1 on two different slices,
    # 2026-09-12), while the 2-tall qwen blocks, which are always aligned,
    # train. Restricting to y0 in {0,2} = chip rows 0-3 or 4-7.
    if x0 != 0 or y0 not in (0, 2):
        sys.exit("rank 0 at (%d,%d) is not the low corner of an aligned 2x2 block" % (x0, y0))
    cands = [([at[(x, y)] for y in (y0, y0 + 1) for x in (0, 1)], "2,2,1")]
else:
    # adjacent pair with rank 0 at the lower coordinate: prefer the y neighbour
    # (2x4 chips, the v6e-8 shape), else the x neighbour. Both are emitted so
    # the pre-flight below can fall back: the y pair in column 0 failed the
    # libtpu index_on_host check on two different slices (682, 2026-09-12)
    # while the same pair in column 1 trained for hours.
    cands = []
    if (x0, y0 + 1) in at:
        cands.append(([at[(x0, y0)], at[(x0, y0 + 1)]], "1,2,1"))
    if (x0 + 1, y0) in at:
        cands.append(([at[(x0, y0)], at[(x0 + 1, y0)]], "2,1,1"))
    if not cands:
        sys.exit("rank 0 at (%d,%d) has no neighbour at a higher coordinate" % (x0, y0))
for ranks, bounds in cands:
    if ranks[0] != 0:
        sys.exit("task 0 resolved to sky rank %d, not 0" % ranks[0])
    print("CAND", ",".join(str(r) for r in ranks), bounds)
print("GRID", " ".join("%d:(%d,%d)" % (r, pos[r][0], pos[r][1]) for r in sorted(pos)))
' "${V6E_TRAINER_VMS:-2}") || {
  echo "topology: $(cat "$probe_dir"/probe.* 2>/dev/null | wc -l) probe results; selection failed; recovering" >&2
  rm -rf "$probe_dir"; exit 33
}
rm -rf "$probe_dir"
grid_map=$(sed -n 's/^GRID //p' <<<"$split")

# Pre-flight: bring a candidate block up once with the trainer's own sub-slice
# environment (jax + libtpu only, ~10 s) BEFORE staging 40 GB of checkpoints
# onto it and starting engines. A block that libtpu rejects otherwise costs
# 30 min (qwen: TPU_RET_CHECK index_on_host()==i on the head after CELL-UP,
# 682 twice) or 75 min (gemma: SLICE_FAILURE_SW_INJECT_ERROR plus a launcher
# waiting on engines the event killed, 677/678). A task whose peer failed
# blocks forever, hence the timeout and the sweep afterwards.
preflight_block() {  # $1 = comma-separated sky ranks in task order, $2 = process bounds
  local ranks=$1 bounds=$2 t r ip addrs="" pf_dir ok
  local -a pf_ranks pf_ips
  IFS=, read -r -a pf_ranks <<<"$ranks"
  IFS=, read -r -a pf_ips <<<"$ips"
  for r in "${pf_ranks[@]}"; do addrs="$addrs${addrs:+,}${pf_ips[$r]}:8478"; done
  local py='import jax; ds=jax.devices(); print("PREFLIGHT_OK", len(ds))'
  local cmd="env -u TPU_VISIBLE_CHIPS -u TPU_HOST_BOUNDS JAX_PLATFORMS=tpu TPU_PROCESS_BOUNDS=$bounds TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_ADDRESSES=$addrs TPU_PROCESS_PORT=8478"
  pf_dir=$(mktemp -d)
  t=0
  for r in "${pf_ranks[@]}"; do
    ip=${pf_ips[$r]}
    ( if [ "$r" -eq 0 ]; then out=$(timeout 100 bash -c "$cmd CLOUD_TPU_TASK_ID=$t \$HOME/topo-venv/bin/python -c '$py'" 2>&1)
      else out=$(timeout 100 ssh "${SSHO[@]}" "$REMOTE_USER@$ip" "$cmd CLOUD_TPU_TASK_ID=$t \$HOME/topo-venv/bin/python -c '$py'" 2>&1); fi
      grep -E 'PREFLIGHT_OK|RET_CHECK|SLICE_FAILURE|already in use|Error' <<<"$out" | head -1 > "$pf_dir/task.$t" ) &
    t=$((t + 1))
  done
  wait
  for r in "${pf_ranks[@]}"; do
    ip=${pf_ips[$r]}
    if [ "$r" -eq 0 ]; then pkill -u "$(id -un)" -f 'topo-venv/bin/python -c' 2>/dev/null || true
    else timeout 40 ssh "${SSHO[@]}" "$REMOTE_USER@$ip" "pkill -u \$(id -un) -f 'topo-venv/bin/python -c' 2>/dev/null; true" >/dev/null 2>&1 || true; fi
  done
  for f in "$pf_dir"/task.*; do echo "preflight[block $ranks, task ${f##*.}]: $(head -c 200 "$f")"; done
  ok=$(cat "$pf_dir"/task.* 2>/dev/null | grep -c "PREFLIGHT_OK $(( ${#pf_ranks[@]} * 4 ))\$" || true)
  rm -rf "$pf_dir"
  [ "${ok:-0}" -eq "${#pf_ranks[@]}" ]
}
TRAIN_WORKERS=""; TRAIN_TPU_PROCESS_BOUNDS=""
while read -r tag cand_ranks cand_bounds; do
  [ "$tag" = CAND ] || continue
  if preflight_block "$cand_ranks" "$cand_bounds"; then
    TRAIN_WORKERS=$cand_ranks; TRAIN_TPU_PROCESS_BOUNDS=$cand_bounds; break
  fi
  echo "preflight: block $cand_ranks (process grid $cand_bounds) failed libtpu init; trying the next candidate"
done <<<"$split"
if [ -z "$TRAIN_WORKERS" ]; then
  echo "topology: $grid_map"
  echo "preflight: no candidate trainer block passes libtpu init on this slice; recovering" >&2
  exit 33
fi
export TRAIN_WORKERS TRAIN_TPU_PROCESS_BOUNDS
export VLLM_WORKERS
VLLM_WORKERS=$(for r in 0 1 2 3 4 5 6 7; do
  case ",$TRAIN_WORKERS," in *",$r,"*) ;; *) echo "$r" ;; esac
done | paste -sd, -)
IFS=, read -r -a train_ranks <<<"$TRAIN_WORKERS"
partner_ranks=("${train_ranks[@]:1}")
IFS=, read -r -a engine_ranks <<<"$VLLM_WORKERS"
echo "topology: sky rank -> host grid position: $grid_map"
echo "cell $CELL on v6e-32 (${#train_ranks[@]}+${#engine_ranks[@]}): trainer ranks $TRAIN_WORKERS (process grid $TRAIN_TPU_PROCESS_BOUNDS, rank 0 = task 0 = client/API); engine ranks $VLLM_WORKERS"

# Rank 0 staged its own orbax copy above; every other trainer VM needs one too,
# since each JAX process restores its own shards.
for pr in "${partner_ranks[@]}"; do
  timeout 3600 ssh "${SSHO[@]}" "$REMOTE_USER@$(cut -d, -f$((pr + 1)) <<<"$ips")" \
    "CELL='$CELL' TRAIN_WORKERS='$TRAIN_WORKERS' JOBMAN_WORKER_ID='$pr' SKYRL_REPO_DIR='$REPO' TUNIX_MAXTEXT_CKPT_CACHE_GCS='${TUNIX_MAXTEXT_CKPT_CACHE_GCS:-}' bash '$REPO/tpu/jobman/ensure_orbax_ckpt.sh'" 2>&1 | sed "s/^/[train rank $pr] /" \
    || { echo "topology: trainer rank $pr could not stage the orbax checkpoint" >&2; exit 33; }
done

# A resume also needs the RL checkpoint ARCHIVES on every trainer VM, not just
# rank 0. tunix_backend.load_checkpoint broadcasts the call and each process
# opens the same local path ($HOME/gcs/skyrl-checkpoints/<model>/<step>.tar.gz)
# itself; on v5p the trainer was one host so only rank 0 ever needed them, and
# here the partner's directory is empty, so the client's first load_weights
# dies with FileNotFoundError (job 658, 19:48Z, right after the LoRA template
# built). launch_cell.sh restores them on rank 0 from SKYRL_CKPT_GCS; mirror
# the newest row of the run's checkpoints.jsonl onto the partner first.
if [ -n "${GCS_RUN:-}" ]; then
  ckpt_gcs="gs://$(cut -d/ -f3 <<<"$GCS_RUN")/skyrl-checkpoints"
  latest=$(timeout 180 gcloud storage cat "$GCS_RUN/tinker_log/*/*/checkpoints.jsonl" 2>/dev/null | python3 -c '
import json, re, sys
pat = re.compile(r"tinker://(model_[0-9a-f]+)/weights/(\d+)")
last = ""
for line in sys.stdin:
    try:
        row = json.loads(line)
    except ValueError:
        continue
    m = pat.match(row.get("state_path") or "")
    if m:
        last = m.group(1) + " " + m.group(2)
print(last)
' || true)
  if [ -n "$latest" ]; then
    read -r ckpt_model ckpt_step <<<"$latest"
    for pr in "${partner_ranks[@]}"; do
      echo "resume: staging $ckpt_model/$ckpt_step archives on trainer rank $pr"
      timeout 1800 ssh "${SSHO[@]}" "$REMOTE_USER@$(cut -d, -f$((pr + 1)) <<<"$ips")" \
        "set -e; d=\$HOME/gcs/skyrl-checkpoints/$ckpt_model; mkdir -p \$d/sampler_weights
         for rel in $ckpt_step.tar.gz sampler_weights/$ckpt_step.tar.gz; do
           if [ -s \"\$d/\$rel\" ]; then echo \"have \$rel\"; continue; fi
           if gcloud storage cp '$ckpt_gcs/$ckpt_model'/\$rel \"\$d/\$rel\" >/dev/null 2>&1; then
             echo \"staged \$rel (\$(du -h \"\$d/\$rel\" | cut -f1))\"
           else
             rm -f \"\$d/\$rel\"; echo \"MISSING \$rel in $ckpt_gcs/$ckpt_model\"
           fi
         done" 2>&1 | sed "s/^/[ckpt rank $pr] /" \
        || { echo "resume: could not stage RL checkpoints on rank $pr" >&2; exit 33; }
    done
  else
    echo "resume: no checkpoints.jsonl rows under $GCS_RUN -- fresh run, nothing to stage"
  fi
fi

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
