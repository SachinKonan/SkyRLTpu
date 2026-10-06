#!/usr/bin/env bash
# Leave a v5p-32 pool worker clean after a FAILED cell, on all four hosts.
#
#   cleanup_v5p32_worker.sh <exit-code>
#   env: CELL, JOBMAN_TPU_INTERNAL_IPS (head first), SSH_KEY_FILE, REMOTE_USER,
#        SKYRL_REPO_DIR (bundle install, same path on every host)
#
# Run by run_v5p32_cell.sh's EXIT trap on rank 0 whenever the cell exits
# non-zero (orbax restore failed, engine bring-up failed, client died,
# engines sick, ...). The pool fork never re-places a failed job, so the next
# job to land here is a different one, often a different model, and must find:
#   - no engines, trainer, client or sidecar from this cell still holding the
#     TPU chips, HBM or the LoRA upload port;
#   - no partial downloads (`_.gstmp`) or gcloud tracker files that would make
#     the next restore resume into the same failure;
#   - caches reconciled for THIS cell's model (foreign model caches gone, HF
#     duplicate collapsed, superseded bundles pruned) so the disk has room for
#     the next job's SkyPilot ray worker and restore -- a host at 0 bytes kills
#     that ray worker before any of our code runs (FAILED_DRIVER).
# A preempted worker is gone anyway; this is for the failures we own.
# Best effort throughout: a cleanup step must never mask the original failure.
set -uo pipefail
rc="${1:-1}"
: "${CELL:?}"
ips="${JOBMAN_TPU_INTERNAL_IPS:-}"
key="${SSH_KEY_FILE:-$HOME/.ssh/jobman_tpu_ed25519}"
user="${REMOTE_USER:-$(id -un)}"
repo="${SKYRL_REPO_DIR:-$HOME/SkyRLTpu-tpuswarm}"
# -F /dev/null: a corrupted ~/.ssh/config on a head ("ost ..." on line 1,
# job 280's node, 2026-09-06) made every default ssh fail with rc 255 and left
# the engine hosts uncleaned; cell_monitor.sh already ignores the file.
SSHO=(-F /dev/null -i "$key" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o LogLevel=ERROR)

# The per-host body. Runs as the pool user; $1 = rank (0 = head).
# Bracketed first letters keep pkill from matching this very command line.
host_cleanup() {
  local rank="$1"
  local before
  before=$(df -Pk / | awk 'NR==2 {print int($4/1048576)}')
  for s in $(tmux ls 2>/dev/null | cut -d: -f1); do
    case "$s" in
      cell|cell-backup|skyrl-tinker|vllm-tpu|vllm-tpu-e*) tmux kill-session -t "=$s" 2>/dev/null ;;
    esac
  done
  pkill -u "$(id -un)" -f '[v]llm_tpu_server|[s]kyrl\.tinker\.(api|engine)|[r]un_ttd_ensemble|[c]ell_monitor\.sh|[c]ell_worker\.sh|[s]tart_vllm_tpu|[s]idecar_' 2>/dev/null
  sleep 2
  pkill -9 -u "$(id -un)" -f '[v]llm_tpu_server|[s]kyrl\.tinker\.(api|engine)|[r]un_ttd_ensemble' 2>/dev/null
  # vLLM's engine subprocess renames itself to "VLLM::EngineCore" and outlives
  # its parent; it kept holding the chips on worker 242 after job 485 was
  # cancelled (2026-09-09) and blocked the next job's preflight. Bracketed so
  # the pattern never matches this shell itself.
  pkill -9 -u "$(id -un)" -f 'VLLM::EngineCor[e]' 2>/dev/null
  # partial downloads and the trackers that would resume into them
  find "$HOME/.cache/huggingface" "$HOME/skyrl-maxtext-ckpts-local" "$HOME/gcs/skyrl-checkpoints" \
    \( -name '*_.gstmp' -o -name '*.gstmp' -o -name '*.incomplete' -o -name '*.partial' \) -delete 2>/dev/null
  rm -rf "$HOME/.config/gcloud/surface_data/storage/tracker_files" 2>/dev/null
  rm -f "$HOME/ENGINE-SICK" 2>/dev/null
  # caches reconciled for this cell's model: foreign models gone, HF duplicate
  # collapsed, superseded bundles and benchmark caches pruned
  if [ -r "$repo/tpu/swarm/reconcile_v5p32_worker.sh" ]; then
    CELL="$CELL" SKYPILOT_NODE_RANK="$rank" SKYRL_REPO_DIR="$repo" \
      bash "$repo/tpu/swarm/reconcile_v5p32_worker.sh" 2>/dev/null | grep -E "removing|reclaimed|keeps" | sed 's/^/  /'
  fi
  echo "cleanup[rank $rank]: free ${before} GB -> $(df -Pk / | awk 'NR==2 {print int($4/1048576)}') GB; procs left: $(pgrep -u "$(id -un)" -fc '[v]llm_tpu_server|[s]kyrl\.tinker' 2>/dev/null)"
}

echo "cleanup: cell $CELL exited $rc -- leaving the worker clean for the next job"
host_cleanup 0

rank=1
for ip in $(tr ',' '\n' <<<"$ips" | awk 'NF' | tail -n +2); do
  if [ -f "$key" ]; then
    timeout 300 ssh "${SSHO[@]}" "$user@$ip" \
      "CELL='$CELL' SKYRL_REPO_DIR='$repo' bash -s $rank" <<EOF 2>&1 | sed "s/^/[$ip] /" || echo "[$ip] cleanup ssh failed"
$(declare -f host_cleanup)
repo="\$SKYRL_REPO_DIR"
host_cleanup "\$1"
EOF
  else
    echo "[$ip] no ssh key at $key; engine host not cleaned"
  fi
  rank=$((rank + 1))
done
echo "cleanup: done (cell $CELL, rc $rc)"
