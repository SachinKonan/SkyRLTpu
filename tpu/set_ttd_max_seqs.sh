#!/usr/bin/env bash
# Change SAMPLE_MAX_NUM_SEQUENCES for the running TTD full run.
#
# Kills the current Tinker API on the TPU so the supervisor redeploys it at the
# new value; the discover client auto-restarts and resumes from its checkpoint.
#
# The run's tmux sessions live on one login node (RUN_NODE); the Claude/Bash
# shell can land on a different login node, so tmux ops are pinned to RUN_NODE to
# avoid spawning a duplicate supervisor on the wrong node.
#
# Usage: bash tpu/set_ttd_max_seqs.sh <N>
set -uo pipefail

MAXSEQ="${1:?usage: set_ttd_max_seqs.sh <N>}"
RUN_NODE="${RUN_NODE:-della9}"
REPO="${REPO:-/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-ttd-discover}"
TPU="${TPU_NAME:-sk7524-ttd-erdos-v5p16-east5a_spot}"
KEY="${SSH_KEY_FILE:-$HOME/.ssh/jobman_tpu_ed25519}"

# 1. Kill the API on the TPU (node-independent) so the supervisor triggers a redeploy.
gcloud alpha compute tpus tpu-vm ssh "sk7524_princeton_edu@${TPU}" \
  --project=vision-mix --zone=us-east5-a --worker=0 --ssh-key-file="$KEY" --quiet \
  --command 'tmux kill-session -t skyrl-tinker 2>/dev/null; echo api-killed' 2>/dev/null | tail -1

# 2. Restart the supervisor at the new max_seqs, on the node holding the run.
ops="cd $REPO; for s in ttd-supervise ttd-client ttd-tunnel; do tmux kill-session -t \$s 2>/dev/null; done; tmux new-session -d -s ttd-supervise \"cd $REPO && SAMPLE_MAX_NUM_SEQUENCES=$MAXSEQ bash tpu/supervise_ttd_run.sh 2>&1 | tee -a runs/ttd_erdos_v5p16/logs/supervise-full.log\"; echo restarted-on-\$(hostname -s)"
if [ "$(hostname -s)" = "$RUN_NODE" ]; then
  bash -c "$ops"
else
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$RUN_NODE" "$ops"
fi

echo "SAMPLE_MAX_NUM_SEQUENCES set to ${MAXSEQ}; supervisor redeploying API (~4-5 min), client will auto-resume."
