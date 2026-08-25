#!/usr/bin/env bash
# One-shot GCS push of every member run dir, executed on the host that owns it.
# (The per-member sidecars also push every 300s; this is the synchronous flush
# the monitor calls on exit paths.)
set -uo pipefail
: "${ARM:?}"; : "${GEN:?}"; : "${JOBMAN_TPU_INTERNAL_IPS:?}"
INT="$JOBMAN_TPU_INTERNAL_IPS"
ip_at() { echo "$INT" | cut -d, -f"$1"; }
KEY="$HOME/.ssh/jobman_tpu_ed25519"
SSHO="-i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15"
for spec in "qwen:$(ip_at ${T_QWEN:-1})" "gemma:$(ip_at ${T_GEMMA:-5})" "muse:$(ip_at ${T_MUSE:-9})"; do
  tag=${spec%%:*}; trainer=${spec##*:}
  run="${ARM}-g${GEN}-${tag}"
  cmd="[ -d ~/skyrl-runs/$run ] && gsutil -m rsync -r -x '.*wandb/.*|.*\.tmp$|.*\.gstmp$' ~/skyrl-runs/$run gs://sk7524-tinker-tpu-us-east5/skyrl-runs/$run >> ~/meta-sync.log 2>&1 || true"
  if [ "$trainer" = "$(ip_at 1)" ]; then bash -c "$cmd"
  else timeout 600 ssh $SSHO sk7524_princeton_edu@"$trainer" "$cmd" || echo "sync $tag failed"; fi
done
