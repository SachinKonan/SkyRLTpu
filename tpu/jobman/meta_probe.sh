#!/usr/bin/env bash
# completion_probe (workers: 0): all three members done? Member run dirs live
# on their TRAINER hosts (client-local), so gemma/muse are checked over ssh.
set -euo pipefail
: "${ARM:?}"; : "${GEN:?}"; : "${JOBMAN_TPU_INTERNAL_IPS:?}"
INT="$JOBMAN_TPU_INTERNAL_IPS"
ip_at() { echo "$INT" | cut -d, -f"$1"; }
KEY="$HOME/.ssh/jobman_tpu_ed25519"
SSHO="-i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15"
STEPS="${NUM_EPOCHS:-15}"
bad=""
for spec in "qwen:$(ip_at ${T_QWEN:-1})" "gemma:$(ip_at ${T_GEMMA:-5})" "muse:$(ip_at ${T_MUSE:-9})"; do
  tag=${spec%%:*}; trainer=${spec##*:}
  run="${ARM}-g${GEN}-${tag}"
  if [ "$trainer" = "$(ip_at 1)" ]; then
    out=$(RUN="$run" TARGET="$STEPS" bash "$HOME/SkyRLTpu-league/tpu/jobman/member_done.sh" 2>/dev/null) || bad="$bad $tag(${out##*: })"
  else
    out=$(timeout 60 ssh $SSHO sk7524_princeton_edu@"$trainer" \
      "RUN='$run' TARGET='$STEPS' bash \$HOME/SkyRLTpu-league/tpu/jobman/member_done.sh" 2>/dev/null) || bad="$bad $tag(${out##*: })"
  fi
done
[ -z "$bad" ] && { echo "complete: all members done"; exit 0; }
echo "incomplete:$bad"; exit 1
