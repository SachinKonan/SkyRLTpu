#!/usr/bin/env bash
# Launch (idempotently) the three meta-member clients, EACH ON ITS OWN TRAINER
# HOST -- the proven stage-cell topology: client next to its tinker server and
# its Ray head, so every address is 127.0.0.1 and reregister is local. (The
# first design ran all clients on w0; cross-host ray.init from a non-cluster
# node hangs in the raylet handshake -- observed live on gen-0 gemma.)
set -uo pipefail
: "${ARM:?}"; : "${GEN:?}"; : "${JOBMAN_TPU_INTERNAL_IPS:?}"
INT="$JOBMAN_TPU_INTERNAL_IPS"
ip_at() { echo "$INT" | cut -d, -f"$1"; }
export PATH="$HOME/.local/bin:$PATH"
STEPS="${NUM_EPOCHS:-15}"
KEY="$HOME/.ssh/jobman_tpu_ed25519"
SSHO="-i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20"

launch_one() {  # $1 tag  $2 cell-alias  $3 trainer-ip  $4 env-suffix
  local tag="$1" cell="$2" trainer="$3" suf="$4"
  local run="${ARM}-g${GEN}-${tag}" sess="cell-$tag"
  local is_local=0; [ "$trainer" = "$(ip_at 1)" ] && is_local=1
  rexec() {  # run a command on the member's trainer host
    if [ "$is_local" = 1 ]; then bash -c "$1"; else
      timeout "${2:-120}" ssh $SSHO sk7524_princeton_edu@"$trainer" "$1"; fi
  }
  if rexec "RUN='$run' TARGET='$STEPS' bash \$HOME/SkyRLTpu-league/tpu/jobman/member_done.sh" 60 >/dev/null 2>&1; then
    echo "member $tag: complete -- not relaunching"; return 0
  fi
  if rexec "tmux has-session -t '$sess' 2>/dev/null" 60; then
    echo "member $tag: client already up"; return 0
  fi

  local extra="TTD_FLATLINE_STOP=1 TTD_FLATLINE_EPS=1e-9 TTD_FLATLINE_CONSEC=3 TTD_FLATLINE_MIN_STEPS=4"
  extra="$extra TTD_SICK_MARKER=\$HOME/ENGINE-SICK-$tag TTD_RESTART_RATIO=0 TTD_RESTART_AT_STEP=-1"
  local init_sp_var="INIT_SP_${suf}" init_jsonl_var="INIT_JSONL_GCS_${suf}"
  local init_sp="${!init_sp_var:-}" init_jsonl_gcs="${!init_jsonl_var:-}" fetch_jsonl=""
  if [ -n "$init_sp" ]; then
    extra="$extra TTD_INIT_STATE_PATH_${suf}=$init_sp"
    if [ -n "$init_jsonl_gcs" ]; then
      # fetch the prev-gen jsonl ON the trainer host, register locally there
      rexec "gsutil -q cp '$init_jsonl_gcs' ~/init_prev_${tag}.jsonl" 120 \
        && fetch_jsonl="\$HOME/init_prev_${tag}.jsonl" \
        || echo "member $tag: WARNING could not fetch prev-gen jsonl"
    fi
  fi

  rexec "export PATH=\$HOME/.local/bin:\$PATH
    ln -sfn \$HOME/SkyRLTpu-league \$HOME/ttd-client
    CELL='$cell' EXPERIMENT_NAME='$run' RUN_DIR_NAME='$run' \
    CELL_SESSION='$sess' \
    EXTRA_REREG_JSONL=\"${fetch_jsonl}\" \
    EXTRA_TTD_ENV='$extra' \
    LEARNING_RATE=1.5e-4 NUM_EPOCHS='$STEPS' \
    TTD_ELITE_SLOTS=2 GROUPS_PER_BATCH=16 GROUP_SIZE=32 \
    bash \$HOME/ttd-client/tpu/launch_cell.sh" 600
}

launch_one qwen  meta-qwen "$(ip_at ${T_QWEN:-1})" QWEN
launch_one gemma g-meta    "$(ip_at ${T_GEMMA:-5})" GEMMA
launch_one muse  m-meta    "$(ip_at ${T_MUSE:-9})" MUSE
