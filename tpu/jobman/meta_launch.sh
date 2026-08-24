#!/usr/bin/env bash
# Launch (idempotently) the three meta-member clients ON W0.
# Each member = a 1-member ensemble client (launch_cell.sh) with:
#   - its own run dir  ${ARM}-g${GEN}-{tag}   (seed pre-staged in GCS by driver)
#   - TTD_M0_BASE_URL at its member's trainer host
#   - reregister routed to that trainer host (REREG_HOST)
#   - flatline stop on, LR 1.5e-4, per-member sick marker
#   - carry arms: TTD_INIT_STATE_PATH_<TAG> + prev-gen jsonl reregister
set -uo pipefail
: "${ARM:?}"; : "${GEN:?}"; : "${JOBMAN_TPU_INTERNAL_IPS:?}"
INT="$JOBMAN_TPU_INTERNAL_IPS"
ip_at() { echo "$INT" | cut -d, -f"$1"; }
export PATH="$HOME/.local/bin:$PATH"
STEPS="${NUM_EPOCHS:-15}"

launch_one() {  # $1 tag  $2 cell-alias  $3 trainer-ip  $4 env-suffix (QWEN/GEMMA/MUSE)
  local tag="$1" cell="$2" trainer="$3" suf="$4"
  local run="${ARM}-g${GEN}-${tag}" sess="cell-$tag"
  # complete? (CONVERGED or budget reached) -> leave alone
  if RUN_ONE="$run" STEPS="$STEPS" python3 - <<'PY'
import glob, json, os, sys
run, target = os.environ["RUN_ONE"], int(os.environ["STEPS"])
if glob.glob(os.path.expanduser(f"~/skyrl-runs/{run}/tinker_log/*/CONVERGED")): sys.exit(0)
latest = None
for p in glob.glob(os.path.expanduser(f"~/skyrl-runs/{run}/tinker_log/*/member_*/checkpoints.jsonl")):
    for line in open(p):
        line=line.strip()
        if not line: continue
        try: row=json.loads(line)
        except ValueError: continue
        b=row.get("batch")
        if isinstance(b,int): latest=b if latest is None else max(latest,b)
        if row.get("name")=="final": sys.exit(0)
sys.exit(0 if (latest is not None and latest>=target) else 1)
PY
  then echo "member $tag: complete -- not relaunching"; return 0; fi
  tmux has-session -t "$sess" 2>/dev/null && { echo "member $tag: client already up"; return 0; }

  local extra="TTD_FLATLINE_STOP=1 TTD_FLATLINE_EPS=1e-9 TTD_FLATLINE_CONSEC=3 TTD_FLATLINE_MIN_STEPS=4"
  extra="$extra TTD_SICK_MARKER=\$HOME/ENGINE-SICK-$tag TTD_RESTART_RATIO=0 TTD_RESTART_AT_STEP=-1"
  # weights-carry (set by driver in jobman env for carry arms)
  local init_sp_var="INIT_SP_${suf}" init_jsonl_var="INIT_JSONL_GCS_${suf}"
  local init_sp="${!init_sp_var:-}" init_jsonl_gcs="${!init_jsonl_var:-}" extra_jsonl=""
  if [ -n "$init_sp" ]; then
    extra="$extra TTD_INIT_STATE_PATH_${suf}=$init_sp"
    if [ -n "$init_jsonl_gcs" ]; then
      gsutil -q cp "$init_jsonl_gcs" ~/init_prev_${tag}.jsonl 2>/dev/null \
        && extra_jsonl=~/init_prev_${tag}.jsonl \
        || echo "member $tag: WARNING could not fetch prev-gen jsonl ($init_jsonl_gcs)"
    fi
  fi

  CELL="$cell" \
  EXPERIMENT_NAME="$run" RUN_DIR_NAME="$run" \
  CELL_SESSION="$sess" \
  TTD_M0_BASE_URL="http://$trainer:8000" \
  REREG_HOST="$([ "$trainer" = "$(ip_at 1)" ] && echo local || echo "$trainer")" \
  EXTRA_REREG_JSONL="$extra_jsonl" \
  EXTRA_TTD_ENV="$extra" \
  RAY_ADDRESS="$trainer:6379" \
  LEARNING_RATE=1.5e-4 NUM_EPOCHS="$STEPS" \
  TTD_ELITE_SLOTS=2 GROUPS_PER_BATCH=16 GROUP_SIZE=32 \
  bash "$HOME/ttd-client/tpu/launch_cell.sh"
}

launch_one qwen  meta-qwen "$(ip_at 1)" QWEN
launch_one gemma g-meta    "$(ip_at 5)" GEMMA
launch_one muse  m-meta    "$(ip_at 9)" MUSE
