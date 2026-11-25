#!/usr/bin/env bash
# Start a generation-only fork of each seed when it reaches step 12 (user 2026-11-25): copy the seed's step-12 PUCT tree
# (visit counts kept) into a new run dir and sample the seed's step-12 adapter (TTD_FROZEN_SAMPLER_PATH, no training client)
# for 13 steps (12 -> 25), same recipe + grading + API key as the seed. Erdős seeds mn9-sN -> mf9-sN, circuit mc-sN -> mcf-sN.
# Idempotent: a fork whose run dir exists is skipped. Run on della-vis1 under systemd (unit tinker-fork-watch).
O=/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling/ops
R=/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling-runs
C=/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling/code
FORK_AT=12
# seed run : fork run : API key : recipe args : pinned skyrl snapshot
SPECS="
mn9-s1:mf9-s1:2:TTD_RECIPE=tvg-fix GRADE_FARMS=ink:skyrl-edef388b25a3-g2bb4
mn9-s2:mf9-s2:2:TTD_RECIPE=tvg-fix GRADE_FARMS=ink:skyrl-edef388b25a3-g2bb4
mn9-s3:mf9-s3:3:TTD_RECIPE=tvg-fix GRADE_FARMS=ink:skyrl-edef388b25a3-g2bb4
mn9-s4:mf9-s4:1:TTD_RECIPE=tvg-fix GRADE_FARMS=ink:skyrl-edef388b25a3-g2bb4
mn9-s5:mf9-s5:1:TTD_RECIPE=tvg-fix GRADE_FARMS=ink:skyrl-edef388b25a3-g2bb4
mc-s1:mcf-s1:TAI:TTD_RECIPE=circuit-fix GRADE_FARMS=inkc:skyrl-edef388b25a3-tfsci
mc-s2:mcf-s2:TAI:TTD_RECIPE=circuit-fix GRADE_FARMS=inkc:skyrl-edef388b25a3-tfsci
mc-s3:mcf-s3:3:TTD_RECIPE=circuit-fix GRADE_FARMS=inkc:skyrl-edef388b25a3-tfsci
"
while true; do
  while IFS=: read -r SEED FORK KEY RECIPE SNAP; do
    [ -n "$SEED" ] || continue
    [ -d "$R/$FORK" ] && continue
    LOG="$R/$SEED/tinker_log/$SEED"
    TREE="$LOG/puct_sampler_step_$(printf %06d $FORK_AT).json"
    SP=$(python3 - "$LOG/member_inkling/checkpoints.jsonl" "$FORK_AT" <<'PY'
import json, sys
try:
    rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
except FileNotFoundError:
    rows = []
row = [r for r in rows if int(r.get('batch', -1)) == int(sys.argv[2]) and r.get('sampler_path')]
print(row[-1]['sampler_path'] if row else '')
PY
)
    [ -n "$SP" ] && [ -f "$TREE" ] || continue
    N=${SEED##*-s}
    echo "$(date -u +%FT%TZ) forking $SEED at step $FORK_AT -> $FORK (key $KEY, adapter $SP)"
    python3 "$O/import_tree.py" "$FORK" "$TREE" --keep-visits || { echo "import failed for $FORK"; continue; }
    printf 'PIN_SKYRL=%s\nPIN_DISCOVER=%s\n' "$C/$SNAP" "$C/discover-0766e2413eb0" > "$R/$FORK/code.pin"
    # shellcheck disable=SC2086
    bash "$O/run_service.sh" run "$FORK" $((25 - FORK_AT)) "$N" TTD_MODEL=mini $RECIPE LEARNING_RATE=0 KL_PENALTY_COEF=0 \
        TTD_FROZEN_POLICY=1 TTD_FROZEN_SAMPLER_PATH="$SP" TINKER_KEY="$KEY"
  done <<< "$SPECS"
  sleep 120
done
