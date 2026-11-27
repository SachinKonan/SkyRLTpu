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
mn9-s1:mf9-s1:2:TTD_RECIPE=tvg-fix GRADE_FARMS=v5p:skyrl-28b4c4aa866a-tfsci
mn9-s2:mf9-s2:2:TTD_RECIPE=tvg-fix GRADE_FARMS=v5p:skyrl-28b4c4aa866a-tfsci
mn9-s3:mf9-s3:3:TTD_RECIPE=tvg-fix GRADE_FARMS=v5p:skyrl-28b4c4aa866a-tfsci
mn9-s4:mf9-s4:1:TTD_RECIPE=tvg-fix GRADE_FARMS=v5p:skyrl-28b4c4aa866a-tfsci
mn9-s5:mf9-s5:1:TTD_RECIPE=tvg-fix GRADE_FARMS=v5p:skyrl-28b4c4aa866a-tfsci
mc-s1:mcf-s1:TAI:TTD_RECIPE=circuit-fix GRADE_FARMS=inkc SKYRL_OPEN_GRADING_HTTP_CONCURRENCY=3 INKC_FARMS_JSON=/scratch/gpfs/ZHUANGL/sk7524/rq4-local/tvg/inkling/ink-circuit-farms-plus.json:skyrl-28b4c4aa866a-tfsci
mc-s2:mcf-s2:TAI:TTD_RECIPE=circuit-fix GRADE_FARMS=inkc SKYRL_OPEN_GRADING_HTTP_CONCURRENCY=3 INKC_FARMS_JSON=/scratch/gpfs/ZHUANGL/sk7524/rq4-local/tvg/inkling/ink-circuit-farms-plus.json:skyrl-28b4c4aa866a-tfsci
mc-s3:mcf-s3:3:TTD_RECIPE=circuit-fix GRADE_FARMS=inkc SKYRL_OPEN_GRADING_HTTP_CONCURRENCY=3 INKC_FARMS_JSON=/scratch/gpfs/ZHUANGL/sk7524/rq4-local/tvg/inkling/ink-circuit-farms-plus.json:skyrl-28b4c4aa866a-tfsci
mc-s4:mcf-s4:HARISH:TTD_RECIPE=circuit-fix GRADE_FARMS=inkc SKYRL_OPEN_GRADING_HTTP_CONCURRENCY=3 INKC_FARMS_JSON=/scratch/gpfs/ZHUANGL/sk7524/rq4-local/tvg/inkling/ink-circuit-farms-plus.json:skyrl-28b4c4aa866a-tfsci
mc-s5:mcf-s5:HARISH:TTD_RECIPE=circuit-fix GRADE_FARMS=inkc SKYRL_OPEN_GRADING_HTTP_CONCURRENCY=3 INKC_FARMS_JSON=/scratch/gpfs/ZHUANGL/sk7524/rq4-local/tvg/inkling/ink-circuit-farms-plus.json:skyrl-28b4c4aa866a-tfsci
"
# Deferred TRAINED seeds (user 2026-11-25: circuit has 5 seeds, the 8 free training slots are full): start each when the
# run holding its key's slot has finished all 25 steps. new seed : seed number : key : run whose finish frees the slot
DEFER="
# (mc-s4 started directly on the HARISH key 2026-11-25)
# (mc-s5 started directly on the HARISH key 2026-11-25)
"
while true; do
  # Circuit runs grade on our 8 ink-cfarms + train-faster's idle circuit farms (user OK 2026-11-26 00:00Z); a circuit run
  # (re)launch rewrites registry-inkc.json from its own list, so re-assert the 70-farm list every loop.
  # The tunnel maps ports by position in the same list, so whenever the farm set changes the tunnel must restart too,
  # or the registry's names point at the wrong farms (2026-11-26: train-faster rebuilds the plus list as farms move).
  RW=$(python3 "$O/write_registry.py" /scratch/gpfs/ZHUANGL/sk7524/rq4-local/tvg/inkling/ink-circuit-farms-plus.json \
      "$O/registry-inkc.json" 27880)
  if [[ -n "$RW" ]]; then echo "$RW"; systemctl --user restart tinker-tunnel-inkc && echo "$(date -u +%FT%TZ) restarted tinker-tunnel-inkc for the new farm list"; fi
  # Inkling Erdős grading pool (user 2026-11-27 02:15Z, "expand its pool to the broader cluster"): train-faster's slim v6e
  # math farms, farms-v5p.json (their registry_sync keeps it current; eftt farms join as they come up). Used by the arms
  # C/D Erdős runs (GRADE_FARMS=v5p) and the MOMO Erdős shim. On a change: rewrite registry-v5p.json and restart its
  # tunnel together (ports are positional).
  IE=$(python3 "$O/write_registry.py" /scratch/gpfs/ZHUANGL/sk7524/rq4-local/tvg/farms-v5p.json "$O/registry-v5p.json" 28000)
  if [[ -n "$IE" ]]; then
    echo "$IE"; systemctl --user restart tinker-tunnel-v5p && echo "$(date -u +%FT%TZ) restarted tinker-tunnel-v5p for the new farm list"
  fi
  while IFS=: read -r NEW N KEY AFTER; do
    [ -n "$NEW" ] || continue
    [ -f "$R/$NEW/env.txt" ] && continue
    done_steps=$(cat "$R/$AFTER/tinker_log/$AFTER/metrics.jsonl" 2>/dev/null | wc -l)
    if [ "$done_steps" -ge 25 ] && ! systemctl --user is-active --quiet "tinker-run-$AFTER"; then
      echo "$(date -u +%FT%TZ) $AFTER finished: starting $NEW (seed $N, key $KEY)"
      mkdir -p "$R/$NEW"
      printf 'PIN_SKYRL=%s\nPIN_DISCOVER=%s\n' "$C/skyrl-edef388b25a3-tfsci" "$C/discover-0766e2413eb0" > "$R/$NEW/code.pin"
      bash "$O/run_service.sh" run "$NEW" 25 "$N" TTD_MODEL=mini TTD_RECIPE=circuit-fix GRADE_FARMS=inkc \
          LEARNING_RATE=5.4e-5 KL_PENALTY_COEF=0 TINKER_KEY="$KEY"
    fi
  done <<< "$DEFER"
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
  # Every ~30 min: delete live runs' checkpoints for steps < 5, keeping each run's two newest steps (prune_early.py).
  LOOPS=$(( ${LOOPS:-0} + 1 ))
  if (( LOOPS % 15 == 1 )); then
    timeout 1200 /scratch/gpfs/ZHUANGL/sk7524/tinker-inkling/venv/bin/python "$O/prune_early.py" 2>&1 \
      | grep -E "^key|FAILED" | sed "s/^/$(date -u +%FT%TZ) prune: /"
  fi
  # Hourly: per-key checkpoint-storage alert at $10/month of our checkpoints (storage_check.py; user 2026-11-26).
  if (( LOOPS % 30 == 2 )); then
    timeout 900 /scratch/gpfs/ZHUANGL/sk7524/tinker-inkling/venv/bin/python "$O/storage_check.py" --alert 10 2>&1 \
      | grep -E "^ALERT|^\{" | sed "s/^/$(date -u +%FT%TZ) storage: /"
  fi
  # Every ~6 min: stall watchdog (user 2026-11-26): email on grading/step stalls or tunnel fd exhaustion; a run that is
  # grading with no completion for 40 min is restarted (resumes from its last checkpoint), at most once per 2 h.
  if (( LOOPS % 3 == 0 )); then
    timeout 300 python3 "$O/stall_watch.py" 2>&1 | sed "s/^/$(date -u +%FT%TZ) stall: /"
  fi
  sleep 120
done
