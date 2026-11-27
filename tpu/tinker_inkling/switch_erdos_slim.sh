#!/usr/bin/env bash
# One-shot (user 2026-11-27 02:15Z, "expand its pool to the broader cluster"): move the 5 Erdős arm runs from our 8
# full-hash ink-gfarms to the shared slim v6e math pool (GRADE_FARMS=v5p = farms-v5p.json). Each run is re-created right
# after it finishes its current step, so no step's work is lost; it resumes from that step's tree snapshot. Exits when done.
set -u
O=/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling/ops
R=/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling-runs
declare -A STEPS=([mf9-s5]=39 [mxe-t5a1]=13 [mxe-t5a2]=13 [mxe-t5a3]=13 [mxe-t5a4]=13)
declare -A SEED=([mf9-s5]=5 [mxe-t5a1]=1 [mxe-t5a2]=2 [mxe-t5a3]=3 [mxe-t5a4]=4)
declare -A N0 DONE=()
NDONE=0
steps() { ls "$R/$1/tinker_log/$1"/member_*/trajectories/*_step_*.jsonl.gz 2>/dev/null | wc -l; }
for r in "${!STEPS[@]}"; do N0[$r]=$(steps "$r"); done
echo "$(date -u +%FT%TZ) start: $(for r in "${!N0[@]}"; do printf '%s=%s ' "$r" "${N0[$r]}"; done)"
while (( NDONE < ${#STEPS[@]} )); do
  for r in "${!STEPS[@]}"; do
    [[ -n "${DONE[$r]:-}" ]] && continue
    n=$(steps "$r")
    if (( n > N0[$r] )) || (( n >= STEPS[$r] )); then
      if (( n < STEPS[$r] )); then
        env=$(systemctl --user show "tinker-run-$r" -p Environment --value)
        read -r -a KV <<< "$env"
        NEW=()
        for kv in "${KV[@]}"; do
          case "$kv" in GRADE_FARMS=*|SKYRL_OPEN_GRADING_HTTP_CONCURRENCY=*) ;; *) NEW+=("$kv");; esac
        done
        NEW+=(GRADE_FARMS=v5p SKYRL_OPEN_GRADING_HTTP_CONCURRENCY=8)
        echo "$(date -u +%FT%TZ) $r finished step $n -> re-create on the slim pool: ${NEW[*]}"
        bash "$O/run_service.sh" run "$r" "${STEPS[$r]}" "${SEED[$r]}" "${NEW[@]}"
      else
        echo "$(date -u +%FT%TZ) $r already done ($n steps); nothing to switch"
      fi
      DONE[$r]=1; NDONE=$((NDONE + 1))
    fi
  done
  sleep 30
done
echo "$(date -u +%FT%TZ) all 5 Erdős arm runs switched"
