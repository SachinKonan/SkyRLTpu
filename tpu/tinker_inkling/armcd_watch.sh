#!/usr/bin/env bash
# Explore-then-commit arms C and D for Inkling-2-Mini (user approval relayed by train-faster-ana, 2026-11-26 ~18:40Z).
# Inkling seed 5 has the best step-12 tree in every block, so the arms commit to it:
#   C: seed 5's frozen-at-12 run continued to step 51 (39 frozen steps): Erdős mf9-s5, circuit mcf-s5 (NUM_STEPS 13 -> 39).
#   D: seed 5's step-12 tree sampled by the frozen step-12 adapters of seeds 1-4, 12 -> 25 (13 steps):
#      Erdős mxe-t5a{1..4}, circuit mxc-t5a{1..4}. Each run uses its adapter's Tinker key (checkpoints are per org).
# Same recipe/grading as the existing frozen runs of each problem. Erdős grades on our v6e ink-gfarms (the v5p is gone; the
# user OK'd v6e for Erdős). Circuit D waits until the 5 trained circuit runs are done (~00Z). C relaunches a finished
# frozen run with the larger NUM_STEPS (it resumes from its last tree snapshot). Every 2 min, idempotent.
set -u
O=/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling/ops
R=/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling-runs
C=/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling/code
SNAP=skyrl-28b4c4aa866a-tfsci
# Erdős arms grade on the shared slim v6e math pool (farms-v5p.json; user 2026-11-27 02:15Z), not just our 8 ink-gfarms.
ERD="TTD_RECIPE=tvg-fix GRADE_FARMS=v5p SKYRL_OPEN_GRADING_HTTP_CONCURRENCY=8"
CIR="TTD_RECIPE=circuit-fix GRADE_FARMS=inkc SKYRL_OPEN_GRADING_HTTP_CONCURRENCY=3 INKC_FARMS_JSON=/scratch/gpfs/ZHUANGL/sk7524/rq4-local/tvg/inkling/ink-circuit-farms-plus.json"
# adapter seed : Tinker key : step-12 sampler path   (from fork_watch.log)
ERD_AD="1:2:tinker://1528f286-0fba-51df-88f8-1d1036cff970:train:0/sampler_weights/000012
2:2:tinker://256201a6-f952-5dfe-bce8-1b6644237d98:train:0/sampler_weights/000012
3:3:tinker://7fe9ea16-1dac-565a-ae1e-1bbacd52d0a3:train:0/sampler_weights/000012
4:1:tinker://3894a5b6-1d28-55c2-abc4-db52b6ebf3c3:train:0/sampler_weights/000012
5:1:tinker://307a1783-ac19-5274-941a-308a43108638:train:0/sampler_weights/000012"
CIR_AD="1:TAI:tinker://a904e838-7d09-5b1b-92b0-4912936a65d3:train:0/sampler_weights/000012
2:TAI:tinker://72b3d154-81a2-50c1-bf8d-372d38662242:train:0/sampler_weights/000012
3:3:tinker://0b2a6a2b-b3bc-5d71-8f1e-f4b165684a3a:train:0/sampler_weights/000012
4:HARISH:tinker://c5920f8a-d30f-5270-b4ae-56f3db9de06a:train:0/sampler_weights/000012
5:HARISH:tinker://b9352b42-346c-5f14-bafc-662d0219e950:train:0/sampler_weights/000012"

steps() { ls "$R/$1/tinker_log/$1"/member_*/trajectories/*_step_*.jsonl.gz 2>/dev/null | wc -l; }
active() { systemctl --user is-active --quiet "tinker-run-$1"; }
ad() { grep "^$2:" <<< "$1"; }   # adapter line for seed $2

launch_frozen() {  # name steps seed recipe key sampler_path
  # shellcheck disable=SC2086
  bash "$O/run_service.sh" run "$1" "$2" "$3" TTD_MODEL=mini $4 LEARNING_RATE=0 KL_PENALTY_COEF=0 \
      TTD_FROZEN_POLICY=1 TTD_FROZEN_SAMPLER_PATH="$6" TINKER_KEY="$5"
}

arm_c() {  # run recipe adapters
  local run=$1 n
  n=$(steps "$run")
  if (( n >= 13 && n < 39 )) && ! active "$run"; then
    local KEY SP
    KEY=$(ad "$3" 5 | cut -d: -f2); SP=$(ad "$3" 5 | cut -d: -f3-)
    echo "$(date -u +%FT%TZ) arm C: $run at $n frozen steps -> relaunch to 39 (key $KEY)"
    launch_frozen "$run" 39 5 "$2" "$KEY" "$SP"
  fi
}

arm_d() {  # prefix tree recipe adapters
  local s new KEY SP
  for s in 1 2 3 4; do
    new="$1-t5a$s"
    [ -d "$R/$new" ] && continue
    KEY=$(ad "$4" "$s" | cut -d: -f2); SP=$(ad "$4" "$s" | cut -d: -f3-)
    echo "$(date -u +%FT%TZ) arm D: $new = seed-5 step-12 tree x seed-$s adapter (key $KEY)"
    python3 "$O/import_tree.py" "$new" "$2" --keep-visits || { echo "import failed for $new"; continue; }
    printf 'PIN_SKYRL=%s\nPIN_DISCOVER=%s\n' "$C/$SNAP" "$C/discover-0766e2413eb0" > "$R/$new/code.pin"
    launch_frozen "$new" 13 "$s" "$3" "$KEY" "$SP"
  done
}

while true; do
  arm_c mf9-s5 "$ERD" "$ERD_AD"
  arm_d mxe "$R/mn9-s5/tinker_log/mn9-s5/puct_sampler_step_000012.json" "$ERD" "$ERD_AD"
  arm_c mcf-s5 "$CIR" "$CIR_AD"
  circ_done=1
  for i in 1 2 3 4 5; do
    if active "mc-s$i" || (( $(steps "mc-s$i") < 25 )); then circ_done=0; fi
  done
  if (( circ_done )); then
    arm_d mxc "$R/mc-s5/tinker_log/mc-s5/puct_sampler_step_000012.json" "$CIR" "$CIR_AD"
  fi
  sleep 120
done
