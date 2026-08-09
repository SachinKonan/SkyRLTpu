#!/usr/bin/env bash
# Stage A cell definitions -- the single source of truth for the 2x3 factorial.
#
#   {GRPO, TTD-Discover} x {N: no regularizer, K: KL penalty, R: decay restart}
#
# Sourced by bringup_v5p32_cell.sh and by the fleet driver. Each cell runs a
# single qwen3.5-27B on its own v5p-32 (w0 trainer + w1-3 vLLM), which is the
# exact topology the ~2.0-2.4 h/step measurements came from.
#
# Batch shapes are fixed PER OBJECTIVE and constant across that objective's three
# regularizer arms, so the regularization contrast -- the primary question -- is
# unconfounded. Only cross-objective comparison inherits the difference.
#   GRPO           16x32 = 512 rollouts/step, elite slots 2
#   TTD-Discover    8x32 = 256 rollouts/step, elite slots 0, adaptive beta
#
# cell|adv_estimator|elite|groups|group_size|kl|restart_ratio
STAGE_A_CELLS=(
  "grpo-n|mean_baseline|2|16|32|0|0"
  "grpo-k|mean_baseline|2|16|32|0.1|0"
  "grpo-r|mean_baseline|2|16|32|0|100"
  "ttd-n|entropic_adaptive_beta|0|8|32|0|0"
  "ttd-k|entropic_adaptive_beta|0|8|32|0.1|0"
  "ttd-r|entropic_adaptive_beta|0|8|32|0|100"
)

# ttd-k is TTD-Discover's published vanilla setting; ttd-n is the new ablation
# that asks whether their KL is load-bearing or incidental. grpo-n is what the
# fleet has been running. grpo-r/ttd-r make the accidental restart deliberate,
# fired by decay rather than on a schedule (fixed K was rejected on evidence:
# one arm had decayed 692x by step 4, another only 14x by step 7).

stage_a_env() {   # usage: stage_a_env <cell>   -> prints KEY=VAL lines
  local want=$1 row
  for row in "${STAGE_A_CELLS[@]}"; do
    IFS='|' read -r c adv elite gpb gsz kl rr <<<"$row"
    [ "$c" = "$want" ] || continue
    echo "TTD_ADV_ESTIMATOR=$adv"
    echo "TTD_ELITE_SLOTS=$elite"
    echo "GROUPS_PER_BATCH=$gpb"
    echo "GROUP_SIZE=$gsz"
    echo "KL_PENALTY_COEF=$kl"
    echo "TTD_RESTART_RATIO=$rr"
    return 0
  done
  echo "unknown cell: $want" >&2
  return 1
}
