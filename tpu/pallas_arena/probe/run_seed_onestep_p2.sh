#!/usr/bin/env bash
# Re-run the clean 2x2 under the PROMPT-V2 (constraints + lib imports).
#
# This is an A/B against the run of 2026-08-28, so everything except the
# prompt must be byte-identical to it. Deliberately NOT run_seed_onestep.sh:
# that script's steps 1-3 regrade the seeds and REWRITE seed-obs-*.txt and
# seed-reward-*.txt, which are pasted verbatim into the prompt. Regenerating
# them would change the prompt body underneath the comparison and there would
# be no way to attribute a difference to the constraints text. The seed files
# are therefore treated as fixed inputs here and never touched.
#
# Held identical to the baseline (each verified from the baseline's own
# artefacts, not assumed):
#   ctx 32768              vLLM max_seq_len in both engine logs
#   THINK_BUDGET 12288     live supervisor env + phase1 completion_tokens
#   GROUP_SIZE 32          32 rows banked per cell
#   v6e-8, SERVE_TP 8      tensor_parallel_size=8 in both engine logs
#   XLA cache prefixes     the exact gs:// paths the baseline logged
#   seed files / obs / reward   untouched on disk
# Changed: LIB_IMPORTS=1, which adds "## Rules your rewrite must satisfy" and
# "## Reusing what you are not changing" and switches the Output contract to
# the one that permits `from lib import ...`.
set -uo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
cd "$REPO"
S=$REPO/tpu/pallas_arena/probe

for f in "$S/seed_splash_flash.py" "$S/seed_rglru_active.py" \
         runs/pallas_arena/seed-obs-splash.txt runs/pallas_arena/seed-obs-rglru.txt \
         runs/pallas_arena/seed-reward-splash.txt runs/pallas_arena/seed-reward-rglru.txt; do
  [ -s "$f" ] || { echo "FATAL: missing prompt input $f"; exit 1; }
done

launch() {  # launch <tag> <cells> <seedfile> <obstag> [gemma]
  ( export SUP_TAG="$1" SMOKE_CELLS="$2" SEED_FILE="$3"
    export SEED_OBS="$REPO/runs/pallas_arena/seed-obs-$4.txt"
    export SEED_REWARD="$(cat "$REPO/runs/pallas_arena/seed-reward-$4.txt")"
    export LIB_IMPORTS=1
    export THINK_BUDGET=12288 QWEN_MAXLEN=32768 GROUP_SIZE=32
    export ACCEL=v6e-8 SERVE_TP=8
    # RUNTIME MUST MATCH THE CHIP. Both sbatch files default to
    # v2-alpha-tpuv5, so every v6e request that does not set this asks for
    # v6e hardware with a v5 runtime image. judge3, which served for 20h,
    # had runtimeVersion v2-alpha-tpuv6e; judge8 (unset -> v5) sat stuck and
    # the two engine failures reported num_chips=0 / 'Failed to get global
    # TPU topology' -- exactly what a mismatched runtime would produce.
    export RUNTIME=v2-alpha-tpuv6e
    # v6e-8 is PERMITTED ONLY IN us-east5-b for this project (measured
    # 2026-08-28: us-east5-a refuses with code 7 "does not have permission
    # to submit requests into this queue"). The default rotation starts in
    # us-east5-a, which cost the first p2 launch 50 minutes of empty retries.
    export BENCH_ZONES="us-east5-b"
    # HOLD THE REQUEST, DO NOT CHURN IT. The default 3h landing window made
    # each arm delete and re-create its QR every 3 hours, which sends it to
    # the back of the queue. Measured 2026-08-30 with 19 requests ahead of us
    # zone-wide: judge8, which simply waited 12.5h, reached PROVISIONING while
    # all four arms -- recreating on schedule -- sat at WAITING_FOR_RESOURCES.
    # A queued request advances only if it is left alone.
    export LAND_DEADLINE_S=64800 LAND_GRACE_TRIES=12
    export BUCKET=gs://sk7524-tinker-tpu-us-east5
    if [ "${5:-}" = "gemma" ]; then
      export QWEN_MODEL=google/gemma-4-31B-it
      export QWEN_HF_GCS="${BUCKET}/hf-cache-gemma4"
      export QWEN_XLA_GCS="${BUCKET}/vllm-xla-cache-gemma4-31b-32k-v6e8-tp8"
      export EXTRA_PIP="transformers==5.14.0" ENABLE_THINKING_KWARG=1
    else
      export QWEN_MODEL=Qwen/Qwen3.5-27B
      export QWEN_HF_GCS="${BUCKET}/hf-cache"
      export QWEN_XLA_GCS="${BUCKET}/vllm-xla-cache-qwen35-32k-b16-v6e8-tp8"
    fi
    nohup bash "$S/bench_supervisor.sh" > "runs/pallas_arena/bench-sup-$1.log" 2>&1 &
    echo "launched $1 (model=$QWEN_MODEL cells=$SMOKE_CELLS lib=on)" )
}

launch qwen-p2-splash   "splash_attention:rf3s" "$S/seed_splash_flash.py"  splash
launch gemma-p2-splash  "splash_attention:rf3s" "$S/seed_splash_flash.py"  splash gemma
launch qwen-p2-rglru    "rg_lru:rf3s"           "$S/seed_rglru_active.py"  rglru
launch gemma-p2-rglru   "rg_lru:rf3s"           "$S/seed_rglru_active.py"  rglru gemma
echo "all four prompt-v2 arms launched $(date +%H:%M:%S)"
