#!/bin/bash
# Preemption-proof T2 collection: bring the farm up, collect with --resume, and repeat until
# the cell is complete. v5p-16 spot in us-east5-a is churning badly (qwen35: 3 preemptions in
# 4 attempts), so a single-shot job just dies mid-cell; this supervisor turns that into a
# retry loop where each round keeps every sample already on disk.
#
#   sbatch t2_supervised.sh <qwen35|gemma4> <problem> <out_dir> [n] [conc] [cell] [extra_body]
#
#SBATCH --job-name=rq1_t2sup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --exclude=neu301
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu-rq1/runs/rq1/logs/%x_%j.log
set -uo pipefail
KEY="$1"; PROBLEM="$2"; OUT="$3"; N="${4:-200}"; CONC="${5:-32}"; CELL="${6:-C}"; EXTRA="${7:-}"
RQ1=/n/fs/vision-mix/sk7524/SkyRLTpu-rq1/tpu/rq1
ROUNDS="${ROUNDS:-8}"

case "$KEY" in
  qwen35) TPU=sk7524-tunix-qwen35-v5p16-r1-east5a_spot; MODEL=Qwen/Qwen3.5-27B ;;
  gemma4) TPU=sk7524-tunix-gemma4-v5p16-r1-east5a_spot; MODEL=google/gemma-4-31B-it ;;
  *) echo "unknown model key $KEY" >&2; exit 2 ;;
esac

# Count only SUCCESSFUL samples: a failed request also leaves a raw/*.json (holding its error),
# so counting files would call a preempted cell complete (measured: ac1_D showed 200 files for
# 137 real samples). Parse the JSON rather than grepping for '"error"' -- generated code
# routinely contains that literal, which would undercount and trigger endless re-collection.
have() {
python3 - "$OUT" <<'PY'
import glob, json, sys
n = 0
for f in glob.glob(sys.argv[1] + "/raw/*.json"):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if not d.get("error") and ((d.get("text") or "") or (d.get("think") or "")):
        n += 1
print(n)
PY
}

for r in $(seq 1 "$ROUNDS"); do
  H=$(have)
  if [ "$H" -ge "$N" ]; then
    echo "[sup] $PROBLEM/$CELL complete: $H/$N samples"; exit 0
  fi
  echo "[sup] === round $r/$ROUNDS: $PROBLEM/$CELL at $H/$N samples ==="
  # farm_up is idempotent: reuses an ACTIVE slice, recreates a dead one, restarts vLLM.
  bash "$RQ1/jobs/farm_up.sh" "$KEY" || { echo "[sup] farm_up failed; retrying"; sleep 120; continue; }
  # t2_smoke.sh works as a plain script too (its #SBATCH lines are just comments).
  bash "$RQ1/jobs/t2_smoke.sh" "$TPU" "$MODEL" "$PROBLEM" "$OUT" "$N" "$CONC" "$CELL" "$EXTRA"
  echo "[sup] round $r finished with rc=$? at $(have)/$N samples"
done

H=$(have)
echo "[sup] EXHAUSTED $ROUNDS rounds at $H/$N samples"
[ "$H" -ge "$N" ] && exit 0 || exit 1
