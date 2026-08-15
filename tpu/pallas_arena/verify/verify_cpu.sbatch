#!/bin/bash
# CPU half of VERIFY-AND-FIX: baseline audit, adversarial re-grade of the
# above-1.0 rg_lru candidates, the failure taxonomy, and the repair experiment.
# No chips, no QRs, no network.
#
#SBATCH -p all
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH --job-name=arena-verify
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/pallas_arena/verify-%j.log

set -uo pipefail
repo_root=/n/fs/vision-mix/sk7524/SkyRLTpu
out_dir="${repo_root}/runs/pallas_arena"
mkdir -p "$out_dir"
cd "$repo_root"
export PYTHONPATH="${repo_root}/tpu:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu
J=${SLURM_JOB_ID:-manual}
RESULTS=${RESULTS:-runs/pallas_arena/seam-results-3651278.jsonl}
STAGES=${STAGES:-audit,verify,taxonomy,repair}
rc=0

run () {  # run <label> <args...>
  local label="$1"; shift
  case ",$STAGES," in *",$label,"*) ;; *) echo "== skip $label"; return 0;; esac
  echo; echo "=== $label $(date +%H:%M:%S) ==="
  uv run --isolated --extra dev --extra jax --with flatbuffers python "$@" 2>&1 | tail -400
  rc=$(( rc + ${PIPESTATUS[0]} ))
}

run audit    -m pallas_arena.verify.audit_baselines
run verify   -m pallas_arena.verify.verify_rglru_cpu \
  --results "$RESULTS" --min-reward 1.0 --full-grade \
  --out "${out_dir}/verify-rglru-${J}.json"
run taxonomy -m pallas_arena.verify.taxonomy \
  --results "$RESULTS" --out "${out_dir}/verify-taxonomy-${J}.json"
run repair   -m pallas_arena.verify.repair_fills \
  --results "$RESULTS" --out "${out_dir}/verify-repair-${J}.json"
run prove    -m pallas_arena.verify.prove_fixes \
  --out "${out_dir}/verify-prove-${J}.json"

# Can the REAL recurrentgemma scan be bound as the rg_lru denominator? Needs the
# package, which is NOT an arena dependency -- hence its own uv invocation.
case ",$STAGES," in *,rgbind,*)
  echo; echo "=== rgbind $(date +%H:%M:%S) ==="
  uv run --isolated --extra dev --extra jax --with flatbuffers \
    --with 'recurrentgemma==1.0.1' python -m pallas_arena.verify.rg_baseline_probe \
    --out "${out_dir}/verify-rgbind-${J}.json" 2>&1 | tail -120
  echo "rgbind rc=${PIPESTATUS[0]}"
  ;;
*) echo "== skip rgbind" ;;
esac

echo; echo "=== verify-cpu done rc=${rc} $(date +%H:%M:%S) ==="
exit "$rc"
