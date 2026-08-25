#!/usr/bin/env bash
# Meta-tree generation driver (login node). One instance per arm.
#
#   meta_driver.sh <ARM> <OP> <GENERATIONS> <WEIGHTS>
#     ARM      e.g. meta-wt16-fresh          (also the run-dir prefix)
#     OP       winner-top16 | mix-top16      (seed operator)
#     GENS     total generations, e.g. 4
#     WEIGHTS  fresh | carry                 (carry = continue_with_fresh_optimizer)
#
# Per generation: build seed from gen-1 final trees (validated C5, lineage-
# deduped, parents stripped) -> upload to all 3 member run dirs -> create the
# jobman job (v5p-128) -> poll GCS until all members done -> next.
# Resumable: state in ~/meta-driver/<ARM>/state.json; safe to re-nohup.
set -uo pipefail
ARM=${1:?arm}; OP=${2:?op}; GENS=${3:?generations}; WEIGHTS=${4:?fresh|carry}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GCS=gs://sk7524-tinker-tpu-us-east5
JOBMAN=~/.venvs/jobman/bin/jobman
SD=~/meta-driver/$ARM; mkdir -p "$SD"
STEPS=${STEPS:-15}
PYBIN="$REPO/third_party/discover/.venv/bin/python"; [ -x "$PYBIN" ] || PYBIN=python3
log() { echo "($(date '+%F %T')) [$ARM] $*" | tee -a "$SD/driver.log"; }

member_done() {  # $1 run -> 0 if done
  gsutil -q stat "$GCS/skyrl-runs/$1/tinker_log/$1/CONVERGED" 2>/dev/null && return 0
  local j; j=$(mktemp)
  gsutil -q cp "$GCS/skyrl-runs/$1/tinker_log/$1/member_*/checkpoints.jsonl" "$j" 2>/dev/null || { rm -f "$j"; return 1; }
  RES=$(J="$j" T="$STEPS" python3 -c '
import json,os
latest=None; final=False
for l in open(os.environ["J"]):
    l=l.strip()
    if not l: continue
    try: r=json.loads(l)
    except ValueError: continue
    b=r.get("batch")
    if isinstance(b,int): latest=b if latest is None else max(latest,b)
    if r.get("name")=="final": final=True
print("done" if (final or (latest is not None and latest>=int(os.environ["T"]))) else "no")')
  rm -f "$j"; [ "$RES" = done ]
}

gen_done() { local g=$1 t; for t in qwen gemma muse; do member_done "$ARM-g$g-$t" || return 1; done; }

fetch_final_snap() {  # $1 run  $2 dest -> 0 on success
  local snap
  snap=$(gsutil ls "$GCS/skyrl-runs/$1/tinker_log/$1/puct_sampler_step_*.json" 2>/dev/null | sort | tail -1)
  [ -n "$snap" ] && gsutil -q cp "$snap" "$2"
}

last_state_path() {  # $1 run -> prints "state_path" of last checkpoint row
  local j; j=$(mktemp)
  gsutil -q cp "$GCS/skyrl-runs/$1/tinker_log/$1/member_*/checkpoints.jsonl" "$j" 2>/dev/null || { rm -f "$j"; return 1; }
  J="$j" python3 -c '
import json,os
sp=""
for l in open(os.environ["J"]):
    l=l.strip()
    if not l: continue
    try: r=json.loads(l)
    except ValueError: continue
    sp=r.get("state_path") or sp
print(sp)'
  rm -f "$j"
}

for (( g=0; g<GENS; g++ )); do
  if gen_done "$g"; then log "gen $g already complete"; continue; fi

  # ---- seed (gen>0) ----------------------------------------------------------
  if [ "$g" -gt 0 ] && [ ! -f "$SD/seed_g$g.done" ]; then
    log "building gen-$g seed ($OP) from gen-$((g-1)) trees"
    trees=()
    for t in qwen gemma muse; do
      if fetch_final_snap "$ARM-g$((g-1))-$t" "$SD/g$((g-1))_$t.json"; then
        trees+=("--tree" "$t=$SD/g$((g-1))_$t.json")
      else
        log "WARNING: no final snapshot for $t in gen $((g-1)) (member ineligible)"
      fi
    done
    [ ${#trees[@]} -gt 0 ] || { log "FATAL: no member produced a tree; refusing to advance"; exit 1; }
    "$PYBIN" "$REPO/tpu/meta/build_meta_seed.py" --op "$OP" --k 16 \
      --out "$SD/seed_g$g.json" "${trees[@]}" | tee -a "$SD/driver.log" || { log "FATAL: seed build failed"; exit 1; }
    for t in qwen gemma muse; do
      run="$ARM-g$g-$t"
      gsutil -q cp "$SD/seed_g$g.json" \
        "$GCS/skyrl-runs/$run/tinker_log/$run/puct_sampler_step_000000.json" || { log "FATAL: seed upload $t"; exit 1; }
    done
    touch "$SD/seed_g$g.done"
  fi

  # ---- create the generation job --------------------------------------------
  if [ ! -f "$SD/job_g$g.created" ]; then
    initargs=()
    if [ "$WEIGHTS" = carry ] && [ "$g" -gt 0 ]; then
      for t in qwen gemma muse; do
        prev="$ARM-g$((g-1))-$t"
        sp=$(last_state_path "$prev" || true)
        if [ -n "${sp:-}" ]; then
          jl="$GCS/skyrl-runs/$prev/tinker_log/$prev/member_$t/checkpoints.jsonl"
          initargs+=("--init" "$t=$sp,$jl")
        else
          log "WARNING: no carry state for $t (fresh weights this gen)"
        fi
      done
    fi
    # spares: STATIC for the whole arm (META_SPARE_FOR, default qwen -- the
    # sampling-dominated member). Per-generation reassignment was removed: on a
    # persistent slice the boundary skips healthy engines, so moving spares
    # would orphan the old member's engines on w12-15 (its registry still lists
    # them, the new member's never learns them) unless BOTH engine stacks are
    # recycled -- a 30-60 min hit per boundary for an unmeasured gain.
    spare="${META_SPARE_FOR:-qwen}"
    # META_ACCEL + META_LAYOUT_ENV (comma list of K=V) select the slice shape;
    # defaults reproduce the v5p-128 layout.
    extraargs=()
    for kv in $(echo "${META_LAYOUT_ENV:-}" | tr , " "); do extraargs+=("--extra-env" "$kv"); done
    python3 "$REPO/tpu/meta/gen_meta_config.py" \
      --template "${META_TEMPLATE:?set META_TEMPLATE to a stageC cell config.yaml}" \
      --arm "$ARM" --gen "$g" --spare-for "$spare" --steps "$STEPS" \
      --accel "${META_ACCEL:-v5p-128}" \
      "${extraargs[@]}" "${initargs[@]}" --out "$SD/cfg_g$g.yaml" || { log "FATAL: config gen"; exit 1; }
    log "creating gen-$g job (spares -> $spare)"
    "$JOBMAN" create "$SD/cfg_g$g.yaml" >> "$SD/driver.log" 2>&1 &
    touch "$SD/job_g$g.created"
  fi

  # ---- wait ------------------------------------------------------------------
  log "waiting on gen $g"
  until gen_done "$g"; do sleep 300; done
  log "gen $g complete"
done

log "ALL $GENS generations complete"
for (( g=0; g<GENS; g++ )); do
  for t in qwen gemma muse; do
    fetch_final_snap "$ARM-g$g-$t" "$SD/final_g${g}_$t.json" 2>/dev/null || true
  done
done
"$PYBIN" - "$SD" "$GENS" <<'PYEOF' | tee -a "$SD/driver.log"
import glob, json, sys
import numpy as np
sd, gens = sys.argv[1], int(sys.argv[2])
def true_c5(h):
    h=np.array(h,dtype=float); n=len(h)
    if n==0 or not np.all(np.isfinite(h)) or np.any(h<0) or np.any(h>1): return None
    t=n/2.0; dx=2.0/n
    if h.sum()!=t:
        if h.sum()<=0: return None
        h=h*(t/h.sum())
        if np.any(h<0) or np.any(h>1): return None
    return float((np.correlate(h,1.0-h,mode="full")*dx).max())
print("=== validated bests per generation ===")
for g in range(gens):
    row=[]
    for f in sorted(glob.glob(f"{sd}/final_g{g}_*.json")):
        tag=f.split("_")[-1][:-5]
        best=None
        for s in json.load(open(f)).get("states") or []:
            c=s.get("construction")
            if not c: continue
            t=true_c5(c)
            if t is not None and (best is None or t<best): best=t
        row.append(f"{tag}={best:.9f}" if best else f"{tag}=none")
    print(f"g{g}: " + "  ".join(row))
PYEOF
