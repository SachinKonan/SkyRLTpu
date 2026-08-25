#!/usr/bin/env bash
# Meta-tree driver v3 (login node): lightweight control loop over PER-MEMBER
# jobman units. Each member = one long-lived jobman job on its own v5p-64
# (1 trainer + 7 vLLM hosts) running the proven stage-cell scripts untouched.
# Generations advance by editing the job snapshot's RUN_DIR_NAME/EXPERIMENT_NAME
# (+ carry-init env) and restarting the controller (runbook layer-2): a
# surviving slice keeps warm engines and the boundary costs ~10 min.
#
#   meta_driver.sh <ARM> <OP> <GENS> <WEIGHTS>
#     OP       winner-top16 | mix-top16
#     WEIGHTS  fresh | carry
# Env: META_TEMPLATE (stageC cell yaml), META_ACCEL (default v5p-64), STEPS.
set -uo pipefail
ARM=${1:?arm}; OP=${2:?op}; GENS=${3:?generations}; WEIGHTS=${4:?fresh|carry}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GCS=gs://sk7524-tinker-tpu-us-east5
JOBMAN=~/.venvs/jobman/bin/jobman
JOBDIR=/n/fs/vision-mix/sk7524/SkyRLTpu/third_party/jobman/jobs/sk7524
SD=~/meta-driver/$ARM; mkdir -p "$SD"
STEPS=${STEPS:-15}
MEMBERS=(qwen gemma muse)
PYBIN="$REPO/third_party/discover/.venv/bin/python"; [ -x "$PYBIN" ] || PYBIN=python3
log() { echo "($(date '+%F %T')) [$ARM] $*" | tee -a "$SD/driver.log"; }

member_done() {  # $1 run -> 0 if done (GCS: CONVERGED, final row, or budget)
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
gen_done() { local g=$1 t; for t in "${MEMBERS[@]}"; do member_done "$ARM-g$g-$t" || return 1; done; }

fetch_final_snap() {
  local snap
  snap=$(gsutil ls "$GCS/skyrl-runs/$1/tinker_log/$1/puct_sampler_step_*.json" 2>/dev/null | sort | tail -1)
  [ -n "$snap" ] && gsutil -q cp "$snap" "$2"
}
last_state_path() {
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

controller_up() { pgrep -f "[j]obman run $1" >/dev/null 2>&1; }
restart_controller() {  # $1 job id
  pgrep -f "[j]obman run $1" | xargs -r kill -9 2>/dev/null; sleep 2
  ( cd "$JOBDIR/.." && nohup bash -c "jobman run $1 | tee -a '$JOBDIR/$1/logs/job.log'" >/dev/null 2>&1 & )
  sleep 3; controller_up "$1"
}

ensure_member_job() {  # $1 tag  $2 gen  $3 init_sp  $4 init_jsonl
  local tag=$1 g=$2 init_sp=$3 init_jsonl=$4
  local run="$ARM-g$g-$tag" idfile="$SD/job_$tag.id" id
  if [ ! -f "$idfile" ]; then
    python3 "$REPO/tpu/meta/gen_member_config.py" \
      --template "${META_TEMPLATE:?}" --arm "$ARM" --gen "$g" --member "$tag" \
      --steps "$STEPS" --accel "${META_ACCEL:-v5p-64}" \
      ${init_sp:+--init-sp "$init_sp"} ${init_jsonl:+--init-jsonl "$init_jsonl"} \
      --out "$SD/cfg_$tag.yaml" || { log "FATAL: config gen $tag"; return 1; }
    id=$("$JOBMAN" create "$SD/cfg_$tag.yaml" 2>&1 | tee -a "$SD/driver.log" | grep -oE 'Created job [0-9]+' | grep -oE '[0-9]+')
    [ -n "$id" ] || { log "FATAL: jobman create $tag produced no id"; return 1; }
    echo "$id" > "$idfile"; log "member $tag: created job $id (gen $g)"
    return 0
  fi
  id=$(cat "$idfile")
  # advance the snapshot to gen g if needed (runbook layer-2: edit + restart)
  local want="$run" have
  have=$(grep -m1 'RUN_DIR_NAME:' "$JOBDIR/$id/config.yaml" | awk '{print $2}' | tr -d "'\"")
  if [ "$have" != "$want" ]; then
    CFG="$JOBDIR/$id/config.yaml" WANT="$want" SP="$init_sp" JL="$init_jsonl" TAG="$tag" python3 - <<'PYEOF' || { log "FATAL: snapshot edit $tag"; return 1; }
import os, yaml
p=os.environ["CFG"]; want=os.environ["WANT"]; sp=os.environ["SP"]; jl=os.environ["JL"]; tag=os.environ["TAG"]
c=yaml.safe_load(open(p)); e=c["resumable"]["env"]
e["RUN_DIR_NAME"]=want; e["EXPERIMENT_NAME"]=want
extra=e.get("EXTRA_TTD_ENV","")
extra=" ".join(x for x in extra.split() if not x.startswith("TTD_INIT_STATE_PATH_"))
if sp: extra+=f" TTD_INIT_STATE_PATH_{tag.upper()}={sp}"
e["EXTRA_TTD_ENV"]=extra
if jl:
    e["INIT_JSONL_GCS"]=jl
    e["EXTRA_REREG_JSONL"]=f"/home/sk7524_princeton_edu/init_prev_{tag}.jsonl"
else:
    e.pop("INIT_JSONL_GCS",None); e.pop("EXTRA_REREG_JSONL",None)
yaml.safe_dump(c, open(p,"w"), sort_keys=False)
print(f"snapshot {p} -> {want}")
PYEOF
    log "member $tag: advanced snapshot to $want; restarting controller $id"
    restart_controller "$id" || log "WARNING: controller $id not confirmed up"
  elif ! controller_up "$id"; then
    log "member $tag: controller $id down -- restarting"
    restart_controller "$id" || log "WARNING: controller $id not confirmed up"
  fi
}

for (( g=0; g<GENS; g++ )); do
  if gen_done "$g"; then log "gen $g already complete"; continue; fi

  # ---- seed (gen>0) ----------------------------------------------------------
  if [ "$g" -gt 0 ] && [ ! -f "$SD/seed_g$g.done" ]; then
    log "building gen-$g seed ($OP) from gen-$((g-1)) trees"
    trees=()
    for t in "${MEMBERS[@]}"; do
      if fetch_final_snap "$ARM-g$((g-1))-$t" "$SD/g$((g-1))_$t.json"; then
        trees+=("--tree" "$t=$SD/g$((g-1))_$t.json")
      else
        log "WARNING: no final snapshot for $t in gen $((g-1)) (ineligible)"
      fi
    done
    [ ${#trees[@]} -gt 0 ] || { log "FATAL: no member produced a tree"; exit 1; }
    "$PYBIN" "$REPO/tpu/meta/build_meta_seed.py" --op "$OP" --k 16 \
      --out "$SD/seed_g$g.json" "${trees[@]}" | tee -a "$SD/driver.log" || { log "FATAL: seed build"; exit 1; }
    for t in "${MEMBERS[@]}"; do
      run="$ARM-g$g-$t"
      gsutil -q cp "$SD/seed_g$g.json" \
        "$GCS/skyrl-runs/$run/tinker_log/$run/puct_sampler_step_000000.json" || { log "FATAL: seed upload $t"; exit 1; }
    done
    touch "$SD/seed_g$g.done"
  fi

  # ---- ensure the three member units are on gen g ---------------------------
  for t in "${MEMBERS[@]}"; do
    init_sp=""; init_jsonl=""
    if [ "$WEIGHTS" = carry ] && [ "$g" -gt 0 ]; then
      prev="$ARM-g$((g-1))-$t"
      init_sp=$(last_state_path "$prev" || true)
      [ -n "$init_sp" ] && init_jsonl="$GCS/skyrl-runs/$prev/tinker_log/$prev/member_$t/checkpoints.jsonl" \
        || log "WARNING: no carry state for $t (fresh this gen)"
    fi
    ensure_member_job "$t" "$g" "$init_sp" "$init_jsonl" || exit 1
  done

  # ---- barrier ---------------------------------------------------------------
  log "waiting on gen $g (3 member units)"
  while ! gen_done "$g"; do
    sleep 300
    for t in "${MEMBERS[@]}"; do   # keep controllers alive through the wait
      idf="$SD/job_$t.id"; [ -f "$idf" ] || continue
      member_done "$ARM-g$g-$t" && continue
      controller_up "$(cat "$idf")" || { log "member $t controller down mid-gen -- restarting"; restart_controller "$(cat "$idf")"; }
    done
  done
  log "gen $g complete"
done

log "ALL $GENS generations complete"
for (( g=0; g<GENS; g++ )); do
  for t in "${MEMBERS[@]}"; do
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
