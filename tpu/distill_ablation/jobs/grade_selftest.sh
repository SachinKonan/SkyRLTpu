#!/bin/bash
set -euo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu; AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"; cd "$AB"
exec "$PY" - <<'PYEOF'
import json, sys
sys.path.insert(0, "/n/fs/vision-mix/sk7524/SkyRLTpu/tpu/distill_ablation")
import grading_mcp as g
def pick(path):
    recs = json.load(open(path))["records"]
    v = [r for r in recs if r["status"]=="valid" and r.get("code")]
    return v[0]["code"] if v else None
for problem, jf, fasts in [("fc46","corpora/initial_fcalgo46.json",[True,False]),
                           ("erdos","corpora/initial_erdos.json",[True])]:
    code = pick(jf)
    if not code: print(f"{problem}: no valid cached solution"); continue
    root,mod,cls,pt,lang = g.PROBLEMS[problem]
    base = g.build_base_construction(problem)
    print(f"{problem}: base_len={len(base) if base else 0}", flush=True)
    for fast in fasts:
        r = g._grade(root,mod,cls,pt,lang,base,code,fast,f"/tmp/gradetest_{problem}")
        print(f"  fast={fast}: valid={r['valid']} score={r['score']} detail={r['detail'][:120]!r}", flush=True)
PYEOF
