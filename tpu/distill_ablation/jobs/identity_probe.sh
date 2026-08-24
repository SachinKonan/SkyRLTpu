#!/bin/bash
# Do codex SUBAGENTS open their own MCP connections? If so we can enforce per-agent budgets
# server-side instead of trusting the `session` string they pass.
set -uo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu; AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"
export PATH="/n/fs/vision-mix/sk7524/.npm-global/bin:$PATH"
cd "$AB/portfolio"
"$PY" -u mapreduce.py --problem fc46 --arm native --n 6 --replicate 98 \
    --orch-style cdc --orch-wall 1800 --max-parallel 6 --grader-conc 8 || echo "^^ FAILED"
D=$REPO/runs/mapreduce/fc46_native_r98
echo "=== identity.jsonl: distinct transport identities vs claimed sessions ==="
"$PY" - <<'PY'
import json
from pathlib import Path
from collections import Counter
f=Path('/n/fs/vision-mix/sk7524/SkyRLTpu/runs/mapreduce/fc46_native_r98/identity.jsonl')
if not f.exists(): print("  (no identity log)"); raise SystemExit
rows=[json.loads(l) for l in f.read_text().splitlines() if l.strip()]
print(f"  calls logged: {len(rows)}")
print(f"  distinct transport conns : {len(set(r.get('conn') for r in rows))}")
print(f"  distinct client_ids      : {len(set(str(r.get('client_id')) for r in rows))}")
print(f"  distinct claimed sessions: {len(set(r.get('claimed_session') for r in rows))}")
print("  claimed session values:", dict(Counter(r.get('claimed_session') for r in rows).most_common(6)))
print("  conn -> #calls:", dict(Counter(r.get('conn') for r in rows).most_common(8)))
hs=[r.get('hdrs') for r in rows if r.get('hdrs')]
if hs: print("  sample headers:", hs[0])
PY
