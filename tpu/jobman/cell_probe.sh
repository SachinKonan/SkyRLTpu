#!/usr/bin/env bash
# jobman completion_probe hook (workers: 0): exit 0 iff the cell has finished
# its budget. Reads the member's checkpoints.jsonl (batch counter) -- the same
# record weight-resume uses, so "complete" and "resumable" cannot disagree.
# Without this probe, loop:true re-runs a finished cell forever (the fc46
# failure mode).
set -euo pipefail
: "${CELL:?}"
NUM_EPOCHS="${NUM_EPOCHS:-15}" RUN="${RUN_DIR_NAME:-stageA-$CELL}" python3 - <<'PY'
import glob, json, os, sys

run = os.environ["RUN"]
target = int(os.environ["NUM_EPOCHS"])
pats = glob.glob(os.path.expanduser(
    f"~/skyrl-runs/{run}/tinker_log/*/member_*/checkpoints.jsonl"))
if not pats:
    print("incomplete: no checkpoints.jsonl yet")
    sys.exit(1)

latest = None
final = False
for p in pats:
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        b = row.get("batch")
        if isinstance(b, int):
            latest = b if latest is None else max(latest, b)
        if row.get("name") == "final":
            final = True

if final or (latest is not None and latest >= target):
    print(f"complete: latest_batch={latest} final={final}")
    sys.exit(0)
print(f"incomplete: latest_batch={latest}/{target}")
sys.exit(1)
PY
