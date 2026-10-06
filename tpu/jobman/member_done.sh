#!/usr/bin/env bash
# Is ONE meta member complete? (runs on the member's trainer host)
# env: RUN (run dir name), TARGET (steps). exit 0 = complete.
set -euo pipefail
RUN="${RUN:?}" TARGET="${TARGET:-15}" python3 - <<'PY'
import glob, json, os, sys
run, target = os.environ["RUN"], int(os.environ["TARGET"])
if glob.glob(os.path.expanduser(f"~/skyrl-runs/{run}/tinker_log/*/CONVERGED")):
    print("done: CONVERGED"); sys.exit(0)
latest, final = None, False
for p in glob.glob(os.path.expanduser(f"~/skyrl-runs/{run}/tinker_log/*/member_*/checkpoints.jsonl")):
    for line in open(p):
        line = line.strip()
        if not line: continue
        try: row = json.loads(line)
        except ValueError: continue
        b = row.get("batch")
        if isinstance(b, int): latest = b if latest is None else max(latest, b)
        if row.get("name") == "final": final = True
if final or (latest is not None and latest >= target):
    print(f"done: batch={latest} final={final}"); sys.exit(0)
print(f"incomplete: batch={latest}/{target}"); sys.exit(1)
PY
