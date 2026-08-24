#!/usr/bin/env bash
# jobman completion_probe (workers: 0): exit 0 iff ALL THREE members of this
# generation are done -- CONVERGED marker (flatline stop), a "final" checkpoint
# row, or batch >= NUM_EPOCHS.
set -euo pipefail
: "${ARM:?}"; : "${GEN:?}"
ARM="$ARM" GEN="$GEN" NUM_EPOCHS="${NUM_EPOCHS:-15}" python3 - <<'PY'
import glob, json, os, sys
arm, gen, target = os.environ["ARM"], os.environ["GEN"], int(os.environ["NUM_EPOCHS"])
incomplete = []
for tag in ("qwen", "gemma", "muse"):
    run = f"{arm}-g{gen}-{tag}"
    if glob.glob(os.path.expanduser(f"~/skyrl-runs/{run}/tinker_log/*/CONVERGED")):
        continue
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
    if not (final or (latest is not None and latest >= target)):
        incomplete.append(f"{tag}={latest}")
if incomplete:
    print("incomplete: " + " ".join(incomplete) + f"/{target}")
    sys.exit(1)
print("complete: all members done")
PY
