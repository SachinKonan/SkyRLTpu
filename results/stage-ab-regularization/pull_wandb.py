"""Pull the full per-life history of the league ctrl/main arms from wandb.

Every guardian relaunch created a NEW wandb run with the same display name, so
the complete record (which the GCS jsonl lost to overwrite-on-failed-restore)
is the union of those runs. A life whose first logged step is 0 (after the
arm's very first run) is a FRESH-WEIGHTS restart; a first step > 0 is a proper
resume (interruption, weights kept).
"""
import json, os, sys

import wandb

ENTITY = "sk7524-princeton-university"
PROJECT = "tpu-tinker-exps"
ARMS = {
    "main": "league1-qwen-gemma-erdos",
    "ctrl": "league1b-ctrl-qwen-gemma-erdos",
}
KEYS = ["pool/best_value", "qwen/env/all/format", "gemma/env/all/format"]

api = wandb.Api(timeout=60)
out = {}
for arm, name in ARMS.items():
    runs = [r for r in api.runs(f"{ENTITY}/{PROJECT}",
                                filters={"displayName": name})]
    runs.sort(key=lambda r: r.createdAt)
    lives = []
    for r in runs:
        rows = []
        for row in r.scan_history():
            rows.append({k: row.get(k) for k in ["_timestamp", "_step"] + KEYS})
        rows = [x for x in rows if x.get("pool/best_value") is not None]
        rows.sort(key=lambda x: x["_timestamp"])
        lives.append({"run_id": r.id, "created": r.createdAt, "rows": rows})
        print(f"{arm}: run {r.id} created {r.createdAt} rows={len(rows)}",
              file=sys.stderr)
    out[arm] = lives

with open(os.path.join(os.path.dirname(__file__), "league_lives.json"), "w") as f:
    json.dump(out, f)
print("saved league_lives.json", file=sys.stderr)
