"""Dump one work item's FULL merged verdict from the live queue as JSON.

    python3 fetch_full.py <work_id> [out.json]
"""
import json
import pathlib
import sys
import urllib.request

wid = sys.argv[1]
out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else None
url = pathlib.Path("/n/fs/vision-mix/sk7524/SkyRLTpu/runs/pallas_arena/rl-queue-url.txt").read_text().strip()
req = urllib.request.Request(url + "/results",
    data=json.dumps({"work_ids": [wid]}).encode(),
    headers={"Content-Type": "application/json"})
rec = (json.load(urllib.request.urlopen(req, timeout=30))["results"] or {}).get(wid) or {}
r = rec.get("result") or rec
if out:
    out.write_text(json.dumps(r, indent=2, default=str))
for k in ("passed", "gate", "reward", "reward_with_bwd", "n_scored_cases",
          "n_bwd_factors", "per_case", "holdout", "excluded_cases",
          "skipped_cases", "case_noise_floors", "case_boot_s", "latencies",
          "mxu_fracs", "speed_of_light_fracs", "grad_scores", "tp_control",
          "tp_timer_ratios", "tp_baseline_impls", "timer"):
    if k in r and r[k] not in (None, {}, []):
        print(f"{k}: {json.dumps(r[k], default=str)[:900]}")
