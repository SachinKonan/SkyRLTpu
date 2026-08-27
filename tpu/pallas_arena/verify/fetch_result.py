"""Fetch one work item's full verdict from the live queue and bank the
seed-obs/seed-reward files the arm launcher reads.

    python3 fetch_result.py <work_id> <tag>
"""
import json
import pathlib
import sys
import urllib.request

wid, tag = sys.argv[1], sys.argv[2]
url = pathlib.Path("/n/fs/vision-mix/sk7524/SkyRLTpu/runs/pallas_arena/rl-queue-url.txt").read_text().strip()
req = urllib.request.Request(url + "/results",
    data=json.dumps({"work_ids": [wid]}).encode(),
    headers={"Content-Type": "application/json"})
rec = (json.load(urllib.request.urlopen(req, timeout=30))["results"] or {}).get(wid) or {}
r = rec.get("result") or rec
rw = r.get("reward_with_bwd") or r.get("reward")
obs = str(r.get("observation") or "").strip()
assert r.get("passed") and obs, (r.get("passed"), len(obs))
out = pathlib.Path("/n/fs/vision-mix/sk7524/SkyRLTpu/runs/pallas_arena")
(out / f"seed-obs-{tag}.txt").write_text(obs)
(out / f"seed-reward-{tag}.txt").write_text(
    f"{float(rw):.3f}x vs the production kernel -- reward accrues only ABOVE this")
print(f"banked: reward={rw} obs={len(obs)} chars")
print(obs[:800])
