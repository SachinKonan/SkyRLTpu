"""Submit ONE candidate to a live arena queue and print its verdict.

For fast iteration against a judge that is already up: the compile gate
(Mosaic alignment, VMEM, budget) is only observable on real silicon, and
CPU interpret mode never exercises it.

    python3 tpu/pallas_arena/verify/submit_one.py <program.py> <tag> [problem]
"""
import json
import pathlib
import sys
import time
import urllib.request

REPO = pathlib.Path("/n/fs/vision-mix/sk7524/SkyRLTpu")
url = (REPO / "runs/pallas_arena/rl-queue-url.txt").read_text().strip()
code = open(sys.argv[1]).read()
tag = sys.argv[2]
problem = sys.argv[3] if len(sys.argv) > 3 else "rg_lru"


def post(path, payload):
    req = urllib.request.Request(url + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=30))


wid = post("/submit", {"problem": problem, "code": code, "tag": tag})["work_id"]
print(f"submitted {wid} ({tag}, {problem}) to {url}", flush=True)
deadline = time.time() + 3600
while time.time() < deadline:
    rec = (post("/results", {"work_ids": [wid]})["results"] or {}).get(wid) or {}
    if rec.get("done"):
        r = rec.get("result") or rec
        print(f"passed={r.get('passed')} gate={r.get('gate')} "
              f"reward={r.get('reward_with_bwd') or r.get('reward')}", flush=True)
        print("violations:", str(r.get("violations"))[:700], flush=True)
        print("observation:", str(r.get("observation"))[:1200], flush=True)
        break
    time.sleep(10)
else:
    print("no verdict within an hour", flush=True)
