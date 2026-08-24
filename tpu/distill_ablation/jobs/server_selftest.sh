#!/bin/bash
set -euo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu; AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"; cd "$AB"
PORT=8791
# start the grader (thread backend) in background
"$PY" grading_mcp.py --problem fc46 --port $PORT --logdir /tmp/srvtest_fc46 --backend thread --max-concurrent 4 \
  > /tmp/srvtest_fc46.log 2>&1 &
SRV=$!
trap "kill $SRV 2>/dev/null || true" EXIT
# wait for it to bind
for i in $(seq 1 40); do curl -s -o /dev/null "http://127.0.0.1:$PORT/mcp" && break; sleep 1; done
"$PY" - <<PYEOF
import asyncio, json, sys
sys.path.insert(0,"$AB")
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
# best-scoring cached fc46 solution
recs=json.load(open("corpora/initial_fcalgo46.json"))["records"]
valid=[r for r in recs if r["status"]=="valid" and r.get("code") and r.get("score")]
sol=max(valid,key=lambda r:r["score"])["code"] if valid else [r for r in recs if r.get("code")][0]["code"]
async def run():
    async with streamablehttp_client("http://127.0.0.1:$PORT/mcp") as (r,w,_):
        async with ClientSession(r,w) as s:
            await s.initialize()
            t=await s.list_tools(); print("TOOLS:",[x.name for x in t.tools])
            for tool in ("grade_fast","grade_full"):
                res=await s.call_tool(tool,{"solution":sol})
                print(f"{tool}:",res.content[0].text[:200])
asyncio.run(run())
PYEOF
echo "--- server log tail ---"; tail -4 /tmp/srvtest_fc46.log
