#!/bin/bash
# Verify the submit() contract: rejects without a matching grade_full, accepts with one,
# rejects a DIFFERENT program than the one verified, and enforces the grade_full cap.
set -uo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu; AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"
cd "$AB"
exec "$PY" -u - <<'PY'
import json, subprocess, socket, sys, time, signal, asyncio
from pathlib import Path
sys.path.insert(0,'.')
import benchmark_cdc as B

root = Path(f"{B.REPO}/runs/submit_test"); root.mkdir(parents=True, exist_ok=True)
port = B.free_port()
p = subprocess.Popen([B.PY, "grading_mcp.py", "--problem", "fc46", "--port", str(port),
                      "--logdir", str(root), "--backend", "thread", "--max-concurrent", "4",
                      "--max-full", "2"],
                     stdout=open(root/"grader.log","w"), stderr=subprocess.STDOUT, env=B._base_env())
for _ in range(240):
    try: socket.create_connection(("127.0.0.1",port),1).close(); break
    except OSError:
        if p.poll() is not None: print("grader died"); sys.exit(1)
        time.sleep(1)

async def sequence(steps):
    """One connection for the whole sequence -- real agents reuse a single MCP connection
    (measured: 17/16/8/5/4 calls per connection), and budgets key off mcp-session-id."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    out=[]
    async with streamablehttp_client(f"http://127.0.0.1:{port}/mcp") as (r,w,_):
        async with ClientSession(r,w) as s:
            await s.initialize()
            for tool, kw in steps:
                res = await s.call_tool(tool, kw)
                out.append(json.loads(res.content[0].text))
    return out

recs = json.loads(Path("corpora/initial_fcalgo46.json").read_text())["records"]
prog = [r for r in recs if r.get("code") and (r.get("score") or 0) > 0][0]["code"]
other = prog + "\n// variant\n"
S = "test_sess"
try:
    res = asyncio.run(sequence([
        ("submit",     dict(program=prog, approach="a", insight="i", session=S)),
        ("grade_full", dict(solution=prog, session=S)),
        ("submit",     dict(program=other, approach="a", insight="i", session=S)),
        ("submit",     dict(program=prog, approach="tabu", insight="critical path", session=S)),
        ("submit",     dict(program=prog, approach="again", insight="dup", session=S)),
        ("grade_full", dict(solution=other, session=S)),
        ("grade_full", dict(solution=prog+"\n//x\n", session=S)),
    ]))
    print(f"1. submit BEFORE grade_full -> accepted={res[0]['accepted']}  (expect False)")
    print(f"2. grade_full -> valid={res[1]['valid']} score={res[1].get('score')}")
    print(f"3. submit a DIFFERENT program -> accepted={res[2]['accepted']}  (expect False)")
    print(f"4. submit the VERIFIED program -> accepted={res[3]['accepted']} score={res[3].get('score')}  (expect True)")
    print(f"5. submit TWICE -> accepted={res[4]['accepted']}  (expect False)")
    print(f"6. grade_full past cap(2) -> valid={res[6]['valid']} detail={res[6].get('detail','')[:70]}")
    sub = (root/"submissions.jsonl")
    print("6. submissions.jsonl:", sub.read_text().strip()[:160] if sub.exists() else "MISSING")
finally:
    p.send_signal(signal.SIGINT)
    try: p.wait(15)
    except subprocess.TimeoutExpired: p.kill()
PY
