#!/bin/bash
# Preflight for the scheme A/B: verify retooled grader (check@10s, session time-left, --no-full
# gating, weak base construction) with NO codex calls.
set -euo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu; AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"
cd "$AB"
exec "$PY" -u - <<'PY'
import json, subprocess, socket, sys, time, signal, asyncio
from pathlib import Path
sys.path.insert(0, '.'); sys.path.insert(0, 'portfolio')
import benchmark_cdc as B

root = Path(f"{B.REPO}/runs/ab_scheme/preflight"); root.mkdir(parents=True, exist_ok=True)
(root/"sessions").mkdir(exist_ok=True)

ws = json.loads(Path("corpora/worse_set.json").read_text())["worse"]
ranked = sorted(range(len(ws)), key=lambda i: -ws[i]["value"]); idx = ranked[len(ranked)//2]
seed = ws[idx]
base = root/"base.json"; base.write_text(json.dumps(list(seed["construction"])))
print(f"seed idx={idx} recorded_c5={-seed['value']:.9f} base_len={len(seed['construction'])}")

# deadline file for session time-left
(root/"sessions"/"pf.json").write_text(json.dumps({"deadline": time.time()+600}))

def boot(no_full):
    port = B.free_port()
    cmd = [B.PY, "grading_mcp.py", "--problem", "erdos", "--port", str(port),
           "--logdir", str(root), "--backend", "thread", "--max-concurrent", "4",
           "--base-json", str(base)]
    if no_full: cmd.append("--no-full")
    p = subprocess.Popen(cmd, stdout=open(root/f"grader_nofull{no_full}.log","w"),
                         stderr=subprocess.STDOUT, env=B._base_env())
    for _ in range(180):
        try: socket.create_connection(("127.0.0.1",port),1).close(); return p,port
        except OSError:
            if p.poll() is not None: raise RuntimeError("grader died")
            time.sleep(1)
    raise RuntimeError("no bind")

async def probe(port, tool, sol):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    async with streamablehttp_client(f"http://127.0.0.1:{port}/mcp") as (r,w,_):
        async with ClientSession(r,w) as s:
            await s.initialize()
            names = [t.name for t in (await s.list_tools()).tools]
            if tool is None: return names, None
            res = await s.call_tool(tool, {"solution": sol, "session": "pf"})
            return names, json.loads(res.content[0].text)

prog = seed["code"]
sol = prog if prog.strip().startswith("```") else f"```python\n{prog}\n```"

# --- arm "fast": --no-full must hide grade_full; check must run ~10s and report time left
p,port = boot(True)
try:
    names,_ = asyncio.run(probe(port, None, None))
    print("tools (--no-full):", names)
    assert "check" in names and "grade_full" not in names, "no-full gating FAILED"
    t0=time.time(); _,out = asyncio.run(probe(port,"check",sol)); el=time.time()-t0
    print(f"check@10s -> valid={out.get('valid')} score={out.get('score')} "
          f"left={out.get('session_seconds_left')} wall={el:.0f}s")
    assert out.get("session_seconds_left") is not None, "session time-left MISSING"
    assert el < 90, f"check too slow ({el:.0f}s)"
finally:
    p.send_signal(signal.SIGINT); p.wait(15)

# --- arm "full": both tools present
p,port = boot(False)
try:
    names,_ = asyncio.run(probe(port, None, None))
    print("tools (full):", names)
    assert "check" in names and "grade_full" in names, "full arm tools FAILED"
finally:
    p.send_signal(signal.SIGINT); p.wait(15)

print("PREFLIGHT PASSED")
PY
