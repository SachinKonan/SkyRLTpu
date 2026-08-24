#!/bin/bash
# Pick the A/B seed empirically: for each budget-adaptive candidate, run check(budget_s=10) and
# report validity + elapsed. We want a seed that is VALID and FAST at 10s (so `check` is genuinely
# a quick smoke test) and still has real headroom to the 0.380928 record. No codex calls.
set -euo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu; AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"
cd "$AB"
exec "$PY" -u - <<'PY'
import json, subprocess, socket, sys, time, signal, asyncio
from pathlib import Path
sys.path.insert(0, '.')
import benchmark_cdc as B

CANDS = [27, 35, 12, 2, 32, 31]
root = Path(f"{B.REPO}/runs/ab_scheme/pickseed"); root.mkdir(parents=True, exist_ok=True)
ws = json.loads(Path("corpora/worse_set.json").read_text())["worse"]

async def call(port, tool, sol):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    async with streamablehttp_client(f"http://127.0.0.1:{port}/mcp") as (r,w,_):
        async with ClientSession(r,w) as s:
            await s.initialize()
            res = await s.call_tool(tool, {"solution": sol})
            return json.loads(res.content[0].text)

print(f"{'idx':>4}{'rec_c5':>14}{'check@10s':>14}{'valid':>7}{'secs':>7}  detail")
out_rows = []
for idx in CANDS:
    e = ws[idx]
    base = root/f"base_{idx}.json"; base.write_text(json.dumps(list(e["construction"])))
    port = B.free_port()
    p = subprocess.Popen([B.PY, "grading_mcp.py", "--problem", "erdos", "--port", str(port),
                          "--logdir", str(root), "--backend", "thread", "--max-concurrent", "2",
                          "--base-json", str(base)],
                         stdout=open(root/f"grader_{idx}.log","w"), stderr=subprocess.STDOUT,
                         env=B._base_env())
    for _ in range(180):
        try: socket.create_connection(("127.0.0.1",port),1).close(); break
        except OSError:
            if p.poll() is not None: break
            time.sleep(1)
    try:
        code = e["code"]
        sol = code if code.strip().startswith("```") else f"```python\n{code}\n```"
        t0=time.time(); r = asyncio.run(call(port,"check",sol)); el=time.time()-t0
        print(f"{idx:>4}{-e['value']:>14.9f}{str(r.get('score')):>14}"
              f"{str(r.get('valid')):>7}{el:>7.0f}  {r.get('detail','')[:70]}", flush=True)
        out_rows.append({"idx": idx, "rec_c5": -e["value"], "s10": r.get("score"),
                         "valid": r.get("valid"), "secs": round(el)})
    finally:
        p.send_signal(signal.SIGINT)
        try: p.wait(15)
        except subprocess.TimeoutExpired: p.kill()

(root/"pickseed.json").write_text(json.dumps(out_rows, indent=2))
ok = [r for r in out_rows if r["valid"] and r["secs"] <= 60]
print("\nVALID and FAST (<=60s):", [r["idx"] for r in ok] or "NONE")
if ok:
    best = min(ok, key=lambda r: r["secs"])
    print(f"RECOMMEND seed idx={best['idx']} rec_c5={best['rec_c5']:.9f} "
          f"check10s={best['s10']} in {best['secs']}s")
PY
