"""Portfolio search driver: SimpleTES-style loop with agentic selection + agentic rollouts.

Per round: (1) AGGREGATOR (gpt-5.6-sol xhigh) browses the store via store_mcp and submits
N seed groups under hard code constraints; (2) N x G EXECUTOR rollouts (gpt-5.4-mini)
each get a minimal prompt (task + group programs/scores + failure patterns) and a
multi-turn session with the grader MCP; (3) the DRIVER commits each rollout's best
independently-graded solution to the DAG, backs up U, saves. Strictly serial codex.

srun/sbatch only. Launch: jobs/portfolio_smoke.sh
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))  # for benchmark_cdc helpers
import benchmark_cdc as B
from prompts import aggregator_prompt, executor_prompt
from store import Store

PF = Path(__file__).parent
MAXIMIZE = B.MAXIMIZE

SEED_PROGRAM = {
    "erdos": """def run(seed=42, budget_s=1000, **kwargs):
    import numpy as np
    h = np.asarray(initial_h_values, float)
    dx = 2.0 / h.size
    c5 = float(np.correlate(h, 1 - h, 'full').max() * dx)
    return h.tolist(), c5, h.size""",
    "ac1": """def propose_candidate():
    import numpy as np
    n = 512
    f = np.ones(n); f /= f.sum() / n
    return f.tolist()""",
    "ac2": """def construct_function():
    import numpy as np
    n = 512
    x = np.linspace(-1, 1, n)
    f = np.maximum(0.0, 1 - np.abs(x))
    return f.tolist()""",
}


def run_codex(model, reasoning, prompt, workdir, mcp_url, mcp_name, wall, out_name):
    cmd = ["codex", "exec", "-m", model, "-c", f"model_reasoning_effort={reasoning}",
           "-s", "workspace-write", "-c", "approval_policy=never", "--json",
           "-C", str(workdir),
           "-c", f'mcp_servers.{mcp_name}.url="{mcp_url}"',
           "-c", f"mcp_servers.{mcp_name}.tool_timeout_sec=1200",
           "-c", f"mcp_servers.{mcp_name}.startup_timeout_sec=60",
           "-c", f'mcp_servers.{mcp_name}.default_tools_approval_mode="approve"']
    t0 = time.time()
    timed_out = False
    with open(workdir / f"{out_name}.jsonl", "w") as out:
        try:
            subprocess.run(cmd, input=prompt, text=True, stdout=out, stderr=subprocess.STDOUT,
                           env=B._base_env(), cwd=str(workdir), timeout=wall)
        except subprocess.TimeoutExpired:
            timed_out = True
    return round(time.time() - t0), timed_out


def grade_via_server(port, solution, tool="grade_full"):
    async def call():
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
        async with streamablehttp_client(f"http://127.0.0.1:{port}/mcp") as (r, w, _):
            async with ClientSession(r, w) as s:
                await s.initialize()
                res = await s.call_tool(tool, {"solution": solution})
                return json.loads(res.content[0].text)
    try:
        return asyncio.run(call())
    except Exception as e:
        return {"score": None, "valid": False, "detail": f"driver-grade failed: {e}"}


def log_slice(grade_log: Path, start_line: int):
    if not grade_log.exists():
        return []
    return [json.loads(x) for x in grade_log.read_text().splitlines()[start_line:]
            if x.strip().startswith("{")]


def log_len(grade_log: Path):
    return len(grade_log.read_text().splitlines()) if grade_log.exists() else 0


def blob_suspect(txt: str) -> bool:
    stripped = re.sub(r"\s+", "", txt)
    if re.search(r"[A-Za-z0-9+/]{400,}={0,2}", stripped):
        return True
    return len(re.findall(r"-?\d+\.\d{6,}", txt)) > 200


def rollout_best(problem, wd, rows, port):
    """Best independently-graded solution of THIS rollout: prefer its grade_full entries,
    else regrade its best grade_fast at full budget."""
    mx = MAXIMIZE[problem]
    better = (lambda a, b: a > b) if mx else (lambda a, b: a < b)
    best_full, best_fast = None, None
    for r in rows:
        if not (r.get("valid") and r.get("score") is not None):
            continue
        cur = (r["score"], r["sol_hash"])
        if r["tool"] == "grade_full" and (best_full is None or better(cur[0], best_full[0])):
            best_full = cur
        if r["tool"] == "grade_fast" and (best_fast is None or better(cur[0], best_fast[0])):
            best_fast = cur
    pick = best_full or best_fast
    if pick is None:
        return None, None, "no valid graded solution"
    sol_file = wd / "solutions" / f"{pick[1]}.txt"
    if not sol_file.exists():
        return None, None, f"solution file missing for {pick[1]}"
    sol = sol_file.read_text()
    if best_full is None:  # only fast-graded -> confirm at full budget
        out = grade_via_server(port, sol)
        if not out.get("valid"):
            return None, None, f"full-budget regrade failed: {out.get('detail','')[:120]}"
        return sol, out["score"], out.get("detail", "")
    return sol, pick[0], ""


def fallback_groups(store: Store, n_groups, n_used_cap=4):
    """Code fallback if the aggregator fails: disjoint singletons, best-U first."""
    valid = [n for n in store.g.nodes
             if store.g.nodes[n]["r"] is not None and store.g.nodes[n]["n_used"] < n_used_cap]
    valid.sort(key=lambda n: store.g.nodes[n]["U"], reverse=store.maximize)
    return [[n] for n in valid[:n_groups]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="erdos", choices=list(SEED_PROGRAM))
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--n-groups", type=int, default=4)
    ap.add_argument("--g-rollouts", type=int, default=2)
    ap.add_argument("--max-group", type=int, default=3)
    ap.add_argument("--n-fast", type=int, default=5)
    ap.add_argument("--n-full", type=int, default=1)
    ap.add_argument("--exec-model", default="gpt-5.4-mini")
    ap.add_argument("--exec-reasoning", default="high")
    ap.add_argument("--agg-model", default="gpt-5.6-sol")
    ap.add_argument("--agg-reasoning", default="xhigh")
    ap.add_argument("--exec-wall", type=int, default=1500)
    ap.add_argument("--agg-wall", type=int, default=900)
    ap.add_argument("--tag", default="")
    ap.add_argument("--backend", default="thread")
    ap.add_argument("--max-concurrent", type=int, default=8)
    args = ap.parse_args()

    run_dir = Path(f"{B.REPO}/runs/portfolio/{args.problem}{args.tag}")
    run_dir.mkdir(parents=True, exist_ok=True)
    grade_log = run_dir / "grade_log.jsonl"
    graph_path = run_dir / "graph.json"

    port = B.free_port()
    grader = B.start_grader(args.problem, port, run_dir, args.backend, args.max_concurrent)
    try:
        # ---- store init (resume if graph exists) ----
        if graph_path.exists():
            store = Store.load(graph_path)
            print(f"[pf] resumed store: {store.g.number_of_nodes()} nodes", flush=True)
        else:
            store = Store(MAXIMIZE[args.problem], graph_path)
            seed_sol = f"```python\n{SEED_PROGRAM[args.problem]}\n```"
            out = grade_via_server(port, seed_sol)
            store.add(seed_sol, out.get("score"), [], feedback=out.get("detail", ""),
                      rnd=0, summary="seed baseline")
            store.backup_U(); store.save()
            print(f"[pf] seed graded: {out.get('score')}", flush=True)

        for rnd in range(1, args.rounds + 1):
            best0 = store.best()
            best0_r = store.g.nodes[best0]["r"] if best0 is not None else None
            n_sel = min(args.n_groups,
                        sum(1 for n in store.g.nodes if store.g.nodes[n]["r"] is not None))
            # ---- 1. aggregator ----
            groups = None
            if store.g.number_of_nodes() >= 3:
                store.save()
                gout = run_dir / f"groups_r{rnd}.json"
                gout.unlink(missing_ok=True)
                sport = B.free_port()
                smcp = subprocess.Popen(
                    [B.PY, str(PF / "store_mcp.py"), "--graph", str(graph_path),
                     "--out", str(gout), "--port", str(sport),
                     "--n-groups", str(n_sel), "--max-group", str(args.max_group)],
                    stdout=open(run_dir / "store_mcp.log", "a"), stderr=subprocess.STDOUT,
                    env=B._base_env(), cwd=str(PF))
                time.sleep(3)
                awd = run_dir / f"agg_r{rnd}"; awd.mkdir(exist_ok=True)
                secs, to = run_codex(args.agg_model, args.agg_reasoning,
                                     aggregator_prompt(args.problem, n_sel, args.max_group),
                                     awd, f"http://127.0.0.1:{sport}/mcp", "store",
                                     args.agg_wall, "codex")
                smcp.send_signal(signal.SIGINT)
                if gout.exists():
                    groups = json.loads(gout.read_text())
                print(f"[pf] r{rnd} aggregator {secs}s groups={groups}", flush=True)
            if not groups:
                groups = fallback_groups(store, n_sel)
                print(f"[pf] r{rnd} FALLBACK groups={groups}", flush=True)

            # ---- 2+3. executors (serial) + commit ----
            for gi, group in enumerate(groups):
                gnodes = [store.meta(i, with_program=True) for i in group]
                for g in range(args.g_rollouts):
                    ewd = run_dir / f"r{rnd}_g{gi}_x{g}"; ewd.mkdir(exist_ok=True)
                    prompt = executor_prompt(args.problem, gnodes, grade_log,
                                             args.n_fast, args.n_full,
                                             wall_min=args.exec_wall // 60)
                    mark = log_len(grade_log)
                    secs, to = run_codex(args.exec_model, args.exec_reasoning, prompt,
                                         ewd, f"http://127.0.0.1:{port}/mcp", "grader",
                                         args.exec_wall, "codex")
                    sol, score, note = rollout_best(args.problem, run_dir,
                                                    log_slice(grade_log, mark), port)
                    if sol is None:  # never graded -> salvage solution.txt if written
                        sf = ewd / "solution.txt"
                        if sf.exists() and sf.read_text().strip():
                            out = grade_via_server(port, sf.read_text())
                            if out.get("valid"):
                                sol, score = sf.read_text(), out["score"]
                                note = "salvaged from solution.txt"
                    if sol is None:
                        print(f"[pf] r{rnd} g{gi} x{g}: FAILED ({note}) {secs}s", flush=True)
                        continue
                    fb = note
                    if blob_suspect(sol):
                        fb = ("SUSPECT-EMBEDDED-DATA; " + fb)[:400]
                    nid = store.add(sol, score, group, feedback=fb, rnd=rnd)
                    print(f"[pf] r{rnd} g{gi} x{g}: node {nid} score={score} "
                          f"{'BLOB?' if 'SUSPECT' in fb else ''} ({secs}s)", flush=True)
                store.mark_used(group)
                store.backup_U(); store.save()

            best1 = store.best()
            best1_r = store.g.nodes[best1]["r"] if best1 is not None else None
            print(f"[pf] ===== round {rnd} done: best {best0_r} -> {best1_r} "
                  f"({store.g.number_of_nodes()} nodes) =====", flush=True)
    finally:
        grader.send_signal(signal.SIGINT)
        try:
            grader.wait(15)
        except subprocess.TimeoutExpired:
            grader.kill()
    b = store.best()
    print(f"[pf] FINAL best node {b}: r={store.g.nodes[b]['r'] if b is not None else None}")


if __name__ == "__main__":
    main()
