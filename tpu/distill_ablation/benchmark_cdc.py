"""Driver for the CDC-style deep-agent benchmark. STRICTLY SERIAL — one codex session live at a
time (protects the codex quota). Runs ON the srun-allocated node and launches codex directly so the
grader (127.0.0.1) and codex are co-located.

Per problem: build the CDC prompt (subprocess, clean path) -> start the grading MCP -> run
`codex exec gpt-5.6-sol` (multiagent-v2, MCP grader registered) under a wall-clock cap -> report the
best `grade_full` from the server log -> cache everything -> stop the grader -> next problem.

Launch:  srun -N1 -n1 -c <cores> --mem <mem> -t <time> \
           <venv>/bin/python benchmark_cdc.py --backend thread [--smoke] [--problems fc46 ...]
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = "/n/fs/vision-mix/sk7524/SkyRLTpu"
AB = f"{REPO}/tpu/distill_ablation"
PY = f"{REPO}/third_party/discover/.venv-ttd-discover/bin/python"
CODEX_BIN_DIR = "/n/fs/vision-mix/sk7524/.npm-global/bin"
MODEL = "gpt-5.6-sol"
ALL = ["erdos", "ac1", "ac2", "fc46", "fc302"]
# per-problem optimization direction for picking the "best" grade_full
MAXIMIZE = {"erdos": False, "ac1": False, "ac2": True, "fc46": True, "fc302": True}


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _base_env():
    env = os.environ.copy()
    env["PATH"] = CODEX_BIN_DIR + ":" + env.get("PATH", "")
    env["TTD_EVAL_BACKEND"] = "local"
    env["TTD_DISCOVER_SYNC"] = "0"
    return env


def build_prompt(problem, workdir):
    pf = workdir / "prompt.txt"
    subprocess.run([PY, f"{AB}/cdc_prompt.py", "--problem", problem, "--out", str(pf)],
                   check=True, env=_base_env())
    return pf


def start_grader(problem, port, workdir, backend, max_concurrent):
    log = open(workdir / "grader.log", "w")
    p = subprocess.Popen(
        [PY, f"{AB}/grading_mcp.py", "--problem", problem, "--port", str(port),
         "--logdir", str(workdir), "--backend", backend, "--max-concurrent", str(max_concurrent)],
        stdout=log, stderr=subprocess.STDOUT, env=_base_env(), cwd=AB)
    for _ in range(180):  # wait up to 3 min for bind (ray/tokenizer import on boot)
        try:
            socket.create_connection(("127.0.0.1", port), 1).close()
            return p
        except OSError:
            if p.poll() is not None:
                raise RuntimeError(f"grader for {problem} died on startup; see {workdir}/grader.log")
            time.sleep(1)
    raise RuntimeError(f"grader for {problem} did not bind on port {port}")


def run_codex(problem, prompt_file, workdir, port, wall_clock, max_agents, reasoning="high"):
    cmd = ["codex", "exec", "-m", MODEL, "-c", f"model_reasoning_effort={reasoning}",
           "--enable", "multi_agent_v2",
           "-c", f"features.multi_agent_v2.max_concurrent_threads_per_session={max_agents}",
           "-s", "workspace-write", "-c", "approval_policy=never", "--json",
           "-o", str(workdir / "final.txt"), "-C", str(workdir),
           "-c", f'mcp_servers.grader.url="http://127.0.0.1:{port}/mcp"',
           "-c", "mcp_servers.grader.tool_timeout_sec=1200",
           "-c", "mcp_servers.grader.startup_timeout_sec=60",
           # auto-approve the grader's tools (valid enum: auto|prompt|writes|approve; non-interactive
           # exec cancels un-approved MCP calls). "auto" = run without prompting.
           "-c", 'mcp_servers.grader.default_tools_approval_mode="approve"']
    t0 = time.time()
    timed_out = False
    with open(prompt_file) as pf, open(workdir / "codex_stdout.jsonl", "w") as out:
        try:
            subprocess.run(cmd, stdin=pf, stdout=out, stderr=subprocess.STDOUT,
                           env=_base_env(), cwd=str(workdir), timeout=wall_clock)
        except subprocess.TimeoutExpired:
            timed_out = True
    return {"secs": round(time.time() - t0), "timed_out": timed_out}


def crosscheck_full(problem, workdir, port):
    """Grade the best grade_fast solution at FULL budget via the running server -> authoritative
    score (safety net if the agent never called grade_full itself)."""
    import asyncio
    logf = workdir / "grade_log.jsonl"
    if not logf.exists():
        return None
    maximize = MAXIMIZE[problem]
    best_hash, best_score = None, None
    for line in open(logf):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r["tool"] == "grade_fast" and r.get("valid") and r.get("score") is not None:
            if best_score is None or (r["score"] > best_score if maximize else r["score"] < best_score):
                best_score, best_hash = r["score"], r["sol_hash"]
    if best_hash is None:
        return None
    solf = workdir / "solutions" / f"{best_hash}.txt"
    if not solf.exists():
        return None
    solution = solf.read_text()

    async def call():
        from mcp.client.streamable_http import streamablehttp_client
        from mcp import ClientSession
        async with streamablehttp_client(f"http://127.0.0.1:{port}/mcp") as (r, w, _):
            async with ClientSession(r, w) as s:
                await s.initialize()
                res = await s.call_tool("grade_full", {"solution": solution})
                return json.loads(res.content[0].text)
    try:
        out = asyncio.run(call())
        return out.get("score") if out.get("valid") else None
    except Exception as e:
        print(f"  crosscheck failed: {e}", flush=True)
        return None


def best_from_log(problem, workdir):
    logf = workdir / "grade_log.jsonl"
    best, n_fast, n_full = None, 0, 0
    maximize = MAXIMIZE[problem]
    if logf.exists():
        for line in open(logf):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r["tool"] == "grade_fast":
                n_fast += 1
            if r["tool"] == "grade_full":
                n_full += 1
                if r.get("valid") and r.get("score") is not None:
                    if best is None or (r["score"] > best if maximize else r["score"] < best):
                        best = r["score"]
    return {"best_full": best, "n_grade_fast": n_fast, "n_grade_full": n_full}


def run_one(problem, args):
    tag = problem + args.tag_suffix + ("_smoke" if args.smoke else "")
    wd = Path(f"{REPO}/runs/benchmark_cdc/{tag}")
    wd.mkdir(parents=True, exist_ok=True)
    wall = args.wall_clock or (600 if args.smoke else 4 * 3600)
    max_agents = 8 if args.smoke else args.max_agents
    print(f"\n===== {problem}  (wall={wall}s, agents={max_agents}, backend={args.backend}) =====", flush=True)
    pf = build_prompt(problem, wd)
    port = free_port()
    grader = start_grader(problem, port, wd, args.backend, args.max_concurrent)
    cross = None
    try:
        info = run_codex(problem, pf, wd, port, wall, max_agents, reasoning=args.reasoning)
        if not args.no_crosscheck:
            cross = crosscheck_full(problem, wd, port)  # grade agent's best at full budget (grader still up)
    finally:
        grader.send_signal(signal.SIGINT)
        try:
            grader.wait(15)
        except subprocess.TimeoutExpired:
            grader.kill()
    logged = best_from_log(problem, wd)
    # authoritative best_full = best of {agent's own grade_full calls, our cross-check}
    cands = [x for x in (logged["best_full"], cross) if x is not None]
    if cands:
        logged["best_full"] = max(cands) if MAXIMIZE[problem] else min(cands)
    logged["crosscheck_full"] = cross
    res = {**info, **logged, "problem": problem, "port": port}
    (wd / "result.json").write_text(json.dumps(res, indent=2))
    print(f"  codex {info['secs']}s (timed_out={info['timed_out']}) | grade_fast={res['n_grade_fast']} "
          f"grade_full={res['n_grade_full']} | BEST grade_full={res['best_full']}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", nargs="*", default=ALL, choices=ALL)
    ap.add_argument("--smoke", action="store_true", help="short bounded run (10min, 8 agents)")
    ap.add_argument("--wall-clock", type=int, default=None, help="override per-problem seconds")
    ap.add_argument("--max-agents", type=int, default=32)
    ap.add_argument("--reasoning", default="high", help="model_reasoning_effort: low|medium|high|xhigh|max")
    ap.add_argument("--tag-suffix", default="", help="suffix on workdir tag so a scaling run doesn't clobber a prior run")
    ap.add_argument("--max-concurrent", type=int, default=24, help="grader parallel grade cap")
    ap.add_argument("--backend", choices=["ray", "thread"], default="thread")
    ap.add_argument("--no-crosscheck", action="store_true",
                    help="skip the driver's full-budget re-grade of the agent's best (faster smokes)")
    args = ap.parse_args()
    results = []
    for problem in args.problems:  # STRICTLY SERIAL — never two codex sessions at once
        try:
            results.append(run_one(problem, args))
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({"problem": problem, "best_full": None, "n_grade_fast": 0,
                            "n_grade_full": 0, "secs": 0, "error": f"{type(e).__name__}: {e}"})
    print("\n===== SUMMARY =====")
    for r in results:
        err = f"  ERROR: {r['error']}" if r.get("error") else ""
        print(f"  {r['problem']:<7} best_full={r['best_full']} "
              f"(fast={r['n_grade_fast']} full={r['n_grade_full']}, {r['secs']}s){err}")


if __name__ == "__main__":
    main()
