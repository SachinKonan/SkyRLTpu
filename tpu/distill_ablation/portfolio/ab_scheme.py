"""A/B: which feedback scheme evolves a suboptimal program better, at group size G in parallel?

  ARM "fast"  -- 10 min wall, ONLY check() (compile + end-to-end run at budget_s=10, ~50s cap),
                 unlimited calls. Cheap, many turns, no production grade during search.
  ARM "full"  -- 30 min wall, check() + grade_full() (budget_s=1000, hard 1100s wall).

FAIRNESS: the arms differ only in the feedback available during search. The REPORTED score for
both is identical -- the driver independently runs each produced program as a PRODUCTION run
(budget_s=1000 / 1100s wall) after the fact. We also re-run it at budget_s=10 to measure
budget-adaptivity (does the program actually use the compute it is given?).

Seed = one weak gpt-oss-20b program from corpora/worse_set.json (c5 band 0.3814-0.3818, i.e. ~9e-4
of headroom to the 0.380928 record). Its construction is ALSO installed as `initial_h_values` via
--base-json, so a candidate cannot score the record for free by returning the base.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import benchmark_cdc as B
from prompts import TASK, mine_failure_patterns

PF = Path(__file__).parent
AB = PF.parent

ARMS = {  # arm -> (wall_seconds, expose_grade_full)
    "fast": (600, False),
    "full": (1800, True),
}


def arm_prompt(problem, arm, seed_program, seed_score, wall, session, grade_log):
    tools = ("`check(solution, session)` -- compiles and RUNS your program end-to-end with "
             "budget_s=10 (killed at 50s) and returns a rough 10s score. Call it as often as you like.")
    if ARMS[arm][1]:
        tools += ("\n`grade_full(solution, session)` -- the PRODUCTION run: budget_s=1000 under a hard "
                  "1100s wall. This is the only score that counts. Slow (up to ~18 min) -- screen with "
                  "check() first, then spend it on your best candidate.")
        final = ("Confirm your final answer with grade_full before you run out of time.")
    else:
        final = ("You have NO access to the production grader during this session -- check() is a smoke "
                 "test only. Your program will be scored later by a single production run.")
    parts = [
        TASK[problem], "",
        "TOOLS (MCP server `grader`) -- always pass session=\"%s\" so the reply tells you your "
        "remaining wall time:\n%s" % (session, tools), "",
        "SCORING -- read carefully. Your program is finally scored by ONE PRODUCTION RUN: "
        "run(budget_s=1000) under a hard 1100s wall, no more and no less. check() only proves it "
        "compiles and runs; a 10s score is a weak proxy. Therefore your program MUST scale its work "
        "to the budget_s argument it receives (use ~80-90% of it) -- a program that finishes in 10s "
        "when handed 1000s throws away 99% of its budget and will not win. Local runs in your own "
        "sandbox are fine for development but count for nothing.", "",
        f"You have {wall // 60} minutes of wall clock for this session. {final}", "",
        f"### starting program (its production score is {seed_score}) -- improve on it substantially:",
        f"```python\n{seed_program}\n```",
    ]
    fp = mine_failure_patterns(Path(grade_log))
    if fp:
        parts += ["", fp]
    parts += ["", "Write your single best program to solution.txt as one ```python block."]
    return "\n".join(parts)


async def _one(port, solution, tool):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    try:
        async with streamablehttp_client(f"http://127.0.0.1:{port}/mcp") as (r, w, _):
            async with ClientSession(r, w) as s:
                await s.initialize()
                res = await s.call_tool(tool, {"solution": solution})
                return json.loads(res.content[0].text)
    except Exception as e:
        return {"score": None, "valid": False, "detail": f"driver-grade failed: {e}"[:200]}


def grade_via_server(port, solution, tool="grade_full"):
    return asyncio.run(_one(port, solution, tool))


def grade_many(port, jobs):
    """jobs = [(key, solution, tool)] graded CONCURRENTLY (the grader has its own semaphore).
    Sequential production runs would cost 10 x up to 18min per arm; concurrent is one wave."""
    async def go():
        res = await asyncio.gather(*[_one(port, sol, tool) for _, sol, tool in jobs])
        return {k: r for (k, _, _), r in zip(jobs, res)}
    return asyncio.run(go())


def extract_program(wd: Path, adir: Path, session: str, maximize: bool):
    """Prefer solution.txt; else fall back to the best solution this session passed to check()
    (critical for the fast arm, which has no grade_full to leave a trail)."""
    f = wd / "solution.txt"
    if f.exists() and f.read_text().strip():
        return f.read_text(), "solution.txt"
    glog = adir / "grade_log.jsonl"
    if not glog.exists():
        return None, "no solution.txt, no grade_log"
    best = None
    for line in glog.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("session") != session or not r.get("valid") or r.get("score") is None:
            continue
        if best is None or (r["score"] > best["score"] if maximize else r["score"] < best["score"]):
            best = r
    if best is None:
        return None, "no valid checked solution"
    sf = adir / "solutions" / f"{best['sol_hash']}.txt"
    if not sf.exists():
        return None, "checked solution file missing"
    return sf.read_text(), f"salvaged from check (10s score {best['score']})"


def start_grader(problem, adir, base_json, expose_full, max_concurrent):
    port = B.free_port()
    cmd = [B.PY, f"{AB}/grading_mcp.py", "--problem", problem, "--port", str(port),
           "--logdir", str(adir), "--backend", "thread",
           "--max-concurrent", str(max_concurrent), "--base-json", str(base_json)]
    if not expose_full:
        cmd.append("--no-full")
    p = subprocess.Popen(cmd, stdout=open(adir / "grader.log", "w"),
                         stderr=subprocess.STDOUT, env=B._base_env(), cwd=str(AB))
    for _ in range(180):
        try:
            __import__("socket").create_connection(("127.0.0.1", port), 1).close()
            return p, port
        except OSError:
            if p.poll() is not None:
                raise RuntimeError(f"grader died; see {adir}/grader.log")
            time.sleep(1)
    raise RuntimeError("grader did not bind")


def stop(p):
    p.send_signal(signal.SIGINT)
    try:
        p.wait(15)
    except subprocess.TimeoutExpired:
        p.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="erdos")
    ap.add_argument("--arms", nargs="+", default=["fast", "full"], choices=list(ARMS))
    ap.add_argument("--group", type=int, default=10, help="parallel rollouts per arm")
    ap.add_argument("--seed-idx", type=int, default=None,
                    help="worse_set.json index; default = median c5 of the band")
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--reasoning", default="high")
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-concurrent", type=int, default=12)
    args = ap.parse_args()

    # ---- weak 20b seed ----
    ws = json.loads((AB / "corpora/worse_set.json").read_text())["worse"]
    ranked = sorted(range(len(ws)), key=lambda i: -ws[i]["value"])  # value = -c5 -> asc c5
    idx = args.seed_idx if args.seed_idx is not None else ranked[len(ranked) // 2]
    seed = ws[idx]
    seed_c5 = -seed["value"]
    seed_prog = seed["code"]
    if "```" in seed_prog:  # strip fence for embedding
        import re
        m = re.search(r"```(?:python)?\s*([\s\S]*?)\s*```", seed_prog)
        seed_prog = m.group(1) if m else seed_prog

    root = Path(f"{B.REPO}/runs/ab_scheme/{args.problem}{args.tag}")
    root.mkdir(parents=True, exist_ok=True)
    base_json = root / "base_construction.json"
    base_json.write_text(json.dumps(list(seed["construction"])))
    print(f"[ab] seed worse_set[{idx}] recorded c5={seed_c5:.9f} (record 0.380928; headroom "
          f"{seed_c5 - 0.380928:.2e}) prog_chars={len(seed_prog)}", flush=True)

    # ---- MEASURE the baseline: the seed program's own production score, with its construction
    # installed as initial_h_values. This (not the recorded metadata value) is what both arms beat.
    bdir = root / "baseline"; bdir.mkdir(exist_ok=True)
    bg, bport = start_grader(args.problem, bdir, base_json, True, args.max_concurrent)
    try:
        seed_sol = f"```python\n{seed_prog}\n```"
        bout = grade_via_server(bport, seed_sol, "grade_full")
        b10 = grade_via_server(bport, seed_sol, "check")
    finally:
        stop(bg)
    seed_prod = bout.get("score") if bout.get("valid") else None
    print(f"[ab] BASELINE seed production score = {seed_prod} (10s={b10.get('score')}) "
          f"detail={bout.get('detail','')[:90]}", flush=True)
    if seed_prod is None:
        print("[ab] WARNING: seed program did not grade validly; arms still run but baseline unknown",
              flush=True)
    baseline_str = f"{seed_prod:.9f}" if seed_prod is not None else f"~{seed_c5:.9f}"

    results = {}
    for arm in args.arms:
        wall, expose_full = ARMS[arm]
        adir = root / arm
        (adir / "sessions").mkdir(parents=True, exist_ok=True)
        glog = adir / "grade_log.jsonl"

        grader, port = start_grader(args.problem, adir, base_json, expose_full,
                                    args.max_concurrent)
        print(f"\n[ab] ===== ARM {arm}: wall={wall}s grade_full={expose_full} "
              f"group={args.group} =====", flush=True)
        procs = []
        try:
            t0 = time.time()
            for g in range(args.group):
                wd = adir / f"x{g}"
                wd.mkdir(exist_ok=True)
                session = f"{arm}_x{g}"
                (adir / "sessions" / f"{session}.json").write_text(
                    json.dumps({"deadline": time.time() + wall}))
                prompt = arm_prompt(args.problem, arm, seed_prog, baseline_str,
                                    wall, session, glog)
                (wd / "prompt.txt").write_text(prompt)
                p = subprocess.Popen(
                    ["codex", "exec", "-m", args.model, "-c",
                     f"model_reasoning_effort={args.reasoning}",
                     "-s", "workspace-write", "-c", "approval_policy=never", "--json",
                     "-C", str(wd),
                     "-c", f'mcp_servers.grader.url="http://127.0.0.1:{port}/mcp"',
                     "-c", "mcp_servers.grader.tool_timeout_sec=1200",
                     "-c", "mcp_servers.grader.startup_timeout_sec=60",
                     "-c", 'mcp_servers.grader.default_tools_approval_mode="approve"'],
                    stdin=subprocess.PIPE, stdout=open(wd / "codex.jsonl", "w"),
                    stderr=subprocess.STDOUT, env=B._base_env(), cwd=str(wd), text=True)
                p.stdin.write(prompt); p.stdin.close()
                procs.append((g, wd, p))
            # all G run in parallel; kill any that overrun the arm wall
            for g, wd, p in procs:
                left = max(30, wall + 120 - int(time.time() - t0))
                try:
                    p.wait(timeout=left)
                except subprocess.TimeoutExpired:
                    p.kill()
            print(f"[ab] arm {arm}: rollouts done in {int(time.time()-t0)}s", flush=True)

            # ---- identical yardstick: driver production-runs every program ----
            mx = B.MAXIMIZE[args.problem]
            progs, rows = {}, []
            for g, wd, _ in procs:
                prog, src = extract_program(wd, adir, f"{arm}_x{g}", mx)
                progs[g] = (prog, src)
            jobs = []
            for g, (prog, _) in progs.items():
                if prog is not None:
                    jobs.append((("prod", g), prog, "grade_full"))   # identical yardstick
                    jobs.append((("s10", g), prog, "check"))         # budget-adaptivity probe
            # The check-only arm's server has --no-full, which also blocks the DRIVER's yardstick
            # call. Swap in a full-enabled grader now that the agents are done.
            if not expose_full:
                stop(grader)
                grader, port = start_grader(args.problem, adir, base_json, True,
                                            args.max_concurrent)
                print(f"[ab] arm {arm}: swapped in full-enabled grader for driver grading",
                      flush=True)
            print(f"[ab] arm {arm}: grading {len(jobs)//2} programs concurrently "
                  f"(production 1000s/1100s)...", flush=True)
            tg = time.time()
            graded = grade_many(port, jobs) if jobs else {}
            print(f"[ab] arm {arm}: grading wave done in {int(time.time()-tg)}s", flush=True)
            for g, (prog, src) in progs.items():
                if prog is None:
                    rows.append({"x": g, "prod": None, "s10": None, "note": src})
                    print(f"[ab] {arm} x{g}: NO PROGRAM ({src})", flush=True)
                    continue
                pr, s1 = graded.get(("prod", g), {}), graded.get(("s10", g), {})
                rows.append({"x": g, "chars": len(prog), "src": src,
                             "prod": pr.get("score") if pr.get("valid") else None,
                             "s10": s1.get("score") if s1.get("valid") else None,
                             "note": pr.get("detail", "")[:120]})
                r = rows[-1]
                adapt = ("n/a" if r["prod"] is None or r["s10"] is None
                         else f"{r['s10'] - r['prod']:+.2e}")
                print(f"[ab] {arm} x{g}: production={r['prod']} (10s={r['s10']}, "
                      f"uses-budget={adapt}) [{src}]", flush=True)
        finally:
            grader.send_signal(signal.SIGINT)
            try:
                grader.wait(15)
            except subprocess.TimeoutExpired:
                grader.kill()

        ok = [r for r in rows if r["prod"] is not None]
        bestv = (max if mx else min)([r["prod"] for r in ok]) if ok else None
        beat = ([r for r in ok if (r["prod"] > seed_prod if mx else r["prod"] < seed_prod)]
                if seed_prod is not None else [])
        results[arm] = {"rows": rows, "valid": len(ok), "n": args.group, "best": bestv,
                        "beat_baseline": len(beat),
                        "mean": (sum(r["prod"] for r in ok) / len(ok)) if ok else None,
                        "calls": len(glog.read_text().splitlines()) if glog.exists() else 0}
        (root / "results.json").write_text(json.dumps(
            {"seed_idx": idx, "seed_recorded_c5": seed_c5, "seed_production": seed_prod,
             "arms": results}, indent=2))
        print(f"[ab] ARM {arm} SUMMARY valid={len(ok)}/{args.group} best={bestv} "
              f"beat_baseline={len(beat)}/{args.group} mean={results[arm]['mean']} "
              f"grade_calls={results[arm]['calls']}", flush=True)

    print(f"\n[ab] ============ A/B RESULT (baseline production = {seed_prod}) ============")
    for arm, r in results.items():
        print(f"  {arm:<5} wall={ARMS[arm][0]:>4}s full={str(ARMS[arm][1]):<5} | "
              f"valid {r['valid']}/{r['n']} | beat-baseline {r['beat_baseline']}/{r['n']} | "
              f"best {r['best']} | mean {r['mean']} | grade calls {r['calls']}")


if __name__ == "__main__":
    main()
