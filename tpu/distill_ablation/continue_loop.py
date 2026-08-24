"""Outer continuation loop: RESUME each problem's existing codex session (full context intact)
and push it to keep improving, until `patience` consecutive iterations show no improvement (or
max-iters). Builds directly off the completed benchmark runs — reuses the grade_log, the saved
solutions, and the codex session. Strictly serial (one codex session live at a time).

Per problem: recover the session id from the run's codex_stdout.jsonl, start the grader on a stable
port (same logdir -> grade_log ACCUMULATES), then repeatedly `codex exec resume <session> "<keep
going, beat X>"`, reading the new best from the accumulated log + full-budget cross-check each round.
"""
from __future__ import annotations

import argparse
import json
import signal
import subprocess
import time
from pathlib import Path

import benchmark_cdc as B  # reuse start_grader / free_port / _base_env / best_from_log / crosscheck_full / MAXIMIZE

FENCE = {"erdos": "python", "ac1": "python", "ac2": "python", "fc46": "cpp", "fc302": "cpp"}


def session_id(workdir):
    f = workdir / "codex_stdout.jsonl"
    if not f.exists():
        return None
    for line in open(f):
        try:
            e = json.loads(line)
            if e.get("type") == "thread.started":
                return e["thread_id"]
        except Exception:
            pass
    return None


def continuation_prompt(problem, best, n_fast, n_full):
    direction = "higher" if B.MAXIMIZE[problem] else "lower"
    return f"""You are CONTINUING your earlier work on this exact problem, with all of your prior context. \
Your best VALIDATED grade_full score so far is {best} ({direction} is better). You have made about \
{n_fast} grade_fast and {n_full} grade_full calls across several approach families.

Do NOT stop, and do NOT merely repeat approaches that already plateaued. Push further: launch NEW \
approach families you have not fully explored, cross-pollinate your strongest ideas, and try \
fundamentally different constructions/algorithms to BEAT {best}. Screen many candidates with \
grade_fast and confirm promising ones with grade_full. Keep an explicit approach-family registry \
and audit every candidate adversarially before trusting its score.

When you have a genuinely new best (or have exhausted distinct new ideas), return your single best \
solution as one ```{FENCE[problem]} code block that passes grade_full."""


def run_resume(session, prompt, workdir, port, wall, max_agents, reasoning="high"):
    # NB: `codex exec resume` does NOT accept -s/-C (sandbox + cwd are inherited from the session);
    # the subprocess cwd= sets the working dir. Only -c/-m/--enable/--json/-o are valid here.
    cmd = ["codex", "exec", "resume", session, "-m", B.MODEL, "-c", f"model_reasoning_effort={reasoning}",
           "--enable", "multi_agent_v2",
           "-c", f"features.multi_agent_v2.max_concurrent_threads_per_session={max_agents}",
           "-c", "approval_policy=never", "--json",
           "-o", str(workdir / "resume_final.txt"),
           "-c", f'mcp_servers.grader.url="http://127.0.0.1:{port}/mcp"',
           "-c", "mcp_servers.grader.tool_timeout_sec=1200",
           "-c", "mcp_servers.grader.startup_timeout_sec=60",
           "-c", 'mcp_servers.grader.default_tools_approval_mode="approve"', "-"]
    t0 = time.time()
    timed_out = False
    with open(workdir / "resume_stdout.jsonl", "a") as out:
        try:
            subprocess.run(cmd, input=prompt, text=True, stdout=out, stderr=subprocess.STDOUT,
                           env=B._base_env(), cwd=str(workdir), timeout=wall)
        except subprocess.TimeoutExpired:
            timed_out = True
    return round(time.time() - t0), timed_out


def loop_one(problem, args):
    wd = Path(f"{B.REPO}/runs/benchmark_cdc/{problem}{args.tag_suffix}")
    session = session_id(wd)
    if not session:
        print(f"  {problem}: no session id found — skipping"); return None
    maximize = B.MAXIMIZE[problem]
    port = B.free_port()
    grader = B.start_grader(problem, port, wd, args.backend, args.max_concurrent)
    hist = []
    best = None
    try:
        logged = B.best_from_log(problem, wd)
        best = logged["best_full"]
        if best is None:
            best = B.crosscheck_full(problem, wd, port)
        hist.append({"stage": "start", "best": best})
        print(f"\n===== CONTINUE {problem} (session {session[:13]}, start best={best}) =====", flush=True)
        no_improve, it = 0, 0
        while no_improve < args.patience and it < args.max_iters:
            it += 1
            lg = B.best_from_log(problem, wd)
            prompt = continuation_prompt(problem, best, lg["n_grade_fast"], lg["n_grade_full"])
            secs, to = run_resume(session, prompt, wd, port, args.iter_wall, args.max_agents, reasoning=args.reasoning)
            cross = B.crosscheck_full(problem, wd, port)
            nl = B.best_from_log(problem, wd)
            cands = [x for x in (nl["best_full"], cross) if x is not None]
            newbest = (max(cands) if maximize else min(cands)) if cands else best
            improved = best is None or (newbest > best + args.eps if maximize else newbest < best - args.eps)
            hist.append({"stage": f"iter{it}", "best": newbest, "improved": bool(improved),
                         "secs": secs, "timed_out": to})
            if improved:
                best = newbest; no_improve = 0
            else:
                no_improve += 1
            print(f"  {problem} iter{it}: best={newbest} "
                  f"{'IMPROVED' if improved else f'flat({no_improve}/{args.patience})'} ({secs}s)", flush=True)
    finally:
        grader.send_signal(signal.SIGINT)
        try:
            grader.wait(15)
        except subprocess.TimeoutExpired:
            grader.kill()
    (wd / "continue_result.json").write_text(json.dumps(
        {"problem": problem, "session": session, "final_best": best, "history": hist}, indent=2, default=str))
    print(f"  {problem} DONE: final_best={best}", flush=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", nargs="*", default=B.ALL, choices=B.ALL)
    ap.add_argument("--iter-wall", type=int, default=1800, help="seconds per resume iteration")
    ap.add_argument("--patience", type=int, default=2, help="stop after this many non-improving iters")
    ap.add_argument("--max-iters", type=int, default=6)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--max-agents", type=int, default=32)
    ap.add_argument("--reasoning", default="high", help="model_reasoning_effort: low|medium|high|xhigh|max")
    ap.add_argument("--tag-suffix", default="", help="operate on the <problem><suffix> workdir (match benchmark_cdc)")
    ap.add_argument("--max-concurrent", type=int, default=24)
    ap.add_argument("--backend", choices=["ray", "thread"], default="thread")
    args = ap.parse_args()
    for p in args.problems:  # STRICTLY SERIAL
        try:
            loop_one(p, args)
        except Exception:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
