"""FastMCP streamable-HTTP grading server for the CDC deep-agent benchmark.

Per-problem (start with --problem). Exposes two tools the codex agent (and its multiagent-v2
sub-agents) call to optimize against the REAL reward:

  grade_fast(solution) -> reduced budget (~60s), cheap/approximate — iterate over many candidates
  grade_full(solution) -> full budget (1000s / native limits), authoritative — the reported metric

Grading reuses each env's `reward_function.get_reward(code, state)` with `eval_backend="local"`.
Concurrency backend is selectable: `--backend ray` (a local Ray cluster this process starts, grade
tasks fan out across the node's CPUs) or `--backend thread` (a bounded thread pool). Every call is
appended to <logdir>/grade_log.jsonl so we can recover the agent's full score trajectory + its best.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

# NB: must be a MODULE-level import. `from __future__ import annotations` turns the
# tool signatures into strings, and FastMCP resolves them against module globals --
# importing Context inside main() makes every @srv.tool() fail with InvalidSignature.
from mcp.server.fastmcp import Context, FastMCP

REPO = "/n/fs/vision-mix/sk7524/SkyRLTpu"
MAIN = f"{REPO}/third_party/discover"
FRONTIER = "/n/fs/vision-mix/sk7524/SkyRLTpu-frontiercs/third_party/discover"
GRPO_SNAPSHOT = f"{REPO}/runs/ttd_obj_grpo_16x32/tinker_log/erdos-20b-16x32-grpo/puct_sampler_step_000019.json"
FAST_BUDGET = int(os.environ.get("TTD_FAST_BUDGET", "10"))   # `check` = compile + full
                       # end-to-end execution smoke test. Per-PROBLEM tunable: fidelity of the
                       # fast ranking is budget-dependent and differs by problem (erdos rho .91
                       # at 10s, ac1 only .68), so a problem may buy rank fidelity with budget.
FAST_WALL = 240        # tolerant kill for `check`: budget-honoring programs return in ~10s, but a
                       # program with a hardcoded time floor still COMPLETES instead of hard-failing
                       # (hard-failing it would force budget-adaptive code only in the check-only arm)
FULL_WALL = 1100       # production run: the ONLY score that counts

# problem -> (discover_root, env_module, env_class, problem_type, lang)
PROBLEMS = {
    "erdos": (MAIN, "examples.erdos_min_overlap.env", "ErdosMinOverlapEnv", "", "python"),
    "ac1":   (MAIN, "examples.ac_inequalities.env", "AutoCorrInequalityEnv", "ac1", "python"),
    "ac2":   (MAIN, "examples.ac_inequalities.env", "AutoCorrInequalityEnv", "ac2", "python"),
    "fc46":  (FRONTIER, "examples.frontier_algo.env", "FrontierAlgoEnv", "46", "cpp"),
    "fc302": (FRONTIER, "examples.frontier_algo.env", "FrontierAlgoEnv", "302", "cpp"),
}


def _strip_fence(sol: str) -> str:
    m = re.search(r"```(?:python|cpp|c\+\+)?\s*([\s\S]*?)\s*```", sol)
    return m.group(1) if m else sol


def _grade(discover_root, env_module, env_class, problem_type, lang, base_construction,
           solution, fast, logdir):
    """Pure grading fn — runs in a ray worker OR a thread. Returns {score, valid, detail}."""
    if discover_root not in sys.path:
        sys.path.insert(0, discover_root)
    EnvClass = getattr(importlib.import_module(env_module), env_class)
    code = _strip_fence(solution or "")
    if lang == "python":
        if fast:
            # cap the candidate's self-budget so grade_fast is quick (needs the program to honor budget_s)
            # NB: the optional `: <annotation>` group is load-bearing. Without it this misses
            # type-annotated contracts like `def f(budget_s: int = 1000)` -- measured: it hit
            # 2 of 187 ac1 programs (all 187 declare a budget) while hitting 200/200 on erdos,
            # whose contract is unannotated. The result was NOT a cheap grade but a full-length
            # run guillotined by FAST_WALL, discarding 116/145 valid programs including the best.
            code = re.sub(r"budget_s\s*(?::\s*[A-Za-z_][\w\[\], ]*\s*)?=\s*\d+(?:\.\d+)?",
                          f"budget_s={FAST_BUDGET}", code)
        payload = f"```python\n{code}\n```"
        eval_timeout = FAST_WALL if fast else FULL_WALL
    else:  # cpp: the frontier judge compiles the raw source
        payload = code
        eval_timeout = 150
        os.environ["TTD_FCALGO_MAX_CASES"] = "1" if fast else "0"
    try:
        state = EnvClass.create_initial_state(problem_type)
        if base_construction is not None:
            state.construction = base_construction
        ev = EnvClass.reward_function(problem_type=problem_type, log_dir=logdir,
                                      num_cpus_per_task=1, eval_timeout=eval_timeout,
                                      eval_backend="local")
        out = ev.get_reward(payload, state)
        valid = out.get("correctness", 0) > 0
        return {"score": out.get("raw_score") if valid else None, "valid": bool(valid),
                "detail": (out.get("msg") or "")[:300]}
    except Exception as e:
        return {"score": None, "valid": False, "detail": f"{type(e).__name__}: {e}"[:300]}


def build_base_construction(problem, base_json=None):
    """Base construction handed to the candidate as `initial_h_values`.
    --base-json <path> overrides it (e.g. a WEAK 20b construction, so there is real headroom and a
    candidate cannot score the record for free by just returning initial_h_values).
    Otherwise: only Erdős overrides (the graded record, matching the 120b probe)."""
    if base_json:
        return list(json.loads(Path(base_json).read_text()))
    if problem != "erdos":
        return None
    sys.path.insert(0, f"{REPO}/tpu/distill_ablation")
    import common
    sampler = common.load_pool_sampler(GRPO_SNAPSHOT, "/tmp/grader_erdos_pool")
    with sampler._lock:
        best = max((s for s in sampler._states if s.value is not None and s.code and s.code.strip()),
                   key=lambda s: s.value)
    return list(best.construction)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True, choices=list(PROBLEMS))
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--backend", choices=["ray", "thread"], default="ray")
    ap.add_argument("--max-concurrent", type=int, default=24)
    ap.add_argument("--base-json", default=None,
                    help="JSON list overriding the base construction (weak-seed A/B)")
    ap.add_argument("--no-full", action="store_true",
                    help="disable grade_full (fast-check-only arm)")
    ap.add_argument("--max-full", type=int, default=3,
                    help="production grades allowed per rollout session (0 = unlimited)")
    args = ap.parse_args()
    Path(args.logdir).mkdir(parents=True, exist_ok=True)
    root, mod, cls, pt, lang = PROBLEMS[args.problem]
    base = build_base_construction(args.problem, args.base_json)
    print(f"[grader] problem={args.problem} lang={lang} base_len="
          f"{len(base) if base else 0} backend={args.backend} port={args.port}", flush=True)

    task_args = (root, mod, cls, pt, lang, base)

    if args.backend == "ray":
        import ray
        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
        _remote = ray.remote(num_cpus=1)(_grade)

        def _run(solution, fast):
            return ray.get(_remote.remote(*task_args, solution, fast, args.logdir))
    else:
        def _run(solution, fast):
            return _grade(*task_args, solution, fast, args.logdir)

    logf = Path(args.logdir) / "grade_log.jsonl"
    loglock = threading.Lock()
    sem = asyncio.Semaphore(args.max_concurrent)

    def _log(rec):
        with loglock, open(logf, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    soldir = Path(args.logdir) / "solutions"
    soldir.mkdir(exist_ok=True)

    sessdir = Path(args.logdir) / "sessions"

    def _time_left(session):
        """Rollout wall remaining, from the deadline file the driver writes before launching."""
        if not session:
            return None
        f = sessdir / f"{session}.json"
        if not f.exists():
            return None
        try:
            return max(0, int(json.loads(f.read_text())["deadline"] - time.time()))
        except Exception:
            return None

    # Per-session record of which exact programs have a VALID production grade, so submit() can
    # require that the submitted program is the one that was actually verified (not a later edit).
    verified: dict[str, dict[str, dict]] = {}
    submitted: set[str] = set()
    vlock = threading.Lock()
    subf = Path(args.logdir) / "submissions.jsonl"

    def _sha(s):
        return hashlib.sha1((s or "").encode()).hexdigest()[:12]

    async def _do(solution, fast, session="", akey=None):
        h = _sha(solution)
        solf = soldir / f"{h}.txt"
        if not solf.exists():  # keep the raw solution so the driver can re-grade the best at full budget
            try:
                solf.write_text(solution or "")
            except Exception:
                pass
        t0 = time.time()
        async with sem:
            res = await asyncio.to_thread(_run, solution, fast)
        _log({"tool": "grade_fast" if fast else "grade_full", "problem": args.problem,
              "sol_hash": h, "score": res["score"], "valid": res["valid"],
              "detail": res.get("detail", "")[:200], "session": session,
              "secs": round(time.time() - t0, 1), "ts": round(time.time(), 1)})
        if (not fast) and res.get("valid"):
            with vlock:
                verified.setdefault(akey or session, {})[h] = {"score": res["score"],
                                                               "ts": time.time()}
        left = _time_left(session)
        if left is not None:
            res = {**res, "session_seconds_left": left}
        return res

    identf = Path(args.logdir) / "identity.jsonl"

    def _ident(session, ctx, tool):
        if ctx is None:
            return
        try:
            rec = {"tool": tool, "claimed_session": session, **_identity(ctx),
                   "ts": round(time.time(), 1)}
            with loglock, open(identf, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        except Exception:
            pass

    def _full_count(session):
        with vlock:
            return len(verified.get(session, {}))

    def _agent_key(ctx, claimed):
        """Per-AGENT identity for budget enforcement.

        Native subagents all inherit the orchestrator's prompt, so they all pass the SAME
        `session` string (measured: 57/58 calls claimed "native_orch"). The MCP transport,
        however, gives each agent its own connection with a distinct `mcp-session-id` header
        (measured: 8 agents -> 8 ids, 1:1 with the connection). So budgets key off the header
        and fall back to the claimed string only if it is unavailable."""
        try:
            req = getattr(ctx.request_context, "request", None)
            if req is not None:
                sid = dict(getattr(req, "headers", {}) or {}).get("mcp-session-id")
                if sid:
                    return f"mcp:{sid}"
        except Exception:
            pass
        try:
            return f"conn:{hex(id(ctx.session))[-8:]}"
        except Exception:
            return claimed or "unknown"

    def _identity(ctx):
        """Transport-level identity, independent of the `session` string the agent passes.
        If codex subagents open their own MCP connections, these differ per subagent and we can
        enforce per-agent budgets server-side instead of trusting self-reported session ids."""
        info = {}
        try:
            info["client_id"] = getattr(ctx, "client_id", None)
        except Exception:
            pass
        try:
            info["conn"] = hex(id(ctx.session))[-8:]
        except Exception:
            pass
        try:
            req = getattr(ctx.request_context, "request", None)
            if req is not None:
                h = dict(getattr(req, "headers", {}) or {})
                info["hdrs"] = {k: v[:40] for k, v in h.items()
                                if k.lower() in ("mcp-session-id", "user-agent", "x-agent-id")}
        except Exception:
            pass
        return info

    srv = FastMCP("grader", host="127.0.0.1", port=args.port)

    @srv.tool()
    async def check(solution: str, session: str = "", ctx: Context = None) -> dict:
        """SMOKE TEST: compiles your program and runs it END-TO-END with budget_s=10 (~10s, killed
        at 50s). Confirms it executes and returns a valid construction, and gives a ROUGH score at
        10s of compute. Cheap -- call it as often as you like. NOTE: this is NOT your final number;
        production runs your program with budget_s=1000 under a hard 1100s wall, so a program tuned
        to finish in 10s will waste 99% of its real budget -- scale work to the budget_s you get.
        Pass `session` (given in your instructions) and the reply includes your remaining wall time.
        Returns {"score": float|null, "valid": bool, "detail": str, "session_seconds_left": int}."""
        _ident(session, ctx, "check")
        return await _do(solution, True, session)

    @srv.tool()
    async def submit(program: str, approach: str, insight: str, session: str = "",
                     ctx: Context = None) -> dict:
        """FINISH YOUR TASK. Records your final program plus a two-line reflection.

        REQUIREMENT: the EXACT program you submit must already have a successful grade_full() in
        this session -- submit fails otherwise. Workflow: check() to screen ideas, grade_full() to
        verify your best one, then submit() that same text.
          approach: what idea this solution tried
          insight:  what should be learned for future improvement
        Returns {"accepted": bool, "score": float|null, "detail": str}."""
        _ident(session, ctx, "submit")
        akey = _agent_key(ctx, session) if ctx is not None else session
        h = _sha(program)
        with vlock:
            if akey and akey in submitted:
                return {"accepted": False, "score": None,
                        "detail": "REJECTED: this session already submitted. One submission each."}
            # Anti-substitution is a property of the PROGRAM, not the connection: the hash is
            # the proof it was production-graded. Look it up globally so an agent that reconnects
            # mid-session (websocket drops are common here) can still submit its own verified work.
            # The per-agent key still governs the grade_full BUDGET below.
            rec = verified.get(akey, {}).get(h)
            if rec is None:
                for _k, _m in verified.items():
                    if h in _m:
                        rec = _m[h]
                        break
        if rec is None:
            n = _full_count(akey)
            return {"accepted": False, "score": None,
                    "detail": (f"REJECTED: this exact program has no successful grade_full() in "
                               f"session '{session}' ({n} verified so far). Call "
                               f"grade_full(solution=<this exact text>, session='{session}') "
                               f"first, then submit the identical text.")}
        (soldir / f"{h}.txt").write_text(program)
        row = {"session": session, "agent_key": akey, "sol_hash": h, "score": rec["score"],
               "approach": (approach or "")[:600], "insight": (insight or "")[:600],
               "ts": round(time.time(), 1)}
        with vlock:
            submitted.add(akey)
        with loglock, open(subf, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        return {"accepted": True, "score": rec["score"],
                "detail": "Submission recorded. You are done -- end the session."}

    if not args.no_full:
        @srv.tool()
        async def grade_full(solution: str, session: str = "", ctx: Context = None) -> dict:
            """PRODUCTION RUN: your program executed with budget_s=1000 under a hard 1100s wall --
            no more, no less. THIS IS THE ONLY SCORE THAT COUNTS. Slow (up to ~18 min), so screen
            with check() first and spend this on a candidate you believe in. Pass `session` to get
            your remaining wall time back.
            Returns {"score": float|null, "valid": bool, "detail": str, "session_seconds_left": int}."""
            akey = _agent_key(ctx, session) if ctx is not None else session
            if args.max_full and _full_count(akey) >= args.max_full:
                return {"score": None, "valid": False,
                        "detail": (f"budget exhausted: {args.max_full} production grades per "
                                   f"session. Submit the best one you already verified.")}
            _ident(session, ctx, "grade_full")
            return await _do(solution, False, session, akey)

    srv.run(transport="streamable-http")


if __name__ == "__main__":
    main()
