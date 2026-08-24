"""Experiment 1: which MAP-REDUCE protocol turns N subagent rollouts into the best solution?

  --arm native      codex multi_agent_v2 decides its own map / verify / reduce. The orchestrator
                    (sol) spawns its own subagents (terra) and reduces however it likes. It is
                    PERMITTED up to N; we report what it actually spawns -- last time it used
                    15 of an allowed 32, so the count is its choice, not ours.
  --arm mechanical  the driver forks exactly N terra rollouts, collects their submissions, and
                    hands a compact digest to one sol orchestrator to reduce.

Both arms: same problem, same seed, same models, same wall budget, and every rollout is
submit()-gated -- a submission is only accepted if that EXACT program already has a successful
grade_full() in the session, so all reported scores are server-verified rather than self-claimed.

Matched on wall-clock and session count, NOT on agent count (arm A will not comply, and that
non-compliance is itself a result).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import benchmark_cdc as B
from sweep import (FENCE, env_with_home, grade_many, load_seed, prepare_codex_home,
                   start_grader, stop, strip_fence, task_text)

PF = Path(__file__).parent
AB = PF.parent

CONTRACT = """
TOOLS (`grader` MCP) -- always pass session="{sess}":
  check(solution, session)      fast screen; cheap, call it freely.
  grade_full(solution, session) the PRODUCTION run -- the only score that counts. Limited to
                                {max_full} calls, so spend them on candidates you believe in.
  submit(program, approach, insight, session)   finish. REJECTED unless that EXACT program text
                                already has a successful grade_full() in this session.

WORKFLOW: write a program -> check() to screen -> grade_full() your best -> submit() that same
text. Your task is not complete until submit() returns accepted=true.
  approach: what idea this solution tried
  insight:  what should be learned for future improvement

Local shell access is restricted on this machine; check() compiles and runs your program for you,
so you do not need a shell. Embedding a precomputed answer as data is invalid.
"""


def rollout_prompt(problem, sess, seed_prog, max_full, wall_min):
    return "\n".join([
        task_text(problem), "",
        CONTRACT.format(sess=sess, max_full=max_full),
        f"You have ~{wall_min} minutes.", "",
        "## Improve substantially on this starting program:",
        f"```{FENCE[problem]}\n{strip_fence(seed_prog)}\n```",
    ])


# Two orchestration styles for the native arm. The MINIMAL one is permissive ("use as many as you
# judge useful") -- and the model judges ~3, regardless of whether the cap is 3 or 50, because it
# treats itself as the solver and subagents as side threads. The CDC one is OpenAI's scaffolding
# from the original benchmark, which is directive about aggressive fan-out AND carries explicit
# anti-convergence machinery (don't reveal the favoured approach, track approach families, redirect
# off crowded ones). Under it the same model spawned 15. Prompt, not protocol, drives fan-out.
CDC_ORCHESTRATION = """You have a large team of agents available through multiagent v2. Use them \
aggressively and dynamically. Your budget is {n} SUBAGENT CALLS IN TOTAL over the whole session \
(counting every role -- generators, verifiers, adversarial checkers, refiners); at most {conc} may \
run CONCURRENTLY. Spend close to the full budget: an unused agent is wasted search. Do NOT use a \
fixed assignment like "N agents for approach X." Manage the search with these heuristics:
- Begin with a genuinely diverse portfolio: substantially different formulations, algorithms, \
heuristics, representations, decompositions, relaxations, and computational sanity checks.
- Do not tell most agents the currently favored approach -- preserve independence in early rounds \
so the team does not all converge on one attractive-but-mediocre idea.
- Maintain an explicit registry of approach families (grouped by the underlying idea, not wording). \
If many agents converge to one family, redirect some toward underexplored ones.
- Keep several incompatible approaches alive across rounds; cross-pollinate only after independent \
agents have developed them far enough to expose their real strengths and gaps.
- Use adversarial agents throughout: every candidate must be checked and its score confirmed with \
the grader before you trust it.
- Require agents to return concrete programs and their measured scores -- reject status reports, \
vague optimism, and untested claims.
- The root agent should repeatedly synthesize, challenge, redirect, and launch new rounds. Do not \
stop after the first wave fails. Keep searching until your time budget is nearly exhausted; then \
submit the single strongest VALIDATED solution you found.

Spawn subagents with spawn_agent(task_name, message, model, reasoning_effort) using
model="gpt-5.6-terra". Subagents inherit your tools, including the grader."""

MINIMAL_ORCHESTRATION = """You have a multi-agent capability: you may spawn subagents with
spawn_agent(task_name, message, model, reasoning_effort). Use model="gpt-5.6-terra". You are
permitted {n} subagent calls in total ({conc} concurrent) -- use as many or as few as you judge useful, in whatever pattern of
waves, verification and merging you think works best. Subagents inherit your tools, including the
grader."""

NATIVE_PROMPT = """You are orchestrating a search for a better solution to the problem below.

{orchestration}

{contract}

Coordinate however you like, then finish by calling submit() yourself with the single best program
you have verified. Remember submit() requires a successful grade_full() of that exact text.

## Starting program to improve on:
```{fence}
{seed}
```

## Problem
{task}
"""

REDUCE_PROMPT = """You are the reducer in a map-reduce search. {n} independent agents each tried to
improve the same starting program. Every score below was INDEPENDENTLY VERIFIED by the harness --
these are not self-reported numbers.

{digest}

Your job: produce a single program that is better than any of them. Study what the successful
approaches have in common, what the failures ran into, and combine the complementary strengths --
do not simply copy the current best.

{contract}

Finish by calling submit() with your final program (grade_full it first). Also include, in your
`insight` field, one sentence on which combination of prior approaches you drew on.

## Problem
{task}
"""


def digest(subs, rows, k=5, maximize=True):
    """All-N table (verified score + reflection + failure), full source for the top-k only.
    Full transcripts are deliberately excluded: 50 of them is ~1M tokens and mostly shell noise,
    and the native arm's orchestrator only ever sees its subagents' final messages either way."""
    ok = sorted([s for s in subs if s.get("score") is not None],
                key=lambda s: s["score"], reverse=maximize)
    L = [f"{len(subs)} rollouts, {len(ok)} produced a verified submission.", "",
         "session | verified_score | approach | insight"]
    for s in subs:
        L.append(f"{s.get('session','?')} | {s.get('score')} | "
                 f"{(s.get('approach') or '')[:160]} | {(s.get('insight') or '')[:160]}")
    fails = [r for r in rows if r.get("note")]
    if fails:
        L += ["", "Failure patterns across rollouts:"]
        for f in fails[:8]:
            L.append(f"  - {f['note'][:150]}")
    L += ["", f"Full source of the top {min(k, len(ok))} verified programs:"]
    for s in ok[:k]:
        L.append(f"\n### {s['session']}  (verified score {s['score']})\n"
                 f"```\n{strip_fence(s.get('program',''))}\n```")
    return "\n".join(L)


def run_codex(model, effort, prompt, wd, port, wall, tag, ch, extra=()):
    cmd = ["codex", "exec", "--strict-config", "-m", model,
           "-c", f"model_reasoning_effort={effort}",
           "-s", "workspace-write", "-c", "approval_policy=never", "--json", "-C", str(wd),
           "-c", f'mcp_servers.grader.url="http://127.0.0.1:{port}/mcp"',
           "-c", "mcp_servers.grader.tool_timeout_sec=1800",
           "-c", "mcp_servers.grader.startup_timeout_sec=60",
           "-c", 'mcp_servers.grader.default_tools_approval_mode="approve"', *extra]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=open(wd / f"{tag}.jsonl", "w"),
                         stderr=subprocess.STDOUT, env=env_with_home(ch), cwd=str(wd), text=True)
    p.stdin.write(prompt); p.stdin.close()
    return p


def submissions(root, sessions=None):
    f = root / "submissions.jsonl"
    if not f.exists():
        return []
    out = []
    for line in f.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if sessions is None or r.get("session") in sessions:
            sf = root / "solutions" / f"{r['sol_hash']}.txt"
            r["program"] = sf.read_text() if sf.exists() else ""
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True, choices=["fc46", "fc302", "erdos"])
    ap.add_argument("--arm", required=True, choices=["native", "mechanical"])
    ap.add_argument("--n", type=int, default=50, help="subagents (permitted for native, exact for mechanical)")
    ap.add_argument("--replicate", type=int, default=0)
    ap.add_argument("--sub-model", default="gpt-5.6-terra")
    ap.add_argument("--sub-effort", default="xhigh")
    ap.add_argument("--orch-model", default="gpt-5.6-sol")
    ap.add_argument("--orch-effort", default="xhigh")
    ap.add_argument("--rollout-wall", type=int, default=1200)
    ap.add_argument("--orch-wall", type=int, default=3600)
    ap.add_argument("--max-parallel", type=int, default=25)
    ap.add_argument("--max-full", type=int, default=3)
    ap.add_argument("--grader-conc", type=int, default=32)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--orch-style", choices=["cdc", "minimal"], default="cdc",
                    help="native arm: CDC directive fan-out (default) vs permissive")
    ap.add_argument("--no-reduce", action="store_true",
                    help="mechanical arm: skip the aggregator; the result is plain best-of-N")
    args = ap.parse_args()

    mx = B.MAXIMIZE[args.problem]
    root = Path(f"{B.REPO}/runs/mapreduce/{args.problem}_{args.arm}_r{args.replicate}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "sessions").mkdir(exist_ok=True)
    ch = prepare_codex_home(root, multi_agent=(args.max_parallel if args.arm == 'native' else 0))
    seed_prog, seed_constr, seed_rec = load_seed(args.problem, None)
    base_json = None
    if seed_constr is not None:
        base_json = root / "base_construction.json"
        base_json.write_text(json.dumps(seed_constr))

    # ONE production grade per AGENT, both arms. Budgets key off the mcp-session-id header, which
    # is 1:1 with each agent's MCP connection and cannot be set by the agent -- so this holds even
    # though every native subagent inherits (and passes) the orchestrator's session string.
    # Total production evals therefore equals the number of agents that actually ran, which is the
    # quantity we are matching. (Earlier this was `n` for native, meaning a TEAM budget -- with
    # per-agent keying that would have granted each subagent 100 grades, ~10000 vs mechanical 100.)
    cap = 1
    grader, port = start_grader_full(args.problem, root, base_json, args.grader_conc, cap)
    t_start = time.time()
    sessions_used = 0
    try:
        sol0 = f"```{FENCE[args.problem]}\n{strip_fence(seed_prog)}\n```"
        g0 = grade_many(port, [(("b", 0), sol0, "grade_full")])[("b", 0)]
        baseline = g0.get("score") if g0.get("valid") else None
        print(f"[mr] {args.problem}/{args.arm} r{args.replicate}: baseline={baseline}", flush=True)

        if args.arm == "native":
            wd = root / "orchestrator"; wd.mkdir(exist_ok=True)
            sess = "native_orch"
            (root / "sessions" / f"{sess}.json").write_text(
                json.dumps({"deadline": time.time() + args.orch_wall}))
            orch = (CDC_ORCHESTRATION if args.orch_style == "cdc"
                    else MINIMAL_ORCHESTRATION).format(n=args.n, conc=args.max_parallel)
            pr = NATIVE_PROMPT.format(
                orchestration=orch,
                fence=FENCE[args.problem], seed=strip_fence(seed_prog),
                task=task_text(args.problem),
                contract=CONTRACT.format(sess=sess, max_full=args.max_full * 10))
            (wd / "prompt.txt").write_text(pr)
            p = run_codex(args.orch_model, args.orch_effort, pr, wd, port, args.orch_wall,
                          "codex", ch,
                          extra=["--enable", "multi_agent_v2"])
            sessions_used += 1
            try:
                p.wait(args.orch_wall)
            except subprocess.TimeoutExpired:
                p.kill()
            spawned = count_spawned(ch, wd)
            subs = submissions(root)
            print(f"[mr] native: subagents actually spawned={spawned}, submissions={len(subs)}",
                  flush=True)
            result = {"spawned": spawned}
        else:
            # ---- MAP: exactly N driver-forked rollouts ----
            sess_names = [f"m{i}" for i in range(args.n)]
            rows = []
            for i in range(0, args.n, args.max_parallel):
                wave = sess_names[i:i + args.max_parallel]
                procs = []
                for s in wave:
                    wd = root / s; wd.mkdir(exist_ok=True)
                    (root / "sessions" / f"{s}.json").write_text(
                        json.dumps({"deadline": time.time() + args.rollout_wall}))
                    pr = rollout_prompt(args.problem, s, seed_prog, cap,
                                        args.rollout_wall // 60)
                    (wd / "prompt.txt").write_text(pr)
                    procs.append((s, run_codex(args.sub_model, args.sub_effort, pr, wd, port,
                                               args.rollout_wall, "codex", ch)))
                    sessions_used += 1
                for s, p in procs:
                    try:
                        p.wait(args.rollout_wall + 120)
                    except subprocess.TimeoutExpired:
                        p.kill()
                print(f"[mr] map wave done ({min(i+args.max_parallel, args.n)}/{args.n})", flush=True)
            subs = submissions(root, set(sess_names))
            for s in sess_names:
                if not any(x["session"] == s for x in subs):
                    rows.append({"note": f"{s}: no accepted submission"})
            print(f"[mr] map: {len(subs)}/{args.n} accepted submissions", flush=True)

            # ---- REDUCE (optional): the clean comparison is plain best-of-N, since the
            # aggregator was measured to add only +1.4% (fc46) / +0.62% (fc302) ----
            if args.no_reduce:
                bm = max((s_["score"] for s_ in subs if s_.get("score") is not None), default=None)
                print(f"[mr] no-reduce: best-of-{args.n} = {bm}", flush=True)
                result = {"map_submissions": len(subs), "reduced": False}
            else:
                wd = root / "reducer"; wd.mkdir(exist_ok=True)
                rs = "reducer"
                (root / "sessions" / f"{rs}.json").write_text(
                    json.dumps({"deadline": time.time() + args.orch_wall}))
                dg = digest(subs, rows, args.topk, mx)
                (root / "digest.txt").write_text(dg)
                pr = REDUCE_PROMPT.format(n=args.n, digest=dg[:120000],
                                          contract=CONTRACT.format(sess=rs, max_full=cap),
                                          task=task_text(args.problem))
                (wd / "prompt.txt").write_text(pr)
                p = run_codex(args.orch_model, args.orch_effort, pr, wd, port, args.orch_wall,
                              "codex", ch)
                sessions_used += 1
                try:
                    p.wait(args.orch_wall)
                except subprocess.TimeoutExpired:
                    p.kill()
                result = {"map_submissions": len(subs), "reduced": True}

        allsubs = submissions(root)
        best = (max if mx else min)([s["score"] for s in allsubs if s.get("score") is not None],
                                    default=None)
        out = {"problem": args.problem, "arm": args.arm, "replicate": args.replicate,
               "baseline": baseline, "best": best, "n_submissions": len(allsubs),
               "sessions_used": sessions_used, "wall_secs": round(time.time() - t_start),
               **result}
        (root / "result.json").write_text(json.dumps(out, indent=2))
        print(f"[mr] RESULT {json.dumps(out)}", flush=True)
    finally:
        stop(grader)


def start_grader_full(problem, logdir, base_json, max_conc, max_full):
    """start_grader + the per-session production-grade cap."""
    port = B.free_port()
    cmd = [B.PY, f"{AB}/grading_mcp.py", "--problem", problem, "--port", str(port),
           "--logdir", str(logdir), "--backend", "thread", "--max-concurrent", str(max_conc),
           "--max-full", str(max_full)]
    if base_json:
        cmd += ["--base-json", str(base_json)]
    p = subprocess.Popen(cmd, stdout=open(logdir / "grader.log", "w"),
                         stderr=subprocess.STDOUT, env=B._base_env(), cwd=str(AB))
    import socket
    for _ in range(240):
        try:
            socket.create_connection(("127.0.0.1", port), 1).close()
            return p, port
        except OSError:
            if p.poll() is not None:
                raise RuntimeError(f"grader died; see {logdir}/grader.log")
            time.sleep(1)
    raise RuntimeError("grader failed to bind")


def count_spawned(ch: Path, wd: Path):
    """Native arm only: spawns are NOT recorded in the parent's event stream -- they exist only as
    separate session files naming the parent thread. (Learned the hard way: the parent transcript
    shows only `wait` calls with empty receiver lists, which looks like zero subagents.)"""
    try:
        root_thread = None
        for line in (wd / "codex.jsonl").read_text().splitlines():
            if '"thread.started"' in line:
                root_thread = json.loads(line).get("thread_id")
                break
        if not root_thread:
            return 0
        n = 0
        for f in (ch / "sessions").rglob("*.jsonl"):
            try:
                if f'"parent_thread_id":"{root_thread}"' in f.read_text():
                    n += 1
            except Exception:
                pass
        return n
    except Exception:
        return -1


if __name__ == "__main__":
    main()
