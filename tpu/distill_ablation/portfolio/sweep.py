"""Matched sweep: state-reuse strategies for agentic code discovery.

Two arms, identical everywhere except HOW past solutions condition the next round:

  --reuse continual   PDR-inspired: ONE shared workspace (OUT.md). Every rollout sees the same
                      distilled lessons. Coordinator rewrites OUT.md from the round digest.
  --reuse simple_tes  Persistent DAG: coordinator samples S states (1 program = refine,
                      2-3 = merge/cross-pollinate) under code-enforced constraints; each rollout
                      is conditioned ONLY on its assigned state.

Rollouts are DRIVER-LAUNCHED `codex exec -m <exec-model>` processes (not native subagents:
spawn_agent only allows gpt-5.6-{sol,terra}, never mini -- and code-launched rollouts also
guarantee the matched S x G count instead of trusting an orchestrator to comply).

Trust boundary: rollouts emit program+reflection in their FINAL MESSAGE (captured via -o, since the
sandbox blocks file writes); the DRIVER independently grades every
program at production budget and audits it for hardcoding. Agent-reported scores are never used.
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
sys.path.insert(0, str(Path(__file__).parent.parent))
import benchmark_cdc as B
from audit_programs import classify
from prompts import TASK, mine_failure_patterns
from store import Store

PF = Path(__file__).parent
AB = PF.parent
FENCE = {"erdos": "python", "ac1": "python", "ac2": "python", "fc46": "cpp", "fc302": "cpp"}


# ---------------------------------------------------------------- seeds
def load_seed(problem, idx):
    """Return (program_text, construction|None, recorded_score)."""
    if problem == "erdos":
        ws = json.loads((AB / "corpora/worse_set.json").read_text())["worse"]
        e = ws[idx if idx is not None else 2]
        return e["code"], list(e["construction"]), -e["value"]
    f = {"fc46": "initial_fcalgo46.json", "fc302": "initial_fcalgo302.json",
         "ac1": "initial_ac1.json", "ac2": "initial_ac2.json"}[problem]
    recs = json.loads((AB / "corpora" / f).read_text())["records"]
    ok = [r for r in recs if r.get("code") and r.get("score") is not None]
    # score==0 on the frontier problems means the program FAILED, not "a weak solution" --
    # 14/22 fc46 records are such zeros, so a naive median lands on garbage.
    if B.MAXIMIZE[problem]:
        ok = [r for r in ok if r["score"] > 0] or ok
    ok.sort(key=lambda r: r["score"], reverse=not B.MAXIMIZE[problem])
    e = ok[idx if idx is not None else len(ok) // 2]      # mid-range: room to improve
    return e["code"], None, e["score"]


def strip_fence(t):
    m = re.search(r"```(?:python|cpp|c\+\+)?\s*([\s\S]*?)\s*```", t or "")
    return m.group(1) if m else (t or "")


def task_text(problem):
    """Plain task statement. erdos/ac* have hand-written minimal versions in prompts.TASK;
    the frontier problems must come from their env (different discover root), so we reuse
    cdc_prompt.get_question() in a subprocess for a clean import path, and cache it."""
    if problem in TASK:
        return TASK[problem]
    cache = AB / "corpora" / f"question_{problem}.txt"
    if cache.exists() and cache.read_text().strip():
        return cache.read_text()
    code = (f"import sys; sys.path.insert(0, {str(AB)!r}); import cdc_prompt; "
            f"q, _ = cdc_prompt.get_question({problem!r}); print(q)")
    r = subprocess.run([B.PY, "-c", code], capture_output=True, text=True,
                       env=B._base_env(), cwd=str(AB), timeout=600)
    q = (r.stdout or "").strip()
    if not q:
        raise RuntimeError(f"could not get question for {problem}: {r.stderr[-400:]}")
    cache.write_text(q)
    return q


# ---------------------------------------------------------------- grading (driver-side)
async def _one(port, sol, tool):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    try:
        async with streamablehttp_client(f"http://127.0.0.1:{port}/mcp") as (r, w, _):
            async with ClientSession(r, w) as s:
                await s.initialize()
                res = await s.call_tool(tool, {"solution": sol})
                return json.loads(res.content[0].text)
    except Exception as e:
        return {"score": None, "valid": False, "detail": f"driver-grade failed: {e}"[:160]}


def grade_many(port, jobs):
    async def go():
        out = await asyncio.gather(*[_one(port, s, t) for _, s, t in jobs])
        return {k: r for (k, _, _), r in zip(jobs, out)}
    return asyncio.run(go()) if jobs else {}


def start_grader(problem, logdir, base_json, max_conc):
    port = B.free_port()
    cmd = [B.PY, f"{AB}/grading_mcp.py", "--problem", problem, "--port", str(port),
           "--logdir", str(logdir), "--backend", "thread", "--max-concurrent", str(max_conc)]
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


def stop(p):
    if p is None:
        return
    p.send_signal(signal.SIGINT)
    try:
        p.wait(15)
    except subprocess.TimeoutExpired:
        p.kill()


def prepare_codex_home(root: Path, multi_agent: int = 0) -> Path:
    """Isolated CODEX_HOME carrying the sandbox fix.

    This cluster has max_user_namespaces=0, so codex's default bubblewrap sandbox cannot start
    ("bwrap: Creating new namespace failed: No space left on device") and rollouts silently lose
    the ability to run ANY shell command -- they burn their whole wall clock and produce nothing.
    `use_legacy_landlock` sandboxes without user namespaces. It must go in config.toml: `-c`
    overrides are not validated and feature keys set that way are silently ignored.

    Second fix: long rollouts were dying to "stream disconnected before completion: idle timeout
    waiting for websocket" while the model reasoned -- only 12.5% usable output at 8-16
    concurrency. The built-in `openai` provider cannot be overridden ("reserved built-in provider
    IDs"), but a CUSTOM provider carrying a long stream_idle_timeout_ms authenticates fine with
    ChatGPT auth and lifted yield to 62.5% (5x) on an otherwise identical probe.
    """
    ch = root / "codex_home"
    ch.mkdir(parents=True, exist_ok=True)
    src = Path.home() / ".codex" / "auth.json"
    if src.exists():
        (ch / "auth.json").write_text(src.read_text())
    extra = ""
    if multi_agent:
        # Sol auto-selects multi-agent V2 with hide_spawn_agent_metadata=true, which STRIPS
        # model/reasoning_effort/agent_type from the model-visible spawn_agent schema -- so the
        # orchestrator silently cannot choose its subagents' model (openai/codex#31814). The
        # tool_namespace setting avoids "collaboration.spawn_agent is reserved for this model".
        # max_concurrent_threads_per_session must be set HERE, not via -c: V2 defaults to 4 threads
        # INCLUDING root, i.e. only 3 concurrent children, which is exactly the fan-out ceiling we
        # kept measuring and mistaking for the model declining to parallelise.
        extra = ("\n[features.multi_agent_v2]\n"
                 "hide_spawn_agent_metadata = false\n"
                 'tool_namespace = "agents"\n'
                 "expose_spawn_agent_model_overrides = true\n"
                 f"max_concurrent_threads_per_session = {multi_agent}\n")
    (ch / "config.toml").write_text(
        'model_provider = "openai-long"\n\n'
        "[features]\nuse_legacy_landlock = true\n" + extra + "\n"
        "[model_providers.openai-long]\n"
        'name = "OpenAI ChatGPT (long idle timeout)"\n'
        'base_url = "https://chatgpt.com/backend-api/codex"\n'
        'wire_api = "responses"\n'
        "requires_openai_auth = true\n"
        "stream_idle_timeout_ms = 1800000\n"
        "stream_max_retries = 20\n"
        "request_max_retries = 10\n")
    return ch


def env_with_home(ch: Path):
    e = B._base_env()
    e["CODEX_HOME"] = str(ch)
    return e



def salvage_from_checks(root: Path, session: str, maximize: bool):
    """A rollout that never writes its deliverable is otherwise lost. Recover the best solution it
    passed to check(): the grader logs every call with its session tag and stores the raw source
    under solutions/<hash>.txt. Costs nothing and rescued 6/10 rollouts in the earlier A/B."""
    log = root / "grade_log.jsonl"
    if not log.exists():
        return None
    best = None
    for line in log.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("session") != session or not r.get("valid") or r.get("score") is None:
            continue
        if best is None or (r["score"] > best["score"] if maximize else r["score"] < best["score"]):
            best = r
    if best is None:
        return None
    f = root / "solutions" / f"{best['sol_hash']}.txt"
    return f.read_text() if f.exists() else None


# ---------------------------------------------------------------- prompts
ROLLOUT_CONTRACT = """
TOOLS (`grader` MCP): check(solution, session) compiles and runs your program end-to-end on a
reduced budget and returns a rough score -- cheap, call it freely. Pass session="{sess}" and the
reply tells you your remaining wall time.

HOW TO WORK: check() is your test harness -- it compiles and executes the program for you, so you
do NOT need to run anything in your own shell. Local shell access is restricted on this machine;
if a shell command fails, do not try to work around it -- just write your program and call
check() to evaluate it. Iterate: write, check(), read the score and feedback, improve, repeat.

SCORING: your program is finally scored by ONE PRODUCTION RUN under the full budget -- that is the
only number that counts. Scale your work to the budget_s you are given (use most of it); a program
that finishes instantly wastes its budget. Embedding a precomputed answer as data (literal arrays,
base64) is INVALID and will be rejected.

DELIVERABLES -- put these in your FINAL MESSAGE (do NOT rely on writing files; the sandbox on
this machine often blocks file writes, and anything only written to disk may be lost):
  1. your single best program as one ```{fence} code block
  2. then exactly two lines:
     Approach: <what idea this solution tried>
     Insight: <what should be learned for future improvement>
"""


def rollout_prompt(problem, sess, workspace, states, grade_log, wall_min):
    p = [task_text(problem), "", ROLLOUT_CONTRACT.format(sess=sess, fence=FENCE[problem]),
         f"You have ~{wall_min} minutes.", ""]
    if workspace:
        p += ["## Accumulated findings so far (shared workspace)", workspace, ""]
    if states:
        if len(states) == 1:
            p.append("## Improve on this program substantially:")
        else:
            p.append(f"## These {len(states)} programs take DIFFERENT approaches. "
                     "Combine their complementary strengths:")
        for s in states:
            p.append(f"\n### node {s['id']} (score {s['r']})\n```{FENCE[problem]}\n"
                     f"{strip_fence(s['program'])}\n```")
    fp = mine_failure_patterns(Path(grade_log))
    if fp:
        p += ["", fp]
    return "\n".join(p)


def digest_text(rows, baseline, maximize, top_n=20):
    """Compact, code-verified view of a round: score table + reports of the best few."""
    better = (lambda a, b: a > b) if maximize else (lambda a, b: a < b)
    ok = [r for r in rows if r["score"] is not None]
    ok.sort(key=lambda r: r["score"], reverse=maximize)
    direction = "HIGHER scores are BETTER (maximise)" if maximize else \
                "LOWER scores are BETTER (minimise)"
    L = [f"Round digest: {len(rows)} rollouts, {len(ok)} produced a valid program.",
         f"OBJECTIVE DIRECTION: {direction}. Baseline to beat: {baseline}.",
         f"'delta' below is (score - baseline), so a "
         f"{'POSITIVE' if maximize else 'NEGATIVE'} delta means BETTER than baseline.",
         "", "id | state | verified_score | delta | verdict"]
    for r in rows:
        d = ("--" if r["score"] is None or baseline is None
             else f"{r['score'] - baseline:+.3e}")
        L.append(f"{r['rid']} | {r['state']} | {r['score']} | {d} | {r['verdict']}")
    L += ["", f"Reports from the {min(top_n, len(ok))} best rollouts:"]
    for r in ok[:top_n]:
        L.append(f"\n--- {r['rid']} (score {r['score']}, {r['verdict']})\n{r['report'][:600]}")
    bad = [r for r in rows if r["score"] is None][:5]
    if bad:
        L += ["", "Sample of failures:"]
        for r in bad:
            L.append(f"--- {r['rid']}: {r['note'][:150]}")
    return "\n".join(L)


DISTILL_PROMPT = """You are the coordinator of an evaluation-driven discovery loop.
The objective direction is stated in the digest below -- read it carefully and never infer the
direction yourself from the numbers.

Below is a code-verified digest of the round that just finished (every score was independently
re-graded by the harness; agent self-reports are not included).

{digest}

{prev}

Rewrite the shared findings workspace that will be given to EVERY rollout next round. Capture what
is actually working, what has been ruled out, contradictions worth resolving, and the most
promising next directions. Be concrete and technical -- name algorithms, parameters, failure modes.
Do NOT include full programs. Output ONLY the new workspace content, nothing else."""

SELECT_PROMPT = """You are the search coordinator for an evaluation-driven discovery loop.
The objective direction is stated in the digest below -- read it carefully and never infer the
direction yourself from the numbers.

A graph of every program tried so far is available via the `store` tools (overview, top_k, node,
relatives). Node fields: r = own score (raw units); p = own-score percentile in [0,1) (higher=better either
direction); U = value propagated up the DAG with gamma=0.8, so U >> p marks a fertile
stepping-stone that spawned winners despite a mediocre own score; n_used = times already seeded.
`top_k(by="rpucg")` ranks by the reference selection score (exploit + explore) -- use it as a
prior, but you may override it when you can see a better structural reason.

{digest}

Pick exactly {n} SEED GROUPS (each 1..{maxg} node ids):
  - a 1-node group -> that program gets refined;
  - a multi-node group -> those programs get MERGED, so choose ones whose ideas are complementary
    (different mechanisms, different failure modes), never near-duplicates.
Maximize coverage of genuinely different approaches; use U to revive fertile stepping-stones;
prefer under-used nodes over re-polishing one lineage.

Submit with submit_seed_groups(groups_json). Constraints are code-enforced; on REJECTED, adjust
and resubmit until ACCEPTED, then end the session."""


def run_codex(model, effort, prompt, wd, mcp, wall, tag, extra=(), codex_home=None):
    cmd = ["codex", "exec", "--strict-config", "-m", model,
           "-c", f"model_reasoning_effort={effort}",
           "-s", "workspace-write", "-c", "approval_policy=never", "--json", "-C", str(wd)]
    if mcp:
        name, url = mcp
        cmd += ["-c", f'mcp_servers.{name}.url="{url}"',
                "-c", f"mcp_servers.{name}.tool_timeout_sec=1200",
                "-c", f"mcp_servers.{name}.startup_timeout_sec=60",
                "-c", f'mcp_servers.{name}.default_tools_approval_mode="approve"']
    cmd += list(extra)
    out = open(wd / f"{tag}.jsonl", "w")
    env = env_with_home(codex_home) if codex_home else B._base_env()
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=out, stderr=subprocess.STDOUT,
                            env=env, cwd=str(wd), text=True), prompt


def launch(procs, prompt_pairs):
    for (p, pr) in prompt_pairs:
        p.stdin.write(pr)
        p.stdin.close()
        procs.append(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True, choices=list(TASK) + ["fc46", "fc302"])
    ap.add_argument("--reuse", required=True, choices=["continual", "simple_tes"])
    ap.add_argument("--states", type=int, default=16)
    ap.add_argument("--rollouts", type=int, default=16, help="per state")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--max-parallel", type=int, default=64)
    ap.add_argument("--exec-model", default="gpt-5.4-mini")
    ap.add_argument("--exec-effort", default="high")
    ap.add_argument("--coord-model", default="gpt-5.6-sol")
    ap.add_argument("--coord-effort", default="xhigh")
    ap.add_argument("--rollout-wall", type=int, default=900)
    ap.add_argument("--coord-wall", type=int, default=1200)
    ap.add_argument("--max-group", type=int, default=3)
    ap.add_argument("--select", choices=["agent", "rpucg"], default="agent",
                    help="simple_tes seed selection: LLM coordinator vs the RPUCG formula")
    ap.add_argument("--seed-idx", type=int, default=None)
    ap.add_argument("--grader-conc", type=int, default=32)
    ap.add_argument("--max-sessions", type=int, default=3000, help="hard cost cap; abort if exceeded")
    ap.add_argument("--tag", default="")
    ap.add_argument("--resume", action="store_true",
                    help="continue from an existing graph.json instead of re-seeding "
                         "(for arms clipped by the sbatch wall limit)")
    args = ap.parse_args()

    maximize = B.MAXIMIZE[args.problem]
    root = Path(f"{B.REPO}/runs/sweep/{args.problem}_{args.reuse}{args.tag}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "sessions").mkdir(exist_ok=True)
    glog = root / "grade_log.jsonl"
    CH = prepare_codex_home(root)

    seed_prog, seed_constr, seed_rec = load_seed(args.problem, args.seed_idx)
    base_json = None
    if seed_constr is not None:
        base_json = root / "base_construction.json"
        base_json.write_text(json.dumps(seed_constr))
    print(f"[sw] {args.problem}/{args.reuse}: seed recorded={seed_rec} chars={len(seed_prog)}",
          flush=True)

    grader, port = start_grader(args.problem, root, base_json, args.grader_conc)
    sessions_used = 0
    traj = []
    try:
        # ---- measured baseline (the number both arms must beat) ----
        sol0 = f"```{FENCE[args.problem]}\n{strip_fence(seed_prog)}\n```"
        g0 = grade_many(port, [(("b", 0), sol0, "grade_full")])[("b", 0)]
        baseline = g0.get("score") if g0.get("valid") else None
        print(f"[sw] baseline production score = {baseline}", flush=True)

        gpath = root / "graph.json"
        start_round = 1
        if args.resume and gpath.exists():
            store = Store.load(gpath)
            done = max((store.g.nodes[n].get("round", 0) for n in store.g.nodes), default=0)
            start_round = done + 1
            b = store.best()
            print(f"[sw] RESUMED: {store.g.number_of_nodes()} nodes, {done} rounds done, "
                  f"best={store.g.nodes[b]['r'] if b is not None else None}, "
                  f"continuing at round {start_round}", flush=True)
        else:
            store = Store(maximize, gpath)
            store.add(sol0, baseline, [], feedback="seed", rnd=0, summary="seed")
            store.backup_U(); store.save()
        workspace = ""
        wf = sorted(root.glob("workspace_r*.md"))
        if args.resume and wf:
            workspace = wf[-1].read_text()
            print(f"[sw] resumed workspace from {wf[-1].name} ({len(workspace)} chars)", flush=True)

        for rnd in range(start_round, args.rounds + 1):
            # ---------- decide this round's states ----------
            groups = []
            if args.reuse == "simple_tes":
                nsel = min(args.states,
                           sum(1 for n in store.g.nodes if store.g.nodes[n]["r"] is not None))
                if args.select == "rpucg":          # formula selector, no LLM call
                    groups = store.rpucg_select(nsel, args.max_group)
                elif store.g.number_of_nodes() >= 3:
                    store.save()
                    gout = root / f"groups_r{rnd}.json"; gout.unlink(missing_ok=True)
                    sport = B.free_port()
                    smcp = subprocess.Popen(
                        [B.PY, str(PF / "store_mcp.py"), "--graph", str(root / "graph.json"),
                         "--out", str(gout), "--port", str(sport), "--n-groups", str(nsel),
                         "--max-group", str(args.max_group)],
                        stdout=open(root / "store_mcp.log", "a"), stderr=subprocess.STDOUT,
                        env=B._base_env(), cwd=str(PF))
                    time.sleep(3)
                    cwd = root / f"coord_r{rnd}"; cwd.mkdir(exist_ok=True)
                    pr = SELECT_PROMPT.format(digest=traj[-1]["digest"] if traj else "(first round)",
                                              n=nsel, maxg=args.max_group)
                    p, txt = run_codex(args.coord_model, args.coord_effort, pr, cwd,
                                       ("store", f"http://127.0.0.1:{sport}/mcp"),
                                       args.coord_wall, "coord", codex_home=CH)
                    p.stdin.write(txt); p.stdin.close(); sessions_used += 1
                    try:
                        p.wait(args.coord_wall)
                    except subprocess.TimeoutExpired:
                        p.kill()
                    stop(smcp)
                    if gout.exists():
                        groups = json.loads(gout.read_text())
                if not groups:  # code fallback: RPUCG formula (one-hop-excluded)
                    groups = store.rpucg_select(nsel, args.max_group)
                print(f"[sw] r{rnd} groups={groups}", flush=True)
            else:
                best = store.best()
                groups = [[best]] * args.states      # one shared context; state is just the incumbent

            # ---------- launch S x G rollouts, bounded parallelism ----------
            jobs = []
            for si, grp in enumerate(groups):
                for k in range(args.rollouts):
                    jobs.append((si, grp, k))
            if sessions_used + len(jobs) > args.max_sessions:
                print(f"[sw] ABORT: session cap {args.max_sessions} would be exceeded", flush=True)
                break
            print(f"[sw] r{rnd}: launching {len(jobs)} rollouts "
                  f"({args.max_parallel} at a time)", flush=True)
            rows = []
            t0 = time.time()
            for i in range(0, len(jobs), args.max_parallel):
                wave = jobs[i:i + args.max_parallel]
                procs = []
                for (si, grp, k) in wave:
                    rid = f"r{rnd}_s{si}_k{k}"
                    wd = root / rid; wd.mkdir(exist_ok=True)
                    sess = rid
                    (root / "sessions" / f"{sess}.json").write_text(
                        json.dumps({"deadline": time.time() + args.rollout_wall}))
                    states = [store.meta(n, with_program=True) for n in grp] \
                        if args.reuse == "simple_tes" else \
                        [store.meta(store.best(), with_program=True)]
                    pr = rollout_prompt(args.problem, sess,
                                        workspace if args.reuse == "continual" else "",
                                        states, glog, args.rollout_wall // 60)
                    (wd / "prompt.txt").write_text(pr)
                    p, txt = run_codex(args.exec_model, args.exec_effort, pr, wd,
                                       ("grader", f"http://127.0.0.1:{port}/mcp"),
                                       args.rollout_wall, "codex",
                                       extra=["-o", str(wd / "final.txt")], codex_home=CH)
                    p.stdin.write(txt); p.stdin.close()
                    procs.append((rid, si, grp, wd, p))
                    sessions_used += 1
                for rid, si, grp, wd, p in procs:
                    left = max(30, args.rollout_wall + 180 - int(time.time() - t0))
                    try:
                        p.wait(timeout=left)
                    except subprocess.TimeoutExpired:
                        p.kill()
                # harvest this wave
                for rid, si, grp, wd, _ in procs:
                    prog, rep_txt = None, ""
                    fin = wd / "final.txt"
                    if fin.exists() and fin.read_text().strip():
                        body = fin.read_text()
                        if "```" in body:
                            prog = body
                        m = re.search(r"Approach:.*?(?:\n\s*Insight:.*)?$", body,
                                      re.S | re.M)
                        rep_txt = m.group(0)[:400] if m else ""
                    if prog is None:
                        for cand in ("solution.txt", "solution.py", "solution.cpp"):
                            f = wd / cand
                            if f.exists() and f.read_text().strip():
                                prog = f.read_text(); break
                    if prog is None:      # salvage: best solution this rollout passed to check()
                        prog = salvage_from_checks(root, rid, maximize)
                    rep = (wd / "report.md")
                    rows.append({"rid": rid, "state": grp, "wd": wd, "prog": prog,
                                 "report": (rep.read_text()[:800] if rep.exists() else rep_txt),
                                 "score": None, "verdict": "NO-PROGRAM", "note": ""})
            print(f"[sw] r{rnd}: rollouts done in {int(time.time()-t0)}s", flush=True)

            # ---------- driver-side verification (never trust agent scores) ----------
            gjobs = [((r["rid"],), f"```{FENCE[args.problem]}\n{strip_fence(r['prog'])}\n```",
                      "grade_full") for r in rows if r["prog"]]
            print(f"[sw] r{rnd}: production-grading {len(gjobs)} programs...", flush=True)
            graded = grade_many(port, gjobs)
            for r in rows:
                if not r["prog"]:
                    r["note"] = "no program (no file, nothing salvageable from check history)"
                    continue
                g = graded.get((r["rid"],), {})
                r["score"] = g.get("score") if g.get("valid") else None
                r["note"] = (g.get("detail") or "")[:150]
                v, why = classify(r["prog"], r["score"], None, baseline)
                r["verdict"] = v if r["score"] is not None else "INVALID"
                r["why"] = why

            # ---------- commit genuine results ----------
            for r in rows:
                if r["score"] is None or r["verdict"] in ("HARDCODED",):
                    continue
                parents = r["state"] if args.reuse == "simple_tes" else [store.best()]
                store.add(f"```{FENCE[args.problem]}\n{strip_fence(r['prog'])}\n```", r["score"],
                          parents, feedback=r["note"], rnd=rnd, summary=r["report"][:250])
            store.backup_U(); store.save()
            for r in rows:
                if args.reuse == "simple_tes" and r["state"]:
                    store.mark_used(r["state"])

            dg = digest_text(rows, baseline, maximize)
            (root / f"digest_r{rnd}.txt").write_text(dg)
            b = store.best()
            bs = store.g.nodes[b]["r"] if b is not None else None
            gen = sum(1 for r in rows if r["verdict"] == "GENUINE")
            hard = sum(1 for r in rows if r["verdict"] in ("HARDCODED", "SUSPECT"))
            traj.append({"round": rnd, "best": bs, "genuine": gen, "hardcoded": hard,
                         "valid": sum(1 for r in rows if r["score"] is not None),
                         "n": len(rows), "digest": dg[:4000]})
            print(f"[sw] ===== r{rnd}: best={bs} genuine={gen}/{len(rows)} hardcoded={hard} "
                  f"sessions={sessions_used} =====", flush=True)
            (root / "trajectory.json").write_text(json.dumps(
                {"problem": args.problem, "reuse": args.reuse, "baseline": baseline,
                 "rounds": [{k: v for k, v in t.items() if k != "digest"} for t in traj]}, indent=2))

            # ---------- distill (continual arm only) ----------
            if args.reuse == "continual" and rnd < args.rounds:
                cwd = root / f"coord_r{rnd}"; cwd.mkdir(exist_ok=True)
                pr = DISTILL_PROMPT.format(
                    digest=dg[:12000],
                    prev=(f"Current workspace:\n{workspace}" if workspace else "(no workspace yet)"))
                p, txt = run_codex(args.coord_model, args.coord_effort, pr, cwd, None,
                                   args.coord_wall, "coord", extra=["-o", str(cwd / "out.md")], codex_home=CH)
                p.stdin.write(txt); p.stdin.close(); sessions_used += 1
                try:
                    p.wait(args.coord_wall)
                except subprocess.TimeoutExpired:
                    p.kill()
                nw = cwd / "out.md"
                if nw.exists() and nw.read_text().strip():
                    workspace = nw.read_text()
                    (root / f"workspace_r{rnd}.md").write_text(workspace)
                print(f"[sw] r{rnd}: workspace {len(workspace)} chars", flush=True)
    finally:
        stop(grader)
    b = store.best() if 'store' in dir() else None
    print(f"[sw] FINAL best={store.g.nodes[b]['r'] if b is not None else None} "
          f"sessions_used={sessions_used}")


if __name__ == "__main__":
    main()
