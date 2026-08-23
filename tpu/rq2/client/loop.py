# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx>=0.27"]
# ///
"""RQ2 discovery loop: one cell = one 10-step search under one treatment.

    render -> sample the farm -> fast-grade -> reflect -> aggregate -> repeat

Weights never move; everything learned lives in the state. The 2x2 is orthogonal by
construction:

    STATE     controls the CONTEXT each sub-agent sees   (puct tree lineage | shared workspace)
    EXECUTION controls the INSTRUCTIONS they are given   (none | orchestrator's allocation)

so an orchestrated run still gets its state's context and a puct run still gets its lineages.
Mixing those would confound the two factors.

Model assignment is made HERE, never by the orchestrator -- otherwise a mixed cell could
silently become a single-model cell and the composition factor would be meaningless.

  uv run loop.py --problem fc46 --state puct --execution simple \
      --composition 50-50 --n 100 --steps 10 --out runs/rq2/fc46_puct_simple_5050_n100
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "fleet"))
import prompts as P  # noqa: E402
import registry as REG  # noqa: E402
from state_reuse import MEMORY_MAX_CHARS, make_state_reuse  # noqa: E402

PACKS = HERE.parent.parent / "rq1" / "client" / "data"
ATTEMPTS = 6
# serving shapes (32k both models -- the SimpleTES reference assumes an effectively unbounded
# window and 32k completion budget, so starving the prompt is the unfaithful choice); two-phase
# budgets must fit inside them
MODEL_CFG = {
    "qwen35": {"model": "Qwen/Qwen3.5-27B", "ctx": 32768, "p1": 12000, "p2": 7000},
    "gemma4": {"model": "google/gemma-4-31B-it", "ctx": 32768, "p1": 9000, "p2": 6000},
}
# ~3 chars/token is conservative for C++/math-heavy text; margin covers template + tail
CHARS_PER_TOKEN = 3.0
WORKSPACE_MAX_CHARS = 12_000   # backstop for the compactor's 3x8-sentence contract


def prompt_char_budget(models_for_bundle):
    """Chars a bundle prompt may occupy so prompt + phase-1 budget fits every model that will
    serve it (50-50: both)."""
    return int(min((MODEL_CFG[m]["ctx"] - MODEL_CFG[m]["p1"] - 512) * CHARS_PER_TOKEN
                   for m in models_for_bundle))
FORCE = ("\n\nI have thought about this enough. Here is my final, complete, self-contained "
         "program:\n\n```{fence}\n")


def sha(s):
    return hashlib.sha1((s or "").encode()).hexdigest()[:12]


def strip_fence(t):
    m = re.findall(r"```(?:python|cpp|c\+\+)?\s*([\s\S]*?)\s*```", t or "")
    return m[-1] if m else ""


TEAM_COMP = {"qq": ("qwen35", "qwen35"), "gg": ("gemma4", "gemma4"),
             "qg": ("qwen35", "gemma4")}
COMP_MODELS = {"qwen": ["qwen35"], "gemma": ["gemma4"], "50-50": ["qwen35", "gemma4"],
               "qq": ["qwen35"], "gg": ["gemma4"], "qg": ["qwen35", "gemma4"]}


def assign_models_by_bundle(bundles, composition):
    """Deterministic; for 50-50 the split is G/2 per model WITHIN EVERY BUNDLE, so each sampled
    state is explored by both models equally -- not merely a global 50/50 that could leave some
    states single-model. Team compositions (qq/gg/qg) assign by the bundle's AGENT: in a mixed
    team, composition == team makeup, so agent 0 is qwen and agent 1 is gemma for every one of
    the agent's bundles."""
    out = []
    for b in bundles:
        if composition == "qwen":
            out += ["qwen35"] * b.k
        elif composition == "gemma":
            out += ["gemma4"] * b.k
        elif composition in TEAM_COMP:
            out += [TEAM_COMP[composition][b.agent_idx]] * b.k
        else:
            half = b.k // 2
            out += ["qwen35"] * half + ["gemma4"] * (b.k - half)
    return out


# ------------------------------------------------------------------ farm sampling (2-phase)
async def _post(cli, url, key, body, pool=None):
    """pool: other same-model URLs to FAIL OVER to when `url` looks dead (connect-class
    errors). A worker preempted mid-step used to burn every retry into its dead IP -- measured
    117 ConnectTimeout losses in one hour of slice-B churn. HTTP-level errors (400 etc.) still
    fail fast on the same URL: those are request problems, not worker problems."""
    urls = [url] + [u for u in (pool or []) if u != url]
    ui = 0
    last = None
    for a in range(ATTEMPTS):
        try:
            r = await cli.post(f"{urls[ui]}/v1/chat/completions",
                               headers={"Authorization": f"Bearer {key}"}, json=body)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code < 500:
                raise                    # live server rejected the request (e.g. 400): no retry helps
            last = e                     # 5xx: transient; retry, next worker
            ui = (ui + 1) % len(urls)
        except Exception as e:
            last = e                     # transport-level (connect/read): try the next worker
            ui = (ui + 1) % len(urls)
        if a == ATTEMPTS - 1:
            raise last
        await asyncio.sleep(min(45, 8 * (a + 1)))


async def one_rollout(i, cli, url, key, mkey, prompt, fence, args, outdir, lock, pool=None):
    """Returns a result dict. Two-phase forcing fires only when phase 1 truncates, so it costs
    nothing for models that finish naturally."""
    cfg = MODEL_CFG[mkey]
    base = {"model": cfg["model"], "temperature": args.temperature,
            "messages": [{"role": "user", "content": prompt}]}
    if mkey == "gemma4":
        base["chat_template_kwargs"] = {"enable_thinking": True}
    sid = f"s{args._step:02d}_r{i:04d}"
    try:
        d = await _post(cli, url, key, {**base, "max_tokens": cfg["p1"]}, pool=pool)
    except Exception as e:
        (outdir / "raw" / f"{sid}.json").write_text(json.dumps(
            {"sid": sid, "model": mkey, "url": url, "phase": 1,
             "error": f"{type(e).__name__}: {str(e)[:400]}"}))
        return {"sid": sid, "model": mkey, "program": None, "score": None,
                "detail": f"request failed: {str(e)[:150]}", "convo": None}
    msg = d["choices"][0]["message"]
    text = msg.get("content") or ""
    finish = d["choices"][0].get("finish_reason")
    forced = ""
    if finish == "length":
        # Size phase 2 from what phase 1 ACTUALLY used. A fixed p2 overflows the context as soon
        # as the prompt is large (fc46's statement plus a seed program), and vLLM answers 400 --
        # measured: 2 of 4 rollouts lost that way. usage.prompt_tokens already counts the state
        # context, so this adapts per problem and per state size.
        u = d.get("usage") or {}
        used = (u.get("prompt_tokens") or 0) + (u.get("completion_tokens") or cfg["p1"])
        room = cfg["ctx"] - used - 256          # margin for the forcing text and template
        p2 = min(cfg["p2"], room)
        if p2 < 512:
            # No room to force an answer; keep phase 1 and let the parser try.
            code = strip_fence(text)
            async with lock:
                (outdir / "raw" / f"{sid}.json").write_text(json.dumps(
                    {"sid": sid, "model": mkey, "url": url, "text": text, "finish": finish,
                     "two_phase": False, "note": f"no room for phase 2 (used {used}/{cfg['ctx']})"}))
            return {"sid": sid, "model": mkey, "program": code or None, "score": None,
                    "detail": "" if code else "truncated, no room to force",
                    "convo": base["messages"] + [{"role": "assistant", "content": text}]}
        forced = FORCE.format(fence=fence)
        try:
            d2 = await _post(cli, url, key, {
                **base, "max_tokens": p2,
                "messages": base["messages"] + [{"role": "assistant", "content": text + forced}],
                "continue_final_message": True, "add_generation_prompt": False}, pool=pool)
            text = text + forced + (d2["choices"][0]["message"].get("content") or "")
        except Exception as e:
            (outdir / "raw" / f"{sid}.json").write_text(json.dumps(
                {"sid": sid, "model": mkey, "url": url, "phase": 2, "text": text,
                 "error": f"{type(e).__name__}: {str(e)[:400]}"}))
            return {"sid": sid, "model": mkey, "program": None, "score": None,
                    "detail": f"phase2 failed: {str(e)[:150]}", "convo": None}
    code = (text.split(forced, 1)[1].split("```")[0] if forced else strip_fence(text))
    async with lock:
        (outdir / "raw" / f"{sid}.json").write_text(json.dumps(
            {"sid": sid, "model": mkey, "url": url, "text": text, "finish": finish,
             "two_phase": bool(forced)}))
    return {"sid": sid, "model": mkey, "program": code or None, "score": None,
            "detail": "" if code else "no code block",
            "convo": base["messages"] + [{"role": "assistant", "content": text}]}


class LivePools:
    """Registry-backed URL pools refreshed at most every TTL seconds. A step used to snapshot
    its pools once at step start, so capacity arriving mid-step (recovered slices) sat idle
    while the snapshot workers queued 400+ deep -- measured during the 2026-08-12 recovery."""

    def __init__(self, registry_path, fallback, ttl=30):
        self.path, self.fallback, self.ttl = registry_path, fallback, ttl
        self._t, self._pools = 0.0, {}

    def get(self, mkey):
        now = time.time()
        if now - self._t > self.ttl:
            try:
                fresh = {m: REG.urls(self.path, model=m) for m in ("qwen35", "gemma4")}
                if any(fresh.values()):
                    self._pools = fresh
            except Exception:
                pass                             # keep serving the last good pools
            self._t = now
        return self._pools.get(mkey) or self.fallback.get(mkey) or []


async def sample_round(prompts, models, urls_by_model, args, outdir, fence):
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    rr = {k: 0 for k in urls_by_model}
    live = LivePools(args.registry, urls_by_model)
    out = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(2400, connect=30)) as cli:
        async def go(i):
            mkey = models[i]
            pool = live.get(mkey)
            url = pool[rr[mkey] % len(pool)]
            rr[mkey] += 1
            async with sem:
                return await one_rollout(i, cli, url, args.api_key, mkey, prompts[i],
                                         fence, args, outdir, lock, pool=pool)
        done = 0
        for fut in asyncio.as_completed([go(i) for i in range(len(prompts))]):
            r = await fut
            out.append(r)
            done += 1
            if done % 25 == 0:
                print(f"[loop]   sampled {done}/{len(prompts)}", flush=True)
    return out


async def reflect_round(results, urls_by_model, args, maximize, baseline):
    """PUCT only: a SECOND call resuming each conversation once the grade is known, so the
    reflection is about a measured outcome. Short decode over a cached prefix."""
    sem = asyncio.Semaphore(args.concurrency)
    rr = {k: 0 for k in urls_by_model}
    live = LivePools(args.registry, urls_by_model)
    async with httpx.AsyncClient(timeout=httpx.Timeout(900, connect=30)) as cli:
        async def go(r):
            if not r.get("convo"):
                return
            mkey = r["model"]
            pool = live.get(mkey)
            url = pool[rr[mkey] % len(pool)]
            rr[mkey] += 1
            body = {"model": MODEL_CFG[mkey]["model"], "temperature": 0.7, "max_tokens": 220,
                    "messages": r["convo"] + [{"role": "user",
                                               "content": P.reflect_prompt(r["score"], maximize, baseline)}]}
            async with sem:
                try:
                    d = await _post(cli, url, args.api_key, body, pool=pool)
                    r["reflection"] = (d["choices"][0]["message"].get("content") or "")[:400]
                except Exception:
                    r["reflection"] = ""
        await asyncio.gather(*[go(r) for r in results])


# ---------------------------------------------------------------------------- grading (fast)
def _grader_shims(registry_path):
    """Healthy per-slice grading shims from the registry (entries with model == 'grader')."""
    return REG.urls(registry_path, model="grader")


async def _grade_remote(results, problem, args, base_construction):
    """Fan grading over the slice shims; route each program to the least-loaded shim by live
    outstanding counts (client-side), Ray balances within the slice. Falls back per-item."""
    shims = _grader_shims(args.registry)
    if not shims:
        return None
    todo = [r for r in results if r.get("program")]
    out_ct = {u: 0 for u in shims}
    sem = asyncio.Semaphore(min(len(todo), 600))
    async with httpx.AsyncClient(timeout=httpx.Timeout(1800, connect=20)) as cli:
        async def go(r):
            async with sem:
                url = min(out_ct, key=out_ct.get)
                out_ct[url] += 1
                try:
                    resp = await cli.post(f"{url}/grade", json={
                        "problem": problem, "solution": r["program"], "fast": True,
                        "fast_budget": args.fast_budget,
                        "base_construction": base_construction})
                    resp.raise_for_status()
                    g = resp.json()
                    r["score"] = g.get("score") if g.get("valid") else None
                    r["detail"] = (g.get("detail") or "")[:200]
                except Exception as e:
                    r["score"] = None
                    r["detail"] = f"remote grade failed: {str(e)[:120]}"
                    r["_regrade_local"] = True
                finally:
                    out_ct[url] -= 1
        await asyncio.gather(*[go(r) for r in todo])
    return results



def grade_round(results, problem, outdir, concurrency, fast_budget):
    """Inline fast grading -- the substitution the whole campaign rests on. Threads are PINNED:
    unconstrained BLAS under high concurrency inflated RQ1 grading by ~9x."""
    sys.path.insert(0, "/n/fs/vision-mix/sk7524/SkyRLTpu/tpu/distill_ablation")
    sys.path.insert(0, str(HERE.parent.parent / "rq1" / "server"))
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"
    os.environ["TTD_FAST_BUDGET"] = str(fast_budget)
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
    from grading_mcp import _grade
    from make_problem_pack import PROBLEMS
    root, mod, cls, ptype, lang, maximize = PROBLEMS[problem]
    meta = json.loads((PACKS / problem / "meta.json").read_text())
    constr = None
    cj = PACKS / problem / "seed_construction.json"
    if cj.exists():
        constr = json.loads(cj.read_text())
    todo = [r for r in results if r.get("program")]
    payload = lambda c: f"```{meta['fence']}\n{c}\n```" if lang == "python" else c
    Pool = ProcessPoolExecutor if lang == "cpp" else ThreadPoolExecutor
    with Pool(max_workers=concurrency) as ex:
        futs = {ex.submit(_grade, root, mod, cls, ptype, lang, constr,
                          payload(r["program"]), True, str(outdir / "gradelogs")): r for r in todo}
        for f in as_completed(futs):
            r = futs[f]
            try:
                g = f.result()
                r["score"] = g["score"] if g.get("valid") else None
                r["detail"] = (g.get("detail") or "")[:200]
            except Exception as e:
                r["score"] = None
                r["detail"] = f"grade crashed: {e}"[:150]
    return results


# ------------------------------------------------------------------------ agent invocations
AUTH_SRC = Path.home() / ".codex" / "auth.json"
AUTH_LOCK = Path.home() / ".codex" / ".rq2-auth.lock"
AUTH_RETRIES = 4


def _auth_sync(codex_home, direction):
    """Keep every cell's copied auth.json in step with the one real credential.

    write_codex_home COPIES ~/.codex/auth.json into each cell, but the OAuth refresh token inside
    is single-use: the first cell to refresh rotates it and every other copy 401s with
    `refresh_token_reused`. That is what silently emptied the memory of 10 RQ2 cells.

    direction "in"  -- pull the current credential just before running codex
    direction "out" -- push a rotated credential back so the next cell picks it up
    Both under one flock, held only for the copy, never across the codex call itself.
    """
    import fcntl
    dst = Path(codex_home) / "auth.json"
    if not AUTH_SRC.exists():
        return
    AUTH_LOCK.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(AUTH_LOCK, "a+") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                if direction == "in":
                    dst.write_text(AUTH_SRC.read_text())
                elif dst.exists() and dst.read_text() != AUTH_SRC.read_text():
                    AUTH_SRC.write_text(dst.read_text())   # we rotated it; publish
            finally:
                fcntl.flock(lk, fcntl.LOCK_UN)
    except Exception as e:                                  # never let auth plumbing kill a run
        print(f"[loop] auth sync ({direction}) failed: {type(e).__name__}: {e}", flush=True)


def _auth_reuse_error(log_path):
    try:
        return "refresh_token_reused" in Path(log_path).read_text()
    except Exception:
        return False


def run_codex(model, effort, prompt, wd, mcp_url, wall, tag, codex_home):
    """Retries when another cell rotated the shared refresh token out from under this one."""
    for attempt in range(AUTH_RETRIES):
        _auth_sync(codex_home, "in")
        rc = _run_codex_once(model, effort, prompt, wd, mcp_url, wall, tag, codex_home)
        _auth_sync(codex_home, "out")
        if not _auth_reuse_error(wd / f"{tag}.jsonl"):
            return rc
        if attempt < AUTH_RETRIES - 1:
            back = min(60, 5 * (attempt + 1)) + random.uniform(0, 5)
            print(f"[loop] codex {tag}: refresh_token_reused, resyncing auth and retrying "
                  f"in {back:.0f}s ({attempt + 1}/{AUTH_RETRIES - 1})", flush=True)
            time.sleep(back)
    print(f"[loop] codex {tag}: still refresh_token_reused after {AUTH_RETRIES} attempts",
          flush=True)
    return rc


def _run_codex_once(model, effort, prompt, wd, mcp_url, wall, tag, codex_home):
    cmd = ["codex", "exec", "--strict-config", "-m", model,
           "-c", f"model_reasoning_effort={effort}", "-s", "danger-full-access",
           "-c", "approval_policy=never", "--json", "-C", str(wd)]
    if mcp_url:
        cmd += ["-c", f'mcp_servers.planner.url="{mcp_url}"',
                "-c", "mcp_servers.planner.tool_timeout_sec=1800",
                "-c", 'mcp_servers.planner.default_tools_approval_mode="approve"']
    env = os.environ.copy()
    npm = Path.home() / ".npm-global" / "bin"
    if npm.exists():
        env["PATH"] = f"{npm}:{env['PATH']}"
    env["CODEX_HOME"] = str(codex_home)
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=open(wd / f"{tag}.jsonl", "w"),
                         stderr=subprocess.STDOUT, env=env, cwd=str(wd), text=True)
    try:
        p.stdin.write(prompt)
        p.stdin.close()
        p.wait(timeout=wall)
    except subprocess.TimeoutExpired:
        p.kill()
    return p.returncode


def orchestrate(step, n, state_md, outdir, args, codex_home):
    """One orchestrator call per step. submit_plan enforces sum(n) == budget EXACTLY, so
    under-spend is structurally impossible; if no valid plan lands in the turn cap the step is
    recorded as a no-op rather than auto-filled (that must stay visible, not silently shrink
    the cell)."""
    import socket
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    wd = outdir / f"orch_{step:02d}"
    wd.mkdir(parents=True, exist_ok=True)
    srv = subprocess.Popen(
        ["uv", "run", "--script", str(HERE / "plan_mcp.py"), "--port", str(port),
         "--logdir", str(wd), "--budget", str(n)],
        stdout=open(wd / "planner.log", "w"), stderr=subprocess.STDOUT)
    for _ in range(90):
        try:
            socket.create_connection(("127.0.0.1", port), 1).close(); break
        except OSError:
            if srv.poll() is not None:
                return None
            time.sleep(1)
    try:
        pr = P.orchestrator_prompt(
            n, args.turns,
            "the current search state, dumped to a file you can grep",
            f"  {state_md}")
        run_codex(args.orch_model, args.orch_effort, pr, wd,
                  f"http://127.0.0.1:{port}/mcp", args.orch_wall, "orch", codex_home)
    finally:
        srv.terminate()
    acc = wd / "plan_accepted.json"
    return json.loads(acc.read_text())["allocations"] if acc.exists() else None


def compact(step, round_md, workspace_path, outdir, args, codex_home):
    wd = outdir / f"compact_{step:02d}"
    wd.mkdir(parents=True, exist_ok=True)
    out = wd / "workspace_new.md"
    pr = P.compactor_prompt(round_md, workspace_path, out, args.turns)
    run_codex(args.compact_model, args.compact_effort, pr, wd, None,
              args.orch_wall, "compact", codex_home)
    if not out.exists():
        return None
    ws = out.read_text()
    # Physical backstop for the 3-sections/8-sentences contract. The first live compaction
    # wrote 168KB (~45k tokens) -- larger than any servable window -- and 400'd the whole next
    # step. Truncation is a logged last resort, not the normal path.
    if len(ws) > WORKSPACE_MAX_CHARS:
        print(f"[loop] compact step {step}: workspace {len(ws)} chars > "
              f"{WORKSPACE_MAX_CHARS} -> TRUNCATED (contract violation)", flush=True)
        ws = ws[:WORKSPACE_MAX_CHARS] + "\n[truncated: workspace exceeded size contract]\n"
        out.write_text(ws)
    return ws


def edit_memory(step, agent, round_md, mem_path, outdir, args, codex_home, shared):
    wd = outdir / f"mem_{step:02d}_a{agent}"
    wd.mkdir(parents=True, exist_ok=True)
    out = wd / "memory_new.md"
    cur = mem_path if Path(mem_path).exists() else "(empty -- first round)"
    pr = P.memory_editor_prompt(round_md, cur, out, args.turns, shared=shared)
    run_codex(args.compact_model, args.compact_effort, pr, wd, None,
              args.orch_wall, f"mem{agent}", codex_home)
    if not out.exists():
        return None
    ws = out.read_text()
    if len(ws) > MEMORY_MAX_CHARS:
        print(f"[loop] memory step {step} agent {agent}: {len(ws)} chars > "
              f"{MEMORY_MAX_CHARS} -> TRUNCATED (contract violation)", flush=True)
        ws = ws[:MEMORY_MAX_CHARS] + "\n[truncated: memory exceeded size contract]\n"
        out.write_text(ws)
    return ws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True)
    ap.add_argument("--state", required=True,
                    choices=["puct", "workspace", "perennial", "team", "team-split"])
    ap.add_argument("--execution", required=True, choices=["simple", "orchestrator"])
    ap.add_argument("--composition", required=True,
                    choices=["qwen", "gemma", "50-50", "qq", "gg", "qg"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--registry", default="/n/fs/vision-mix/sk7524/SkyRLTpu/runs/rq2/fleet/registry.json")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--grade-concurrency", type=int, default=16)
    ap.add_argument("--fast-budget", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument("--B", type=int, default=None,
                    help="sampled states per step; with --G this sets n = B*G")
    ap.add_argument("--G", type=int, default=None,
                    help="rollouts per sampled state (group size); alias for --k")
    ap.add_argument("--k", type=int, default=5,
                    help="candidates per bundle (SimpleTES batch size: K candidates share one "
                         "inspiration set; the local best commits + reflects)")
    ap.add_argument("--chains", type=int, default=4,
                    help="SimpleTES chain count C (reference default 4). With C < B a chain "
                         "serves B/C bundles per step; inspiration selection SAMPLES with "
                         "rank-skewed probability (state_reuse.SAMPLE_GAMMA) so those bundles "
                         "diverge. Deeper chains (~1+steps*B/C members) give RPUCG real "
                         "selection pressure vs C=B's ~11-member stubs.")
    ap.add_argument("--orch-model", default="gpt-5.6-sol")
    ap.add_argument("--orch-effort", default="xhigh")
    ap.add_argument("--compact-model", default="gpt-5.3-codex-spark")
    ap.add_argument("--compact-effort", default="medium")
    ap.add_argument("--orch-wall", type=int, default=2400)
    args = ap.parse_args()
    if args.B is not None and args.G is not None:
        args.n, args.k = args.B * args.G, args.G     # the B x G framing: n rollouts as B bundles of G

    out = Path(args.out).resolve()
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "gradelogs").mkdir(exist_ok=True)
    pack = PACKS / args.problem
    meta = json.loads((pack / "meta.json").read_text())
    task = (pack / "prompt_completion.md").read_text()
    fence, maximize, baseline = meta["fence"], meta["maximize"], meta.get("seed_score")
    seed = (pack / ("seed.py" if meta["lang"] == "python" else "seed.cpp")).read_text()

    codex_home = out / "codex_home"
    if (args.execution == "orchestrator" or args.state == "workspace"
            or args.state in ("perennial", "team", "team-split")):   # memory editor runs codex too
        sys.path.insert(0, str(HERE.parent.parent / "rq1" / "client"))
        import preflight
        # Sandbox probe (job 3670962): landlock blocks ALL file access for the agent
        # ("permission profiles requiring direct runtime enforcement are incompatible"),
        # and without landlock bwrap cannot start (max_user_namespaces=0). The only mode on
        # this cluster where the orchestrator can read state and write its plan is
        # danger-full-access with no sandbox config at all.
        preflight.write_codex_home(out, landlock=False, long_provider=True)

    st = make_state_reuse(args.state, out / "state", maximize, seed, baseline,
                          num_chains=args.chains) if args.state in ("puct", "simpletes") \
        else make_state_reuse(args.state, out / "state", maximize, seed, baseline)
    (out / "manifest.json").write_text(json.dumps({
        "problem": args.problem, "state": args.state, "execution": args.execution,
        "composition": args.composition, "n": args.n, "steps": args.steps,
        "fast_budget": args.fast_budget, "started": time.strftime("%F %T")}, indent=2))

    trace = out / "trace.jsonl"
    done_steps = set()
    if trace.exists():
        done_steps = {json.loads(l)["step"] for l in trace.read_text().splitlines() if l.strip()}

    step = 1
    while step <= args.steps:
        # An outage must STALL a step, never consume it: rehearsal 3673116 lost steps 1-3 to
        # rollouts dispatched against a stale-healthy registry (0/512, all ConnectTimeout) and
        # steps 8-10 to a `continue` that advanced the counter while "waiting" -- the cell
        # finished at the seed score without doing a single step of work.
        if step in done_steps:
            print(f"[loop] step {step}: already done, skipping", flush=True)
            step += 1
            continue
        t0 = time.time()
        args._step = step
        reg = REG.read(args.registry)
        urls = {m: REG.urls(args.registry, model=m) for m in ("qwen35", "gemma4")}
        need = COMP_MODELS[args.composition]
        if any(not urls[m] for m in need):
            print(f"[loop] step {step}: farm has no healthy {need} endpoints; waiting", flush=True)
            time.sleep(120)
            continue                                 # same step, retried

        bundles = st.sample(args.n, args.k)
        # flatten: candidate i -> its bundle, all k candidates of a bundle share one prompt
        budget = prompt_char_budget(need)
        if args.execution == "orchestrator":
            budget -= 1500                       # room for the appended group instruction
        bundle_of, base_prompts, trimmed = [], [], 0
        for b in bundles:
            bp = st.render_bundle(b, task, fence) + P.ROLLOUT_TAIL.format(fence=fence)
            # Safety net, not the sizing mechanism: at 32k this should almost never fire. The
            # alternative is what smoke 3672525-28 measured -- silent vLLM 400s eating 60-100%
            # of a step's budget. Drop the least-preferred inspirations first, then trim the
            # context block.
            while len(bp) > budget and b.inspirations:
                b.inspirations = b.inspirations[:-1]
                bp = st.render_bundle(b, task, fence) + P.ROLLOUT_TAIL.format(fence=fence)
                trimmed += 1
            if len(bp) > budget and b.context:
                overshoot = len(bp) - budget
                b.context = b.context[:max(0, len(b.context) - overshoot - 64)]
                bp = st.render_bundle(b, task, fence) + P.ROLLOUT_TAIL.format(fence=fence)
                trimmed += 1
            for _ in range(b.k):
                bundle_of.append(b)
                base_prompts.append(bp)
        if trimmed:
            print(f"[loop] step {step}: PROMPT BUDGET trimmed {trimmed} block(s) "
                  f"(budget {budget} chars)", flush=True)
        allocations = None
        if args.execution == "orchestrator":
            state_md = (out / "state" / f"round_{step-1:02d}.md") if args.state == "workspace" \
                else (out / "state" / "graph.json")
            allocations = orchestrate(step, args.n, state_md, out, args, codex_home)
            if allocations is None:
                with open(trace, "a") as fh:
                    fh.write(json.dumps({"step": step, "plan_submitted": False, "n_requested": args.n,
                                         "note": "no valid plan within turn cap; step is a no-op",
                                         "secs": round(time.time() - t0)}) + "\n")
                print(f"[loop] step {step}: NO PLAN -> no-op (budget unspent, recorded)", flush=True)
                step += 1                            # no-plan is a CONSUMED no-op step by design
                continue
            instr = []
            for g in allocations:
                instr += [g["instruction"]] * g["n"]
            base_prompts = [p + "\n\n## Additional instruction for this group\n" + instr[i]
                            for i, p in enumerate(base_prompts)]

        models = assign_models_by_bundle(bundles, args.composition)
        print(f"[loop] step {step}: sampling {args.n} ({args.composition})", flush=True)
        results = asyncio.run(sample_round(base_prompts, models, urls, args, out, fence))
        req_failed = sum(1 for r in results
                         if r.get("program") is None
                         and str(r.get("detail", "")).startswith(("request failed", "phase2 failed")))
        if req_failed == len(results):
            # 100% transport failure = the fleet died under us mid-step (stale-healthy
            # registry). Nothing was generated; retry the SAME step once endpoints return.
            print(f"[loop] step {step}: ALL {req_failed} rollouts failed at transport -> "
                  "outage, retrying step", flush=True)
            time.sleep(120)
            continue
        print(f"[loop] step {step}: grading", flush=True)
        base_constr = None
        cj = PACKS / args.problem / "seed_construction.json"
        if cj.exists():
            base_constr = json.loads(cj.read_text())
        remote = asyncio.run(_grade_remote(results, args.problem, args, base_constr))
        if remote is None:
            print(f"[loop] step {step}: no grader shims healthy -> grading locally", flush=True)
            results = grade_round(results, args.problem, out, args.grade_concurrency, args.fast_budget)
        else:
            retry = [r for r in results if r.pop("_regrade_local", False)]
            if retry:
                print(f"[loop] step {step}: {len(retry)} remote failures -> local retry", flush=True)
                grade_round(retry, args.problem, out, args.grade_concurrency, args.fast_budget)
        # regroup by bundle (sid carries the flat candidate index)
        by_idx = {int(r["sid"].rsplit("r", 1)[1]): r for r in results}
        grouped = []
        i = 0
        for b in bundles:
            grouped.append((b, [by_idx[j] for j in range(i, i + b.k) if j in by_idx]))
            i += b.k
        # reflection: only where the state says it matters (SimpleTES: batch winners)
        targets = st.reflection_targets(grouped)
        if targets:
            asyncio.run(reflect_round(targets, urls, args, maximize, baseline))
        upkw = {}
        if args.state == "workspace":
            upkw["compact_fn"] = lambda md, wp: compact(step, md, wp, out, args, codex_home)
        elif args.state in ("perennial", "team", "team-split"):
            upkw["memory_fn"] = (lambda a, md, mp, _s=step: edit_memory(
                _s, a, md, mp, out, args, codex_home, shared=args.state != "perennial"))
        st.update(grouped, step, **upkw)
        prog, best = st.best()
        ok = [r for r in results if r.get("score") is not None]
        with open(trace, "a") as fh:
            fh.write(json.dumps({"step": step, "plan_submitted": allocations is not None,
                                 "groups": len(allocations) if allocations else None,
                                 "n_requested": args.n, "valid": len(ok), "best": best,
                                 # per-agent memory-editor success; [] for states with no memory
                                 "mem_ok": getattr(st, "mem_ok", []),
                                 "secs": round(time.time() - t0)}) + "\n")
        import collections
        why = collections.Counter((r.get("detail") or "ok")[:60] for r in results if r.get("score") is None)
        print(f"[loop] step {step}: {len(ok)}/{args.n} valid, best={best} "
              f"({int(time.time()-t0)}s)", flush=True)
        for d, c in why.most_common(4):
            print(f"[loop]    {c:4d} x {d}", flush=True)
        step += 1

    prog, best = st.best()
    (out / "result.json").write_text(json.dumps(
        {"problem": args.problem, "state": args.state, "execution": args.execution,
         "composition": args.composition, "n": args.n, "steps": args.steps,
         "best_fast_score": best, "best_program": prog}, indent=2))
    print(f"[loop] DONE best(fast)={best} -> {out}/result.json", flush=True)


if __name__ == "__main__":
    main()
