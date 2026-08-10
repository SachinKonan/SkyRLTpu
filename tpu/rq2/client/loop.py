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
from state_reuse import make_state_reuse  # noqa: E402

PACKS = HERE.parent.parent / "rq1" / "client" / "data"
ATTEMPTS = 6
# cached serving shapes; two-phase budgets must fit inside them (see the plan)
MODEL_CFG = {
    "qwen35": {"model": "Qwen/Qwen3.5-27B", "ctx": 22528, "p1": 12000, "p2": 7000},
    "gemma4": {"model": "google/gemma-4-31B-it", "ctx": 16384, "p1": 9000, "p2": 6000},
}
FORCE = ("\n\nI have thought about this enough. Here is my final, complete, self-contained "
         "program:\n\n```{fence}\n")


def sha(s):
    return hashlib.sha1((s or "").encode()).hexdigest()[:12]


def strip_fence(t):
    m = re.findall(r"```(?:python|cpp|c\+\+)?\s*([\s\S]*?)\s*```", t or "")
    return m[-1] if m else ""


def assign_models_by_bundle(bundles, composition):
    """Deterministic; for 50-50 the split is G/2 per model WITHIN EVERY BUNDLE, so each sampled
    state is explored by both models equally -- not merely a global 50/50 that could leave some
    states single-model."""
    out = []
    for b in bundles:
        if composition == "qwen":
            out += ["qwen35"] * b.k
        elif composition == "gemma":
            out += ["gemma4"] * b.k
        else:
            half = b.k // 2
            out += ["qwen35"] * half + ["gemma4"] * (b.k - half)
    return out


# ------------------------------------------------------------------ farm sampling (2-phase)
async def _post(cli, url, key, body):
    last = None
    for a in range(ATTEMPTS):
        try:
            r = await cli.post(f"{url}/v1/chat/completions",
                               headers={"Authorization": f"Bearer {key}"}, json=body)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            if a == ATTEMPTS - 1:
                raise last
            await asyncio.sleep(min(45, 8 * (a + 1)))


async def one_rollout(i, cli, url, key, mkey, prompt, fence, args, outdir, lock):
    """Returns a result dict. Two-phase forcing fires only when phase 1 truncates, so it costs
    nothing for models that finish naturally."""
    cfg = MODEL_CFG[mkey]
    base = {"model": cfg["model"], "temperature": args.temperature,
            "messages": [{"role": "user", "content": prompt}]}
    if mkey == "gemma4":
        base["chat_template_kwargs"] = {"enable_thinking": True}
    sid = f"s{args._step:02d}_r{i:04d}"
    try:
        d = await _post(cli, url, key, {**base, "max_tokens": cfg["p1"]})
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
                "continue_final_message": True, "add_generation_prompt": False})
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


async def sample_round(prompts, models, urls_by_model, args, outdir, fence):
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    rr = {k: 0 for k in urls_by_model}
    out = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(2400, connect=30)) as cli:
        async def go(i):
            mkey = models[i]
            pool = urls_by_model[mkey]
            url = pool[rr[mkey] % len(pool)]
            rr[mkey] += 1
            async with sem:
                return await one_rollout(i, cli, url, args.api_key, mkey, prompts[i],
                                         fence, args, outdir, lock)
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
    async with httpx.AsyncClient(timeout=httpx.Timeout(900, connect=30)) as cli:
        async def go(r):
            if not r.get("convo"):
                return
            mkey = r["model"]
            pool = urls_by_model[mkey]
            url = pool[rr[mkey] % len(pool)]
            rr[mkey] += 1
            body = {"model": MODEL_CFG[mkey]["model"], "temperature": 0.7, "max_tokens": 220,
                    "messages": r["convo"] + [{"role": "user",
                                               "content": P.reflect_prompt(r["score"], maximize, baseline)}]}
            async with sem:
                try:
                    d = await _post(cli, url, args.api_key, body)
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
def run_codex(model, effort, prompt, wd, mcp_url, wall, tag, codex_home):
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
    return out.read_text() if out.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True)
    ap.add_argument("--state", required=True, choices=["puct", "workspace"])
    ap.add_argument("--execution", required=True, choices=["simple", "orchestrator"])
    ap.add_argument("--composition", required=True, choices=["qwen", "gemma", "50-50"])
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
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--orch-model", default="gpt-5.6-sol")
    ap.add_argument("--orch-effort", default="xhigh")
    ap.add_argument("--compact-model", default="gpt-5.3-codex-spark")
    ap.add_argument("--compact-effort", default="medium")
    ap.add_argument("--orch-wall", type=int, default=2400)
    args = ap.parse_args()
    if args.B is not None and args.G is not None:
        args.n, args.k = args.B * args.G, args.G     # the B x G framing: n rollouts as B bundles of G
        # C = B: one bundle per chain per step ("sample one from each chain"). With C < B the
        # extra bundles per chain would select against IDENTICAL visit counts -- selection is
        # deterministic, so they would all carry the same 5 inspirations and effective B
        # collapses to C. The async reference never hits this because visits update between a
        # chain's consecutive batches.
        args.chains = args.B

    out = Path(args.out).resolve()
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "gradelogs").mkdir(exist_ok=True)
    pack = PACKS / args.problem
    meta = json.loads((pack / "meta.json").read_text())
    task = (pack / "prompt_completion.md").read_text()
    fence, maximize, baseline = meta["fence"], meta["maximize"], meta.get("seed_score")
    seed = (pack / ("seed.py" if meta["lang"] == "python" else "seed.cpp")).read_text()

    codex_home = out / "codex_home"
    if args.execution == "orchestrator" or args.state == "workspace":
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

    for step in range(1, args.steps + 1):
        if step in done_steps:
            print(f"[loop] step {step}: already done, skipping", flush=True)
            continue
        t0 = time.time()
        args._step = step
        reg = REG.read(args.registry)
        urls = {m: REG.urls(args.registry, model=m) for m in ("qwen35", "gemma4")}
        need = {"qwen": ["qwen35"], "gemma": ["gemma4"], "50-50": ["qwen35", "gemma4"]}[args.composition]
        if any(not urls[m] for m in need):
            print(f"[loop] step {step}: farm has no healthy {need} endpoints; waiting", flush=True)
            time.sleep(120)
            continue

        bundles = st.sample(args.n, args.k)
        # flatten: candidate i -> its bundle, all k candidates of a bundle share one prompt
        bundle_of, base_prompts = [], []
        for b in bundles:
            bp = st.render_bundle(b, task, fence) + P.ROLLOUT_TAIL.format(fence=fence)
            for _ in range(b.k):
                bundle_of.append(b)
                base_prompts.append(bp)
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
                continue
            instr = []
            for g in allocations:
                instr += [g["instruction"]] * g["n"]
            base_prompts = [p + "\n\n## Additional instruction for this group\n" + instr[i]
                            for i, p in enumerate(base_prompts)]

        models = assign_models_by_bundle(bundles, args.composition)
        print(f"[loop] step {step}: sampling {args.n} ({args.composition})", flush=True)
        results = asyncio.run(sample_round(base_prompts, models, urls, args, out, fence))
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
        st.update(grouped, step,
                  compact_fn=(lambda md, wp: compact(step, md, wp, out, args, codex_home))
                  if args.state == "workspace" else None)
        prog, best = st.best()
        ok = [r for r in results if r.get("score") is not None]
        with open(trace, "a") as fh:
            fh.write(json.dumps({"step": step, "plan_submitted": allocations is not None,
                                 "groups": len(allocations) if allocations else None,
                                 "n_requested": args.n, "valid": len(ok), "best": best,
                                 "secs": round(time.time() - t0)}) + "\n")
        import collections
        why = collections.Counter((r.get("detail") or "ok")[:60] for r in results if r.get("score") is None)
        print(f"[loop] step {step}: {len(ok)}/{args.n} valid, best={best} "
              f"({int(time.time()-t0)}s)", flush=True)
        for d, c in why.most_common(4):
            print(f"[loop]    {c:4d} x {d}", flush=True)

    prog, best = st.best()
    (out / "result.json").write_text(json.dumps(
        {"problem": args.problem, "state": args.state, "execution": args.execution,
         "composition": args.composition, "n": args.n, "steps": args.steps,
         "best_fast_score": best, "best_program": prog}, indent=2))
    print(f"[loop] DONE best(fast)={best} -> {out}/result.json", flush=True)


if __name__ == "__main__":
    main()
