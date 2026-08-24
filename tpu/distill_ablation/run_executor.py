"""Executor stage: a Codex model implements each scientist PLAN into a program, graded locally.

For each (base, plan) in plans_<ctx>.json, run `codex exec -m <model> -c model_reasoning_effort=high`
in a fresh workdir (inside the trusted repo) with a FAITHFUL contract (translate the plan, fix
correctness, do NOT change strategy or optimize C5), read solution.py, grade it against the base.
One --executor-model per invocation (run 3x for mini/terra/luna). Run on a compute node.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import common

CODEX_BIN_DIR = "/n/fs/vision-mix/sk7524/.npm-global/bin"

EXECUTOR_PROMPT = """You are a faithful code executor. Implement the improvement PLAN below into a \
Python file named `solution.py`. You are a TRANSLATOR, not an optimizer.

## Problem and current program
{problem}

## Improvement plan to implement
{plan}

## Your rules
- Write ONLY a file named `solution.py` in the current directory, defining \
`run(seed=42, budget_s=1000, **kwargs)` that returns `(h_values, c5_bound, n_points)`.
- `initial_h_values` (the current construction, a numpy array) and `evaluate_erdos_solution(...)` are \
provided as globals at grading time; do not redefine them.
- You MAY run solution.py to confirm it executes and returns valid output, and MAY fix bugs (syntax, \
API misuse, shape/constraint errors). If you run it, pass a SMALL budget (e.g. `run(budget_s=15)`) \
just to check it executes — do NOT run the full optimization budget.
- You MUST NOT change the strategy, swap the optimizer, add optimizations the plan does not specify, \
or tune parameters to lower C5 beyond what the plan states. Implement the plan faithfully.
"""

# No-run variant: read-only sandbox, one-shot codegen (no execution => hard guarantee the executor
# cannot tune against C5). We parse the program from stdout and cache it to solution.py ourselves.
EXECUTOR_PROMPT_NORUN = """You are a faithful code executor. Translate the improvement PLAN below into \
a single Python program. You are a TRANSLATOR, not an optimizer.

## Problem and current program
{problem}

## Improvement plan to implement
{plan}

## Your rules
- Output the COMPLETE program as ONE ```python code block and nothing else — no prose, no explanation.
- Define `run(seed=42, budget_s=1000, **kwargs)` returning `(h_values, c5_bound, n_points)`.
- `initial_h_values` (the current construction, a numpy array) and `evaluate_erdos_solution(...)` are \
provided as globals at grading time; do not redefine them.
- Do NOT run, execute, or test anything. Just write the code. It must be correct as written.
- You MUST NOT change the strategy, swap the optimizer, add optimizations the plan does not specify, \
or tune parameters to lower C5 beyond what the plan states. Implement the plan faithfully.
"""


def codex_execute(prompt: str, model: str, workdir: Path, timeout: int, mode: str = "run"):
    workdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PATH"] = CODEX_BIN_DIR + ":" + env.get("PATH", "")
    sandbox = "read-only" if mode == "norun" else "workspace-write"
    cmd = ["codex", "exec", "-m", model, "-c", "model_reasoning_effort=high",
           "-s", sandbox, "-c", "approval_policy=never"]
    sol = workdir / "solution.py"
    err = ""
    try:
        r = subprocess.run(cmd, input=prompt, cwd=str(workdir), capture_output=True,
                           text=True, timeout=timeout, env=env)
        err = (r.stderr or "")[-400:]
    except subprocess.TimeoutExpired:
        err = "TIMEOUT"  # NOT fatal: codex often wrote solution.py before we killed it
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    # ALWAYS cache the raw codex stdout for analysis (memory: cache-raw-model-generations).
    try:
        (workdir / "codex_stdout.txt").write_text(r.stdout or "")
    except Exception:
        pass
    if mode == "norun":
        # read-only => no file written; the program is in the final message. Parse and cache it.
        m = re.search(r"```python\s+([\s\S]*?)\s*```", r.stdout or "")
        if m and m.group(1).strip():
            code = m.group(1)
            try:
                sol.write_text(code)
            except Exception:
                pass
            return code, err
        return None, ((r.stderr or "") + (r.stdout or ""))[-400:]
    # run mode: prefer a solution.py on disk — even on timeout it may be complete.
    if sol.exists() and sol.read_text().strip():
        return sol.read_text(), err
    if err != "TIMEOUT":
        m = re.search(r"```python\s+([\s\S]*?)\s*```", r.stdout)
        if m:
            return m.group(1), err
        return None, ((r.stderr or "") + (r.stdout or ""))[-400:]
    return None, err


async def run(args):
    common.load_dotenv_key()
    from examples.erdos_min_overlap.env import ErdosMinOverlapEnv, ErdosMinOverlapRewardEvaluator
    from ttt_discover.tinker_utils.state import state_from_dict

    scratch = f"{common.REPO_ROOT}/runs/distill_ablation/_exec/{args.tag}"
    Path(scratch).mkdir(parents=True, exist_ok=True)
    renderer, tok, cfg = common.env_bits(scratch, eval_timeout=args.eval_timeout)
    sampler = common.load_pool_sampler(args.pool_snapshot, scratch + "/pool")
    ST = ErdosMinOverlapEnv.state_type

    data = common.read_json(args.plans)
    by_id = {r["base_id"]: r for r in data["items"]}
    # rebuild base States from worse_set (need .construction for grading)
    bases = {w["id"]: state_from_dict(w, state_type=ST)
             for w in common.read_json(args.worse_set)["worse"]}

    tasks = []  # (base_state, plan_text, plan_idx, base_c5)
    for bid, rec in by_id.items():
        base = bases[bid]
        for j, plan in enumerate(rec["plans"]):
            tasks.append((base, plan, j, rec["base_c5"]))

    sem = asyncio.Semaphore(args.exec_concurrency)
    loop = asyncio.get_event_loop()
    ex = ThreadPoolExecutor(max_workers=args.exec_concurrency)

    def grade(code_text, base):
        ev = ErdosMinOverlapRewardEvaluator(problem_type="", log_dir=scratch, num_cpus_per_task=1,
                                            eval_timeout=args.eval_timeout, eval_backend="local")
        try:
            out = ev.get_reward(code_text, base)
            return out.get("raw_score") if out.get("correctness", 0) > 0 else None
        except Exception:
            return None

    prompt_tmpl = EXECUTOR_PROMPT_NORUN if args.exec_mode == "norun" else EXECUTOR_PROMPT

    async def one(base, plan, j, base_c5):
        env = ErdosMinOverlapEnv(renderer, initial_state=base, sampler=sampler, config=cfg)
        problem = env.get_question()  # full (with run() Rules) — the coder needs it
        prompt = prompt_tmpl.format(problem=problem, plan=plan)
        wd = Path(scratch) / f"wd_{base.id[:8]}_{j}_{uuid.uuid4().hex[:6]}"
        async with sem:
            code, err = await loop.run_in_executor(ex, codex_execute, prompt, args.executor_model,
                                                   wd, args.exec_timeout, args.exec_mode)
        c5 = None
        if code and not args.no_grade:
            fenced = f"```python\n{code}\n```"
            c5 = await loop.run_in_executor(ex, grade, fenced, base)
        # ALWAYS cache the raw executor output (memory: cache-raw-model-generations)
        return {"base_id": base.id, "base_c5": base_c5, "plan_idx": j,
                "executor": args.executor_model, "got_code": code is not None,
                "code": code, "code_len": len(code or ""), "workdir": str(wd),
                "c5": c5, "improved": c5 is not None and (base_c5 - c5) > 1e-4,
                "delta": (base_c5 - c5) if c5 is not None else None, "err": (err or "")[:400]}

    t0 = time.time()
    results = await asyncio.gather(*[one(*t) for t in tasks])
    ex.shutdown(wait=False)
    common.write_json(args.out, {"executor": args.executor_model, "context": data["context"],
                                 "n": len(results), "results": results})

    n = len(results)
    got = sum(r["got_code"] for r in results)
    valid = [r for r in results if r["c5"] is not None]
    imp = [r for r in results if r["improved"]]
    # per-base best-of-N
    per_base = {}
    for r in results:
        if r["c5"] is not None:
            per_base.setdefault(r["base_id"], []).append((r["base_c5"], r["c5"]))
    bases_imp = sum(1 for v in per_base.values() if any(bc - c > 1e-4 for bc, c in v))
    print(f"\n===== EXECUTOR {args.executor_model} / ctx={data['context']}  ({time.time()-t0:.0f}s) =====")
    print(f"  got_code {got}/{n} | VALID {len(valid)}/{n} | improved(plan) {len(imp)}/{n} | "
          f"bases-improved(best-of) {bases_imp}/{len(by_id)}")
    if valid:
        c5s = sorted(r["c5"] for r in valid)
        print(f"  c5: best={c5s[0]:.6f} median={c5s[len(c5s)//2]:.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", required=True, help="plans_<ctx>.json from gen_plans.py")
    ap.add_argument("--executor-model", required=True, help="gpt-5.4-mini | gpt-5.6-terra | gpt-5.6-luna")
    ap.add_argument("--tag", required=True, help="scratch/output tag")
    ap.add_argument("--exec-mode", choices=["run", "norun"], default="norun",
                    help="norun = read-only one-shot codegen (no execution, hard no-tune guarantee)")
    ap.add_argument("--no-grade", action="store_true",
                    help="codegen + cache only; grade later with regrade_from_disk (parallel phase)")
    ap.add_argument("--exec-concurrency", type=int, default=5)
    ap.add_argument("--exec-timeout", type=int, default=300, help="per codex call wall-clock")
    ap.add_argument("--eval-timeout", type=int, default=1100)
    ap.add_argument("--worse-set", default="tpu/distill_ablation/corpora/worse_set.json")
    ap.add_argument("--pool-snapshot", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
