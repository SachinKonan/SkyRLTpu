"""Why did the invalid gens fail? Re-grade the non-valid records from an initial_*.json,
patching budget_s low so it's fast, and capture the reward-evaluator's failure message.
Reports the distribution of rejection reasons. No prompt/model calls — just grading.
"""
from __future__ import annotations
import argparse, collections, importlib, json, re
import common

ENVS = {"erdos_min_overlap": ("examples.erdos_min_overlap.env", "ErdosMinOverlapEnv"),
        "ac_inequalities": ("examples.ac_inequalities.env", "AutoCorrInequalityEnv")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--initial", required=True)
    ap.add_argument("--env", choices=list(ENVS), required=True)
    ap.add_argument("--problem-type", default="")
    ap.add_argument("--budget", type=int, default=20, help="patch budget_s=1000 -> this")
    ap.add_argument("--eval-timeout", type=int, default=90)
    ap.add_argument("--pool-snapshot", default="")
    args = ap.parse_args()

    common.load_dotenv_key()
    mod, cls = ENVS[args.env]
    EnvClass = getattr(importlib.import_module(mod), cls)
    scratch = f"{common.REPO_ROOT}/runs/behavior_probe/_diag/{args.problem_type or args.env}"
    from pathlib import Path; Path(scratch).mkdir(parents=True, exist_ok=True)

    d = json.load(open(args.initial))
    if args.pool_snapshot:
        sampler = common.load_pool_sampler(args.pool_snapshot, scratch + "/pool")
        with sampler._lock:
            base = max((s for s in sampler._states if s.value is not None and s.code), key=lambda s: s.value)
    else:
        base = EnvClass.create_initial_state(args.problem_type)

    bad = [r for r in d["records"] if r["status"] != "valid" and r.get("code")]
    print(f"diagnosing {len(bad)} invalid gens for {args.env}/{args.problem_type or '-'}")
    reasons = collections.Counter()
    from concurrent.futures import ThreadPoolExecutor
    def diag(r):
        code = re.sub(r"budget_s\s*=\s*1000\b", f"budget_s={args.budget}", r["code"])
        ev = EnvClass.reward_function(problem_type=args.problem_type, log_dir=scratch,
                                      num_cpus_per_task=1, eval_timeout=args.eval_timeout, eval_backend="local")
        try:
            out = ev.get_reward(f"```python\n{code}\n```", base)
            if out.get("correctness", 0) > 0:
                return "actually_valid_at_low_budget"
            msg = (out.get("msg") or "").strip()
            return (msg[:80] or "empty_msg")
        except Exception as e:
            return f"exc:{type(e).__name__}:{str(e)[:50]}"
    with ThreadPoolExecutor(max_workers=min(len(bad), 12)) as ex:
        for res in ex.map(diag, bad):
            reasons[res] += 1
    print("\n--- failure reasons ---")
    for msg, n in reasons.most_common():
        print(f"  {n:>2}x  {msg}")


if __name__ == "__main__":
    main()
