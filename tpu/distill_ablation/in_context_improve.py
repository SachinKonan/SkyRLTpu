"""In-context foreign-guided self-improvement (outcome-filtered).

gpt-oss improves its own base program with better programs IN CONTEXT (foreign:
Qwen+Nemo, or own: gpt-oss champions, or vanilla: none). It writes its OWN
think+program (executable, no parrot/competence risk); we grade all N gens and
keep the best-of-N per base if it beats the base. Measures the YIELD: does foreign
context help gpt-oss find improvements more than its own / than nothing.

Controlled: same base set across arms; only the in-context betters differ.
"""

from __future__ import annotations

import argparse
import asyncio
import ssl
import time

import numpy as np

import common

REF_HEADER = """

## Reference programs (higher-scoring solutions for the SAME problem). Study the
## TECHNIQUES they use that your current program may be missing, then write your
## OWN improved program. Do NOT copy their code verbatim and do NOT mention these
## reference programs in your solution.
{refs}

Now write your improved program following the rules above."""

REF_BLOCK = """
### Reference {i} (achieved C5 = {c5:.6f}):
```python
{code}
```"""


async def run(args):
    common.load_dotenv_key()
    import tinker
    from examples.erdos_min_overlap.env import ErdosMinOverlapEnv, ErdosMinOverlapRewardEvaluator
    from ttt_discover.tinker_utils.state import state_from_dict
    from ttt_discover.tinker_utils.completers import TwoPhaseTokenCompleter
    from concurrent.futures import ThreadPoolExecutor

    svc = tinker.ServiceClient()
    scratch = f"{common.REPO_ROOT}/runs/distill_ablation/_incontext/{args.arm}"
    renderer, tok, cfg = common.env_bits(scratch, eval_timeout=args.eval_timeout)
    sampler = common.load_pool_sampler(args.pool_snapshot, scratch + "/pool")
    ST = ErdosMinOverlapEnv.state_type

    bases = [state_from_dict(w, state_type=ST)
             for w in common.read_json(args.worse_set)["worse"][: args.n_bases]]
    base_ids = {b.id for b in bases}

    # betters sources
    qwen = nemo = champs = []
    if args.arm == "foreign":
        fb = common.read_json(args.foreign_betters)["betters"]
        qwen = sorted([b for b in fb if b["origin"] == "qwen"], key=lambda b: b["value"], reverse=True)
        nemo = sorted([b for b in fb if b["origin"] == "nemo"], key=lambda b: b["value"], reverse=True)
    elif args.arm == "own":
        with sampler._lock:
            champs = sorted(
                [{"value": s.value, "code": s.code} for s in sampler._states
                 if s.id not in base_ids and s.value is not None and s.code and s.code.strip()],
                key=lambda d: d["value"], reverse=True)

    sc = svc.create_sampling_client(base_model=common.STUDENT_MODEL)
    completer = TwoPhaseTokenCompleter(
        sampling_client=sc, tokenizer=tok, phase1_max_tokens=args.phase1_max_tokens,
        temperature=1.0, context_window=common.CONTEXT_WINDOW)
    stop = renderer.get_stop_sequences()

    def betters_for(base):
        if args.arm == "vanilla":
            return []
        if args.arm == "foreign":
            out = []
            q = [x for x in qwen if x["value"] > base.value]
            n = [x for x in nemo if x["value"] > base.value]
            if q: out.append(q[0])
            if n: out.append(n[0])
            return out
        return [c for c in champs if c["value"] > base.value][:2]

    def build_prompt(base, betters):
        env = ErdosMinOverlapEnv(renderer, initial_state=base, sampler=sampler, config=cfg)
        q = env.get_question()
        if not betters:
            return q
        refs = "".join(REF_BLOCK.format(i=i + 1, c5=-b["value"], code=b["code"][:args.max_ref_chars])
                       for i, b in enumerate(betters))
        return q + REF_HEADER.format(refs=refs)

    sem = asyncio.Semaphore(args.base_concurrency)
    loop = asyncio.get_event_loop()

    def grade(text, base):
        ev = ErdosMinOverlapRewardEvaluator(problem_type="", log_dir=scratch,
                                            num_cpus_per_task=1, eval_timeout=args.eval_timeout,
                                            eval_backend="local")
        try:
            out = ev.get_reward(text, base)
            return out.get("raw_score") if out.get("correctness", 0) > 0 else None
        except Exception:
            return None

    async def do_base(base):
        betters = betters_for(base)
        prompt = build_prompt(base, betters)
        mi = renderer.build_generation_prompt([{"role": "user", "content": prompt}])
        async with sem:
            gens = await asyncio.gather(*[completer(mi, stop) for _ in range(args.n_gens)],
                                        return_exceptions=True)
            texts = [tok.decode(list(g.tokens)) for g in gens if not isinstance(g, Exception)]
            with ThreadPoolExecutor(max_workers=max(1, len(texts))) as ex:
                c5s = await asyncio.gather(*[loop.run_in_executor(ex, grade, t, base) for t in texts])
        # ALWAYS cache the full raw generations (think+program) — expensive to make,
        # needed for analysis (memory: cache-raw-model-generations).
        import os
        gd = os.path.join(scratch, "gens"); os.makedirs(gd, exist_ok=True)
        for gi, t in enumerate(texts):
            with open(os.path.join(gd, f"{base.id[:8]}_gen{gi}.txt"), "w") as fh:
                fh.write(t)
        valid = [c for c in c5s if c is not None]
        base_c5 = -base.value
        best = min(valid) if valid else None
        return {"base_id": base.id, "base_c5": base_c5, "n_betters": len(betters),
                "n_valid": len(valid), "results": c5s, "best": best,
                "gens": texts,  # full raw think+program per attempt
                "delta": (base_c5 - best) if best is not None else None,
                "improved": best is not None and (base_c5 - best) > 1e-4}

    t0 = time.time()
    results = await asyncio.gather(*[do_base(b) for b in bases])
    common.write_json(args.out, {"arm": args.arm, "n_bases": len(bases),
                                 "n_gens": args.n_gens, "results": results})

    improved = sum(r["improved"] for r in results)
    deltas = [r["delta"] for r in results if r["delta"] is not None]
    print(f"\n===== ARM={args.arm}  ({time.time()-t0:.0f}s) =====")
    print(f"  bases IMPROVED (best-of-{args.n_gens}): {improved}/{len(bases)}")
    print(f"  mean best-of-{args.n_gens} Δc5 (over bases with a valid gen): "
          f"{np.mean(deltas):+.2e}  (n={len(deltas)})")
    for r in results:
        d = f"{r['delta']:+.2e}" if r["delta"] is not None else "  none  "
        print(f"    base_c5={r['base_c5']:.5f} betters={r['n_betters']} valid={r['n_valid']}/{args.n_gens} "
              f"bestΔ={d} {'IMPROVED' if r['improved'] else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["foreign", "own", "vanilla"], required=True)
    ap.add_argument("--n-bases", type=int, default=16)
    ap.add_argument("--n-gens", type=int, default=5)
    ap.add_argument("--phase1-max-tokens", type=int, default=26000)
    ap.add_argument("--eval-timeout", type=int, default=1100)
    ap.add_argument("--base-concurrency", type=int, default=4)
    ap.add_argument("--max-ref-chars", type=int, default=6000)
    ap.add_argument("--worse-set", default="tpu/distill_ablation/corpora/worse_set.json")
    ap.add_argument("--foreign-betters", default="tpu/distill_ablation/corpora/foreign_betters.json")
    ap.add_argument("--pool-snapshot", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ssl.create_default_context()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
