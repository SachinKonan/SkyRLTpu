"""Diagnostic: for a few foreign-context bases, save the FULL think-trace + program
gpt-oss generated, so we can separate comprehension (did it form a good plan from
the foreign examples?) from execution (did the code implement it / did it run?).
"""

from __future__ import annotations

import argparse
import asyncio
import re
import ssl
from pathlib import Path

import numpy as np

import common

REF_HEADER = """

## Reference programs (higher-scoring solutions for the SAME problem). Study the
## TECHNIQUES they use that your current program may be missing, then write your
## OWN improved program. Do NOT copy their code verbatim and do NOT mention these
## reference programs in your solution.
{refs}

Now write your improved program following the rules above."""
REF_BLOCK = "\n### Reference {i} (achieved C5 = {c5:.6f}):\n```python\n{code}\n```"

TECH = {"smoothmax": r"logsumexp|softmax|smooth.?max|log.?sum.?exp",
        "lbfgs": r"l-?bfgs|method=.?['\"]?L-BFGS|minimize\(",
        "anneal": r"anneal|temperature|basinhop",
        "projection": r"water.?fill|project|bisect|lagrang"}


async def run(args):
    common.load_dotenv_key()
    import tinker
    from examples.erdos_min_overlap.env import ErdosMinOverlapEnv, ErdosMinOverlapRewardEvaluator
    from ttt_discover.tinker_utils.state import state_from_dict
    from ttt_discover.tinker_utils.completers import TwoPhaseTokenCompleter
    from concurrent.futures import ThreadPoolExecutor

    svc = tinker.ServiceClient()
    outdir = Path(f"{common.REPO_ROOT}/runs/distill_ablation/_diag_ic")
    outdir.mkdir(parents=True, exist_ok=True)
    renderer, tok, cfg = common.env_bits(str(outdir), eval_timeout=args.eval_timeout)
    sampler = common.load_pool_sampler(args.pool_snapshot, str(outdir / "pool"))
    ST = ErdosMinOverlapEnv.state_type
    ev = ErdosMinOverlapRewardEvaluator(problem_type="", log_dir=str(outdir),
                                        num_cpus_per_task=1, eval_timeout=args.eval_timeout,
                                        eval_backend="local")

    bases = [state_from_dict(w, state_type=ST)
             for w in common.read_json(args.worse_set)["worse"][: args.n_bases]]
    fb = common.read_json(args.foreign_betters)["betters"]
    qwen = sorted([b for b in fb if b["origin"] == "qwen"], key=lambda b: b["value"], reverse=True)
    nemo = sorted([b for b in fb if b["origin"] == "nemo"], key=lambda b: b["value"], reverse=True)

    sc = svc.create_sampling_client(base_model=common.STUDENT_MODEL)
    completer = TwoPhaseTokenCompleter(sampling_client=sc, tokenizer=tok,
                                       phase1_max_tokens=26000, temperature=1.0,
                                       context_window=common.CONTEXT_WINDOW)
    stop = renderer.get_stop_sequences()
    loop = asyncio.get_event_loop()

    for bi, base in enumerate(bases):
        betters = ([q for q in qwen if q["value"] > base.value][:1]
                   + [n for n in nemo if n["value"] > base.value][:1])
        env = ErdosMinOverlapEnv(renderer, initial_state=base, sampler=sampler, config=cfg)
        refs = "".join(REF_BLOCK.format(i=i+1, c5=-b["value"], code=b["code"][:6000])
                       for i, b in enumerate(betters))
        prompt = env.get_question() + REF_HEADER.format(refs=refs)
        mi = renderer.build_generation_prompt([{"role": "user", "content": prompt}])
        gens = await asyncio.gather(*[completer(mi, stop) for _ in range(args.n_gens)])
        texts = [tok.decode(list(g.tokens)) for g in gens]

        def grade(t):
            try:
                out = ev.get_reward(t, base)
                return (out.get("raw_score") if out.get("correctness", 0) > 0 else None,
                        (out.get("msg") or "")[:100])
            except Exception as e:
                return None, f"{type(e).__name__}"
        with ThreadPoolExecutor(max_workers=len(texts)) as ex:
            graded = await asyncio.gather(*[loop.run_in_executor(ex, grade, t) for t in texts])

        print(f"\n{'='*70}\nBASE {bi}: c5={-base.value:.6f}  betters c5={[round(-b['value'],5) for b in betters]}")
        for gi, (text, (c5, msg)) in enumerate(zip(texts, graded)):
            think = text.split("</think>")[0]
            code = ev._extract_code(text.split("</think>")[-1]) or ""
            techs_think = {k for k, rx in TECH.items() if re.search(rx, think, re.I)}
            techs_code = {k for k, rx in TECH.items() if re.search(rx, code, re.I)}
            (outdir / f"base{bi}_gen{gi}.txt").write_text(text)
            delta = (-base.value - c5) if c5 is not None else None
            print(f"  gen{gi}: c5={c5} Δ={f'{delta:+.2e}' if delta is not None else 'invalid'} "
                  f"| plan_techs={sorted(techs_think)} | code_techs={sorted(techs_code)} | {msg[:60]}")
    print(f"\n[diag] raw traces saved to {outdir}/base*_gen*.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bases", type=int, default=3)
    ap.add_argument("--n-gens", type=int, default=3)
    ap.add_argument("--eval-timeout", type=int, default=1100)
    ap.add_argument("--worse-set", default="tpu/distill_ablation/corpora/worse_set.json")
    ap.add_argument("--foreign-betters", default="tpu/distill_ablation/corpora/foreign_betters.json")
    ap.add_argument("--pool-snapshot", required=True)
    args = ap.parse_args()
    ssl.create_default_context()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
