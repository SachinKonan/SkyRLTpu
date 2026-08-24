"""Scientist stage: gpt-oss generates improvement PLANS (not code) for each base.

The plan prompt = env.get_question(base) with TWO surgical edits (both asserted to fire):
  1. strip the `## Rules` block (the run() contract + code-style rules) — the planner
     doesn't write run(), so it must not see it;
  2. replace the trailing code_section ("Reason about how you could further improve...")
     with the SAME paragraph + open-ended plan guidance (+ foreign refs for --context foreign).
Everything else (problem, record, current program, scores, initial_h_values hint) is byte-identical.
Saves plans to corpora/plans_<ctx>.json. Tinker sampling only (no grading).
"""

from __future__ import annotations

import argparse
import asyncio
import re
import ssl

import common

# exact trailing instruction in env.get_question for a base that HAS code (env.py:142-145)
CODE_SECTION = (
    "Reason about how you could further improve this construction.\n"
    "Ideally, try to do something different than the above algorithm. Could be using "
    "different algorithmic ideas, adjusting your heuristics, adjusting / sweeping your "
    "hyperparemeters, etc. \nUnless you make a meaningful improvement, you will not be rewarded."
)

PLAN_GUIDANCE = """

Rather than writing the code yourself, describe your improvement as a PLAN for an expert coder to
implement. A good plan typically:
- identifies the key issues or limitations of the current approach;
- proposes one or more strategies to improve it — these can be small, targeted changes or larger,
  conceptual rethinks of the approach;
- points to where in the current program the changes should be made;
- optionally, sketches pseudocode for the key parts.

Decide for yourself what the plan should contain; the above are suggestions, not requirements. Do
not write the full program. Output ONLY the plan, wrapped in <plan>...</plan>."""

REF_HEADER = """

## Reference programs (higher-scoring solutions for the SAME problem). Draw the TECHNIQUES they
use that the current approach is missing INTO YOUR PLAN. Do not copy their code and do not mention
them in the plan.{refs}"""
REF_BLOCK = "\n### Reference {i} (achieved C5 = {c5:.6f}):\n```python\n{code}\n```"


def make_plan_prompt(q: str, refs: str) -> str:
    assert "## Rules" in q and "**Lower is better**" in q, "problem template changed"
    q = q.split("## Rules")[0] + "**Lower is better**" + q.split("**Lower is better**", 1)[1]
    assert CODE_SECTION in q, "code_section not found — env.get_question changed"
    return q.replace(CODE_SECTION, CODE_SECTION + refs + PLAN_GUIDANCE)


def extract_plan(text: str) -> str:
    # closed <plan>…</plan>, else unclosed <plan>…(truncated), else the longest
    # harmony channel segment (the plan may sit in analysis or final).
    for pat in (r"<plan>(.*?)</plan>", r"<plan>(.*)"):
        m = re.search(pat, text, re.S)
        if m and len(m.group(1).strip()) >= 80:
            return m.group(1).strip()
    parts = re.split(r"<\|channel\|>\w+<\|message\|>", text)
    return max((p.strip() for p in parts), key=len, default=text.strip()).strip()


async def run(args):
    common.load_dotenv_key()
    import tinker
    from examples.erdos_min_overlap.env import ErdosMinOverlapEnv
    from ttt_discover.tinker_utils.state import state_from_dict
    from ttt_discover.tinker_utils.completers import TwoPhaseTokenCompleter

    svc = tinker.ServiceClient()
    scratch = f"{common.REPO_ROOT}/runs/distill_ablation/_plans/{args.context}"
    renderer, tok, cfg = common.env_bits(scratch)
    sampler = common.load_pool_sampler(args.pool_snapshot, scratch + "/pool")
    ST = ErdosMinOverlapEnv.state_type
    bases = [state_from_dict(w, state_type=ST)
             for w in common.read_json(args.worse_set)["worse"][: args.n_bases]]

    refs_by_base = {}
    if args.context == "foreign":
        fb = common.read_json(args.foreign_betters)["betters"]
        qwen = sorted([b for b in fb if b["origin"] == "qwen"], key=lambda b: b["value"], reverse=True)
        nemo = sorted([b for b in fb if b["origin"] == "nemo"], key=lambda b: b["value"], reverse=True)

    sc = svc.create_sampling_client(base_model=common.STUDENT_MODEL)
    completer = TwoPhaseTokenCompleter(sampling_client=sc, tokenizer=tok,
                                       phase1_max_tokens=args.phase1_max_tokens,
                                       temperature=1.0, context_window=common.CONTEXT_WINDOW)
    stop = renderer.get_stop_sequences()

    def refs_for(base):
        if args.context != "foreign":
            return ""
        chosen = []
        q = [x for x in qwen if x["value"] > base.value]
        n = [x for x in nemo if x["value"] > base.value]
        if q: chosen.append(q[0])
        if n: chosen.append(n[0])
        blocks = "".join(REF_BLOCK.format(i=i + 1, c5=-c["value"], code=c["code"][:args.max_ref_chars])
                         for i, c in enumerate(chosen))
        return REF_HEADER.format(refs=blocks)

    async def plans_for(base):
        env = ErdosMinOverlapEnv(renderer, initial_state=base, sampler=sampler, config=cfg)
        prompt = make_plan_prompt(env.get_question(), refs_for(base))
        mi = renderer.build_generation_prompt([{"role": "user", "content": prompt}])
        gens = await asyncio.gather(*[completer(mi, stop) for _ in range(args.n_plans)],
                                    return_exceptions=True)
        # ALWAYS cache the raw generations too (memory: cache-raw-model-generations)
        raw = [tok.decode(list(g.tokens)) for g in gens if not isinstance(g, Exception)]
        plans = [extract_plan(t) for t in raw]
        return {"base_id": base.id, "base_c5": -base.value, "base_code": base.code,
                "plans": plans, "raw": raw}

    results = await asyncio.gather(*[plans_for(b) for b in bases])
    common.write_json(args.out, {"context": args.context, "n_bases": len(bases),
                                 "n_plans": args.n_plans, "items": results})
    npl = sum(len(r["plans"]) for r in results)
    lens = [len(p) for r in results for p in r["plans"]]
    print(f"[gen_plans:{args.context}] {npl} plans over {len(results)} bases -> {args.out}")
    print(f"  plan length chars: min={min(lens)} med={sorted(lens)[len(lens)//2]} max={max(lens)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", choices=["vanilla", "foreign"], required=True)
    ap.add_argument("--n-bases", type=int, default=16)
    ap.add_argument("--n-plans", type=int, default=3)
    ap.add_argument("--phase1-max-tokens", type=int, default=26000)
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
