"""Diagnostic: for each cross-model pair, run the teacher and categorize the
critique as survived / grounding-dropped / leak-dropped — and for the
LEAK-DROPPED ones, print the exact offending identifier(s) + the critique, so we
can judge whether the leak filter threw away good technique-transfer or genuine
verbatim copying.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import re
import ssl

import common


async def run(args):
    common.load_dotenv_key()
    import tinker
    from examples.erdos_min_overlap.env import ErdosMinOverlapEnv
    from ttt_discover.tinker_utils.state import state_from_dict
    from ttt_discover.tinker_utils.completers import TwoPhaseTokenCompleter
    from ttt_discover.tinker_utils.misc_utils import get_tokenizer
    from ttt_discover.rl import context_distill as cd

    scratch = f"{common.REPO_ROOT}/runs/distill_ablation/_diagleak"
    sampler = common.load_pool_sampler(args.snapshot, scratch)
    ho = set(common.read_json(args.heldout)["ids"])
    fb = common.read_json(args.foreign_betters)["betters"]
    if args.better_origin:
        fb = [b for b in fb if b["origin"] == args.better_origin]
    with sampler._lock:
        sampler._states = [s for s in sampler._states if s.id not in ho]
        for s in sampler._states:
            s.origin = "gptoss"
        for b in fb:
            sampler._states.append(state_from_dict(b, state_type=ErdosMinOverlapEnv.state_type))

    renderer, tok, cfg = common.env_bits(scratch)
    builder = common.make_seed_builder(sampler, renderer, cfg, sampler._states[0], 1)
    pairs, _ = cd.select_cross_distill_pairs(
        sampler, [builder], "gptoss", args.max_pairs, 3, 10000, random.Random(0))
    print(f"[diagleak] {len(pairs)} cross pairs")

    svc = tinker.ServiceClient()
    sc = svc.create_sampling_client(base_model=common.STUDENT_MODEL)
    tokenizer = get_tokenizer(common.STUDENT_MODEL)
    completer = TwoPhaseTokenCompleter(sampling_client=sc, tokenizer=tokenizer,
                                       phase1_max_tokens=26000, temperature=1.0,
                                       context_window=common.CONTEXT_WINDOW)
    stop = renderer.get_stop_sequences()

    async def teach(pair):
        mi = renderer.build_generation_prompt([{"role": "user", "content": pair.teacher_prompt}])
        if mi.length >= 26000 - 2048:
            return None
        return await completer(mi, stop)

    results = await asyncio.gather(*[teach(p) for p in pairs], return_exceptions=True)

    cats = {"survived": 0, "grounding": 0, "leak": 0, "err": 0}
    leak_cases = []
    for pair, res in zip(pairs, results):
        if isinstance(res, Exception) or res is None:
            cats["err"] += 1; continue
        toks = list(res.tokens)
        span = cd._final_span_start(toks, tokenizer)
        if span is None:
            cats["err"] += 1; continue
        final = tokenizer.decode(toks[span:]).split(cd.FINAL_SPLIT)[-1]
        ok, _ = cd._grounding_ok(final, pair.worse.code)
        if not ok:
            cats["grounding"] += 1; continue
        offenders = cd._leak_offenders(final, pair.worse.code, pair.betters)
        if offenders:
            cats["leak"] += 1
            whats = re.findall(r"<what>(.*?)</what>", final, re.S)
            leak_cases.append((offenders, whats))
        else:
            cats["survived"] += 1

    print(f"\n[diagleak] categories: {cats}")
    print(f"\n===== LEAK-DROPPED CRITIQUES ({len(leak_cases)}): offender + advice =====")
    for i, (off, whats) in enumerate(leak_cases):
        print(f"\n--- dropped {i}: OFFENDERS = {sorted(off)} ---")
        for w in whats[:4]:
            print(f"  WHAT: {' '.join(w.split())[:240]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--foreign-betters", required=True)
    ap.add_argument("--better-origin", default="qwen")
    ap.add_argument("--max-pairs", type=int, default=40)
    ap.add_argument("--heldout", default="tpu/distill_ablation/heldout_seeds.json")
    args = ap.parse_args()
    ssl.create_default_context()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
