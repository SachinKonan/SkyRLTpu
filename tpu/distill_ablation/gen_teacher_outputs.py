"""Generate teacher critiques for cross-model (or self) pairs and STORE EVERY raw
output (no gating). This is the ONLY paid step. Filtering into a corpus, trying
different gate settings, and inspecting dropped critiques all happen offline
(free) via build_corpus_offline.py — never re-pay to regenerate.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import ssl
import time

import common


def build_fair_pairs(builder, worse_states, betters_states, num_betters, max_code_chars):
    """FAIR pairing: each FIXED worse (shared across arms) with the top-num_betters
    betters from THIS arm's source. Same worse everywhere -> only the better's
    technique differs. Returns (pairs, gaps_in_c5)."""
    from ttt_discover.rl import context_distill as cd
    env_type, renderer, sampler, config = cd._thunk_parts(builder)
    penv = env_type(renderer, initial_state=env_type.state_type(
        timestep=-1, construction=None, code="", value=None), sampler=sampler, config=config)
    problem_block = penv.get_question()
    maximize = penv.is_maximize()
    betters_sorted = sorted([b for b in betters_states if b.value is not None],
                            key=lambda b: b.value, reverse=True)
    pairs, gaps = [], []
    for worse in worse_states:
        elig = [b for b in betters_sorted if b.value > worse.value]
        if not elig:
            continue
        chosen = elig[:num_betters]
        betters_section = "\n".join(
            cd.BETTER_BLOCK.format(idx=i + 1, score=f"{cd._raw(b.value, maximize):.6f}",
                                   code=b.code[:max_code_chars])
            for i, b in enumerate(chosen))
        teacher_prompt = cd.TEACHER_TEMPLATE.format(
            n_betters=len(chosen), problem=problem_block,
            worse_score=f"{cd._raw(worse.value, maximize):.6f}",
            worse_code=worse.code[:max_code_chars], betters_section=betters_section)
        prompt_mi = cd._build_generator_prompt(builder, worse)
        pairs.append(cd.DistillPair(worse=worse, betters=chosen, maximize=maximize,
                                    prompt_model_input=prompt_mi, teacher_prompt=teacher_prompt))
        gaps.append((-worse.value) - (-chosen[0].value))  # worse_c5 - best_better_c5
    return pairs, gaps


async def run(args):
    common.load_dotenv_key()
    import tinker
    from examples.erdos_min_overlap.env import ErdosMinOverlapEnv
    from ttt_discover.tinker_utils.state import state_from_dict
    from ttt_discover.tinker_utils.completers import TwoPhaseTokenCompleter
    from ttt_discover.tinker_utils.misc_utils import get_tokenizer
    from ttt_discover.rl import context_distill as cd

    scratch = f"{common.REPO_ROOT}/runs/distill_ablation/_teacher/{args.name}"
    sampler = common.load_pool_sampler(args.snapshot, scratch)
    ho = set(common.read_json(args.heldout)["ids"])
    with sampler._lock:
        sampler._states = [s for s in sampler._states if s.id not in ho]

    if args.foreign_betters and not args.worse_set:  # legacy cross-model (random worse)
        fb = common.read_json(args.foreign_betters)["betters"]
        if args.better_origin:
            fb = [b for b in fb if b["origin"] == args.better_origin]
        with sampler._lock:
            for s in sampler._states:
                s.origin = "gptoss"
            for b in fb:
                sampler._states.append(state_from_dict(b, state_type=ErdosMinOverlapEnv.state_type))

    renderer, tok0, cfg = common.env_bits(scratch)
    builder = common.make_seed_builder(sampler, renderer, cfg, sampler._states[0], 1)
    rng = random.Random(args.seed)

    if args.worse_set:  # FAIR: fixed shared worse set x this arm's source betters
        import numpy as np
        ws = common.read_json(args.worse_set)
        worse_states = [state_from_dict(w, state_type=ErdosMinOverlapEnv.state_type)
                        for w in ws["worse"]]
        worse_ids = set(ws["ids"])
        if args.source == "own":  # betters = gpt-oss champions (not in the worse set)
            with sampler._lock:
                betters_states = sorted(
                    [s for s in sampler._states if s.id not in worse_ids
                     and s.value is not None and s.code and s.code.strip()],
                    key=lambda s: s.value, reverse=True)[: args.n_better_pool]
        else:  # betters = foreign programs of this source (qwen/nemo)
            fb = common.read_json(args.foreign_betters)["betters"]
            fb = [b for b in fb if b["origin"] == args.source]
            betters_states = [state_from_dict(b, state_type=ErdosMinOverlapEnv.state_type)
                              for b in fb]
        pairs, gaps = build_fair_pairs(builder, worse_states, betters_states,
                                       args.num_betters, 10000)
        g = np.array(gaps)
        print(f"[teacher:{args.name}] FAIR source={args.source}: {len(pairs)} pairs, "
              f"gap(c5) mean={g.mean():.2e} [{g.min():.2e}, {g.max():.2e}]")
    elif args.foreign_betters:
        pairs, _ = cd.select_cross_distill_pairs(sampler, [builder], "gptoss",
                                                 args.max_pairs, args.num_betters, 10000, rng)
        print(f"[teacher:{args.name}] {len(pairs)} pairs")
    else:
        pairs, _ = cd.select_distill_pairs(sampler, [builder], args.max_pairs,
                                           args.num_betters, 10000, rng)
        print(f"[teacher:{args.name}] {len(pairs)} pairs")

    svc = tinker.ServiceClient()
    teacher = svc.create_sampling_client(base_model=args.teacher_model)
    tokenizer = get_tokenizer(common.STUDENT_MODEL)
    completer = TwoPhaseTokenCompleter(sampling_client=teacher, tokenizer=tokenizer,
                                       phase1_max_tokens=args.teacher_phase1_tokens,
                                       temperature=1.0, context_window=common.CONTEXT_WINDOW)
    stop = renderer.get_stop_sequences()

    async def teach(pair):
        mi = renderer.build_generation_prompt([{"role": "user", "content": pair.teacher_prompt}])
        if mi.length >= args.teacher_phase1_tokens - 2048:
            return "too_long"
        return await completer(mi, stop)

    t0 = time.time()
    outs = []
    for i in range(0, len(pairs), args.chunk):
        chunk = pairs[i:i + args.chunk]
        res = await asyncio.gather(*[teach(p) for p in chunk], return_exceptions=True)
        for pair, r in zip(chunk, res):
            rec = {
                "worse_id": pair.worse.id, "worse_value": pair.worse.value,
                "worse_code": pair.worse.code,
                "prompt_ids": [int(x) for x in pair.prompt_model_input.to_ints()],
                "betters": [{"origin": getattr(b, "origin", None), "value": b.value,
                             "code": b.code} for b in pair.betters],
            }
            if isinstance(r, Exception) or r == "too_long" or r is None:
                rec["status"] = "gen_failed"
                rec["teacher_tokens"] = []
            else:
                toks = list(r.tokens)
                span = cd._final_span_start(toks, tokenizer)
                rec["status"] = "ok" if span is not None else "no_final"
                rec["teacher_tokens"] = toks if span is None else toks[span:]
            outs.append(rec)
        print(f"[teacher:{args.name}] chunk {i//args.chunk}: {len(outs)} stored, {time.time()-t0:.0f}s")

    common.write_json(args.out, {
        "name": args.name, "teacher_model": args.teacher_model, "snapshot": args.snapshot,
        "better_origin": args.better_origin, "n": len(outs), "outputs": outs,
    })
    ok = sum(1 for o in outs if o["status"] == "ok")
    print(f"[teacher:{args.name}] DONE {ok}/{len(outs)} usable -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--teacher-model", default=common.STUDENT_MODEL)
    ap.add_argument("--foreign-betters")
    ap.add_argument("--better-origin")
    ap.add_argument("--worse-set", help="fixed shared worse set (fair pairing)")
    ap.add_argument("--source", choices=["own", "qwen", "nemo"],
                    help="better source for --worse-set: own gpt-oss / qwen / nemo")
    ap.add_argument("--n-better-pool", type=int, default=20,
                    help="own source: how many top gpt-oss champions to draw betters from")
    ap.add_argument("--max-pairs", type=int, default=80)
    ap.add_argument("--num-betters", type=int, default=3)
    ap.add_argument("--teacher-phase1-tokens", type=int, default=26000)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heldout", default="tpu/distill_ablation/heldout_seeds.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ssl.create_default_context()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
