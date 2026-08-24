"""Generate a pool of FOREIGN 'better' programs (Nemotron + Qwen) for the
cross-model distillation experiment. Two-phase forcing + concurrent grading +
eval_timeout=1100 (all the fixes). Saves each VALID program as a State-dict
tagged with origin=<model>, so build_corpus can pair gpt-oss 'worse' programs
against foreign 'better' programs via select_cross_distill_pairs.
"""

from __future__ import annotations

import argparse
import asyncio
import ssl
import time
import uuid

import numpy as np

import common


def verify_c5(construction):
    h = np.asarray(construction, dtype=np.float64)
    n = len(h); t = n / 2.0
    h2 = h * (t / h.sum()) if h.sum() != t else h
    return float(np.max(np.correlate(h2, 1.0 - h2, mode="full") * (2.0 / n)))


async def gen_model(svc, model, tag, questions, gen_seeds, n_samples, max_tokens,
                    evaluator, keep_c5_below):
    from ttt_discover.tinker_utils.misc_utils import get_tokenizer
    from ttt_discover.tinker_utils.completers import QwenTwoPhaseTokenCompleter
    from concurrent.futures import ThreadPoolExecutor

    tok = get_tokenizer(model)
    sc = svc.create_sampling_client(base_model=model)
    completer = QwenTwoPhaseTokenCompleter(
        sampling_client=sc, tokenizer=tok, phase1_max_tokens=max_tokens,
        temperature=0.7, context_window=common.CONTEXT_WINDOW)

    # sample all (concurrent per seed)
    gens = []  # (final_text, gen_seed_state)
    for q, gseed in zip(questions, gen_seeds):
        mi_ids = tok.encode(tok.apply_chat_template(
            [{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False),
            add_special_tokens=False)
        import tinker
        mi = tinker.ModelInput.from_ints(mi_ids)
        outs = await asyncio.gather(*[completer(mi, ["<|im_end|>"]) for _ in range(n_samples)])
        for g in outs:
            text = tok.decode(list(g.tokens))
            final = text.split("</think>")[-1] if "</think>" in text else text
            gens.append((final, gseed))

    # grade all concurrently (fresh evaluator per grade)
    loop = asyncio.get_event_loop()
    from examples.erdos_min_overlap.env import ErdosMinOverlapRewardEvaluator

    def _grade(final, gseed):
        ev = ErdosMinOverlapRewardEvaluator(
            problem_type="", log_dir=evaluator.log_dir, num_cpus_per_task=1,
            eval_timeout=evaluator.eval_timeout, eval_backend="local")
        try:
            return ev.get_reward(final, gseed)
        except Exception as e:
            return {"msg": f"{type(e).__name__}", "correctness": 0}

    with ThreadPoolExecutor(max_workers=min(24, len(gens) or 1)) as ex:
        graded = await asyncio.gather(*[
            loop.run_in_executor(ex, _grade, final, gseed) for final, gseed in gens])

    betters = []
    for (final, gseed), out in zip(gens, graded):
        if out.get("correctness", 0) <= 0:
            continue
        con = out.get("result_construction")
        if not con:
            continue
        c5 = verify_c5(con)  # recompute, don't trust
        if c5 >= keep_c5_below:
            continue
        code = evaluator._extract_code(final) or ""
        betters.append({
            "type": "State", "id": str(uuid.uuid4()), "timestep": -1,
            "value": -c5, "construction": list(con), "code": code,
            "parent_values": [], "parents": [], "observation": "",
            "origin": tag,
        })
    return betters


async def run(args):
    common.load_dotenv_key()
    import tinker
    from examples.erdos_min_overlap.env import ErdosMinOverlapEnv, ErdosMinOverlapRewardEvaluator
    from ttt_discover.tinker_utils.state import state_from_dict

    svc = tinker.ServiceClient()
    scratch = f"{common.REPO_ROOT}/runs/distill_ablation/_genforeign"
    renderer, tok0, cfg = common.env_bits(scratch, eval_timeout=args.eval_timeout)
    sampler = common.load_pool_sampler(args.pool_snapshot, scratch + "/pool")
    evaluator = ErdosMinOverlapRewardEvaluator(
        problem_type="", log_dir=scratch, num_cpus_per_task=1,
        eval_timeout=args.eval_timeout, eval_backend="local")

    # generation seeds: spread over difficulty from the pool, EXCLUDING held-out
    ho = set(common.read_json(args.heldout)["ids"])
    with sampler._lock:
        pool = [s for s in sampler._states
                if s.code and s.code.strip() and s.value is not None and s.id not in ho]
    pool.sort(key=lambda s: s.value)  # ascending value = worst (coarse) first
    idx = np.linspace(0, len(pool) - 1, args.n_seeds).round().astype(int)
    gen_seeds = [pool[int(i)] for i in idx]
    questions = []
    for st in gen_seeds:
        env = ErdosMinOverlapEnv(renderer, initial_state=st, sampler=sampler, config=cfg)
        questions.append(env.get_question())
    print(f"[genforeign] {len(gen_seeds)} generation seeds, c5="
          f"{[round(-s.value,5) for s in gen_seeds]}")

    all_betters = []
    for model, tag in args.models:
        t0 = time.time()
        b = await gen_model(svc, model, tag, questions, gen_seeds, args.n_samples,
                            args.max_tokens, evaluator, args.keep_c5_below)
        all_betters += b
        c5s = sorted(-x["value"] for x in b)
        print(f"[genforeign] {tag}: {len(b)} kept betters ({time.time()-t0:.0f}s)  "
              f"c5={[round(c,5) for c in c5s]}")

    common.write_json(args.out, {"n": len(all_betters), "betters": all_betters})
    print(f"[genforeign] wrote {len(all_betters)} betters -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["nemo", "qwen"],
                    help="subset of {nemo,qwen}")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=26000)
    ap.add_argument("--eval-timeout", type=int, default=1100)
    ap.add_argument("--keep-c5-below", type=float, default=0.3835,
                    help="only keep valid programs at least this competitive")
    ap.add_argument("--pool-snapshot", required=True)
    ap.add_argument("--heldout", default="tpu/distill_ablation/heldout_seeds.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    name_map = {
        "nemo": ("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", "nemo"),
        "qwen": ("Qwen/Qwen3.6-35B-A3B", "qwen"),
    }
    args.models = [name_map[m] for m in args.models]
    ssl.create_default_context()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
