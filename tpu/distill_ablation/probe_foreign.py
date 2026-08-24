"""Probe: can foreign models (Qwen3.6, Nemotron-120B) write VALID, competitive
Erdos programs? go/no-go gate before the cross-model distillation experiment.

Renders the Erdos prompt with each model's OWN chat template (verified:
apply_chat_template(tokenize=False) -> encode), samples a few completions,
extracts the ```python block, and grades locally. Reports validity + c5.
Run on a compute node (grading executes the programs).
"""

from __future__ import annotations

import argparse
import asyncio
import ssl
import time

import common


def render_prompt(tokenizer, question: str):
    import tinker
    s = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        add_generation_prompt=True, tokenize=False,
    )
    ids = tokenizer.encode(s, add_special_tokens=False)
    return tinker.ModelInput.from_ints(ids)


async def probe_model(svc, model, questions, seeds, n_samples, max_tokens, evaluator):
    import asyncio
    from ttt_discover.tinker_utils.misc_utils import get_tokenizer
    from ttt_discover.tinker_utils.completers import QwenTwoPhaseTokenCompleter
    tok = get_tokenizer(model)
    sc = svc.create_sampling_client(base_model=model)
    # Two-phase forcing: think up to max_tokens, then FORCE </think> and sample the
    # answer (the mechanism that makes reasoning models emit a program). Works for
    # any <think> ChatML model (Qwen3.x, Nemotron-3).
    completer = QwenTwoPhaseTokenCompleter(
        sampling_client=sc, tokenizer=tok, phase1_max_tokens=max_tokens,
        temperature=0.7, context_window=common.CONTEXT_WINDOW,
    )

    # 1) sample all generations (concurrent per seed)
    gens = []  # (final_text, seed, n_tokens, closed_think)
    for q, seed in zip(questions, seeds):
        mi = render_prompt(tok, q)
        outs = await asyncio.gather(*[completer(mi, ["<|im_end|>"]) for _ in range(n_samples)])
        for g in outs:
            toks = list(g.tokens); text = tok.decode(toks)
            final = text.split("</think>")[-1] if "</think>" in text else text
            gens.append((final, seed, len(toks), "</think>" in text))

    # 2) grade ALL concurrently (programs run up to eval_timeout; timeouts must not
    #    serialize). Fresh evaluator per grade to avoid the shared _last_stdout race.
    from concurrent.futures import ThreadPoolExecutor
    from examples.erdos_min_overlap.env import ErdosMinOverlapRewardEvaluator
    loop = asyncio.get_event_loop()

    def _grade(final, seed):
        ev = ErdosMinOverlapRewardEvaluator(
            problem_type="", log_dir=evaluator.log_dir, num_cpus_per_task=1,
            eval_timeout=evaluator.eval_timeout, eval_backend="local")
        try:
            return ev.get_reward(final, seed)
        except Exception as e:
            return {"msg": f"grade err {type(e).__name__}: {e}", "correctness": 0}

    with ThreadPoolExecutor(max_workers=min(24, len(gens) or 1)) as ex:
        graded = await asyncio.gather(*[
            loop.run_in_executor(ex, _grade, final, seed) for final, seed, _, _ in gens])

    results = []
    for (final, seed, ntok, closed), out in zip(gens, graded):
        c5 = out.get("raw_score") if out.get("correctness", 0) > 0 else None
        seed_c5 = -seed.value
        results.append({
            "model": model, "seed_c5": seed_c5, "n_gen_tokens": ntok,
            "closed_think": closed, "has_fence": "```python" in final,
            "correct": out.get("correctness", 0), "c5": c5,
            "improved": (c5 is not None) and (seed_c5 - c5) > 1e-4,
            "msg": (out.get("msg") or "")[:120],
        })
    return results


async def run(args):
    common.load_dotenv_key()
    import tinker
    from examples.erdos_min_overlap.env import ErdosMinOverlapEnv, ErdosMinOverlapRewardEvaluator
    from ttt_discover.tinker_utils.state import state_from_dict

    svc = tinker.ServiceClient()
    scratch = f"{common.REPO_ROOT}/runs/distill_ablation/_probe"
    renderer, tok, cfg = common.env_bits(scratch, eval_timeout=args.eval_timeout)
    sampler = common.load_pool_sampler(args.eval_pool_snapshot, scratch + "/pool")
    evaluator = ErdosMinOverlapRewardEvaluator(
        problem_type="", log_dir=scratch, num_cpus_per_task=1,
        eval_timeout=args.eval_timeout, eval_backend="local",
    )

    # a few held-out seeds spanning difficulty -> their get_question() text
    ho = common.read_json(args.heldout)["seeds"]
    picks = [ho[0], ho[6], ho[12]][: args.n_seeds]  # hard, mid, near
    seeds, questions = [], []
    for entry in picks:
        st = state_from_dict(entry["state"], state_type=ErdosMinOverlapEnv.state_type)
        env = ErdosMinOverlapEnv(renderer, initial_state=st, sampler=sampler, config=cfg)
        seeds.append(st); questions.append(env.get_question())

    all_res = []
    for model in args.models:
        t0 = time.time()
        res = await probe_model(svc, model, questions, seeds, args.n_samples,
                                 args.max_tokens, evaluator)
        all_res += res
        print(f"\n===== {model}  ({time.time()-t0:.0f}s) =====")
        valid = [r for r in res if r["c5"] is not None]
        imp = [r for r in res if r["improved"]]
        print(f"  VALID {len(valid)}/{len(res)}  |  IMPROVED-seed {len(imp)}/{len(res)}  "
              f"|  closed</think> {sum(r['closed_think'] for r in res)}/{len(res)}")
        if valid:
            c5s = sorted(r["c5"] for r in valid)
            print(f"  c5: best={c5s[0]:.6f}  median={c5s[len(c5s)//2]:.6f}  all={[round(c,5) for c in c5s]}")
        for r in res:
            flag = "IMPROVE" if r["improved"] else ("valid" if r["c5"] is not None else "  -  ")
            print(f"    [{flag:>7}] seed_c5={r['seed_c5']:.5f} gen_tok={r['n_gen_tokens']:>6} "
                  f"c5={r['c5']}  {r['msg'][:50]}")

    common.write_json(f"{scratch}/probe_results.json", all_res)
    print(f"\n[probe] wrote {scratch}/probe_results.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[
        "Qwen/Qwen3.6-35B-A3B", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"])
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=14000)
    ap.add_argument("--eval-timeout", type=int, default=900)
    ap.add_argument("--heldout", default="tpu/distill_ablation/heldout_seeds.json")
    ap.add_argument("--eval-pool-snapshot", required=True)
    args = ap.parse_args()
    ssl.create_default_context()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
