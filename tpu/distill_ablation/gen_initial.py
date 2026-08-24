"""Understand INITIAL behavior: generate gpt-oss-120b responses under the ttt-discover
BASE prompt (unchanged) for ONE problem, grade locally, cache every raw generation.

No training, no prompt changes, no PUCT loop — just sample the model's "improve this
solution" responses from the problem's base/SOTA state and look at them.

Base state per problem:
  - erdos_min_overlap: best state from a PUCT pool snapshot (--pool-snapshot) — improve the record.
  - ac_inequalities (ac1/ac2): env.create_initial_state ships the SOTA seed as state.code.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import re
import ssl
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import common

ENVS = {
    "erdos_min_overlap": ("examples.erdos_min_overlap.env", "ErdosMinOverlapEnv"),
    "ac_inequalities": ("examples.ac_inequalities.env", "AutoCorrInequalityEnv"),
    "frontier_algo": ("examples.frontier_algo.env", "FrontierAlgoEnv"),
}


def env_bits_generic(env_type, problem_type, log_path, eval_timeout, num_cpus=1):
    from ttt_discover.tinker_utils import renderers
    from ttt_discover.tinker_utils.dataset_builder import DatasetConfig
    from ttt_discover.tinker_utils.misc_utils import get_tokenizer

    tok = get_tokenizer(common.TOKENIZER_MODEL)
    renderer = renderers.get_renderer(common.RENDERER_NAME, tokenizer=tok)
    Path(log_path).mkdir(parents=True, exist_ok=True)
    cfg = DatasetConfig(
        problem_type=problem_type, env_type=env_type, batch_size=1,
        model_name_for_tokenizer=common.TOKENIZER_MODEL, renderer_name=common.RENDERER_NAME,
        group_size=1, num_cpus_per_task=num_cpus, eval_backend="local",
        eval_timeout=eval_timeout, log_path=log_path, convo_prefix=None,
    )
    return renderer, tok, cfg


def last_code(text: str, lang: str = "python"):
    tags = {"python": r"python", "cpp": r"cpp|c\+\+"}.get(lang, re.escape(lang))
    ms = re.findall(rf"```(?:{tags})\s+([\s\S]*?)\s*```", text)
    return ms[-1] if ms else None


class _DummySampler:  # env.__init__ requires non-None; get_question never touches it
    pass


async def run(args):
    if args.discover_root:  # e.g. frontiercs worktree that ships examples.frontier_algo
        import sys as _sys
        _sys.path.insert(0, args.discover_root)
    common.load_dotenv_key()
    import tinker
    from ttt_discover.tinker_utils.completers import TwoPhaseTokenCompleter

    mod, cls = ENVS[args.env]
    EnvClass = getattr(importlib.import_module(mod), cls)
    scratch = f"{common.REPO_ROOT}/runs/behavior_probe/{args.tag}"
    renderer, tok, cfg = env_bits_generic(EnvClass, args.problem_type, scratch, args.eval_timeout)

    if args.pool_snapshot:
        sampler = common.load_pool_sampler(args.pool_snapshot, scratch + "/pool")
        with sampler._lock:
            cand = [s for s in sampler._states if s.value is not None and s.code and s.code.strip()]
        base = max(cand, key=lambda s: s.value)  # best = the record we improve from
    else:
        base = EnvClass.create_initial_state(args.problem_type)
        sampler = _DummySampler()

    env = EnvClass(renderer, initial_state=base, sampler=sampler, config=cfg)
    prompt = env.get_question()
    is_max = env.is_maximize()
    mi = renderer.build_generation_prompt([{"role": "user", "content": prompt}])
    stop = renderer.get_stop_sequences()

    svc = tinker.ServiceClient()
    sc = svc.create_sampling_client(base_model=args.model)
    completer = TwoPhaseTokenCompleter(sampling_client=sc, tokenizer=tok,
                                       phase1_max_tokens=args.phase1_max_tokens,
                                       temperature=1.0, context_window=common.CONTEXT_WINDOW)

    gens = await asyncio.gather(*[completer(mi, stop) for _ in range(args.n_gens)],
                                return_exceptions=True)
    texts, errs = [], 0
    for g in gens:
        if isinstance(g, Exception):
            errs += 1
        else:
            texts.append(tok.decode(list(g.tokens)))
    gd = Path(scratch) / "gens"; gd.mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(texts):
        (gd / f"gen{i}.txt").write_text(t)

    loop = asyncio.get_event_loop()

    def grade(text):
        ev = EnvClass.reward_function(problem_type=args.problem_type, log_dir=scratch,
                                      num_cpus_per_task=1, eval_timeout=args.eval_timeout,
                                      eval_backend="local")
        code = last_code(text, args.code_lang)
        if not code:
            return ("no_code", None)
        # frontier_algo compiles the string directly (pure C++); math envs extract a fenced block
        payload = code if args.code_lang != "python" else f"```python\n{code}\n```"
        try:
            out = ev.get_reward(payload, base)
            if out.get("correctness", 0) > 0:
                return ("valid", out.get("raw_score"))
            return ("invalid", None)
        except Exception as e:
            return (f"err:{type(e).__name__}", None)

    with ThreadPoolExecutor(max_workers=min(max(1, len(texts)), args.grade_concurrency)) as ex:
        results = await asyncio.gather(*[loop.run_in_executor(ex, grade, t) for t in texts])

    records = [{"idx": i, "status": s, "score": sc, "has_code": last_code(t, args.code_lang) is not None,
                "code": last_code(t, args.code_lang), "gen_len": len(t)}
               for i, (t, (s, sc)) in enumerate(zip(texts, results))]
    common.write_json(args.out, {"env": args.env, "problem_type": args.problem_type,
                                 "model": args.model, "is_maximize": is_max,
                                 "base_value": base.value, "n_gens": args.n_gens,
                                 "gen_errors": errs, "records": records})

    scores = [r["score"] for r in records if r["score"] is not None]
    valid = len(scores)
    n = len(records)
    print(f"\n===== INITIAL BEHAVIOR  {args.env}/{args.problem_type or '-'}  model={args.model} =====")
    print(f"  gens={n} (sample_errs={errs}) | VALID {valid}/{n} ({100*valid/max(n,1):.0f}%) | "
          f"maximize={is_max}")
    if scores:
        ss = sorted(scores, reverse=is_max)
        print(f"  raw_score: best={ss[0]:.6f} median={ss[len(ss)//2]:.6f} worst={ss[-1]:.6f}")
    print(f"  base(record) value={base.value:.6f}  base code_len={len(base.code or '')}")
    print(f"  cached {len(texts)} raw gens -> {gd}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=list(ENVS), required=True)
    ap.add_argument("--problem-type", default="")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--n-gens", type=int, default=24)
    ap.add_argument("--phase1-max-tokens", type=int, default=26000)
    ap.add_argument("--eval-timeout", type=int, default=1100)
    ap.add_argument("--grade-concurrency", type=int, default=24)
    ap.add_argument("--pool-snapshot", default="")
    ap.add_argument("--discover-root", default="", help="override discover path (e.g. frontiercs worktree)")
    ap.add_argument("--code-lang", default="python", choices=["python", "cpp"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ssl.create_default_context()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
