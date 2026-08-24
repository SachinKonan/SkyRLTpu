"""Diagnostic: dump the FULL raw generation from Qwen/Nemotron on one Erdos seed,
so we can see the think-trace + program + exact failure. Proper thinking-mode
sampling (temp 0.6, top_p 0.95, top_k 20). Saves each raw generation to a .txt.
"""

from __future__ import annotations

import argparse
import asyncio
import ssl
from pathlib import Path

import common


async def run(args):
    common.load_dotenv_key()
    import tinker
    from examples.erdos_min_overlap.env import ErdosMinOverlapEnv, ErdosMinOverlapRewardEvaluator
    from ttt_discover.tinker_utils.misc_utils import get_tokenizer
    from ttt_discover.tinker_utils.state import state_from_dict

    svc = tinker.ServiceClient()
    outdir = Path(f"{common.REPO_ROOT}/runs/distill_ablation/_diag")
    outdir.mkdir(parents=True, exist_ok=True)
    renderer, tok0, cfg = common.env_bits(str(outdir), eval_timeout=args.eval_timeout)
    sampler = common.load_pool_sampler(args.eval_pool_snapshot, str(outdir / "pool"))
    evaluator = ErdosMinOverlapRewardEvaluator(
        problem_type="", log_dir=str(outdir), num_cpus_per_task=1,
        eval_timeout=args.eval_timeout, eval_backend="local")

    ho = common.read_json(args.heldout)["seeds"]
    entry = ho[8]  # a mid seed with headroom
    seed = state_from_dict(entry["state"], state_type=ErdosMinOverlapEnv.state_type)
    env = ErdosMinOverlapEnv(renderer, initial_state=seed, sampler=sampler, config=cfg)
    question = env.get_question()

    from ttt_discover.tinker_utils.completers import QwenTwoPhaseTokenCompleter
    for model in args.models:
        tok = get_tokenizer(model)
        sc = svc.create_sampling_client(base_model=model)
        s = tok.apply_chat_template([{"role": "user", "content": question}],
                                    add_generation_prompt=True, tokenize=False)
        mi = tinker.ModelInput.from_ints(tok.encode(s, add_special_tokens=False))
        # Two-phase forcing: think up to phase1_max_tokens, then FORCE </think> and
        # sample the answer from remaining context (the same mechanism that makes
        # gpt-oss produce valid programs). Works for any <think> ChatML model.
        completer = QwenTwoPhaseTokenCompleter(
            sampling_client=sc, tokenizer=tok,
            phase1_max_tokens=args.max_tokens, temperature=0.7,
            context_window=common.CONTEXT_WINDOW,
        )
        result = await completer(mi, ["<|im_end|>"])
        toks = list(result.tokens)
        text = tok.decode(toks)
        ntok = len(toks)
        closed = "</think>" in text
        final = text.split("</think>")[-1] if closed else text
        code = evaluator._extract_code(final)
        try:
            out = evaluator.get_reward(final, seed)
        except Exception as e:
            out = {"msg": f"{type(e).__name__}: {e}", "correctness": 0}

        tag = model.split("/")[-1]
        (outdir / f"{tag}.raw.txt").write_text(text)
        (outdir / f"{tag}.code.txt").write_text(code or "(none extracted)")
        print(f"\n{'='*70}\n{model}")
        print(f"  tokens={ntok} think_closed={closed} final_len={len(final)} "
              f"code_extracted={code is not None} code_len={len(code or '')}")
        print(f"  correctness={out.get('correctness')} raw_score={out.get('raw_score')}")
        print(f"  MSG: {(out.get('msg') or '')[:300]}")
        print(f"  stdout: {(out.get('stdout') or '')[:300]}")
        print(f"  -> full raw: {outdir}/{tag}.raw.txt ; code: {outdir}/{tag}.code.txt")
        if code:
            print("  --- EXTRACTED CODE (first 1500 chars) ---")
            print("\n".join("  | " + ln for ln in code[:1500].splitlines()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[
        "Qwen/Qwen3.6-35B-A3B", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"])
    ap.add_argument("--max-tokens", type=int, default=26000)
    ap.add_argument("--eval-timeout", type=int, default=900)
    ap.add_argument("--heldout", default="tpu/distill_ablation/heldout_seeds.json")
    ap.add_argument("--eval-pool-snapshot", required=True)
    args = ap.parse_args()
    ssl.create_default_context()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
