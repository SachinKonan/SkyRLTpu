"""One arm of the distillation ablation: CE-finetune BASE gpt-oss-20b on a
distill corpus subset (RL frozen), then run the held-out improver eval.

Paid Tinker work: the short CE finetune + the held-out rollout sampling.
Grading is local (in-process across the node's cores) — run under sbatch, never
on the login node.

Usage:
  # A0 control (no finetune):
  python finetune_and_eval.py --arm A0 --eval-only --out-dir runs/distill_ablation/A0
  # dose arm:
  python finetune_and_eval.py --arm A2 --corpus corpora/self_superset.json \
      --subset-size 32 --epochs 8 --out-dir runs/distill_ablation/A2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import ssl
import time
from pathlib import Path

import common


def load_subset(corpus_path, subset_size, seed):
    """Fixed-order subset so A1 ⊂ A2 ⊂ A3 (nested dose sweep)."""
    import random
    corpus = common.read_json(corpus_path)
    recs = list(corpus["records"])
    random.Random(seed).shuffle(recs)
    if subset_size and subset_size < len(recs):
        recs = recs[:subset_size]
    return recs, corpus


def build_datums(recs):
    """Re-normalize the 0/1 masks so CE = 1.0 * meanNLL over THIS subset."""
    if not recs:
        raise SystemExit("empty corpus subset — nothing to fine-tune on")
    total = sum(sum(r["mask01"]) for r in recs)
    scale = 1.0 / total if total > 0 else 0.0
    datums = [common.record_to_datum(r, scale) for r in recs]
    masks = [r["mask01"] for r in recs]
    return datums, masks, total


async def ce_finetune(tc, datums, masks, epochs, lr):
    """Full-batch CE gradient descent for `epochs` optim steps; returns NLL trace."""
    from ttt_discover.rl.train import (
        enqueue_forward_backward, enqueue_optim_step,
        consume_forward_backward, consume_optim_step,
    )
    import torch
    trace = []
    for ep in range(epochs):
        fb = await enqueue_forward_backward(tc, datums, "cross_entropy")
        opt = await enqueue_optim_step(tc, lr)
        logps = await consume_forward_backward(fb)
        await consume_optim_step(opt)
        nll_sum = tok_sum = 0.0
        for lp, mask in zip(logps, masks):
            m = torch.tensor(mask, dtype=torch.float32)
            n = min(len(lp), len(m))
            nll_sum += float((-lp[:n] * m[:n]).sum().item())
            tok_sum += float(m[:n].sum().item())
        trace.append(nll_sum / tok_sum if tok_sum else float("nan"))
        print(f"  [finetune] epoch {ep}: meanNLL={trace[-1]:.4f}")
    return trace


async def eval_heldout(sc, seeds, sampler, renderer, cfg, k, phase1_max_tokens, concurrency):
    """Sample K improver rollouts per held-out seed; grade locally; collect records."""
    from ttt_discover.rl.train import do_group_rollout_and_filter_constant_reward
    sem = asyncio.Semaphore(concurrency)

    async def one(entry):
        from ttt_discover.tinker_utils.state import state_from_dict
        from examples.erdos_min_overlap.env import ErdosMinOverlapEnv
        seed = state_from_dict(entry["state"], state_type=ErdosMinOverlapEnv.state_type)
        builder = common.make_seed_builder(sampler, renderer, cfg, seed, num_envs=k)
        async with sem:
            tg = await do_group_rollout_and_filter_constant_reward(
                sc, builder, temperature=1.0, do_remove_constant_reward_groups=False,
                step_idx=0, model_name=common.STUDENT_MODEL,
                phase1_max_tokens=phase1_max_tokens, context_window=common.CONTEXT_WINDOW,
            )
        out = []
        if tg is None:
            return out
        for traj in tg.trajectories_G:
            m = traj.transitions[0].metrics if traj.transitions else {}
            out.append({
                "seed_id": entry["state"]["id"], "stratum": entry["stratum"],
                "seed_c5": entry["c5"], "seed_n": entry["n"],
                "correctness": m.get("correctness", 0.0),
                "rollout_c5": m.get("raw_score"),
                "parsed_code": (m.get("parsed_code") or "")[:20000],
            })
        return out

    results = await asyncio.gather(*[one(e) for e in seeds])
    return [r for group in results for r in group]


async def run(args):
    common.load_dotenv_key()
    import tinker
    svc = tinker.ServiceClient()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = str(out_dir / "_scratch")
    renderer, tokenizer, cfg = common.env_bits(scratch, eval_timeout=args.eval_timeout)
    # throwaway sampler for the eval envs (update_states(save=False) is harmless)
    eval_sampler = common.load_pool_sampler(args.eval_pool_snapshot, scratch + "/evalpool")

    nll_trace, total_tokens, n_datums = [], 0, 0
    if args.eval_only:
        print(f"[{args.arm}] eval-only: base {common.STUDENT_MODEL}, no finetune")
        sc = svc.create_sampling_client(base_model=common.STUDENT_MODEL)
    else:
        recs, corpus = load_subset(args.corpus, args.subset_size, args.subset_seed)
        datums, masks, total_tokens = build_datums(recs)
        n_datums = len(datums)
        print(f"[{args.arm}] finetune on {n_datums} datums ({total_tokens:.0f} tokens), "
              f"epochs={args.epochs} lr={args.lr}")
        tc = await svc.create_lora_training_client_async(common.STUDENT_MODEL, rank=args.lora_rank)
        nll_trace = await ce_finetune(tc, datums, masks, args.epochs, args.lr)
        sc = await tc.save_weights_and_get_sampling_client_async()

    # probe NLL (shared frozen critique set) on this (fine-tuned) policy
    probe_nll = None
    if args.probe:
        from ttt_discover.rl.context_distill import probe_nll_async
        import torch
        probe_recs = common.read_json(args.probe)["probe"]
        probe = []
        for r in probe_recs:
            mi = tinker.ModelInput.from_ints(r["model_input_ids"])
            targets = torch.tensor(r["target_tokens"], dtype=torch.int64)
            mask = torch.tensor(r["mask01"], dtype=torch.float32)
            probe.append((mi, targets, mask))
        probe_nll = await probe_nll_async(probe, sc)
        print(f"[{args.arm}] probe_nll={probe_nll:.4f}")

    # held-out improver eval
    seeds = common.read_json(args.heldout)["seeds"]
    if args.max_seeds:
        seeds = seeds[: args.max_seeds]
    t0 = time.time()
    rollouts = await eval_heldout(
        sc, seeds, eval_sampler, renderer, cfg,
        args.k, args.phase1_max_tokens, args.eval_concurrency,
    )
    print(f"[{args.arm}] eval: {len(rollouts)} rollouts over {len(seeds)} seeds "
          f"in {time.time()-t0:.0f}s")

    with open(out_dir / "rollouts.jsonl", "w") as f:
        for r in rollouts:
            f.write(json.dumps(r) + "\n")
    common.write_json(str(out_dir / "summary.json"), {
        "arm": args.arm, "eval_only": args.eval_only,
        "corpus": args.corpus, "subset_size": args.subset_size,
        "n_datums": n_datums, "total_tokens": total_tokens,
        "epochs": args.epochs, "lr": args.lr,
        "nll_trace": nll_trace, "probe_nll": probe_nll,
        "k": args.k, "phase1_max_tokens": args.phase1_max_tokens,
        "n_rollouts": len(rollouts),
    })
    print(f"[{args.arm}] DONE -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--eval-only", action="store_true", help="A0: base, no finetune")
    ap.add_argument("--corpus", help="corpus json from build_corpus.py")
    ap.add_argument("--subset-size", type=int, default=0, help="0 = full corpus")
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--k", type=int, default=8, help="rollouts per held-out seed")
    ap.add_argument("--max-seeds", type=int, default=0, help="limit eval to first N seeds (smoke)")
    ap.add_argument("--phase1-max-tokens", type=int, default=26000)
    ap.add_argument("--eval-timeout", type=int, default=1100)
    ap.add_argument("--eval-concurrency", type=int, default=4)
    ap.add_argument("--heldout", default="tpu/distill_ablation/heldout_seeds.json")
    ap.add_argument("--probe", help="corpus json to read the shared frozen probe from")
    ap.add_argument("--eval-pool-snapshot", required=True,
                    help="any puct_sampler_step_*.json (throwaway sampler for eval envs)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    ssl.create_default_context()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
