#!/usr/bin/env python
"""Grade the cached Muse-Glimmer rollouts on CPU, with the sweep's graders.

Reproduces ``Environment.step``'s grading path exactly -- same code extraction
(``last_codeblock_postprocess``), same evaluator class, same
``eval_timeout`` (1100, what the Qwen3.5 sweep used) and the same parent
``State`` object, so the numbers are comparable to the pool the parents came
from. Only the execution backend differs: ``local`` instead of ``ray``,
because a slurm array task is already the scheduler here.

Input is the raw-generation JSONL the TPU host produced; nothing is
regenerated and the raw text is never rewritten. Output is one JSONL of
grades, shardable via ``--shard/--num-shards`` for a slurm array.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

REPO = "/n/fs/vision-mix/sk7524/SkyRLTpu"
DISCOVER = os.path.join(REPO, "third_party", "discover")
for _p in (DISCOVER, os.path.join(DISCOVER, "examples")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--eval-timeout", type=int, default=1100)
    ap.add_argument("--num-cpus-per-task", type=int, default=1)
    ap.add_argument("--log-dir", default="/n/fs/vision-mix/sk7524/muse-rs/gradelog")
    ap.add_argument("--deadline-epoch", type=int, default=0)
    args = ap.parse_args()

    os.environ.setdefault("TTD_EVAL_BACKEND", "local")

    import importlib
    from ttt_discover.tinker_utils.dataset_builder import last_codeblock_postprocess
    from ttt_discover.tinker_utils.state import State
    from ttt_discover.tinker_utils import rosetta_stone

    man = json.load(open(args.manifest))
    env_mod = importlib.import_module(man["env_module"])
    env_cls = getattr(env_mod, man["env_class"])
    evaluator_cls = env_cls.reward_function
    maximize = man["maximize"]
    # Code extraction MUST use the environment's own languages. Hardcoding
    # ["python"] scored all 320 frontier_algo (JSSP) rollouts 0 with "No C++
    # program with main() found in response" -- it is a C++17 problem
    # (`_get_code_languages() -> ["cpp","c++"]`, keep_separators False).
    langs = man.get("code_languages")
    keep = man.get("keep_code_separators")
    if langs is None:
        langs = env_cls._get_code_languages(None)
    if keep is None:
        keep = env_cls._should_keep_code_separators(None)
    print(f"code_languages={langs} keep_separators={keep}", flush=True)
    os.makedirs(args.log_dir, exist_ok=True)

    recs = []
    with open(args.gen) as f:
        for line in f:
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    recs = [r for r in recs if "error" not in r]
    recs.sort(key=lambda r: r["item_id"])
    # Shard by a hash of the item id, NOT by position: generation streams in
    # while grading runs, so a positional split would reassign items between
    # passes and grade some of them twice (each shard only knows its own
    # output file).
    from zlib import crc32

    mine = [r for r in recs
            if crc32(r["item_id"].encode()) % args.num_shards == args.shard]

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["item_id"])
                except Exception:
                    pass
    mine = [r for r in mine if r["item_id"] not in done]
    print(f"shard {args.shard}/{args.num_shards}: {len(mine)} to grade", flush=True)

    fh = open(args.out, "a", buffering=1)
    for i, r in enumerate(mine):
        if args.deadline_epoch and time.time() > args.deadline_epoch:
            print("deadline reached, stopping", flush=True)
            break
        t0 = time.time()
        # Same splitter the renderer uses, so grading and generation can never
        # disagree about where reasoning ends and the answer begins.
        parsed = rosetta_stone.parse(r.get("text_full", ""), "muse_glimmer")
        code = last_codeblock_postprocess(
            parsed.content, codeblock_seps=langs, keep_separators=keep
        )
        out = {
            "item_id": r["item_id"],
            "arm": r["arm"],
            "kind": r["kind"],
            "state_id": r["state_id"],
            "run": r["run"],
            "parent_value": r["parent_value"],
            "has_code": bool(code and code.strip()),
            "answer_chars": len(parsed.content or ""),
        }
        if not (code and code.strip()):
            out.update(format_ok=False, correctness=0.0, raw_score=None,
                       value=None, msg="no code block", grade_s=0.0)
            fh.write(json.dumps(out) + "\n")
            continue
        state = State.from_dict(man["states"][r["state_id"]])
        try:
            ev = evaluator_cls(
                problem_type=man["problem_type"],
                log_dir=args.log_dir,
                eval_timeout=args.eval_timeout,
                num_cpus_per_task=args.num_cpus_per_task,
                eval_backend="local",
            )
            res = ev.get_reward(code, state=state)
            corr = float(res.get("correctness", 0.0))
            raw = res.get("raw_score")
            out.update(
                format_ok=True,
                correctness=corr,
                reward=res.get("reward"),
                raw_score=raw,
                # Pool convention: State.value is signed so higher is better.
                value=(raw if maximize else -raw) if (corr > 0 and raw is not None) else None,
                msg=str(res.get("msg", ""))[:400],
            )
        except Exception as e:
            out.update(format_ok=True, correctness=0.0, raw_score=None, value=None,
                       msg=f"grader exception: {e!r}"[:400],
                       tb=traceback.format_exc()[-800:])
        out["grade_s"] = round(time.time() - t0, 1)
        # Grading timeouts are a CONFOUND, not just a cost: if xhigh writes
        # slower programs its rollouts time out more, score 0, and the arm
        # looks worse for a reason that has nothing to do with solution
        # quality. Flag them so the asymmetry can be checked directly.
        _m = str(out.get("msg", "")).lower()
        out["timed_out"] = bool(
            "timeout" in _m or "timed out" in _m
            or out["grade_s"] >= 0.95 * args.eval_timeout
        )
        fh.write(json.dumps(out) + "\n")
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(mine)} last={out['grade_s']}s", flush=True)
    fh.close()
    print("SHARD DONE", flush=True)


if __name__ == "__main__":
    main()
