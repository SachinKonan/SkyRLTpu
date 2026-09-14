"""Exercise grading from the frozen training client's environment before sampling."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Match the frozen application imports, including its ray_train package.
    sys.path[:0] = [str(args.source / "third_party/discover"),
                   str(args.source / "tpu"), str(args.source)]
    # Give a direct import error rather than Ray's placeholder-signature error.
    from tpu.swarm.ray_train.grader_tasks import grade_candidate  # noqa: F401
    from pallas_arena.rl.env import RecurrentGemmaRewardEvaluator
    import ray

    try:
        os.environ["ARENA_WAIT_TIMEOUT"] = "150"
        before = set((args.output / "arena").glob("*.json"))
        evaluator = RecurrentGemmaRewardEvaluator("rg_lru", args.output, eval_timeout=180)
        result = evaluator.get_reward("def kernel(x, a, reset): return x", None)
        if result["correctness"] != 0 or result["reward"] != 0:
            raise RuntimeError(f"Grader accepted the non-Pallas transport probe: {result}")
        verdicts = list(set((args.output / "arena").glob("*.json")) - before)
        if len(verdicts) != 1 or json.loads(verdicts[0].read_text())["result"].get("gate") != "pregate":
            raise RuntimeError("Grader transport probe did not return the expected pregate rejection")
        print(json.dumps(dict(event="grader_client_check_passed", result=result)), flush=True)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
