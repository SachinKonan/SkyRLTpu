"""Simple offline batch experiment runner (agent generates candidate files).

Not the TTT-Discover RL trainer. It is the reusable external scalar-reward
`environment` to plug into one.
"""
import argparse
import json
from pathlib import Path
from fairdiscover.trial import trial


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", required=True, help="Directory containing *.py")
    p.add_argument("--stage", required=True)
    p.add_argument("--runs", required=True)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--max-reward-ind", type=int, default=2048)
    p.add_argument("--train-timeout", type=int, default=3600)
    args = p.parse_args()
    root = Path(args.runs)
    root.mkdir(parents=True, exist_ok=True)
    ledger = root / "leaderboard.jsonl"
    best = None
    for candidate in sorted(Path(args.candidates).glob("*.py")):
        result = trial(candidate, args.stage, root / candidate.stem,
                       steps=args.steps, max_reward_ind=args.max_reward_ind,
                       train_timeout=args.train_timeout)
        row = {"candidate": str(candidate), **result}
        with open(ledger, "a", encoding="utf8") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps({"candidate": candidate.name, "reward": result["reward"]}))
        if best is None or result["reward"] > best["reward"]:
            best = row
    print("Best", json.dumps(best))


if __name__ == "__main__":
    main()
