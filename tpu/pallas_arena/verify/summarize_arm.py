"""Summarize one seeded arm: failure taxonomy + reward distribution.

    python3 summarize_arm.py <graded.json>

Input is grade_gens_via_queue's output: {cell: {n, graded, passed,
best_reward, rows:[{idx, outcome, passed, reward_with_bwd, ...}]}}.

The seed is the reference point. Its reward IS 1.0, so a candidate at 1.0
matched the kernel it was handed and only >1.0 is an improvement over
DeepMind's production kernel.
"""
import collections
import json
import re
import sys

cells = json.load(open(sys.argv[1]))

for cell, d in cells.items():
    rows = d.get("rows") or []
    print(f"=== {cell} ===")
    print(f"generations: {d.get('n')}  graded: {d.get('graded')}  passed: {d.get('passed')}")

    gate = collections.Counter()
    msgs = collections.Counter()
    rewards = []
    for r in rows:
        if r.get("passed"):
            gate["PASSED"] += 1
            rw = r.get("reward_with_bwd")
            if rw is not None:
                rewards.append(float(rw))
            continue
        out = str(r.get("outcome") or "")
        m = re.match(r"\[(\w+)\]", out)
        g = m.group(1) if m else ("gen_error" if "error" in out.lower() else out[:24] or "?")
        gate[g] += 1
        msgs[out[:78]] += 1

    n = len(rows)
    print("\nWHERE THEY DIED")
    where = {"pregate": "CPU, no chip touched", "candidate_compile": "silicon: Mosaic compile",
             "correctness": "silicon: vs the fp32 oracle", "runtime_halt": "silicon: core halt",
             "PASSED": "survived every gate", "judge_fault": "judge could not measure"}
    for g, c in gate.most_common():
        print(f"  {g:20s} {c:3d}  {100*c/n:4.0f}%  {where.get(g, '')}")

    if rewards:
        rewards.sort()
        better = [r for r in rewards if r > 1.0]
        print(f"\nREWARDS (seed = 1.0)")
        print(f"  {[round(r, 4) for r in rewards]}")
        print(f"  median {rewards[len(rewards)//2]:.4f}  best {rewards[-1]:.4f}")
        print(f"  BEAT THE SEED: {len(better)}/{n}"
              + (f"  {[round(r,4) for r in better]}" if better else "  -- none"))
    else:
        print("\nREWARDS: no survivor produced a reward")

    print("\nFAILURE MESSAGES")
    for v, c in msgs.most_common(10):
        print(f"  [{c}x] {v}")
    print()
