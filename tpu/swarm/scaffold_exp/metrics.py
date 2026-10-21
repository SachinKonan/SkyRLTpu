"""Sample outcomes and best-so-far per step for scaffold runs, one definition for both arms.

A step issues 512 samples (16 x 32); SimpleTES has no steps, so its step n is samples issued
512n..512(n+1). Every issued sample lands in exactly one outcome:

  no_code      no python block in the answer (incl. answers cut off by the total budget)
  exec_failed  graded; the program crashed or timed out    (grader: "Evaluation failed")
  invalid      graded; ran, construction broke a constraint (grader: "Invalid solution")
  valid        graded; ran and returned a valid construction

validity = valid / issued. Goal: all 512 valid. Both arms commit a step from the first 95% of
its samples (eager); ours logs only those, so its "issued" is the ~487 logged rows of a step. Two flags on top (not outcomes):
thinking_capped (the server forced the answer at the thinking budget) and cut_off (generation
stopped at the total budget).

Sources. Ours: trajectories/<member>_step_N.jsonl.gz, one row per issued sample; the flags are
read from the text (forced = the answer cue right after the think close; cut_off = the cue
present and its code fence never closed). SimpleTES: llm_calls.jsonl choices (no_code and the
flags, exact) and the newest checkpoint's nodes (graded outcomes and scores), placed on the
sample axis by attempts.jsonl (samples issued when each node was graded). Runs from before
attempts.jsonl spread graded nodes evenly over generation_attempts.

  python -m tpu.swarm.scaffold_exp.metrics <runs root> <run glob> --out metrics.json
"""
import argparse
import collections
import glob
import gzip
import json
import os
import re

UNIT = 512
OUTCOMES = ("no_code", "exec_failed", "invalid", "valid")
CUE = "Here is the final complete program:\n\n```python\n"
CLOSES = ("\n</think>\n\n", "to=user<|message|>")


def outcome(correct, code, message):
    if not (code or "").strip():
        return "no_code"
    if correct == 1:
        return "valid"
    return "invalid" if (message or "").startswith("Invalid solution") else "exec_failed"


def ours(run_dir, run):
    steps = {}
    for path in glob.glob(f"{run_dir}/client/tinker_log/{run}/member_*/trajectories/*_step_*.jsonl.gz"):
        step = int(re.search(r"_step_(\d+)", path).group(1))
        counts = steps.setdefault(step, collections.Counter())
        for line in gzip.open(path, "rt"):
            row = json.loads(line)
            counts[outcome(row["correctness"], row["parsed_code"], row["message"])] += 1
            raw = row.get("response_raw") or ""
            forced = any(close + CUE in raw for close in CLOSES)
            counts["thinking_capped"] += forced
            counts["cut_off"] += forced and raw.split(CUE, 1)[1].count("```") == 0
            counts["issued"] += 1
    best = {}
    for path in glob.glob(f"{run_dir}/client/tinker_log/{run}/puct_sampler_step_*.json"):
        values = [float(s["value"]) for s in json.load(open(path))["states"] if s.get("value") is not None]
        if values:
            best[int(re.search(r"(\d+)\.json$", path).group(1))] = -max(values)
    return steps, best


def newest_checkpoint(folder):
    def evaluations(path):
        try:
            return json.load(open(path + "/metadata.json")).get("completed_evaluations", 0)
        except (OSError, ValueError):
            return -1
    found = [c for c in glob.glob(f"{folder}/checkpoints/*/*/db_state_*") if os.path.exists(c + "/nodes.json.gz")]
    return max(found, key=evaluations) if found else None


def simpletes(folder):
    checkpoint = newest_checkpoint(folder)
    if checkpoint is None:
        return {}, {}
    meta = json.load(open(checkpoint + "/metadata.json"))
    nodes = [n for n in json.load(gzip.open(checkpoint + "/nodes.json.gz", "rt")) if n.get("parent_ids")]
    nodes.sort(key=lambda n: n["created_at"])
    attempts = {}
    if os.path.exists(f"{folder}/attempts.jsonl"):
        for line in open(f"{folder}/attempts.jsonl"):
            row = json.loads(line)
            attempts[row["id"]] = row["attempts"]
    issued = int(meta["generation_attempts"])
    steps, best, current = {}, {}, float("inf")
    for i, node in enumerate(nodes):
        sample = attempts.get(node["id"], (i + 1) * issued / max(len(nodes), 1)) - 1
        step = int(sample // UNIT)
        metrics = node.get("metrics") or {}
        valid = bool(metrics.get("validity"))
        kind = "valid" if valid else ("invalid" if str(metrics.get("error", "")).startswith("Invalid solution")
                                      else "exec_failed")
        steps.setdefault(step, collections.Counter())[kind] += 1
        if valid:
            current = min(current, -float(node["score"]))
        best[step + 1] = current
    # Issued samples per step; the ungraded ones had no code. Flags from the per-choice call log.
    for step in range(issued // UNIT):
        counts = steps.setdefault(step, collections.Counter())
        counts["issued"] = UNIT
        counts["no_code"] = max(0, UNIT - sum(counts[k] for k in OUTCOMES[1:]))
    choices = [c for line in open(f"{folder}/llm_calls.jsonl") for row in [json.loads(line)]
               if row.get("kind") == "generate" for c in row.get("choices", [])]
    flags = dict(thinking_capped=sum(bool(c.get("forced")) for c in choices) / max(len(choices), 1),
                 cut_off=sum(c.get("finish_reason") == "length" for c in choices) / max(len(choices), 1))
    return {s: c for s, c in steps.items() if s < issued // UNIT}, best, flags


def summarize(steps, last=3):
    full = sorted(s for s, c in steps.items() if c["issued"] >= 0.9 * UNIT)
    window = collections.Counter()
    for step in full[-last:]:
        window.update(steps[step])
    n = window["issued"] or 1
    return dict(steps=len(full), **{k: window[k] / n for k in (*OUTCOMES, "thinking_capped", "cut_off")})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root")
    parser.add_argument("pattern")
    parser.add_argument("--out", default="metrics.json")
    args = parser.parse_args()
    out = {}
    for base in sorted(glob.glob(os.path.join(args.root, args.pattern))):
        run = os.path.basename(base)
        if "-stes-" in run:
            result = simpletes(f"{base}/simpletes")
            if not result[0]:
                continue
            steps, best, flags = result
        else:
            steps, best = ours(f"{base}/runs/{run}", run)
            flags = None
        if not steps:
            continue
        summary = summarize(steps)
        if flags:
            summary.update(flags)
        out[run] = dict(summary=summary, steps={s: dict(c) for s, c in sorted(steps.items())},
                        best={s: best[s] for s in sorted(best)})
        print(f"{run:36s} steps={summary['steps']:3d} " + " ".join(
            f"{k}={summary[k]:.2f}" for k in (*OUTCOMES, "thinking_capped", "cut_off")))
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
