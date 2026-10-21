"""Sample outcomes and best-so-far per step for scaffold runs, one definition for both arms.

A step issues 512 samples (16 x 32); SimpleTES has no steps, so its step n is the samples it
issued at positions 512n..512n+511. Every issued sample lands in exactly one outcome:

  no_code      no python block in the answer (incl. answers cut off by the total budget)
  exec_failed  graded; the program crashed or timed out    (grader: "Evaluation failed")
  invalid      graded; ran, construction broke a constraint (grader: "Invalid solution")
  valid        graded; ran and returned a valid construction

validity = valid / issued. Goal: all 512 valid. Two flags on top (not outcomes):
thinking_capped (the server forced the answer at the thinking budget) and cut_off (generation
stopped at the total budget). best[s] = best C5 after s steps' samples (best[0] = the start).

Sources. Ours: trajectories/<member>_step_N.jsonl.gz plus trajectories/late/ (samples graded
after the eager step closed), one row per sample; the flags are read from the text (forced =
the answer cue right after the think close; cut_off = forced and the code fence never closed);
best from puct_sampler_step_N.json. SimpleTES: the newest checkpoint's nodes (outcomes and
scores), placed on the sample axis by their batch in dispatch.jsonl; issued samples of a step
that were never graded had no code. Its flags come from llm_calls.jsonl choices in completion
order (the last steps' worth of choices for the summary window). Runs without dispatch.jsonl
(before 2026-10-21) spread graded nodes evenly over generation_attempts.

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
FLAGS = ("thinking_capped", "cut_off")
CUE = "Here is the final complete program:\n\n```python\n"
CLOSES = ("\n</think>\n\n", "to=user<|message|>")
WINDOW = 3


def outcome(row):
    if row.get("error"):
        return "exec_failed"
    if not (row.get("parsed_code") or "").strip():
        return "no_code"
    if row.get("correctness") == 1:
        return "valid"
    return "invalid" if str(row.get("message") or "").startswith("Invalid solution") else "exec_failed"


def ours(run_dir, run):
    steps = {}
    log = f"{run_dir}/client/tinker_log/{run}"
    for path in glob.glob(f"{log}/member_*/trajectories/*_step_*.jsonl.gz") + glob.glob(
            f"{log}/member_*/trajectories/late/*_step_*.jsonl.gz"):
        step = int(re.search(r"_step_(\d+)", path).group(1))
        counts = steps.setdefault(step, collections.Counter())
        for line in gzip.open(path, "rt"):
            row = json.loads(line)
            counts[outcome(row)] += 1
            raw = row.get("response_raw") or ""
            forced = any(close + CUE in raw for close in CLOSES)
            counts["thinking_capped"] += forced
            counts["cut_off"] += forced and raw.split(CUE, 1)[1].count("```") == 0
            counts["issued"] += 1
    best = {}
    for path in glob.glob(f"{log}/puct_sampler_step_*.json"):
        values = [float(s["value"]) for s in json.load(open(path))["states"] if s.get("value") is not None]
        if values:
            best[int(re.search(r"(\d+)\.json$", path).group(1))] = -max(values)
    complete = {s for s, c in steps.items() if c["issued"] >= 0.9 * UNIT}
    return {s: c for s, c in steps.items() if s in complete}, best


def newest_checkpoint(folder):
    def evaluations(path):
        try:
            return json.load(open(path + "/metadata.json")).get("completed_evaluations", 0)
        except (OSError, ValueError):
            return -1
    found = [c for c in glob.glob(f"{folder}/checkpoints/*/*/db_state_*") if os.path.exists(c + "/nodes.json.gz")]
    return max(found, key=evaluations) if found else None


def batch_starts(folder):
    """gen_id -> first issued-sample position of its batch; a re-issued range belongs to the later batch."""
    owner = {}
    path = f"{folder}/dispatch.jsonl"
    if not os.path.exists(path):
        return None
    for line in open(path):
        row = json.loads(line)
        for position in range(row["end"] - row["k"], row["end"]):
            owner[position] = row["gen_id"]
    starts = {}
    for position in sorted(owner):
        starts.setdefault(owner[position], position)
    return starts


def simpletes(folder):
    checkpoint = newest_checkpoint(folder)
    if checkpoint is None:
        return {}, {}, {}
    issued = int(json.load(open(checkpoint + "/metadata.json"))["generation_attempts"])
    nodes = json.load(gzip.open(checkpoint + "/nodes.json.gz", "rt"))
    roots = [n for n in nodes if not n.get("parent_ids")]
    children = sorted((n for n in nodes if n.get("parent_ids")), key=lambda n: n["created_at"])
    starts = batch_starts(folder)
    placed = []
    for i, node in enumerate(children):
        if starts is None:
            position = int(i * issued / max(len(children), 1))
        elif node.get("gen_id") in starts:
            position = starts[node["gen_id"]]
        else:
            continue          # its range was re-issued after a restart: not on the axis
        placed.append((position, node))
    placed.sort(key=lambda item: item[0])
    # The newest step may still have batches in flight unless the run finished.
    total = issued // UNIT if os.path.exists(f"{folder}/summary.json") else max(0, issued // UNIT - 1)
    steps = {s: collections.Counter(issued=UNIT) for s in range(total)}
    current = min((-float(n["score"]) for n in roots if (n.get("metrics") or {}).get("validity", 1)),
                  default=float("inf"))
    best = {0: current}
    for position, node in placed:
        step = position // UNIT
        metrics = node.get("metrics") or {}
        valid = bool(metrics.get("validity"))
        if step in steps:
            steps[step]["valid" if valid else "invalid" if str(metrics.get("error", "")).startswith(
                "Invalid solution") else "exec_failed"] += 1
        if valid:
            current = min(current, -float(node["score"]))
        best[step + 1] = current
    for step in range(1, total + 1):
        best.setdefault(step, best[step - 1])
    best = {s: v for s, v in best.items() if s <= total}
    for counts in steps.values():
        counts["no_code"] = UNIT - sum(counts[k] for k in OUTCOMES[1:])
    choices = [c for line in open(f"{folder}/llm_calls.jsonl") for row in [json.loads(line)]
               if row.get("kind") == "generate" for c in row.get("choices", [])][-WINDOW * UNIT:]
    flags = dict(thinking_capped=sum(bool(c.get("forced")) for c in choices) / max(len(choices), 1),
                 cut_off=sum(c.get("finish_reason") == "length" for c in choices) / max(len(choices), 1))
    return steps, best, flags


def summarize(steps, flags=None):
    window = collections.Counter()
    for step in sorted(steps)[-WINDOW:]:
        window.update(steps[step])
    n = window["issued"] or 1
    summary = dict(steps=len(steps), **{k: window[k] / n for k in (*OUTCOMES, *FLAGS)})
    summary.update(flags or {})
    return summary


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
            steps, best, flags = simpletes(f"{base}/simpletes")
        else:
            (steps, best), flags = ours(f"{base}/runs/{run}", run), None
        if not steps:
            continue
        summary = summarize(steps, flags)
        out[run] = dict(summary=summary, steps={s: dict(c) for s, c in sorted(steps.items())},
                        best={s: best[s] for s in sorted(best)})
        print(f"{run:36s} steps={summary['steps']:3d} " + " ".join(
            f"{k}={summary[k]:.2f}" for k in (*OUTCOMES, *FLAGS)))
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
