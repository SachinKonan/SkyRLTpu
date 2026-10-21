"""Sample outcomes and best-so-far per step for scaffold runs, one definition for both arms.

A step is 512 issued samples (16 x 32). SimpleTES has no steps: its batches of 32, in dispatch
order, fill steps of 16 batches. Every issued sample lands in exactly one outcome:

  no_code      no python block in the answer (incl. answers cut off by the total budget)
  exec_failed  graded; the program crashed or timed out    (grader: "Evaluation failed")
  invalid      graded; ran, construction broke a constraint (grader: "Invalid solution")
  valid        graded; ran and returned a valid construction
  pending      not resolved yet (still generating or grading); a step is complete when none are

validity = valid / issued. Goal: all 512 valid. Two flags on top (not outcomes):
thinking_capped (the server forced the answer at the thinking budget) and cut_off (generation
stopped at the total budget). best[s] = best C5 after s steps' samples, each sample counted at
the step that issued it (late results included); best[0] is the start.

Ours: trajectories/<member>_step_N.jsonl.gz plus trajectories/late/ (graded after the eager step
closed), one row per sample; flags from the text (forced = the answer cue right after the think
close; cut_off = forced and the code fence never closed); best[0] from the first sampler snapshot.
SimpleTES: dispatch.jsonl (batches in order), llm_calls.jsonl generate rows tagged with gen_id
(no_code and flags per sample) and the newest checkpoint's nodes (graded outcomes). A batch
dispatched before a resume with no node in the checkpoint was lost and re-dispatched; it is not
on the axis. Runs without this log (before 2026-10-21) get best-so-far only, spread evenly.

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


def c5_of(message):
    match = re.match(r"C5 bound: ([0-9.eE+-]+)", str(message or ""))
    return float(match.group(1)) if match else None


def cumulative(start, step_best, last):
    best, current = {0: start}, start
    for step in range(last + 1):
        current = min(current, step_best.get(step, float("inf")))
        best[step + 1] = current
    return best


def ours(run_dir, run):
    log = f"{run_dir}/client/tinker_log/{run}"
    steps, step_best = {}, {}
    for path in glob.glob(f"{log}/member_*/trajectories/*_step_*.jsonl.gz") + glob.glob(
            f"{log}/member_*/trajectories/late/*_step_*.jsonl.gz"):
        step = int(re.search(r"_step_(\d+)", path).group(1))
        counts = steps.setdefault(step, collections.Counter())
        for line in gzip.open(path, "rt"):
            row = json.loads(line)
            kind = outcome(row)
            counts[kind] += 1
            counts["issued"] += 1
            raw = row.get("response_raw") or ""
            forced = any(close + CUE in raw for close in CLOSES)
            counts["thinking_capped"] += forced
            counts["cut_off"] += forced and raw.split(CUE, 1)[1].count("```") == 0
            c5 = c5_of(row.get("message")) if kind == "valid" else None
            if c5 is not None:
                step_best[step] = min(step_best.get(step, float("inf")), c5)
    if not steps:
        return {}, {}
    snapshots = sorted(glob.glob(f"{log}/puct_sampler_step_*.json"))
    start = float("inf")
    if snapshots:
        values = [float(s["value"]) for s in json.load(open(snapshots[0]))["states"] if s.get("value") is not None]
        start = -max(values) if values else start
    complete = {s: c for s, c in steps.items() if c["issued"] >= 0.9 * UNIT}
    return complete, cumulative(start, step_best, max(steps))


def newest_checkpoint(folder):
    def evaluations(path):
        try:
            return json.load(open(path + "/metadata.json")).get("completed_evaluations", 0)
        except (OSError, ValueError):
            return -1
    found = [c for c in glob.glob(f"{folder}/checkpoints/*/*/db_state_*") if os.path.exists(c + "/nodes.json.gz")]
    return max(found, key=evaluations) if found else None


def read_jsonl(path):
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else []


def simpletes(folder):
    """(steps, best, legacy): steps is None for runs without the per-batch sample log."""
    checkpoint = newest_checkpoint(folder)
    if checkpoint is None:
        return {}, {}, False
    nodes = json.load(gzip.open(checkpoint + "/nodes.json.gz", "rt"))
    roots = [n for n in nodes if not n.get("parent_ids")]
    start = min((-float(n["score"]) for n in roots if (n.get("metrics") or {}).get("validity", 1)),
                default=float("inf"))
    children = [n for n in nodes if n.get("parent_ids")]
    dispatch = read_jsonl(f"{folder}/dispatch.jsonl")
    calls = [r for r in read_jsonl(f"{folder}/llm_calls.jsonl") if r.get("kind") == "generate"]
    if not dispatch:
        issued = int(json.load(open(checkpoint + "/metadata.json"))["generation_attempts"])
        children.sort(key=lambda n: n["created_at"])
        step_best = {}
        for i, node in enumerate(children):
            if (node.get("metrics") or {}).get("validity"):
                step = int(i * issued / max(len(children), 1)) // UNIT
                step_best[step] = min(step_best.get(step, float("inf")), -float(node["score"]))
        return None, cumulative(start, step_best, issued // UNIT - 1), True
    by_batch = collections.defaultdict(list)
    for node in children:
        by_batch[node.get("gen_id")].append(node)
    resumes = [r["resumed_at"] for r in read_jsonl(f"{folder}/resumes.jsonl")]
    generated = collections.defaultdict(list)
    for row in calls:
        if row.get("overflow"):
            generated[row.get("gen_id")].extend([dict(no_code=True)] * row["n"])
        else:
            generated[row.get("gen_id")].extend(row.get("choices", []))
    steps, step_best, position = {}, {}, 0
    for row in dispatch:
        gen_id = row["gen_id"]
        if gen_id not in by_batch and any(t > row["time"] for t in resumes):
            continue                       # lost to a restart and re-dispatched under a new gen_id
        step, k = position // UNIT, row["k"]
        position += k
        counts = steps.setdefault(step, collections.Counter())
        counts["issued"] += k
        choices = generated.get(gen_id, [])
        counts["no_code"] += sum(1 for c in choices if c.get("no_code") or c.get("bad_choice"))
        counts["thinking_capped"] += sum(1 for c in choices if c.get("forced"))
        counts["cut_off"] += sum(1 for c in choices if c.get("finish_reason") == "length")
        for node in by_batch.get(gen_id, []):
            metrics = node.get("metrics") or {}
            if metrics.get("validity"):
                counts["valid"] += 1
                step_best[step] = min(step_best.get(step, float("inf")), -float(node["score"]))
            else:
                counts["invalid" if str(metrics.get("error", "")).startswith("Invalid solution")
                       else "exec_failed"] += 1
    for counts in steps.values():
        counts["pending"] = max(0, counts["issued"] - sum(counts[k] for k in OUTCOMES))
    complete = {s: c for s, c in steps.items() if c["issued"] == UNIT and c["pending"] == 0}
    return complete, cumulative(start, step_best, max(steps) if steps else -1), False


def summarize(steps):
    window = collections.Counter()
    for step in sorted(steps)[-WINDOW:]:
        window.update(steps[step])
    n = window["issued"] or 1
    return dict(steps=len(steps), **{k: window[k] / n for k in (*OUTCOMES, *FLAGS)})


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
            steps, best, legacy = simpletes(f"{base}/simpletes")
        else:
            (steps, best), legacy = ours(f"{base}/runs/{run}", run), False
        if not best:
            continue
        entry = dict(best=best, legacy=legacy)
        if steps:
            entry.update(summary=summarize(steps), steps={s: dict(c) for s, c in sorted(steps.items())})
            print(f"{run:36s} steps={len(steps):3d} " + " ".join(
                f"{k}={entry['summary'][k]:.2f}" for k in (*OUTCOMES, *FLAGS)))
        else:
            print(f"{run:36s} best only (no per-sample log)")
        out[run] = entry
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
