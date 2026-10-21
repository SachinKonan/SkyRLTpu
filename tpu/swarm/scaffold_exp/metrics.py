"""Sample outcomes and best-so-far per step for scaffold runs, one definition for both arms.

A step is 512 issued samples (16 x 32). SimpleTES has no steps: its batches of 32, in dispatch
order, fill steps of 16 batches. Every issued sample lands in exactly one outcome:

  no_code      no python block in the answer (incl. answers cut off by the total budget)
  exec_failed  graded; the program crashed or timed out    (grader: "Evaluation failed")
  invalid      graded; ran, construction broke a constraint (grader: "Invalid solution")
  valid        graded; ran and returned a valid construction
  pending      not resolved yet (still generating or grading); a step is complete when none are
  lost         never resolved: dropped by a restart or cancelled after the step closed

validity = valid / issued. Goal: all 512 valid. Two flags on top (not outcomes):
thinking_capped (the server forced the answer at the thinking budget) and cut_off (generation
stopped at the total budget). best[s] = best C5 after s steps' samples, each sample counted at
the step that issued it (late results included); best[0] is the start.

Ours: trajectories/<member>_step_N.jsonl.gz plus trajectories/late/ (graded after the eager step
closed), one row per sample, 512 issued per step; flags from the text (forced = the answer cue
right after the think close; cut_off = forced and the code fence never closed). Rows still missing
three steps later were cancelled: lost (a grade is capped at ~18 min; a step takes longer). best[0] from the first sampler snapshot.
SimpleTES: samples.jsonl, written by the scaffold arm: batches in dispatch order (uid per
dispatch), samples without code per batch, terminal generation failures (counted as no_code),
one row per graded candidate. After a resume, a batch dispatched before it is on the axis only if
it was finalized (finalize_at samples resolved) by the checkpoint's write time; its samples
resolved later were lost with the crash. Flags from llm_calls.jsonl rows of the same uid.
Runs without a complete samples.jsonl (before 2026-10-21, or resumed from such a run) get
best-so-far only, spread evenly. Known limit: after a SimpleTES crash-restart, "finalized by the
checkpoint" is inferred from resolved samples and the checkpoint file's write time, not logged.

  python -m tpu.swarm.scaffold_exp.metrics <runs root> <run glob> --out metrics.json
"""
import argparse
import collections
import glob
import gzip
import json
import os
import re
from datetime import datetime, timezone

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
    last = max(steps)
    complete = {}
    for step, counts in steps.items():
        missing = UNIT - counts["issued"]
        if missing and step > last - 3:
            continue                       # late samples may still be grading (a grade is < 20 min, a step longer)
        counts["lost"] = missing
        counts["issued"] = UNIT
        complete[step] = counts
    return complete, cumulative(start, step_best, last)


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
    """(steps, best, legacy): steps is None for runs without samples.jsonl."""
    checkpoint = newest_checkpoint(folder)
    events = read_jsonl(f"{folder}/samples.jsonl")
    if checkpoint is None and not events:
        return {}, {}, False
    # v35-v38 wrote no checkpoints (see driver.keep_previous_checkpoint); the sample log and the
    # round-0 snapshot (the roots) are enough.
    nodes = json.load(gzip.open(checkpoint + "/nodes.json.gz", "rt")) if checkpoint else []
    roots = [n for n in nodes if not n.get("parent_ids")]
    if not roots and os.path.exists(f"{folder}/steps/simpletes_step_000000.json"):
        roots = [dict(score=s["value"], metrics=s.get("metrics") or {})
                 for s in json.load(open(f"{folder}/steps/simpletes_step_000000.json"))["states"]]
    start = min((-float(n["score"]) for n in roots if (n.get("metrics") or {}).get("validity", 1)),
                default=float("inf"))
    logged = {row.get("node") for row in events if row["event"] == "graded"}
    first = min((row["time"] for row in events if row["event"] == "dispatch"), default=None)
    unlogged = [n for n in nodes if n.get("parent_ids") and n["id"] not in logged
                and (first is None or n["created_at"] < datetime.fromtimestamp(first, timezone.utc).replace(tzinfo=None).isoformat())]
    if not events or unlogged:
        children = sorted((n for n in nodes if n.get("parent_ids")), key=lambda n: n["created_at"])
        issued = int(json.load(open(checkpoint + "/metadata.json"))["generation_attempts"])
        step_best = {}
        for i, node in enumerate(children):
            if (node.get("metrics") or {}).get("validity"):
                step = int(i * issued / max(len(children), 1)) // UNIT
                step_best[step] = min(step_best.get(step, float("inf")), -float(node["score"]))
        return None, cumulative(start, step_best, issued // UNIT - 1), True
    resumes = sorted((r["resumed_at"], r.get("checkpoint_time", 0.0))
                     for r in read_jsonl(f"{folder}/resumes.jsonl"))
    batches, order = {}, []
    for row in events:
        if row["event"] == "dispatch":
            batches[row["batch"]] = dict(row, resolved=[])
            order.append(row["batch"])
            continue
        batch = batches.get(row.get("batch"))
        if batch is None:
            continue
        if row["event"] == "generated":
            batch["resolved"] += [(row["time"], "no_code", None)] * row["no_code"]
        elif row["event"] == "generation_failed":
            batch["resolved"] += [(row["time"], "no_code", None)] * row["k"]
        elif row["event"] == "graded":
            batch["resolved"].append((row["time"], row["outcome"], row.get("score")))
    flags = collections.defaultdict(collections.Counter)
    for row in read_jsonl(f"{folder}/llm_calls.jsonl"):
        if row.get("kind") == "generate" and row.get("batch"):
            for choice in row.get("choices", []):
                flags[row["batch"]]["thinking_capped"] += bool(choice.get("forced"))
                flags[row["batch"]]["cut_off"] += choice.get("finish_reason") == "length"
    # A summary from before the newest resume (budget extended) does not end this lifetime.
    summary = f"{folder}/summary.json"
    finished = os.path.exists(summary) and os.path.getmtime(summary) > max((at for at, _ in resumes), default=0)
    steps, step_best, position = {}, {}, 0
    for uid in order:
        batch = batches[uid]
        # The first resume after this dispatch decides whether the batch survived it.
        after = next(((at, saved) for at, saved in resumes if at > batch["time"]), None)
        resolved = batch["resolved"]
        if after is not None:
            resolved = [r for r in resolved if r[0] <= after[1]]
            if len(resolved) < batch["finalize_at"]:
                continue                   # not finalized in the checkpoint: re-dispatched after the resume
        step = position // UNIT
        position += batch["k"]
        counts = steps.setdefault(step, collections.Counter())
        counts["issued"] += batch["k"]
        counts.update(kind for _, kind, _ in resolved)
        counts.update(flags.get(uid, {}))
        for _, kind, score in resolved:
            if kind == "valid":
                step_best[step] = min(step_best.get(step, float("inf")), -score)
        unresolved = batch["k"] - len(resolved)
        counts["lost" if after is not None or finished else "pending"] += unresolved
    complete = {s: c for s, c in steps.items() if c["issued"] == UNIT and not c["pending"]}
    return complete, cumulative(start, step_best, max(steps) if steps else -1), False


def summarize(steps):
    window = collections.Counter()
    for step in sorted(steps)[-WINDOW:]:
        window.update(steps[step])
    n = window["issued"] or 1
    return dict(steps=len(steps), **{k: window[k] / n for k in (*OUTCOMES, "lost", *FLAGS)})


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
                f"{k}={entry['summary'][k]:.2f}" for k in (*OUTCOMES, "lost", *FLAGS)))
        else:
            print(f"{run:36s} best only (no per-sample log)")
        out[run] = entry
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
