"""Report for the seam+dialect run: the metric, the taxonomy, and the deltas.

The metric is NOT pass rate. `tailored` passed 16/16 with a within-group reward
spread of 0.0042 against noise floors of 0.0069/0.0158 -- the best cell ever
measured by pass rate and worthless as an RL environment, because every
candidate converged and there was nothing to learn. So each cell is ranked on
within-group spread ABOVE the judge's own measured noise floor, and a cell that
passes everything at one score is reported as the failure it is.

Three things this adds to `ladder_report`:

  * **verbatim failure signatures**, normalised just enough to count (numbers
    and quoted shapes collapsed), because "which error" is the finding when the
    answer to "did it pass" is no;
  * **deltas against the two runs this one is a response to** -- the seam run
    (job 3651278) and the prompt ladder (job 3687041) -- on the SAME tasks, so
    "this addition removed that failure class" is a measurement and not a story;
  * **finish_reason**, because a prompt that leaves too little context
    truncates candidates and truncation reads exactly like model failure.

Usage:
  python -m pallas_arena.probe.seam_dialect_report --jsonl sd-results-N.jsonl \\
      --boot-report sd-judge-boot-N.json \\
      --compare runs/pallas_arena/seam-results-3651278.jsonl \\
      --compare runs/pallas_arena/ladder-results-3687041.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from pallas_arena.probe.ladder_report import _judged, _reward, build, ladder_table

TARGETS = ("splash_attention", "ragged_paged_attention", "megablox_gmm")

_NUM = re.compile(r"\d+")
_QUOT = re.compile(r"'[^']*'")


def signature(rec: dict) -> str:
    """The candidate's failure, collapsed to a countable class.

    Keeps the exception type and the first line of the message, with digits and
    quoted names blanked, so `Ref shape: (8, 4, 128)` and `Ref shape: (4, 4,
    128)` are one class and `out_spec` and `out_shapes` are one class.
    """
    obs = (rec.get("observation") or "").strip()
    if not obs:
        return f"gate:{rec.get('gate')}"
    body = obs.split("\n")
    line = next((ln for ln in body[1:] if ln.strip()), body[0])
    line = _QUOT.sub("'X'", _NUM.sub("N", line.strip()))
    return line[:130]


def taxonomy(recs: list[dict], key=lambda r: f"{r.get('task')}|{r.get('variant')}") -> dict:
    out: dict[str, Counter] = defaultdict(Counter)
    for r in recs:
        if _judged(r) and r.get("gate") != "all":
            out[key(r)][signature(r)] += 1
    return out


def taxonomy_table(tax: dict, top: int = 6) -> str:
    rows = ["| cell | n | verbatim failure signature |", "|---|---|---|"]
    for cell, c in sorted(tax.items()):
        for sig, n in c.most_common(top):
            rows.append(f"| {cell} | {n} | `{sig.replace('|', chr(92) + '|')}` |")
    return "\n".join(rows)


def delta_table(this: dict, others: dict[str, dict]) -> str:
    """Per TASK (variants pooled), which signatures appeared and which went."""
    def pooled(tax):
        out: dict[str, Counter] = defaultdict(Counter)
        for cell, c in tax.items():
            out[cell.split("|")[0]].update(c)
        return out

    a = pooled(this)
    bs = {name: pooled(t) for name, t in others.items()}
    rows = ["| task | signature | " + " | ".join(f"{n}" for n in bs) + " | **this run** |",
            "|---" * (3 + len(bs)) + "|"]
    for task in sorted(set(a) | {t for b in bs.values() for t in b}):
        sigs = set(a.get(task, {}))
        for b in bs.values():
            sigs |= set(b.get(task, {}))
        for sig in sorted(sigs, key=lambda s: -(a.get(task, Counter()).get(s, 0)
                                                + sum(b.get(task, Counter()).get(s, 0) for b in bs.values()))):
            prev = [str(b.get(task, Counter()).get(sig, 0)) for b in bs.values()]
            now = a.get(task, Counter()).get(sig, 0)
            if now == 0 and all(p == "0" for p in prev):
                continue
            rows.append(f"| {task} | `{sig.replace('|', chr(92) + '|')}` | " + " | ".join(prev) + f" | **{now}** |")
    return "\n".join(rows)


def truncation_table(recs: list[dict]) -> str:
    by: dict[str, Counter] = defaultdict(Counter)
    tok: dict[str, list] = defaultdict(list)
    for r in recs:
        cell = f"{r.get('task')}|{r.get('variant')}"
        by[cell][r.get("finish_reason")] += 1
        tok[cell].append(r.get("max_new_tokens"))
    rows = ["| cell | finish_reason | completion budget |", "|---|---|---|"]
    for cell in sorted(by):
        b = ", ".join(f"{k}={v}" for k, v in by[cell].most_common())
        rows.append(f"| {cell} | {b} | {max(tok[cell]) if tok[cell] else '-'} |")
    return "\n".join(rows)


def fill_table(recs: list[dict]) -> str:
    by: dict[str, list] = defaultdict(list)
    for r in recs:
        by[f"{r.get('task')}|{r.get('variant')}"].append(r)
    rows = ["| cell | median fill chars | median fill tokens | extraction | missing names |", "|---|---|---|---|---|"]
    for cell, rs in sorted(by.items()):
        fc = sorted(r.get("fill_chars", 0) for r in rs)
        ex = Counter(r.get("extraction") for r in rs).most_common(2)
        miss = sum(1 for r in rs if r.get("seam_missing_names"))
        rows.append(f"| {cell} | {fc[len(fc)//2] if fc else 0} | {(fc[len(fc)//2] if fc else 0)//4} | "
                    f"{', '.join(f'{k}={v}' for k, v in ex)} | {miss}/{len(rs)} |")
    return "\n".join(rows)


def passing_programs(recs: list[dict], limit: int = 6000) -> str:
    by: dict[str, list] = defaultdict(list)
    for r in recs:
        if r.get("gate") == "all":
            by[f"{r.get('task')}|{r.get('variant')}"].append(r)
    if not by:
        return "\n_(no cell produced a passing kernel)_\n"
    out = []
    for cell, rs in sorted(by.items()):
        best = max(rs, key=_reward)
        out.append(f"\n### {cell} -- reward {_reward(best):.4f} ({len(rs)} passing)\n")
        out.append("```\n" + (best.get("observation") or "")[:1200] + "\n```\n")
        out.append("The model's FILL, verbatim:\n\n```python\n" + (best.get("fill") or best.get("code") or "")[:limit] + "\n```\n")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--boot-report", default=None)
    ap.add_argument("--compare", action="append", default=[],
                    help="an earlier run's jsonl to diff the failure taxonomy against")
    ap.add_argument("--group-size", type=int, default=16)
    ap.add_argument("--out", default="-")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    recs = [json.loads(ln) for ln in Path(args.jsonl).read_text().splitlines() if ln.strip()]
    floors: dict = {}
    if args.boot_report and Path(args.boot_report).exists():
        rep = json.loads(Path(args.boot_report).read_text())
        floors = {t: r.get("noise_floor") for t, r in rep.items() if isinstance(r, dict)}

    stats = build(recs, floors, args.group_size)
    tax = taxonomy(recs)
    others = {}
    for p in args.compare:
        path = Path(p)
        if not path.exists():
            continue
        prev = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
        prev = [r for r in prev if r.get("task") in TARGETS]
        others[path.name.split("-results-")[0] + " " + path.stem.split("-")[-1]] = taxonomy(prev)

    md = "\n\n".join([
        "## The metric\n\n" + ladder_table(stats),
        "## Failure taxonomy, verbatim\n\n" + taxonomy_table(tax),
        ("## Delta against the earlier runs\n\n" + delta_table(tax, others)) if others else "",
        "## Truncation check\n\n" + truncation_table(recs),
        "## What the model actually wrote\n\n" + fill_table(recs),
        "## Passing kernels\n" + passing_programs(recs),
    ])
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"cells": stats, "noise_floors": floors,
             "taxonomy": {k: dict(v) for k, v in tax.items()}}, indent=1, default=str))
    if args.out == "-":
        print(md)
    else:
        Path(args.out).write_text(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
