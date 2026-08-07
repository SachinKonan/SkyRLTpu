"""Turn probe results into the markdown tables PROBE-REPORT.md carries.

Reads the driver's summary JSON and its per-candidate JSONL and emits:
  * the full 12-cell metric table,
  * the gate histogram per cell,
  * the score distribution per cell,
  * the best generation and one representative failure per cell.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def _f(x, n=4):
    return "-" if x is None else f"{x:.{n}f}"


def table(summary: dict) -> str:
    rows = [
        "| model | variant | task | n | export | compile | correct | nonzero R | mean | median | max | "
        "**non-uniform groups** |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, c in summary["per_config"].items():
        model, variant, task = name.split("|")
        s = c["score_all"]
        rows.append(
            f"| {model} | {variant} | {task} | {c['n_judged']} | {_pct(c['export_rate'])} | "
            f"{_pct(c['compile_rate'])} | {_pct(c['correctness_rate'])} | {_pct(c['nonzero_reward_rate'])} | "
            f"{_f(s.get('mean'))} | {_f(s.get('median'))} | {_f(s.get('max'))} | "
            f"**{c['nonuniform_groups']}/{c['complete_groups']}** |"
        )
    return "\n".join(rows)


def gate_tables(summary: dict) -> str:
    out = []
    gates = sorted({g for c in summary["per_config"].values() for g in c["gate_histogram"]})
    out.append("| config | " + " | ".join(f"`{g}`" for g in gates) + " |")
    out.append("|---" * (len(gates) + 1) + "|")
    for name, c in summary["per_config"].items():
        h = c["gate_histogram"]
        out.append(f"| {name} | " + " | ".join(str(h.get(g, 0)) for g in gates) + " |")
    return "\n".join(out)


def dist_tables(summary: dict) -> str:
    buckets = ["=0", "(0,0.1)", "[0.1,0.25)", "[0.25,0.5)", "[0.5,0.75)", "[0.75,0.9)", "[0.9,1.0)",
               "[1.0,1.25)", ">=1.25"]
    out = ["| config | " + " | ".join(buckets) + " |", "|---" * (len(buckets) + 1) + "|"]
    for name, c in summary["per_config"].items():
        h = c["score_histogram"]
        out.append(f"| {name} | " + " | ".join(str(h.get(b, 0)) for b in buckets) + " |")
    return "\n".join(out)


def rollups(summary: dict) -> str:
    out = []
    for key, title in (("by_variant", "prompt variant"), ("by_model", "model"), ("by_task", "task")):
        out.append(f"\n**By {title}**\n")
        out.append("| " + title + " | n | export | compile | correct | mean | max | non-uniform groups |")
        out.append("|---|---|---|---|---|---|---|---|")
        for k, c in summary[key].items():
            s = c["score_all"]
            out.append(
                f"| {k} | {c['n_judged']} | {_pct(c['export_rate'])} | {_pct(c['compile_rate'])} | "
                f"{_pct(c['correctness_rate'])} | {_f(s.get('mean'))} | {_f(s.get('max'))} | "
                f"{c['nonuniform_groups']}/{c['complete_groups']} |"
            )
    return "\n".join(out)


def examples(recs: list[dict], max_chars: int = 2600) -> str:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by[r.get("config", "?")].append(r)
    out = []
    for name, rs in sorted(by.items()):
        judged = [r for r in rs if r.get("gate")]
        if not judged:
            continue
        best = max(judged, key=lambda r: (r.get("reward") or 0.0, r.get("gate") == "all"))
        gate_counts = Counter(r.get("gate") for r in judged)
        modal_gate = gate_counts.most_common(1)[0][0]
        rep = next((r for r in judged if r.get("gate") == modal_gate), judged[0])
        out.append(f"\n### {name}\n")
        out.append(f"Best candidate — gate `{best.get('gate')}`, reward {best.get('reward')}:\n")
        out.append("```\n" + (best.get("observation") or "(no observation)") + "\n```\n")
        out.append(f"```python\n{(best.get('code') or '(no code)')[:max_chars]}\n```\n")
        out.append(f"Representative failure — modal gate `{modal_gate}` "
                   f"({gate_counts[modal_gate]}/{len(judged)}):\n")
        out.append("```\n" + (rep.get("observation") or "(no observation)") + "\n```\n")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", default="-")
    args = ap.parse_args()
    summary = json.loads(Path(args.summary).read_text())
    recs = [json.loads(ln) for ln in Path(args.jsonl).read_text().splitlines() if ln.strip()]
    md = "\n\n".join([
        "## Full metric table\n\n" + table(summary),
        "## Gate histogram\n\n" + gate_tables(summary),
        "## Score distribution\n\n" + dist_tables(summary),
        "## Roll-ups\n" + rollups(summary),
        "## Examples\n" + examples(recs),
        "## Headline\n\n```json\n" + json.dumps(summary["headline"], indent=1) + "\n```",
    ])
    if args.out == "-":
        print(md)
    else:
        Path(args.out).write_text(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
