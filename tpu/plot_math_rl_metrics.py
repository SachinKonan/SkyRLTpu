#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SERIES = (
    "test/env/all/correct",
    "env/all/correct",
    "test/env/all/reward/total",
    "env/all/reward/total",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if "step" in row:
                rows.append(row)
    if not rows:
        raise SystemExit(f"No step metrics found in {path}")
    return sorted(rows, key=lambda row: row["step"])


def numeric_pairs(rows: list[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, int | float):
            xs.append(int(row["step"]))
            ys.append(float(value))
    return xs, ys


def write_summary(rows: list[dict[str, Any]], keys: list[str], path: Path) -> None:
    summary: dict[str, Any] = {
        "num_rows": len(rows),
        "first_step": rows[0]["step"],
        "last_step": rows[-1]["step"],
        "series": {},
    }
    for key in keys:
        xs, ys = numeric_pairs(rows, key)
        if not ys:
            continue
        summary["series"][key] = {
            "num_points": len(ys),
            "last_step": xs[-1],
            "last": ys[-1],
            "max": max(ys),
            "max_step": xs[ys.index(max(ys))],
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot reward/correctness curves from math-RL metrics.jsonl")
    parser.add_argument("metrics", type=Path, help="Path to metrics.jsonl")
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--summary", type=Path, help="Optional summary JSON path")
    parser.add_argument(
        "--series",
        action="append",
        default=[],
        help="Metric key to plot. Can be repeated. Defaults to common math-RL reward/correct keys.",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.metrics)
    keys = args.series or list(DEFAULT_SERIES)
    plotted_keys = [key for key in keys if numeric_pairs(rows, key)[1]]
    if not plotted_keys:
        available = sorted(k for row in rows for k, v in row.items() if isinstance(v, int | float))
        raise SystemExit(f"No requested series found. Available numeric keys include: {available[:80]}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for key in plotted_keys:
        xs, ys = numeric_pairs(rows, key)
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=3, label=key)
    ax.set_xlabel("step")
    ax.set_ylabel("metric")
    ax.set_title("Qwen3.5-9B MATH RL")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)

    if args.summary:
        write_summary(rows, plotted_keys, args.summary)


if __name__ == "__main__":
    main()
