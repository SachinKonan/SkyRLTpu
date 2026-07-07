#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


RAW_SCORE_KEYS = (
    "env/all/raw_score/min",
    "raw_score/min",
    "raw_score",
)
REWARD_KEYS = (
    "reward/max",
    "reward/mean",
)


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _first_metric(row: dict[str, Any], keys: tuple[str, ...]) -> tuple[str | None, float | None]:
    for key in keys:
        value = _as_float(row.get(key))
        if value is not None:
            return key, value
    return None, None


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Erdos discover metrics.")
    parser.add_argument("metrics", type=Path, help="Path to metrics.jsonl")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path")
    parser.add_argument("--plot", type=Path, default=None, help="Optional PNG plot output path")
    args = parser.parse_args()

    rows = load_rows(args.metrics)
    if not rows:
        raise SystemExit(f"No metrics rows found in {args.metrics}")

    curve: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        step = row.get("step", idx)
        reward_key, reward = _first_metric(row, REWARD_KEYS)
        raw_key, raw_score = _first_metric(row, RAW_SCORE_KEYS)
        curve.append(
            {
                "step": step,
                "reward_key": reward_key,
                "reward": reward,
                "raw_score_key": raw_key,
                "raw_score": raw_score,
                "sampling_sec": _as_float(row.get("time/sampling")),
                "training_sec": _as_float(row.get("time/training")),
                "total_sec": _as_float(row.get("time/total")),
                "total_ac_tokens": _as_float(row.get("total_ac_tokens")),
                "ac_tokens_per_turn": _as_float(row.get("ac_tokens_per_turn")),
            }
        )

    reward_points = [x for x in curve if x["reward"] is not None]
    raw_points = [x for x in curve if x["raw_score"] is not None]

    if reward_points:
        best_reward = max(reward_points, key=lambda x: x["reward"])
        print(
            "best reward: "
            f"step={best_reward['step']} {best_reward['reward_key']}={best_reward['reward']:.6g}"
        )
    else:
        print("best reward: no reward metric found")

    if raw_points:
        best_raw = min(raw_points, key=lambda x: x["raw_score"])
        print(
            "best raw score: "
            f"step={best_raw['step']} {best_raw['raw_score_key']}={best_raw['raw_score']:.6g}"
        )
    else:
        print("best raw score: no raw-score metric found")

    latest = curve[-1]
    print(
        "latest: "
        f"step={latest['step']} reward={latest['reward']} raw_score={latest['raw_score']} "
        f"total_sec={latest['total_sec']}"
    )

    csv_path = args.csv
    if csv_path is None:
        csv_path = args.metrics.with_suffix(".curve.csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(curve[0].keys()))
        writer.writeheader()
        writer.writerows(curve)
    print(f"wrote csv: {csv_path}")

    plot_path = args.plot
    if plot_path is None:
        plot_path = args.metrics.with_suffix(".curve.png")
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipped plot")
        return

    fig, ax1 = plt.subplots(figsize=(10, 5))
    if reward_points:
        ax1.plot([x["step"] for x in reward_points], [x["reward"] for x in reward_points], label="reward")
        ax1.set_ylabel("reward")
    ax1.set_xlabel("step")

    if raw_points:
        ax2 = ax1.twinx()
        ax2.plot(
            [x["step"] for x in raw_points],
            [x["raw_score"] for x in raw_points],
            color="tab:red",
            label="raw score",
        )
        ax2.set_ylabel("raw score")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    print(f"wrote plot: {plot_path}")


if __name__ == "__main__":
    main()
