#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


GCS_LISTING_RE = re.compile(
    r"^\s*(?P<size>\d+)\s+(?P<time>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\s+(?P<path>gs://.*)$"
)
ITERATION_RE = re.compile(r"iteration_(?P<step>\d{6})/")
MODEL_RE = re.compile(r"tinker://(?P<model>[^/]+)/")


@dataclass
class IterationTimes:
    start: datetime | None = None
    finish: datetime | None = None


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def read_latest_metrics(path: Path) -> list[dict[str, Any]]:
    by_step: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(path):
        step = row.get("step")
        if isinstance(step, int) and isinstance(row.get("env/all/reward/total"), (int, float)):
            by_step[step] = row
    if not by_step:
        raise SystemExit(f"No train reward rows found in {path}")
    return [by_step[step] for step in sorted(by_step)]


def read_iteration_times(path: Path) -> dict[int, IterationTimes]:
    times: dict[int, IterationTimes] = {}
    fallback_times: dict[int, list[datetime]] = {}
    with path.open() as f:
        for line in f:
            match = GCS_LISTING_RE.match(line)
            if not match:
                continue
            gcs_path = match.group("path")
            iter_match = ITERATION_RE.search(gcs_path)
            if not iter_match:
                continue
            step = int(iter_match.group("step"))
            timestamp = parse_time(match.group("time"))
            size = int(match.group("size"))
            info = times.setdefault(step, IterationTimes())
            fallback_times.setdefault(step, []).append(timestamp)
            is_iteration_dir = size == 0 and gcs_path.endswith(f"iteration_{step:06d}/")
            if is_iteration_dir:
                info.start = timestamp if info.start is None else min(info.start, timestamp)
            else:
                info.finish = timestamp if info.finish is None else max(info.finish, timestamp)

    for step, seen_times in fallback_times.items():
        info = times[step]
        if info.start is None:
            info.start = min(seen_times)
        if info.finish is None and len(seen_times) > 1:
            info.finish = max(seen_times)
    return times


def read_restart_markers(path: Path) -> list[dict[str, Any]]:
    markers: list[dict[str, Any]] = []
    previous_model: str | None = None
    for row in sorted(read_jsonl(path), key=lambda item: item.get("batch", 0)):
        state_path = row.get("state_path")
        batch = row.get("batch")
        if not isinstance(state_path, str) or not isinstance(batch, int):
            continue
        match = MODEL_RE.search(state_path)
        if not match:
            continue
        model = match.group("model")
        if previous_model is not None and model != previous_model:
            markers.append(
                {
                    "step": batch - 1,
                    "checkpoint": row.get("name"),
                    "from_model": previous_model,
                    "to_model": model,
                }
            )
        previous_model = model
    return markers


def make_plot(
    metrics: list[dict[str, Any]],
    iteration_times: dict[int, IterationTimes],
    restart_markers: list[dict[str, Any]],
    out_path: Path,
    summary_path: Path | None,
) -> None:
    complete_steps = [
        int(row["step"])
        for row in metrics
        if int(row["step"]) in iteration_times and iteration_times[int(row["step"])].finish is not None
    ]
    if not complete_steps:
        raise SystemExit("No metrics rows could be aligned to completed iteration timestamps")

    first_start = min(
        iteration_times[step].start
        for step in complete_steps
        if iteration_times[step].start is not None
    )
    if first_start is None:
        raise SystemExit("No iteration start timestamps found")

    aligned: list[dict[str, Any]] = []
    for row in metrics:
        step = int(row["step"])
        iter_time = iteration_times.get(step)
        if iter_time is None or iter_time.finish is None:
            continue
        hours = (iter_time.finish - first_start).total_seconds() / 3600.0
        aligned.append(
            {
                "step": step,
                "hours": hours,
                "reward": float(row["env/all/reward/total"]),
                "sampling_min": float(row.get("time/sampling", 0.0)) / 60.0,
                "train_min": float(row.get("time/train_step", 0.0)) / 60.0,
            }
        )

    marker_points: list[dict[str, Any]] = []
    aligned_by_step = {point["step"]: point for point in aligned}
    for marker in restart_markers:
        point = aligned_by_step.get(marker["step"])
        if point is None:
            continue
        marker_with_time = dict(marker)
        iter_time = iteration_times.get(marker["step"])
        marker_time = iter_time.start if iter_time and iter_time.start else iter_time.finish if iter_time else None
        marker_with_time["hours"] = (
            (marker_time - first_start).total_seconds() / 3600.0 if marker_time is not None else point["hours"]
        )
        marker_with_time["reward"] = point["reward"]
        marker_points.append(marker_with_time)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_reward, ax_time) = plt.subplots(1, 2, figsize=(15, 5.5), sharex=True)
    fig.suptitle("Qwen3.5-9B MathRL autoresume timeline")

    epoch_start_hours = []
    for step in complete_steps:
        start = iteration_times[step].start
        if start is not None:
            epoch_start_hours.append((start - first_start).total_seconds() / 3600.0)
    for ax in (ax_reward, ax_time):
        for hour in epoch_start_hours:
            ax.axvline(hour, color="0.8", linewidth=0.5, alpha=0.18, zorder=0)

    hours = [point["hours"] for point in aligned]
    ax_reward.plot(
        hours,
        [point["reward"] for point in aligned],
        marker="o",
        markersize=3,
        linewidth=1.8,
        label="train reward",
    )
    ax_reward.set_title("Train reward")
    ax_reward.set_xlabel("wall-clock hours since first iteration")
    ax_reward.set_ylabel("env/all/reward/total")
    ax_reward.grid(True, alpha=0.25)

    ax_time.plot(
        hours,
        [point["sampling_min"] for point in aligned],
        marker="o",
        markersize=3,
        linewidth=1.6,
        label="sampling",
    )
    ax_time.plot(
        hours,
        [point["train_min"] for point in aligned],
        marker="o",
        markersize=3,
        linewidth=1.6,
        label="train step",
    )
    ax_time.set_title("Per-epoch timing")
    ax_time.set_xlabel("wall-clock hours since first iteration")
    ax_time.set_ylabel("minutes / epoch")
    ax_time.grid(True, alpha=0.25)

    for idx, marker in enumerate(marker_points):
        label = "preempt/resume marker" if idx == 0 else None
        for ax in (ax_reward, ax_time):
            ax.axvline(marker["hours"], color="#d62728", linestyle="--", linewidth=1.1, alpha=0.75, label=label)
            label = None
        ax_reward.text(
            marker["hours"],
            0.98,
            f"p{marker['step']}",
            transform=ax_reward.get_xaxis_transform(),
            color="#b22222",
            fontsize=7,
            rotation=90,
            ha="right",
            va="top",
        )

    ax_reward.legend(loc="upper left")
    ax_time.legend(loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)

    if summary_path is not None:
        last = aligned[-1]
        best = max(aligned, key=lambda point: point["reward"])
        summary = {
            "source_note": "Wall-clock hours use GCS iteration object timestamps. Preemption/resume markers are inferred from checkpoint model-id changes.",
            "num_points": len(aligned),
            "first_step": aligned[0]["step"],
            "last_step": last["step"],
            "first_iteration_start_utc": first_start.isoformat(),
            "last_iteration_finish_utc": max(
                iteration_times[point["step"]].finish for point in aligned if iteration_times[point["step"]].finish
            ).isoformat(),
            "wall_clock_hours": last["hours"],
            "latest": last,
            "best_reward": best,
            "preemption_resume_markers": marker_points,
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MathRL reward and timing over real wall-clock hours")
    parser.add_argument("metrics", type=Path, help="metrics.jsonl")
    parser.add_argument("--iteration-listing", type=Path, required=True, help="Output from gsutil ls -l -r iteration_*/")
    parser.add_argument("--checkpoints", type=Path, required=True, help="checkpoints.jsonl")
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--summary", type=Path, help="Optional summary JSON path")
    args = parser.parse_args()

    make_plot(
        read_latest_metrics(args.metrics),
        read_iteration_times(args.iteration_listing),
        read_restart_markers(args.checkpoints),
        args.out,
        args.summary,
    )


if __name__ == "__main__":
    main()
