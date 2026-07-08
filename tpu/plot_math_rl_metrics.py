#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Any


DEFAULT_SERIES = (
    "test/env/all/correct",
    "env/all/correct",
    "test/env/all/reward/total",
    "env/all/reward/total",
)

DERIVED_SERIES = (
    "derived/batch_generated_tokens_per_sec",
    "derived/trajectory_generated_tokens_per_sec",
    "env/all/ac_tokens_per_turn",
    "env/all/ob_tokens_per_turn",
    "time/sampling",
    "time/train_step",
)

TENSORCORE_METRICS = (
    "tpu.googleapis.com/accelerator/tensorcore_utilization",
    "tpu.googleapis.com/tpu/mxu/utilization",
)
MEMORY_USED_METRIC = "tpu.googleapis.com/accelerator/memory_used"
MEMORY_TOTAL_METRIC = "tpu.googleapis.com/accelerator/memory_total"

LABELS = {
    "test/env/all/correct": "eval correct",
    "env/all/correct": "train correct",
    "test/env/all/reward/total": "eval reward",
    "env/all/reward/total": "train reward",
    "derived/batch_generated_tokens_per_sec": "batch generated tok/s",
    "derived/trajectory_generated_tokens_per_sec": "per-trajectory generated tok/s",
    "env/all/ac_tokens_per_turn": "avg generated tokens",
    "env/all/ob_tokens_per_turn": "avg prompt tokens",
    "time/sampling": "sampling wall time (s)",
    "time/train_step": "train step time (s)",
    "derived/tpu_tensorcore_utilization": "tensor core util (%)",
    "derived/tpu_memory_utilization": "HBM memory util (%)",
}


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


def add_derived_metrics(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        total_ac_tokens = row.get("env/all/total_ac_tokens")
        sampling_seconds = row.get("time/sampling")
        ac_tokens_per_turn = row.get("env/all/ac_tokens_per_turn")
        sample_latency = row.get("time/policy_sample:mean")

        if isinstance(total_ac_tokens, (int, float)) and isinstance(sampling_seconds, (int, float)) and sampling_seconds:
            row["derived/batch_generated_tokens_per_sec"] = float(total_ac_tokens) / float(sampling_seconds)
        if isinstance(ac_tokens_per_turn, (int, float)) and isinstance(sample_latency, (int, float)) and sample_latency:
            row["derived/trajectory_generated_tokens_per_sec"] = float(ac_tokens_per_turn) / float(sample_latency)


def numeric_pairs(rows: list[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)):
            xs.append(int(row["step"]))
            ys.append(float(value))
    return xs, ys


def summarize_series(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    xs, ys = numeric_pairs(rows, key)
    if not ys:
        return None
    return {
        "num_points": len(ys),
        "last_step": xs[-1],
        "last": ys[-1],
        "mean": sum(ys) / len(ys),
        "min": min(ys),
        "max": max(ys),
        "max_step": xs[ys.index(max(ys))],
    }


def parse_rfc3339(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_monitoring(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text())


def metric_time_points(monitoring: dict[str, Any] | None, metric_names: tuple[str, ...]) -> list[tuple[datetime, float]]:
    if not monitoring:
        return []
    metrics = monitoring.get("metrics", {})
    for metric_name in metric_names:
        metric = metrics.get(metric_name)
        if not metric:
            continue
        points: list[tuple[datetime, float]] = []
        for point in metric.get("time_points", []):
            end_time = point.get("end_time")
            value = point.get("mean")
            if not end_time or not isinstance(value, (int, float)):
                continue
            points.append((parse_rfc3339(end_time), float(value)))
        if points:
            return points
    return []


def memory_utilization_points(monitoring: dict[str, Any] | None) -> list[tuple[datetime, float]]:
    if not monitoring:
        return []
    metrics = monitoring.get("metrics", {})
    used_points = metric_time_points(monitoring, (MEMORY_USED_METRIC,))
    total_points = metric_time_points(monitoring, (MEMORY_TOTAL_METRIC,))
    if not used_points or not total_points:
        return []

    total_by_time = {end_time: value for end_time, value in total_points if value}
    points: list[tuple[datetime, float]] = []
    for end_time, used in used_points:
        total = total_by_time.get(end_time)
        if total:
            points.append((end_time, 100.0 * used / total))
    return points


def step_end_times(run_dir: Path | None, rows: list[dict[str, Any]]) -> dict[int, datetime]:
    if run_dir is None:
        return {}

    end_times: dict[int, datetime] = {}
    for row in rows:
        step = int(row["step"])
        iteration_dir = run_dir / f"iteration_{step:06d}"
        if not iteration_dir.is_dir():
            continue
        mtimes = [p.stat().st_mtime for p in iteration_dir.iterdir() if p.is_file()]
        if not mtimes:
            mtimes = [iteration_dir.stat().st_mtime]
        end_times[step] = datetime.fromtimestamp(max(mtimes), timezone.utc)
    return end_times


def align_points_to_steps(
    rows: list[dict[str, Any]],
    end_times: dict[int, datetime],
    points: list[tuple[datetime, float]],
    output_key: str,
) -> None:
    if not end_times or not points:
        return

    points = sorted(points)
    for row in rows:
        step = int(row["step"])
        end_time = end_times.get(step)
        total_seconds = row.get("time/total")
        if not end_time or not isinstance(total_seconds, (int, float)):
            continue
        start_time = end_time - timedelta(seconds=float(total_seconds))
        values = [value for point_time, value in points if start_time <= point_time <= end_time]
        if values:
            row[output_key] = sum(values) / len(values)


def add_monitoring_metrics(rows: list[dict[str, Any]], monitoring: dict[str, Any] | None, run_dir: Path | None) -> None:
    end_times = step_end_times(run_dir, rows)
    align_points_to_steps(
        rows,
        end_times,
        metric_time_points(monitoring, TENSORCORE_METRICS),
        "derived/tpu_tensorcore_utilization",
    )
    align_points_to_steps(
        rows,
        end_times,
        memory_utilization_points(monitoring),
        "derived/tpu_memory_utilization",
    )


def write_summary(
    rows: list[dict[str, Any]],
    keys: list[str],
    path: Path,
    *,
    monitoring: dict[str, Any] | None = None,
) -> None:
    summary: dict[str, Any] = {
        "num_rows": len(rows),
        "first_step": rows[0]["step"],
        "last_step": rows[-1]["step"],
        "series": {},
    }
    for key in keys + list(DERIVED_SERIES) + [
        "derived/tpu_tensorcore_utilization",
        "derived/tpu_memory_utilization",
    ]:
        stats = summarize_series(rows, key)
        if stats is not None:
            summary["series"][key] = stats
    if monitoring:
        summary["monitoring"] = {
            "project": monitoring.get("project"),
            "location": monitoring.get("location"),
            "start": monitoring.get("start"),
            "end": monitoring.get("end"),
            "metrics": {},
        }
        for metric_name, metric in monitoring.get("metrics", {}).items():
            summary["monitoring"]["metrics"][metric_name] = {
                key: metric[key]
                for key in ("num_series", "num_points", "overall_mean", "overall_max", "latest_mean")
                if key in metric
            }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def plot_lines(ax: Any, rows: list[dict[str, Any]], keys: list[str]) -> bool:
    plotted = False
    for key in keys:
        xs, ys = numeric_pairs(rows, key)
        if not ys:
            continue
        ax.plot(xs, ys, marker="o", linewidth=1.6, markersize=2.8, label=LABELS.get(key, key))
        plotted = True
    if plotted:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    return plotted


def plot_generation_throughput(ax: Any, rows: list[dict[str, Any]]) -> None:
    batch_xs, batch_ys = numeric_pairs(rows, "derived/batch_generated_tokens_per_sec")
    traj_xs, traj_ys = numeric_pairs(rows, "derived/trajectory_generated_tokens_per_sec")
    if not batch_ys and not traj_ys:
        add_missing_text(ax, "generation throughput unavailable")
        return

    lines = []
    labels = []
    if batch_ys:
        line = ax.plot(
            batch_xs,
            batch_ys,
            marker="o",
            linewidth=1.6,
            markersize=2.8,
            color="tab:blue",
            label=LABELS["derived/batch_generated_tokens_per_sec"],
        )[0]
        lines.append(line)
        labels.append(line.get_label())
        ax.set_ylabel("batch tokens / second", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")

    if traj_ys:
        twin_ax = ax.twinx()
        line = twin_ax.plot(
            traj_xs,
            traj_ys,
            marker="o",
            linewidth=1.6,
            markersize=2.8,
            color="tab:orange",
            label=LABELS["derived/trajectory_generated_tokens_per_sec"],
        )[0]
        lines.append(line)
        labels.append(line.get_label())
        twin_ax.set_ylabel("per-trajectory tokens / second", color="tab:orange")
        twin_ax.tick_params(axis="y", labelcolor="tab:orange")

    ax.grid(True, alpha=0.25)
    ax.legend(lines, labels, fontsize=8, loc="best")


def add_missing_text(ax: Any, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot reward/correctness curves from math-RL metrics.jsonl")
    parser.add_argument("metrics", type=Path, help="Path to metrics.jsonl")
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--summary", type=Path, help="Optional summary JSON path")
    parser.add_argument("--monitoring", type=Path, help="Optional TPU monitoring JSON from tpu/fetch_tpu_monitoring.py")
    parser.add_argument("--run-dir", type=Path, help="Optional run directory with iteration_* mtimes for utilization alignment")
    parser.add_argument(
        "--series",
        action="append",
        default=[],
        help="Metric key to plot. Can be repeated. Defaults to common math-RL reward/correct keys.",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.metrics)
    add_derived_metrics(rows)
    monitoring = load_monitoring(args.monitoring)
    add_monitoring_metrics(rows, monitoring, args.run_dir)

    keys = args.series or list(DEFAULT_SERIES)
    plotted_keys = [key for key in keys if numeric_pairs(rows, key)[1]]
    if not plotted_keys:
        available = sorted(k for row in rows for k, v in row.items() if isinstance(v, (int, float)))
        raise SystemExit(f"No requested series found. Available numeric keys include: {available[:80]}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    flat_axes = axes.ravel()

    plot_lines(flat_axes[0], rows, plotted_keys)
    flat_axes[0].set_title("Reward / Correctness")
    flat_axes[0].set_ylabel("metric")

    plot_generation_throughput(flat_axes[1], rows)
    flat_axes[1].set_title("Generation Throughput")

    plot_lines(flat_axes[2], rows, ["env/all/ac_tokens_per_turn", "env/all/ob_tokens_per_turn"])
    flat_axes[2].set_title("Average Sequence Length")
    flat_axes[2].set_ylabel("tokens / trajectory")

    plot_lines(flat_axes[3], rows, ["time/sampling", "time/train_step"])
    flat_axes[3].set_title("Step Timing")
    flat_axes[3].set_ylabel("seconds")

    if plot_lines(
        flat_axes[4],
        rows,
        ["derived/tpu_tensorcore_utilization", "derived/tpu_memory_utilization"],
    ):
        flat_axes[4].set_ylabel("percent")
    else:
        add_missing_text(flat_axes[4], "TPU utilization unavailable\npass --monitoring and --run-dir")
    flat_axes[4].set_title("TPU Utilization")

    add_missing_text(
        flat_axes[5],
        "Derived from metrics.jsonl:\n"
        "batch tok/s = total generated tokens / sampling wall time\n"
        "per-trajectory tok/s = avg generated tokens / avg sample latency",
    )
    flat_axes[5].set_title("Diagnostics Notes")

    for ax in flat_axes[4:]:
        ax.set_xlabel("step")
    fig.suptitle("Qwen3.5-9B MATH RL", fontsize=16)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)

    if args.summary:
        write_summary(rows, plotted_keys, args.summary, monitoring=monitoring)


if __name__ == "__main__":
    main()
