#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


DEFAULT_METRICS = (
    "tpu.googleapis.com/accelerator/tensorcore_utilization",
    "tpu.googleapis.com/accelerator/duty_cycle",
    "tpu.googleapis.com/accelerator/memory_bandwidth_utilization",
    "tpu.googleapis.com/accelerator/memory_used",
    "tpu.googleapis.com/accelerator/memory_total",
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def rfc3339(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_rfc3339(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def auth_token() -> str:
    return subprocess.check_output(
        ["gcloud", "auth", "print-access-token"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def point_value(point: dict[str, Any]) -> float | None:
    value = point.get("value", {})
    raw = value.get("doubleValue", value.get("int64Value"))
    if raw is None:
        return None
    return float(raw)


def fetch_time_series(
    *,
    project: str,
    token: str,
    metric: str,
    start: str,
    end: str,
    location: str | None,
) -> list[dict[str, Any]]:
    filters = [f'metric.type="{metric}"']
    if location:
        filters.append(f'resource.labels.location="{location}"')
    params = {
        "filter": " AND ".join(filters),
        "interval.startTime": start,
        "interval.endTime": end,
        "pageSize": "200",
    }
    url = f"https://monitoring.googleapis.com/v3/projects/{project}/timeSeries?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    out: list[dict[str, Any]] = []

    while True:
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=60) as response:
                    data = json.load(response)
                break
            except urllib.error.HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504} or attempt == 5:
                    raise
                time.sleep(2**attempt)
        out.extend(data.get("timeSeries", []))
        token_page = data.get("nextPageToken")
        if not token_page:
            return out
        next_params = dict(params)
        next_params["pageToken"] = token_page
        url = f"https://monitoring.googleapis.com/v3/projects/{project}/timeSeries?{urllib.parse.urlencode(next_params)}"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})


def summarize_metric(metric: str, series: list[dict[str, Any]]) -> dict[str, Any]:
    values: list[float] = []
    latest_values: list[float] = []
    series_summaries: list[dict[str, Any]] = []
    points_by_time: dict[str, list[float]] = {}

    for item in series:
        points = item.get("points", [])
        nums = [value for point in points if (value := point_value(point)) is not None]
        if not nums:
            continue
        for point in points:
            value = point_value(point)
            end_time = point.get("interval", {}).get("endTime")
            if value is None or not end_time:
                continue
            point_dt = parse_rfc3339(end_time)
            minute_dt = point_dt.replace(second=0, microsecond=0)
            points_by_time.setdefault(rfc3339(minute_dt), []).append(value)
        values.extend(nums)
        latest_values.append(nums[0])
        labels = {
            **item.get("resource", {}).get("labels", {}),
            **item.get("metric", {}).get("labels", {}),
        }
        series_summaries.append(
            {
                "labels": labels,
                "num_points": len(nums),
                "latest": nums[0],
                "min": min(nums),
                "mean": sum(nums) / len(nums),
                "max": max(nums),
            }
        )

    summary: dict[str, Any] = {
        "metric": metric,
        "num_series": len(series),
        "num_points": len(values),
        "series": series_summaries,
    }
    if values:
        summary.update(
            {
                "latest_mean": sum(latest_values) / len(latest_values),
                "overall_min": min(values),
                "overall_mean": sum(values) / len(values),
                "overall_max": max(values),
            }
        )
    if points_by_time:
        summary["time_points"] = [
            {
                "end_time": end_time,
                "count": len(point_values),
                "min": min(point_values),
                "mean": sum(point_values) / len(point_values),
                "max": max(point_values),
            }
            for end_time, point_values in sorted(points_by_time.items())
        ]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Cloud TPU utilization metrics from Cloud Monitoring.")
    parser.add_argument("--project", default=None, help="GCP project. Defaults to gcloud config project.")
    parser.add_argument("--location", default="us-east5-a", help="TPU worker location/zone filter.")
    parser.add_argument("--minutes", type=int, default=120, help="Lookback window in minutes.")
    parser.add_argument("--start", help="Explicit interval start time, RFC3339 UTC, e.g. 2026-07-08T03:20:00Z.")
    parser.add_argument("--end", help="Explicit interval end time, RFC3339 UTC, e.g. 2026-07-08T18:15:00Z.")
    parser.add_argument("--metric", action="append", default=[], help="Metric type to fetch. Can be repeated.")
    parser.add_argument("--out", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()

    project = args.project
    if project is None:
        project = subprocess.check_output(
            ["gcloud", "config", "get-value", "project"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    if not project:
        raise SystemExit("No project set. Pass --project or configure gcloud.")

    if bool(args.start) != bool(args.end):
        raise SystemExit("Pass both --start and --end, or neither.")
    if args.start and args.end:
        start_dt = parse_rfc3339(args.start)
        end_dt = parse_rfc3339(args.end)
    else:
        end_dt = utc_now()
        start_dt = end_dt - timedelta(minutes=args.minutes)
    if start_dt >= end_dt:
        raise SystemExit("--start must be before --end")
    start = rfc3339(start_dt)
    end = rfc3339(end_dt)
    token = auth_token()

    result: dict[str, Any] = {
        "project": project,
        "location": args.location,
        "start": start,
        "end": end,
        "metrics": {},
    }

    for metric in args.metric or DEFAULT_METRICS:
        series = fetch_time_series(
            project=project,
            token=token,
            metric=metric,
            start=start,
            end=end,
            location=args.location,
        )
        result["metrics"][metric] = summarize_metric(metric, series)

    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    else:
        try:
            sys.stdout.write(text)
        except BrokenPipeError:
            return


if __name__ == "__main__":
    main()
