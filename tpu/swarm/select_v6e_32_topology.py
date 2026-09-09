#!/usr/bin/env python3
"""Select the validated 2x2-host trainer block from a v6e-32 topology probe.

A v6e-32 is 32 chips on 8 hosts of 2x2 chips (a 4x8 or 8x4 chip grid, so a
2x4 or 4x2 host grid). Ray v2 ranks follow SkyPilot's node list, which is NOT
the physical worker order (asia replica 72, 2026-09-09: ranks 0-3 landed on
workers 0, 6, 2, 7 and libtpu aborted the slice with CHIP_DRIVER_ERROR). The
trainer must be a contiguous 2x2 block of hosts declared to libtpu as
TPU_PROCESS_BOUNDS=2,2,1 with CLOUD_TPU_TASK_ID x-fastest, so pick the block
that contains rank 0 (the API/client host) and order it that way; every other
host serves.
"""
from __future__ import annotations

import json
import sys


def host_positions(records: list[dict]) -> dict[int, tuple[int, int]]:
    by_rank: dict[int, tuple[int, int]] = {}
    for record in records:
        rank = int(record["process_id"])
        if rank in by_rank:
            raise ValueError(f"duplicate Sky rank {rank}")
        coords = [tuple(int(v) for v in c) for c in record["coords"]]
        if len(coords) != 4 or len(set(coords)) != 4:
            raise ValueError(f"Sky rank {rank} did not report four unique chips: {coords}")
        xs = sorted({c[0] for c in coords})
        ys = sorted({c[1] for c in coords})
        zs = {c[2] for c in coords} if all(len(c) >= 3 for c in coords) else {0}
        if len(xs) != 2 or len(ys) != 2 or xs[1] - xs[0] != 1 or ys[1] - ys[0] != 1 or len(zs) != 1:
            raise ValueError(f"Sky rank {rank} is not a 2x2x1 host: {coords}")
        if xs[0] % 2 or ys[0] % 2:
            raise ValueError(f"Sky rank {rank} host is not aligned to the 2x2 chip grid: {coords}")
        by_rank[rank] = (xs[0] // 2, ys[0] // 2)
    if set(by_rank) != set(range(8)):
        raise ValueError(f"expected Sky ranks 0..7, got {sorted(by_rank)}")
    positions = set(by_rank.values())
    if len(positions) != 8:
        raise ValueError("two Sky ranks reported the same host position")
    hx = {p[0] for p in positions}
    hy = {p[1] for p in positions}
    if not ({len(hx), len(hy)} == {2, 4} and hx == set(range(len(hx))) and hy == set(range(len(hy)))):
        raise ValueError(f"probe is not a 2x4 or 4x2 host grid: {sorted(positions)}")
    return by_rank


def select_split(records: list[dict]) -> tuple[list[int], list[int]]:
    by_rank = host_positions(records)
    x0, y0 = by_rank[0]
    width = max(p[0] for p in by_rank.values()) + 1
    height = max(p[1] for p in by_rank.values()) + 1
    x1 = x0 + 1 if x0 + 1 < width else x0 - 1
    y1 = y0 + 1 if y0 + 1 < height else y0 - 1
    block = {(x, y) for x in (x0, x1) for y in (y0, y1)}
    at = {pos: rank for rank, pos in by_rank.items()}
    # TPU_PROCESS_BOUNDS=2,2,1: task id = local_x + 2 * local_y.
    xs, ys = sorted((x0, x1)), sorted((y0, y1))
    train = [at[(x, y)] for y in ys for x in xs]
    if 0 not in train or len(block) != 4:
        raise ValueError("could not form a 2x2 host block containing Sky rank 0")
    serving = sorted(set(range(8)) - set(train))
    return train, serving


def main() -> None:
    records = [json.loads(line) for line in sys.stdin if line.strip()]
    train, serving = select_split(records)
    print(f"TRAIN_WORKERS={','.join(map(str, train))}")
    print(f"VLLM_WORKERS={','.join(map(str, serving))}")


if __name__ == "__main__":
    try:
        main()
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid v6e-32 topology probe: {exc}") from exc
