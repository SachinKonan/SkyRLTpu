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


def candidate_splits(records: list[dict]) -> list[tuple[list[int], list[int]]]:
    """Every contiguous 2x2 host block, best first.

    The block containing Sky rank 0 (the executor head) comes first; the
    others follow so the controller can fall back when a host refuses to
    join a multi-host mesh (asia workers 74/65, 2026-09-09: one host per
    worker fails libtpu's ``index_on_host() == i`` check only inside a
    process grid, and with a two-column host grid every block that holds
    the head also holds that host's row-mate). Task ids run x-fastest
    within each block; the trainer API lives on the block's first host.
    """
    by_rank = host_positions(records)
    width = max(p[0] for p in by_rank.values()) + 1
    height = max(p[1] for p in by_rank.values()) + 1
    at = {pos: rank for rank, pos in by_rank.items()}
    x0, y0 = by_rank[0]
    ordered = []
    for ys0 in range(height - 1):
        for xs0 in range(width - 1):
            xs, ys = (xs0, xs0 + 1), (ys0, ys0 + 1)
            train = [at[(x, y)] for y in ys for x in xs]
            serving = sorted(set(range(8)) - set(train))
            holds_head = 0 in train
            # Prefer the head's block, then blocks nearest the head.
            key = (0 if holds_head else 1, abs(ys0 - y0) + abs(xs0 - x0))
            ordered.append((key, train, serving))
    ordered.sort(key=lambda item: item[0])
    return [(train, serving) for _, train, serving in ordered]


def select_split(records: list[dict]) -> tuple[list[int], list[int]]:
    return candidate_splits(records)[0]


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
