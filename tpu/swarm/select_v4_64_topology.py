#!/usr/bin/env python3
"""Select a valid 4-host training row from a v4-64 topology probe."""

from __future__ import annotations

import json
import sys
from collections import defaultdict


def select_split(records: list[dict]) -> tuple[list[int], list[int]]:
    by_rank: dict[int, list[tuple[int, int, int]]] = {}
    for record in records:
        rank = int(record["process_id"])
        if rank in by_rank:
            raise ValueError(f"duplicate Sky rank {rank}")
        coords = [tuple(int(value) for value in coord) for coord in record["coords"]]
        if len(coords) != 4 or len(set(coords)) != 4:
            raise ValueError(f"Sky rank {rank} did not report four unique chips: {coords}")
        if any(len(coord) != 3 for coord in coords):
            raise ValueError(f"Sky rank {rank} reported non-3D coordinates: {coords}")
        by_rank[rank] = coords

    expected_ranks = set(range(8))
    if set(by_rank) != expected_ranks:
        raise ValueError(f"expected Sky ranks 0..7, got {sorted(by_rank)}")

    expected_coords = {
        (x, y, z) for x in range(2) for y in range(4) for z in range(4)
    }
    actual_coords = {coord for coords in by_rank.values() for coord in coords}
    if actual_coords != expected_coords:
        raise ValueError("probe did not report the complete v4-64 2x4x4 chip topology")

    rows: dict[tuple[tuple[int, ...], tuple[int, ...]], list[tuple[int, int]]] = (
        defaultdict(list)
    )
    for rank, coords in by_rank.items():
        xs = tuple(sorted({coord[0] for coord in coords}))
        ys = tuple(sorted({coord[1] for coord in coords}))
        zs = {coord[2] for coord in coords}
        if len(xs) != 2 or len(ys) != 2 or len(zs) != 1:
            raise ValueError(f"Sky rank {rank} is not a 2x2x1 host: {coords}")
        rows[(xs, ys)].append((next(iter(zs)), rank))

    if len(rows) != 2 or any(len(row) != 4 for row in rows.values()):
        raise ValueError(f"expected two four-host rows, got {dict(rows)}")

    train_row = next((row for row in rows.values() if any(rank == 0 for _, rank in row)), None)
    if train_row is None:
        raise ValueError("Sky rank 0 is not present in either physical row")
    train_row.sort()
    if [z for z, _ in train_row] != [0, 1, 2, 3]:
        raise ValueError(f"trainer row does not span z=0..3: {train_row}")

    ordered = [rank for _, rank in train_row]
    rank_zero_index = ordered.index(0)
    train = ordered[rank_zero_index:] + ordered[:rank_zero_index]
    serving = sorted(expected_ranks - set(train))
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
        raise SystemExit(f"invalid v4-64 topology probe: {exc}") from exc
