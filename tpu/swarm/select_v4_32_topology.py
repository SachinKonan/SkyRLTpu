#!/usr/bin/env python3
"""Verify that a v4-32 slice's four hosts line up with a 1,1,4 process grid.

A remote-only trainer runs TP8 x FSDP2 over all sixteen chips with
``TPU_PROCESS_BOUNDS=1,1,4``: process ``i`` must own the 2x2 chip slab at
``z == i``. SkyPilot rank order is not guaranteed to follow the physical
order, so the controller probes every host and fails fast with a clear
message instead of dying inside libtpu.
"""
from __future__ import annotations

import json
import sys


def verify_full_slice_order(records: list[dict]) -> list[int]:
    """Return the physical z coordinate per Sky rank, or raise ValueError."""
    by_rank: dict[int, list[tuple[int, int, int]]] = {}
    for record in records:
        rank = int(record["process_id"])
        if rank in by_rank:
            raise ValueError(f"duplicate Sky rank {rank}")
        coords = [tuple(int(value) for value in coord) for coord in record["coords"]]
        if len(coords) != 4 or len(set(coords)) != 4 or any(len(c) != 3 for c in coords):
            raise ValueError(f"Sky rank {rank} did not report four unique 3D chips: {coords}")
        by_rank[rank] = coords
    if set(by_rank) != set(range(4)):
        raise ValueError(f"expected Sky ranks 0..3, got {sorted(by_rank)}")
    expected = {(x, y, z) for x in range(2) for y in range(2) for z in range(4)}
    actual = {coord for coords in by_rank.values() for coord in coords}
    if actual != expected:
        raise ValueError("probe did not report the complete v4-32 2x2x4 chip topology")
    order = []
    for rank in range(4):
        zs = {coord[2] for coord in by_rank[rank]}
        if len(zs) != 1:
            raise ValueError(f"Sky rank {rank} spans several z rows: {sorted(zs)}")
        order.append(next(iter(zs)))
    if order != list(range(4)):
        raise ValueError(
            "v4-32 hosts are not in physical z order for TPU_PROCESS_BOUNDS=1,1,4: "
            f"rank->z is {order}; relaunch so SkyPilot ranks follow the slice order")
    return order


def main() -> int:
    records = [json.loads(line) for line in sys.stdin if line.strip()]
    print(json.dumps({"rank_to_z": verify_full_slice_order(records)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
