#!/usr/bin/env python3
"""Line a v4-32 slice's four hosts up with a 1,1,4 process grid.

A remote-only trainer runs TP8 x FSDP2 over all sixteen chips with
``TPU_PROCESS_BOUNDS=1,1,4``: process ``i`` must own the 2x2 chip slab at
``z == i``. SkyPilot rank order does not follow the physical order (a pool
worker reported rank->z ``[1, 2, 3, 0]``, 2026-09-24), so the controller
probes every host, orders the trainer ranks by physical row, as the v4-64 and
v6e-32 splits already do, then re-probes under the real process bounds and
verifies process ``i`` landed on row ``i`` before libtpu is trusted with it.
"""
from __future__ import annotations

import json
import sys


def slice_rows(records: list[dict]) -> list[int]:
    """Validate a full 2x2x4 probe; return the z row of each reporter by process_id."""
    by_process: dict[int, list[tuple[int, int, int]]] = {}
    for record in records:
        process = int(record["process_id"])
        if process in by_process:
            raise ValueError(f"duplicate Sky rank {process}")
        coords = [tuple(int(value) for value in coord) for coord in record["coords"]]
        if len(coords) != 4 or len(set(coords)) != 4 or any(len(c) != 3 for c in coords):
            raise ValueError(f"Sky rank {process} did not report four unique 3D chips: {coords}")
        by_process[process] = coords
    if set(by_process) != set(range(4)):
        raise ValueError(f"expected Sky ranks 0..3, got {sorted(by_process)}")
    expected = {(x, y, z) for x in range(2) for y in range(2) for z in range(4)}
    actual = {coord for coords in by_process.values() for coord in coords}
    if actual != expected:
        raise ValueError("probe did not report the complete v4-32 2x2x4 chip topology")
    rows = []
    for process in range(4):
        zs = {coord[2] for coord in by_process[process]}
        if len(zs) != 1:
            raise ValueError(f"Sky rank {process} spans several z rows: {sorted(zs)}")
        rows.append(next(iter(zs)))
    return rows


def full_slice_rank_order(records: list[dict]) -> list[int]:
    """Sky ranks sorted by physical z row: entry ``i`` is the host on row ``i``.

    ``records`` come from a probe over ``range(4)``, so each ``process_id``
    is that host's Sky rank.
    """
    rows = slice_rows(records)
    return sorted(range(4), key=rows.__getitem__)


def verify_full_slice_order(records: list[dict]) -> list[int]:
    """Require process ``i`` to sit on z row ``i``; return the rows or raise."""
    rows = slice_rows(records)
    if rows != list(range(4)):
        raise ValueError(
            "v4-32 processes are not in physical z order for TPU_PROCESS_BOUNDS=1,1,4: "
            f"process->z is {rows}")
    return rows


def main() -> int:
    records = [json.loads(line) for line in sys.stdin if line.strip()]
    print(json.dumps({"rank_to_z": slice_rows(records), "train_ranks": full_slice_rank_order(records)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
