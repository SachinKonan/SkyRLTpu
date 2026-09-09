#!/usr/bin/env python3
"""Select the z-adjacent two-host trainer pair from a v5p-32 topology probe.

A v5p-32 is 16 chips on 4 hosts of 2x2 chips stacked along z (a 2x2x4 chip
torus; host k holds the 2x2x1 layer at z=k). Ray v2 ranks follow SkyPilot's
node list, which is NOT the physical layer order (job 482, worker 200,
2026-09-09: Sky ranks 0 and 1 sat on non-adjacent layers and libtpu failed the
1,1,2 process mesh with "Mesh build was incomplete"). A two-host trainer must
be two consecutive layers declared to libtpu as TPU_PROCESS_BOUNDS=1,1,2 with
CLOUD_TPU_TASK_ID ascending in z, so pick the layer next to rank 0 (the
API/client host) and order the pair by z; the other two hosts serve.
"""
from __future__ import annotations

import json
import sys

HOSTS = 4


def host_layers(records: list[dict]) -> dict[int, int]:
    by_rank: dict[int, int] = {}
    for record in records:
        rank = int(record["process_id"])
        if rank in by_rank:
            raise ValueError(f"duplicate Sky rank {rank}")
        coords = [tuple(int(v) for v in c) for c in record["coords"]]
        if len(coords) != 4 or len(set(coords)) != 4 or any(len(c) != 3 for c in coords):
            raise ValueError(f"Sky rank {rank} did not report four unique 3-d chips: {coords}")
        xs = sorted({c[0] for c in coords})
        ys = sorted({c[1] for c in coords})
        zs = {c[2] for c in coords}
        if len(xs) != 2 or len(ys) != 2 or xs[1] - xs[0] != 1 or ys[1] - ys[0] != 1 or len(zs) != 1:
            raise ValueError(f"Sky rank {rank} is not a 2x2x1 host layer: {coords}")
        by_rank[rank] = zs.pop()
    if set(by_rank) != set(range(HOSTS)):
        raise ValueError(f"expected Sky ranks 0..{HOSTS - 1}, got {sorted(by_rank)}")
    if set(by_rank.values()) != set(range(HOSTS)):
        raise ValueError(f"probe is not a 2x2x4 stack of host layers: z={sorted(by_rank.values())}")
    return by_rank


def candidate_splits(records: list[dict]) -> list[tuple[list[int], list[int]]]:
    """Every z-adjacent host pair, best first (the pair holding Sky rank 0,
    then the rest), so the controller can fall back when a host cannot join
    a multi-host mesh. Task id = local_z; the trainer API lives on the pair's
    first host."""
    by_rank = host_layers(records)
    z0 = by_rank[0]
    at = {z: rank for rank, z in by_rank.items()}
    ordered = []
    for z in range(HOSTS - 1):
        train = [at[z], at[z + 1]]
        serving = sorted(set(range(HOSTS)) - set(train))
        key = (0 if 0 in train else 1, abs(z - z0))
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
        raise SystemExit(f"invalid v5p-32 topology probe: {exc}") from exc
