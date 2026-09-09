"""v6e-32 trainer block selection from a topology probe (asia replica 72, 2026-09-09)."""
import itertools
import random

import pytest

from tpu.swarm.select_v6e_32_topology import select_split


def _records(host_grid, rank_of_position):
    """host_grid=(W,H) hosts; each host holds a 2x2 chip block at (2hx, 2hy)."""
    records = []
    for hx in range(host_grid[0]):
        for hy in range(host_grid[1]):
            rank = rank_of_position[(hx, hy)]
            coords = [[2 * hx + dx, 2 * hy + dy, 0] for dy in range(2) for dx in range(2)]
            records.append({"process_id": rank, "coords": coords})
    random.Random(rank).shuffle(records)
    return records


@pytest.mark.parametrize("grid", [(4, 2), (2, 4)])
@pytest.mark.parametrize("seed", range(6))
def test_block_contains_rank0_is_contiguous_and_x_fastest(grid, seed):
    positions = [(x, y) for x in range(grid[0]) for y in range(grid[1])]
    ranks = list(range(8)); random.Random(seed).shuffle(ranks)
    rank_of = dict(zip(positions, ranks))
    train, serving = select_split(_records(grid, rank_of))
    pos_of = {r: p for p, r in rank_of.items()}
    assert 0 in train and len(train) == 4 and sorted(train + serving) == list(range(8))
    block = [pos_of[r] for r in train]
    xs = sorted({p[0] for p in block}); ys = sorted({p[1] for p in block})
    assert len(xs) == 2 and len(ys) == 2 and xs[1] - xs[0] == 1 and ys[1] - ys[0] == 1
    # task id = local_x + 2*local_y
    assert block == [(x, y) for y in ys for x in xs]


def test_replica_72_scattered_ranks_get_a_real_block():
    # Sky ranks 0..7 sat on physical workers 0,6,2,7,1,3,5,4 (a 4x2 host grid,
    # worker w at (w % 4, w // 4)); ranks 0-3 were not a block.
    worker_of_rank = {0: 0, 1: 6, 2: 2, 3: 7, 4: 1, 5: 3, 6: 5, 7: 4}
    rank_of = {(w % 4, w // 4): r for r, w in worker_of_rank.items()}
    train, serving = select_split(_records((4, 2), rank_of))
    # Block = workers 0,1,4,5 == host positions (0,0),(1,0),(0,1),(1,1), whose
    # Sky ranks are 0,4,7,6; task ids run x-fastest within the block.
    assert train == [0, 4, 7, 6] and serving == [1, 2, 3, 5]


def test_rejects_non_v6e_shapes():
    with pytest.raises(ValueError):
        select_split([{"process_id": r, "coords": [[r, 0, 0], [r, 1, 0], [r, 2, 0], [r, 3, 0]]} for r in range(8)])
    with pytest.raises(ValueError):
        select_split(_records((4, 2), {(x, y): x + 4 * y for x in range(4) for y in range(2)})[:7])


def test_candidates_cover_every_block_head_first():
    from tpu.swarm.select_v6e_32_topology import candidate_splits
    # asia worker 65 (2026-09-09): 2x4 host grid, rank 0 at (0,1); host (1,1)
    # fails the mesh check, so the usable block is rows 2-3 without the head.
    rank_of = {(0, 1): 0, (1, 1): 2, (0, 2): 1, (1, 2): 5, (1, 3): 6, (0, 3): 4, (1, 0): 3, (0, 0): 7}
    cands = candidate_splits(_records((2, 4), rank_of))
    assert len(cands) == 3  # 2-wide grid: three row pairs
    assert 0 in cands[0][0] and cands[0][0] == select_split(_records((2, 4), rank_of))[0]
    assert all(0 in t for t, _ in cands[:2]) and 0 not in cands[2][0]
    assert cands[2][0] == [rank_of[(0, 2)], rank_of[(1, 2)], rank_of[(0, 3)], rank_of[(1, 3)]]
    for train, serving in cands:
        assert sorted(train + serving) == list(range(8)) and len(train) == 4
