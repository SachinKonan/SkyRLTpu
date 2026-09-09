"""v5p-32 two-host trainer selection from a topology probe (job 482, worker 200, 2026-09-09)."""
import random

import pytest

from tpu.swarm.select_v5p_32_topology import select_split


def _records(rank_of_layer):
    records = []
    for z, rank in rank_of_layer.items():
        coords = [[dx, dy, z] for dy in range(2) for dx in range(2)]
        records.append({"process_id": rank, "coords": coords})
    random.Random(rank).shuffle(records)
    return records


@pytest.mark.parametrize("seed", range(8))
def test_pair_contains_rank0_is_z_adjacent_and_z_ordered(seed):
    ranks = list(range(4)); random.Random(seed).shuffle(ranks)
    rank_of = dict(zip(range(4), ranks))
    train, serving = select_split(_records(rank_of))
    layer_of = {r: z for z, r in rank_of.items()}
    assert 0 in train and len(train) == 2 and sorted(train + serving) == list(range(4))
    zs = [layer_of[r] for r in train]
    assert zs[1] - zs[0] == 1  # consecutive layers, task id ascending in z


def test_worker_200_scattered_ranks():
    # Sky ranks 0..3 on layers 0, 2, 1, 3: ranks 0 and 1 were two layers apart
    # and libtpu reported the rank-1 host's four chips as unassigned.
    train, serving = select_split(_records({0: 0, 2: 1, 1: 2, 3: 3}))
    assert train == [0, 2] and serving == [1, 3]


def test_rank0_on_top_layer_pairs_downward():
    train, serving = select_split(_records({3: 0, 0: 1, 1: 2, 2: 3}))
    assert train == [3, 0] and serving == [1, 2]


def test_rejects_non_v5p_shapes():
    with pytest.raises(ValueError):
        select_split([{"process_id": r, "coords": [[2 * r, 0, 0], [2 * r + 1, 0, 0], [2 * r, 1, 0], [2 * r + 1, 1, 0]]}
                      for r in range(4)])
    with pytest.raises(ValueError):
        select_split(_records({0: 0, 1: 1, 2: 2, 3: 3})[:3])


def test_candidates_are_all_adjacent_pairs_head_first():
    from tpu.swarm.select_v5p_32_topology import candidate_splits
    cands = candidate_splits(_records({0: 2, 1: 0, 2: 3, 3: 1}))  # rank 0 on layer 1
    assert len(cands) == 3
    assert [0 in t for t, _ in cands] == [True, True, False]
    assert cands[0] == select_split(_records({0: 2, 1: 0, 2: 3, 3: 1}))
    for train, serving in cands:
        assert sorted(train + serving) == list(range(4)) and len(train) == 2
