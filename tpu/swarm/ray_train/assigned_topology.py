"""Verify only the explicitly assigned TPU partition, never an occupied slice."""
from itertools import product
from math import prod


def verify_partition(records, process_bounds, chip_bounds):
    hosts = tuple(map(int, process_bounds.split(',')))
    chips = tuple(map(int, chip_bounds.split(',')))
    if len(hosts) != 3 or len(chips) != 3 or min(*hosts, *chips) < 1:
        raise ValueError('invalid partition bounds')
    by_rank = {int(r['process_id']): r for r in records}
    if len(by_rank) != len(records) or set(by_rank) != set(range(prod(hosts))):
        raise ValueError('missing or duplicate assigned process')
    expected_blocks = {
        frozenset(product(*(range(p * c, (p + 1) * c) for p, c in zip(position, chips))))
        for position in product(*(range(h) for h in hosts))
    }
    for rank, record in by_rank.items():
        # libtpu discovers physical coordinates independently of process IDs.
        # Require each complete physical host exactly once, without forcing the
        # existing checkpoint/API leader to occupy the mesh's lowest corner.
        actual = [tuple(c) for c in record['coords']]
        block = frozenset(actual)
        if len(actual) != prod(chips) or block not in expected_blocks:
            raise ValueError(f'assigned process {rank} has wrong chip coordinates: {actual}')
        expected_blocks.remove(block)
    if expected_blocks:
        raise ValueError('missing assigned chip block')
    return True
