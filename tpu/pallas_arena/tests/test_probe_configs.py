"""probe/configs.py names must resolve against the LIVE shape cases.

The v6e-8 bring-up (job 3715217) died mid-run on
KeyError: "megablox_gmm: no shape case named 'probe-m8192-e8-8x7b'" -- the
provenance renames changed case names and the TASK_CASES sweep list silently
kept the old ones. Name drift between the config and the problems is a CPU
fact; it must fail here, not on a paid QR.
"""

from __future__ import annotations

import pytest

from pallas_arena.judge.problems import get_problem
from pallas_arena.probe.configs import TASK_CASES


@pytest.mark.parametrize("task", sorted(TASK_CASES))
def test_every_swept_case_name_resolves(task):
    p = get_problem(task)
    for name in TASK_CASES[task]:
        p.case_by_name(name)  # raises KeyError on drift


@pytest.mark.parametrize("task", sorted(TASK_CASES))
def test_swept_sets_have_a_holdout(task):
    p = get_problem(task)
    assert any(p.case_by_name(n).holdout for n in TASK_CASES[task]), task


def test_every_declared_tp_case_validates():
    """Every tp case's declared width must divide its sharded axes, at BOTH
    widths (8 for v6e-8, 4 for v5p-8's 4 JAX devices -- v5p counts
    TensorCores in the type name but exposes megacore chips as devices).
    tp_declared_width raises on an indivisible declaration; this makes that a
    battery failure instead of a judge boot failure."""
    from pallas_arena.judge.problems import ARENA_TASKS

    seen = 0
    for task in ARENA_TASKS:
        p = get_problem(task)
        for case in p.shape_cases():
            if not case.tp:
                continue
            w = p.tp_declared_width(case)
            assert w == case.tp, (task, case.name)
            shard = p.abstract_inputs_tp(case, w)
            full = p.abstract_inputs(case)
            assert any(
                tuple(a.shape) != tuple(b.shape) for a, b in zip(shard, full)
            ), f"{task}/{case.name}: TP case shards nothing"
            seen += 1
    assert seen >= 8, f"expected tp cases across the slate, found {seen}"
