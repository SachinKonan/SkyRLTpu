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
