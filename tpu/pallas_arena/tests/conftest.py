"""Shared fixtures for the pallas-arena CPU battery.

Runs anywhere jax-CPU runs, but per cluster rules the full battery executes
via sbatch (run_phase1_tests.sbatch), never on the login node.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ARENA_ROOT = Path(__file__).resolve().parents[1]   # tpu/pallas_arena
IMPORT_ROOT = ARENA_ROOT.parent                    # tpu/
REPO_ROOT = IMPORT_ROOT.parent

for p in (str(IMPORT_ROOT), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# force CPU for this battery (and for grader children via inheritance)
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def fast_grade_kwargs():
    """Small-but-real grading settings so each forked child stays ~seconds."""
    return dict(
        smoke=True,
        cases=["tiny"],
        timing_pairs=6,
        timing_warmup=1,
        determinism_runs=5,
        correctness_seeds=2,
        timeout_s=120.0,
        rlimit_gb=8.0,
    )


@pytest.fixture()
def local_cache(tmp_path):
    from pallas_arena.judge.cache import RewardCache

    return RewardCache(str(tmp_path / "reward-cache"))
