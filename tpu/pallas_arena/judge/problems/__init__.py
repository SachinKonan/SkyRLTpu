"""Problem registry: one module per arena task (DESIGN.md slate)."""

from __future__ import annotations

import importlib

_PROBLEM_MODULES = {
    "rmsnorm": "pallas_arena.judge.problems.rmsnorm",
    "splash_attention": "pallas_arena.judge.problems.splash_attention",
    "ragged_paged_attention": "pallas_arena.judge.problems.ragged_paged_attention",
    "megablox_gmm": "pallas_arena.judge.problems.megablox_gmm",
    "flce": "pallas_arena.judge.problems.flce",
    "rg_lru": "pallas_arena.judge.problems.rg_lru",
}


# THE GRADED SLATE: the tasks that become RL environments and carry rewards.
#
# Four kernels, each with a real production implementation to beat:
#   splash_attention        Google's Pallas splash MHA
#   megablox_gmm            megablox grouped matmul (MoE)
#   ragged_paged_attention  JAX's paged-decode attention
#   rg_lru                  recurrentgemma's Pallas LRU scan
#
# DROPPED from the slate (2026-08-17): `flce` and `rmsnorm`. Both carried a
# SINGLE baseline -- no per-shape election -- which is the exact configuration
# that made every historical reward invalid on the other tasks (megablox was
# mistuned 13-55x, splash ~10x). Neither declared `tp_specs`, so neither could
# ever meet the tensor-parallel bar, and both ran the ours-specific reward
# while the rest run general, so their numbers were never comparable anyway.
ARENA_TASKS: tuple[str, ...] = (
    "splash_attention",
    "megablox_gmm",
    "ragged_paged_attention",
    "rg_lru",
)

# `rmsnorm` stays IMPORTABLE but ungraded: it is the CPU battery's fixture
# backbone (HONEST_RMSNORM, WRONG_GRAD_RMSNORM, TIMER_TAMPERER_RMSNORM and the
# rest of tests/candidates.py), and ~40 harness tests grade against it because
# it is the one task cheap enough to run end-to-end on a CPU in seconds.
# Removing the module would delete the harness's own test scaffolding, which is
# not what dropping it from the slate means. `flce` is likewise kept loadable
# so its reference/gradient/adversarial tests still run.
TEST_ONLY_PROBLEMS: tuple[str, ...] = ("rmsnorm", "flce")


def problem_names() -> list[str]:
    """Every LOADABLE problem, graded or not. Use ``ARENA_TASKS`` for the
    graded slate -- anything that hands out reward or defines an RL env wants
    that, not this."""
    return list(_PROBLEM_MODULES)


def arena_tasks() -> list[str]:
    return list(ARENA_TASKS)


def get_problem(name: str):
    if name not in _PROBLEM_MODULES:
        raise KeyError(f"unknown problem {name!r}; known: {sorted(_PROBLEM_MODULES)}")
    return importlib.import_module(_PROBLEM_MODULES[name]).PROBLEM
