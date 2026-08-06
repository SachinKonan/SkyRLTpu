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


def problem_names() -> list[str]:
    return list(_PROBLEM_MODULES)


def get_problem(name: str):
    if name not in _PROBLEM_MODULES:
        raise KeyError(f"unknown problem {name!r}; known: {sorted(_PROBLEM_MODULES)}")
    return importlib.import_module(_PROBLEM_MODULES[name]).PROBLEM
