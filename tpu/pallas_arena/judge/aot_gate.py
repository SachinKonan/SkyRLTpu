"""Client-side CPU AOT pre-gate.

Kills the 40-70% of early-RL kernels that don't even compile, at zero chip
cost, BEFORE a candidate is POSTed to the judge:
  1. AST screen (banned imports/calls, pallas_call requirement) — instant;
  2. a forked child (same isolation as real grading) that execs the
     candidate and runs jax.jit(...).lower() on abstract shapes — catches
     python errors, bad signatures, shape/dtype mismatches and pallas
     grid/BlockSpec inconsistencies at trace time. On a TPU host the same
     child continues into the full Mosaic .compile().

Returns (ok, reason). Uses grader.grade(mode="pregate") so the timeout /
RLIMIT_AS / seed-secrecy story is identical to real grading.
"""

from __future__ import annotations

from pallas_arena.judge import grader
from pallas_arena.judge.gates import ast_gate
from pallas_arena.judge.problems import get_problem


def aot_pregate(
    problem_name: str,
    code: str,
    *,
    smoke: bool = False,
    timeout_s: float = 90.0,
    rlimit_gb: float = grader.DEFAULT_RLIMIT_GB,
    enforce_pallas: bool | None = None,
) -> tuple[bool, str]:
    problem = get_problem(problem_name)
    if enforce_pallas is None:
        enforce_pallas = problem.require_pallas
    violations = ast_gate(
        code,
        banned_import_prefixes=problem.all_banned_prefixes,
        banned_call_names=problem.banned_call_names,
        require_pallas=enforce_pallas,
    )
    if violations:
        return False, "; ".join(violations)

    result = grader.grade(
        problem_name,
        code,
        mode="pregate",
        smoke=smoke,
        timeout_s=timeout_s,
        rlimit_gb=rlimit_gb,
        enforce_pallas=enforce_pallas,
        cache=None,
    )
    if result.get("passed"):
        return True, "ok"
    why = "; ".join(result.get("violations", [])) or result.get("error", "pregate failed")
    return False, f"[{result.get('gate', '?')}] {why}"
