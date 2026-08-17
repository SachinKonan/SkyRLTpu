"""What is every arena task's score DENOMINATOR actually bound to?

A reward of 1.19 means nothing until you know what the 1.0 is. This calls
`problem.baseline(...)` on a smoke shape for all five tasks on whatever host it
runs on, reports the `baseline_impl` label each one records, and -- separately
from the runtime answer -- statically reports whether the production-kernel
branch is even REACHABLE (two of the five raise `BaselineUnavailable`
unconditionally inside their own `try`, so they can never bind the real
kernel no matter what is installed).

Usage:  JAX_PLATFORMS=cpu python -m pallas_arena.verify.audit_baselines
"""

from __future__ import annotations

import inspect
import json
import re
import sys

import jax

from pallas_arena.judge.problems import get_problem, problem_names

# The graded slate is defined ONCE, in the registry -- a second copy here
# silently drifted (it still listed flce after the slate changed).
from pallas_arena.judge.problems import ARENA_TASKS  # noqa: F401

# Which shape case to call baseline() on per task: the smallest smoke case that
# exists, so a CPU host can actually run it.
_SMOKE_PREF = ("tiny", "tiny-ragged", "smoke")


def _smoke_case(problem):
    cases = problem.shape_cases()
    for name in _SMOKE_PREF:
        for c in cases:
            if c.name == name:
                return c
    for c in cases:
        if getattr(c, "smoke", False):
            return c
    return cases[0]


def _static_reachability(problem) -> dict:
    """Can the production branch of baseline() ever be taken on this host?

    A `raise BaselineUnavailable(...)` that is NOT inside an `if` and sits
    after the production import means the fallback is hard-wired: the label
    "we try the real kernel first" is false.
    """
    try:
        src = inspect.getsource(type(problem).baseline)
    except Exception:
        return {"source": False}
    raises = re.findall(r"raise\s+BaselineUnavailable\(([^\n]*)", src)
    # An unconditional raise: a `raise BaselineUnavailable` line whose
    # indentation is the same as the preceding import block inside `try`, with
    # no `if` on any line between the production import and the raise.
    hardwired = False
    lines = src.splitlines()
    for i, ln in enumerate(lines):
        if "raise BaselineUnavailable" not in ln:
            continue
        # scan backwards to the enclosing try:, looking for a conditional
        cond = False
        for prev in reversed(lines[:i]):
            s = prev.strip()
            if s.startswith("try:"):
                break
            if s.startswith(("if ", "elif ", "@pl.when", "else:")):
                cond = True
                break
        if not cond:
            hardwired = True
    return {
        "source": True,
        "n_baseline_unavailable_raises": len(raises),
        "raise_messages": [r.strip().rstrip(")").strip('"').strip("'") for r in raises],
        "production_branch_unreachable": hardwired,
    }


def main() -> int:
    out = {}
    print(f"backend: {jax.default_backend()}  devices: {jax.devices()}")
    for name in [n for n in ARENA_TASKS if n in problem_names()]:
        problem = get_problem(name)
        rec = {"task": name}
        rec.update(_static_reachability(problem))
        case = _smoke_case(problem)
        rec["case"] = case.name
        try:
            key = jax.random.PRNGKey(0)
            inputs = problem.make_inputs(key, case)
            type(problem).baseline_impl = "?"
            o = problem.baseline(*inputs)
            jax.block_until_ready(o)
            rec["ran"] = True
            rec["baseline_impl_after_call"] = getattr(type(problem), "baseline_impl", None)
        except Exception as e:  # BaselineUnavailable on a CPU host is a valid answer
            rec["ran"] = False
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["baseline_impl_after_call"] = getattr(type(problem), "baseline_impl", None)
        out[name] = rec
        print(json.dumps(rec, default=str))
    print("\n=== SUMMARY ===")
    for name, rec in out.items():
        flag = "HARD-WIRED FALLBACK" if rec.get("production_branch_unreachable") else "tries production first"
        print(f"{name:24s} impl={str(rec.get('baseline_impl_after_call')):28s} {flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
