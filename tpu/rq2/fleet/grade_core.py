"""Standalone grading core -- `_grade` without the MCP dependency, importable on farm hosts.

The canonical grader (tpu/distill_ablation/grading_mcp.py) imports FastMCP at module level, so
it cannot run on a host whose venv only carries the evaluation stack. This is the same logic
with roots resolved from GRADER_HOME, so the SAME file works on neuronic (absolute checkout
paths) and on a farm host (the unpacked bundle).

Layout expected under GRADER_HOME (see make_grader_bundle.sh):
    discover/            third_party/discover subset  (ttt_discover + python examples)
    frontiercs/          frontiercs discover subset    (examples/frontier_algo + problems)
"""
from __future__ import annotations

import importlib
import os
import re
import sys
import time

HOME = os.environ.get("GRADER_HOME", "")
if HOME:
    MAIN = os.path.join(HOME, "discover")
    FRONTIER = os.path.join(HOME, "frontiercs")
else:                                   # neuronic fallback: the real checkouts
    MAIN = "/n/fs/vision-mix/sk7524/SkyRLTpu/third_party/discover"
    FRONTIER = "/n/fs/vision-mix/sk7524/SkyRLTpu-frontiercs/third_party/discover"

FAST_BUDGET_DEFAULT = 10
FAST_WALL, FULL_WALL = 240, 1100

PROBLEMS = {
    "erdos": (MAIN, "examples.erdos_min_overlap.env", "ErdosMinOverlapEnv", "", "python", False),
    "ac1":   (MAIN, "examples.ac_inequalities.env", "AutoCorrInequalityEnv", "ac1", "python", False),
    "fc46":  (FRONTIER, "examples.frontier_algo.env", "FrontierAlgoEnv", "46", "cpp", True),
    "fc159": (FRONTIER, "examples.frontier_algo.env", "FrontierAlgoEnv", "159", "cpp", True),
    "ud":    (MAIN, "examples.frontier_erdos_ud.env", "FrontierErdosUDEnv", "65536", "python", True),
}


def _strip_fence(sol):
    m = re.search(r"```(?:python|cpp|c\+\+)?\s*([\s\S]*?)\s*```", sol or "")
    return m.group(1) if m else (sol or "")


def _load_env(root, mod, cls):
    """Both trees ship an `examples` package; purge and re-point per call so each problem's env
    comes from ITS root (a worker process grades one problem at a time)."""
    for k in [k for k in sys.modules if k == "examples" or k.startswith("examples.")]:
        del sys.modules[k]
    while root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    return getattr(importlib.import_module(mod), cls)


def grade(problem, solution, fast=True, fast_budget=None, base_construction=None,
          logdir="/tmp/grade_core", backend="local"):
    """-> {score, valid, detail, secs}. backend="ray" uses the league evaluator's payload mode
    (TTD_RAY_PAYLOAD=1, RAY_ADDRESS env) to fan grading across the slice's Ray cluster."""
    root, mod, cls, ptype, lang, _mx = PROBLEMS[problem]
    fb = int(fast_budget or FAST_BUDGET_DEFAULT)
    t0 = time.time()
    try:
        E = _load_env(root, mod, cls)
        code = _strip_fence(solution)
        if lang == "python":
            if fast:
                # annotated-contract-aware rewrite (the RQ1 gate bug): budget_s: int = 1000 too
                code = re.sub(r"budget_s\s*(?::\s*[A-Za-z_][\w\[\], ]*\s*)?=\s*\d+(?:\.\d+)?",
                              f"budget_s={fb}", code)
            payload = f"```python\n{code}\n```"
            eval_timeout = FAST_WALL if fast else FULL_WALL
        else:
            payload = code
            eval_timeout = 150
            os.environ["TTD_FCALGO_MAX_CASES"] = "1" if fast else "0"
        state = E.create_initial_state(ptype)
        if base_construction is not None:
            state.construction = base_construction
        ev = E.reward_function(problem_type=ptype, log_dir=logdir, num_cpus_per_task=1,
                               eval_timeout=eval_timeout, eval_backend=backend)
        out = ev.get_reward(payload, state)
        valid = out.get("correctness", 0) > 0
        return {"score": out.get("raw_score") if valid else None, "valid": bool(valid),
                "detail": (out.get("msg") or "")[:300], "secs": round(time.time() - t0, 1)}
    except Exception as e:
        return {"score": None, "valid": False, "detail": f"{type(e).__name__}: {e}"[:300],
                "secs": round(time.time() - t0, 1)}
