"""Public RG-LRU task and verdict translation; no JAX or Discover imports."""
from __future__ import annotations

import ast
import math
from pathlib import Path
from urllib.parse import urlsplit

ARENA = Path(__file__).resolve().parents[1]
ENV_NAMES = ("recurrent_gemma", "recurrentgemma", "pallas_rglru", "rg_lru")


def validate_queue_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("RecurrentGemma requires ARENA_QUEUE_URL=http(s)://<judge-queue>:<port>")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("ARENA_QUEUE_URL must not contain credentials, query, or fragment")
    return url.rstrip("/")


def public_contract():
    """Read literal case declarations and oracle source without importing JAX.

    The training client does not need a TPU compiler. Keeping the prompt tied
    to the judge's source avoids a second, drifting copy of the recurrence.
    """
    source = (ARENA / "judge/problems/rg_lru.py").read_text()
    tree = ast.parse(source)
    oracle = "import jax\nimport jax.numpy as jnp\n\n" + "\n\n".join(
        ast.get_source_segment(source, node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("_apply_reset", "rg_lru_scan_reference")
    )
    problem = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RGLRUProblem")
    shapes = next(n for n in problem.body if isinstance(n, ast.FunctionDef) and n.name == "shape_cases")
    cases = []
    for node in ast.walk(shapes):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "ShapeCase":
            flags = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
            if flags.get("probe") and not flags.get("tp"):
                cases.append((ast.literal_eval(node.args[0]), ast.literal_eval(node.args[1])))
    if not cases or "def rg_lru_scan_reference" not in oracle:
        raise RuntimeError("RG-LRU public contract changed; update the RL prompt reader")
    return oracle, cases


def build_prompt() -> str:
    oracle, cases = public_contract()
    shapes = "\n".join(f"- {name}: {dims}" for name, dims in cases)
    return f"""Optimize RecurrentGemma's RG-LRU gated diagonal recurrent scan in JAX/Pallas on TPU.
This task covers the recurrence with precomputed gates, not the entire model.

Define kernel(x, a, reset) -> h. x is [b,t,d] bfloat16; a is [b,t,d]
float32 in [0,1); reset is [b,t] bool; h must be [b,t,d] float32.
The initial hidden state is zero. At a reset, h_t = x_t: no earlier state
crosses the boundary. This is the exact correctness oracle:

```python
{oracle}
```

One complete program must work on all these shapes, including ragged time:
{shapes}

Forward AND backward are required. The judge differentiates kernel with
respect to both x and a using a nonuniform cotangent. Implement custom_vjp
if needed; do not stop gradients or substitute a different recurrence.
Tests include hidden random inputs, near-unit gates, zero gates, and dense
resets. Correctness uses calibrated per-element maximum and tail errors.

A real pl.pallas_call is mandatory. Pure XLA scans are not submissions.
Do not import recurrentgemma, the judge, or a library scan implementation.
Return self-contained source, with all helper definitions included; imports
from an external `lib` module are not available. Do not access files/network,
inspect the grader, select code by hidden data, or change the timing machinery.

The TPU judge compares against the fastest calibrated honest baseline per
shape (production RecurrentGemma scan or XLA associative scan). The raw score
is the geometric mean of baseline/candidate latency ratios over forward and
backward, including the ragged holdout. Above 1 means faster. The judge applies
its noise-floor rule to produce the training reward; invalid kernels earn 0.

Use float32 recurrence accumulation. TPU VMEM is limited: tile time and feature
axes rather than materializing an entire long sequence in scratch. The last
two tile dimensions normally align to (8,128). Use refs for writes and JAX
control flow for traced conditions. The provided seed uses a time-blocked
parallel affine scan and a custom backward; you may restructure either.

Return the final program in exactly one fenced ```python code block containing kernel(x, a, reset), its forward and backward implementation, all required imports, and all helper function definitions.
Do not split the program across blocks or include separate usage examples, partial snippets, alternative implementations, or additional code blocks.
Close the code fence after the complete program and end your answer.
"""


class ArenaInfrastructureError(RuntimeError):
    """No trustworthy candidate verdict was produced."""


def translate_verdict(result: dict) -> dict:
    from pallas_arena.judge.observation import build_observation

    if result.get("ok") is not True or result.get("judge_fault") or result.get("gate") in ("harness", "worker", "judge_fault"):
        raise ArenaInfrastructureError(f"Arena judge failed: {result.get('gate')}: {result.get('violations', [])}")
    if type(result.get("passed")) is not bool:
        raise ArenaInfrastructureError("Arena verdict is missing passed")
    passed = result["passed"]
    reward = float(result.get("reward_with_bwd", result.get("reward", float("nan"))))
    score = 0.0
    if passed:
        expected = {name for name, _ in public_contract()[1]}
        forward = {**result.get("per_case", {}), **result.get("holdout", {})}
        backward = result.get("grad_scores", {})
        if (set(forward) != expected or set(backward) != expected
                or result.get("excluded_cases") or result.get("skipped_cases")
                or result.get("n_bwd_factors") != len(expected)
                or result.get("baseline_mode") != "all"
                or "reward_with_bwd" not in result):
            raise ArenaInfrastructureError("Arena returned an incomplete or mismatched forward/backward grading contract")
        factors = [float(x) for x in (*forward.values(), *backward.values())]
        if not all(math.isfinite(x) and x > 0 for x in factors):
            raise ArenaInfrastructureError("Arena returned an invalid latency ratio")
        score = math.exp(sum(map(math.log, factors)) / len(factors))
    if not math.isfinite(reward) or reward < 0 or not math.isfinite(score) or (passed and score <= 0):
        raise ArenaInfrastructureError("Arena verdict has invalid reward/score")
    if not passed and reward != 0:
        raise ArenaInfrastructureError("Rejected kernel received a nonzero reward")
    observation = build_observation({**result, "reward": reward}, max_chars=350 if passed else 480)
    if passed:
        slowest = min(backward, key=backward.get)
        observation += (f"\nCombined raw={score:.4f}x; worst backward "
                        f"{slowest}={backward[slowest]:.4f}x")
    return dict(reward=reward, correctness=float(passed), raw_score=score,
                result_construction=None, msg=observation, stdout=observation,
                metrics={"arena/cache_hit": float(bool(result.get("cache_hit")))})
