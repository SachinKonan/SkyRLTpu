"""Minimal SimpleTES-style prompts + failure-pattern mining.

Prompt = plain task + eval config + seed-group programs/scores/feedback
         + code-mined failure patterns. NO hand-crafted strategy hints.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

# --- plain task instructions (contract kept because our grader requires run()) ---
TASK = {
    "erdos": """Find a step function h: [0,2] -> [0,1] that MINIMIZES the overlap
C5 = max(np.correlate(h, 1-h, mode='full') * dx), dx = 2/n_points.
Constraints: 0 <= h[i] <= 1 and sum(h)*dx = 1 exactly. Lower is better.
Return a ```python block defining run(seed=42, budget_s=1000, **kwargs) ->
(h_values, c5_bound, n_points). Helpers top-level; numpy/scipy/cvxpy allowed;
no file/network IO. evaluate_erdos_solution() and initial_h_values are pre-imported.""",
    "ac1": """Find a non-negative step function f on [-1/4,1/4] (list of floats) that
MINIMIZES 2*n*max(f*f) / sum(f)^2 where f*f is the discrete autoconvolution.
Lower is better. Return a ```python block defining propose_candidate() -> list[float].
Helpers top-level; numpy/scipy allowed; no file/network IO.""",
    "ac2": """Construct a non-negative function f on [-1/4,1/4] (list of floats) to
MAXIMIZE ||f*f||_2^2 / (||f*f||_1 * ||f*f||_inf). Higher is better.
Return a ```python block defining construct_function() -> list[float].
Helpers top-level; numpy/scipy allowed; no file/network IO.""",
}

EVAL_CFG = """SCORING -- read carefully. The ONLY score that counts comes from the MCP tools of the
`grader` server: grade_fast(solution) (~60s screen) and grade_full(solution) (~1000s,
official). Pass the whole ```python block as `solution`. Local runs in your sandbox are
fine for development but count for NOTHING -- a rollout that never calls the grader
scores zero. You have {n_fast} grade_fast + {n_full} grade_full calls and roughly
{wall_min} minutes of wall clock: call grade_fast EARLY (within the first few minutes),
iterate, and ALWAYS spend your grade_full on your final answer well before time is up.
Your solution must COMPUTE its construction within the stated budget; embedding a
precomputed construction as data (arrays/base64) is invalid and will be rejected."""


def mine_failure_patterns(grade_log: Path, k: int = 4, window: int = 250) -> str:
    """Top-k recent failure reasons from grade_log.jsonl -> negative constraints."""
    if not grade_log.exists():
        return ""
    rows = grade_log.read_text().splitlines()[-window:]
    fails = Counter()
    for line in rows:
        try:
            r = json.loads(line)
        except Exception:
            continue
        if not r.get("valid"):
            d = (r.get("detail") or "invalid")[:110]
            d = re.sub(r"0x[0-9a-f]+|\d{4,}", "#", d)
            fails[d] += 1
    if not fails:
        return ""
    lines = [f"- ({c}x) {msg}" for msg, c in fails.most_common(k)]
    return "Recent failure patterns -- avoid these:\n" + "\n".join(lines)


def executor_prompt(problem: str, group_nodes: list[dict], grade_log: Path,
                    n_fast: int = 5, n_full: int = 1, wall_min: int = 25) -> str:
    parts = [TASK[problem], "",
             EVAL_CFG.format(n_fast=n_fast, n_full=n_full, wall_min=wall_min), ""]
    if len(group_nodes) == 1:
        parts.append("Improve on this prior solution (score shown). Meaningful change, not a re-run:")
    else:
        parts.append(f"These {len(group_nodes)} prior solutions take DIFFERENT approaches. "
                     "Combine their complementary strengths into one better solution:")
    for nd in group_nodes:
        fb = f" | feedback: {nd['feedback']}" if nd.get("feedback") else ""
        parts.append(f"\n### prior solution [node {nd['id']}] score={nd['r']}{fb}\n"
                     f"```python\n{nd['program']}\n```")
    fp = mine_failure_patterns(grade_log)
    if fp:
        parts += ["", fp]
    parts += ["", "Finish by writing your single best solution to solution.txt as one ```python block."]
    return "\n".join(parts)


def aggregator_prompt(problem: str, n_groups: int, max_group: int) -> str:
    direction = "minimize" if problem in ("erdos", "ac1") else "maximize"
    return f"""You are the search coordinator for an evaluation-driven discovery loop on: {TASK[problem].splitlines()[0]} ({direction}).

A graph of all prior attempts is available via the `store` tools (overview, top_k,
node, relatives). Node fields: r = own score; U = descendant-backed value (U better
than r means the node spawned winners -- a fertile stepping-stone even if mediocre
itself); n_used = times already seeded.

Analyze the graph, then pick exactly {n_groups} SEED GROUPS (each 1..{max_group} node ids):
- a 1-node group -> that solution gets refined;
- a multi-node group -> those solutions get MERGED; pick approaches whose ideas are
  complementary (different mechanisms, different failure modes), never near-duplicates.
Maximize coverage of genuinely different approaches across your {n_groups} groups; use U to
resurrect fertile stepping-stones; prefer under-used nodes over re-polishing one lineage.

Submit via submit_seed_groups(groups_json). Hard constraints are code-enforced; if
REJECTED, adjust and resubmit until ACCEPTED, then end the session."""
