# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp>=1.9,<2"]
# ///
"""`submit_plan` -- the ONLY tool the orchestrator gets, and the thing that makes the
orchestrated arms budget-comparable to the simple-map arms.

In exp-1 an orchestrator offered 100 subagent calls spent 3. Under-spending is fatal to this
study: simple-map always spends exactly n, so an orchestrated arm that spends a fraction is not
losing on protocol, it is losing on compute, and the comparison becomes unreadable. Capping at
the tool layer prevents OVERspend but cannot compel spend -- so instead the orchestrator's
deliverable IS the allocation, and a plan whose group sizes do not total the budget EXACTLY is
rejected. Under-spend becomes structurally impossible while the orchestrator keeps full freedom
over the thing actually under test: how many distinct instructions, and how much budget each gets.

The plan arrives as a FILE PATH, not an inline argument: at n=500 a plan with per-group
instructions is large, the agent is already working in a workspace beside the solutions MD, and
the file is a durable artifact for the audit trail.

One server per (cell, step); budget is fixed at startup and the process exits with the step.
If no valid plan lands before the driver's turn cap, the budget goes UNSPENT and is recorded --
no auto-fill. That step is a no-op for the cell and must stay visible rather than silently
shrinking it.

  uv run plan_mcp.py --port P --logdir DIR --budget 500
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

# NB: MODULE-level import. `from __future__ import annotations` stringifies the tool signature
# and FastMCP resolves it against module globals -- importing Context inside main() makes every
# @srv.tool() fail with InvalidSignature.
from mcp.server.fastmcp import Context, FastMCP

MAX_INSTRUCTION = 4000


def validate(raw, budget):
    """-> (allocations, None) | (None, reason). Ordered so the agent gets the FIRST real
    problem, not a cascade."""
    if not isinstance(raw, list):
        return None, f"top level must be a JSON array of groups, got {type(raw).__name__}"
    if not raw:
        return None, "plan is empty: it must contain at least one group"
    out = []
    for i, g in enumerate(raw):
        if not isinstance(g, dict):
            return None, f"group {i} must be an object with keys 'instruction' and 'n'"
        extra = set(g) - {"instruction", "n"}
        if extra:
            return None, f"group {i} has unexpected key(s): {sorted(extra)}. Only 'instruction' and 'n' are allowed"
        if "instruction" not in g or "n" not in g:
            return None, f"group {i} is missing {'instruction' if 'instruction' not in g else 'n'}"
        if not isinstance(g["instruction"], str) or not g["instruction"].strip():
            return None, f"group {i}: 'instruction' must be a non-empty string"
        if isinstance(g["n"], bool) or not isinstance(g["n"], int):
            return None, f"group {i}: 'n' must be an integer, got {type(g['n']).__name__}"
        if g["n"] < 1:
            return None, f"group {i}: 'n' must be >= 1, got {g['n']}"
        out.append({"instruction": g["instruction"][:MAX_INSTRUCTION], "n": g["n"]})
    total = sum(g["n"] for g in out)
    if total != budget:
        d = total - budget
        return None, (f"group sizes total {total} but the budget is exactly {budget} "
                      f"({'over' if d > 0 else 'under'} by {abs(d)}). "
                      f"Adjust the group sizes so they sum to {budget} and resubmit.")
    return out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--budget", type=int, required=True)
    args = ap.parse_args()

    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    attempts_f = logdir / "plan_attempts.jsonl"
    accepted_f = logdir / "plan_accepted.json"
    lock = threading.Lock()
    state = {"accepted": False, "attempts": 0}

    def log(rec):
        with lock, open(attempts_f, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    srv = FastMCP("planner", host="127.0.0.1", port=args.port)

    @srv.tool()
    async def submit_plan(path: str, ctx: Context = None) -> dict:
        """SUBMIT YOUR ALLOCATION and finish. Write a JSON file, then pass its path here.

        The file must be a JSON array of groups:
            [{"instruction": "<what this group of sub-agents should try>", "n": <int>}, ...]

        Every group runs `n` sub-agents given that instruction alongside the problem state.
        REQUIREMENT: the group sizes must sum to EXACTLY your budget. A plan that totals
        anything else is rejected with the reason, and you should fix it and resubmit.
        You choose how many groups there are and how the budget is divided between them.

        Returns {"accepted": bool, "detail": str, "budget": int}."""
        with lock:
            state["attempts"] += 1
            attempt = state["attempts"]
            if state["accepted"]:
                return {"accepted": False, "budget": args.budget,
                        "detail": "REJECTED: a plan was already accepted for this step."}

        def reject(reason):
            log({"attempt": attempt, "path": path, "accepted": False, "reason": reason,
                 "ts": round(time.time(), 1)})
            return {"accepted": False, "budget": args.budget, "detail": f"REJECTED: {reason}"}

        p = Path(path)
        if not p.is_file():
            return reject(f"no readable file at '{path}'. Write the plan to disk first, "
                          f"then pass its absolute path.")
        try:
            raw = json.loads(p.read_text())
        except Exception as e:
            return reject(f"'{path}' is not valid JSON: {e}")

        allocations, why = validate(raw, args.budget)
        if why:
            return reject(why)

        with lock:
            state["accepted"] = True
        accepted_f.write_text(json.dumps(
            {"allocations": allocations, "budget": args.budget, "attempts": attempt,
             "source_path": str(p), "ts": round(time.time(), 1)}, indent=2))
        log({"attempt": attempt, "path": path, "accepted": True,
             "groups": len(allocations), "total": args.budget, "ts": round(time.time(), 1)})
        return {"accepted": True, "budget": args.budget,
                "detail": (f"Plan accepted: {len(allocations)} groups totalling {args.budget} "
                           f"sub-agents. You are done -- end the session.")}

    srv.run(transport="streamable-http")


if __name__ == "__main__":
    main()
