# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp>=1.9,<2", "httpx>=0.27"]
# ///
"""Capture-only MCP server for RQ1 sample collection. PORTABLE: no repo imports, run with
`uv run capture_mcp.py` anywhere.

Agents in RQ1 never touch our grader -- they test locally and hand back exactly one solution:

  submit(program, approach, insight, session)   the ONLY capture channel. One accepted
                                                submission per agent (keyed on the transport's
                                                mcp-session-id header, which is 1:1 with the
                                                agent connection and unforgeable by the agent).

With --farm-url the server additionally exposes (arm F only):

  sample_farm(instruction, n)   draw n completions from the OSS farm (vLLM OpenAI endpoint),
                                debiting a global draw budget. Full raw texts are saved to
                                farm_raw/; the tool returns extracted code + a thinking tail.

Everything is logged: submissions.jsonl, identity.jsonl, farm_calls.jsonl.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
import time
from pathlib import Path

# NB: must be a MODULE-level import. `from __future__ import annotations` turns the
# tool signatures into strings, and FastMCP resolves them against module globals --
# importing Context inside main() makes every @srv.tool() fail with InvalidSignature.
from mcp.server.fastmcp import Context, FastMCP


def _sha(s: str) -> str:
    return hashlib.sha1((s or "").encode()).hexdigest()[:12]


def _strip_fence(t: str) -> str:
    m = re.findall(r"```(?:python|cpp|c\+\+)?\s*([\s\S]*?)\s*```", t or "")
    return m[-1] if m else (t or "")


def _agent_key(ctx, claimed):
    """Per-AGENT identity. Subagents all inherit the orchestrator's prompt, so they pass the
    SAME `session` string; the MCP transport however gives each agent its own connection with
    a distinct `mcp-session-id` header (measured 1:1). Key budgets off the header."""
    try:
        req = getattr(ctx.request_context, "request", None)
        if req is not None:
            sid = dict(getattr(req, "headers", {}) or {}).get("mcp-session-id")
            if sid:
                return f"mcp:{sid}"
    except Exception:
        pass
    try:
        return f"conn:{hex(id(ctx.session))[-8:]}"
    except Exception:
        return claimed or "unknown"


def _identity(ctx):
    info = {}
    try:
        info["conn"] = hex(id(ctx.session))[-8:]
    except Exception:
        pass
    try:
        req = getattr(ctx.request_context, "request", None)
        if req is not None:
            h = dict(getattr(req, "headers", {}) or {})
            info["hdrs"] = {k: v[:40] for k, v in h.items()
                            if k.lower() in ("mcp-session-id", "user-agent", "x-agent-id")}
    except Exception:
        pass
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--max-submits-per-agent", type=int, default=1)
    # ---- arm F: farm proxy ----
    ap.add_argument("--farm-url", default=None, help="vLLM OpenAI base url; enables sample_farm")
    ap.add_argument("--farm-key", default="EMPTY")
    ap.add_argument("--farm-model", default=None)
    ap.add_argument("--farm-budget", type=int, default=100, help="total draws allowed")
    ap.add_argument("--farm-prompt-file", default=None,
                    help="base completion prompt; sample_farm appends the instruction to it")
    ap.add_argument("--farm-max-tokens", type=int, default=28000)
    ap.add_argument("--farm-temperature", type=float, default=1.0)
    args = ap.parse_args()

    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    soldir = logdir / "solutions"
    soldir.mkdir(exist_ok=True)
    subf = logdir / "submissions.jsonl"
    identf = logdir / "identity.jsonl"
    lock = threading.Lock()
    submitted: dict[str, int] = {}

    def _log(path, rec):
        with lock, open(path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def _ident(session, ctx, tool):
        if ctx is None:
            return
        try:
            _log(identf, {"tool": tool, "claimed_session": session, **_identity(ctx),
                          "ts": round(time.time(), 1)})
        except Exception:
            pass

    srv = FastMCP("capture", host="127.0.0.1", port=args.port)

    @srv.tool()
    async def submit(program: str, approach: str, insight: str, session: str = "",
                     ctx: Context = None) -> dict:
        """FINISH YOUR TASK: record your single final program plus a two-line reflection.
        Call this EXACTLY ONCE, when you are confident in your best program -- there is no
        grading service here; your program is graded later by the organizers at the production
        budget stated in the problem, so test it locally yourself first.
          program:  your complete final program (source text; a ``` fence is fine)
          approach: one line -- what idea this solution tried
          insight:  one line -- what should be learned for future improvement
        Returns {"accepted": bool, "detail": str}."""
        _ident(session, ctx, "submit")
        akey = _agent_key(ctx, session) if ctx is not None else (session or "unknown")
        body = _strip_fence(program)
        if not body.strip():
            return {"accepted": False, "detail": "REJECTED: empty program."}
        h = _sha(body)
        with lock:
            n = submitted.get(akey, 0)
            if n >= args.max_submits_per_agent:
                return {"accepted": False,
                        "detail": "REJECTED: this agent already submitted. One submission each."}
            submitted[akey] = n + 1
        (soldir / f"{h}.txt").write_text(body)
        _log(subf, {"session": session, "agent_key": akey, "sol_hash": h,
                    "approach": (approach or "")[:600], "insight": (insight or "")[:600],
                    "source": "submit", "ts": round(time.time(), 1)})
        return {"accepted": True,
                "detail": f"Submission {h} recorded. You are done -- end the session."}

    if args.farm_url:
        import httpx
        base_prompt = (Path(args.farm_prompt_file).read_text()
                       if args.farm_prompt_file else "")
        farm_raw = logdir / "farm_raw"
        farm_raw.mkdir(exist_ok=True)
        farmf = logdir / "farm_calls.jsonl"
        budget = {"left": args.farm_budget, "i": 0}

        @srv.tool()
        async def sample_farm(instruction: str = "", n: int = 5, ctx: Context = None) -> dict:
            """Draw n completions (1..10) from the open-source model farm. The farm model gets the
            full problem prompt; `instruction` is APPENDED to it (use it to steer: e.g. a specific
            algorithm family to try, or a prior program to improve -- empty means the plain
            problem). Each draw is one independent completion (thinking + a final code block).
            Draws come out of your TOTAL budget stated in your instructions; the reply tells you
            how many remain. Slow: ~1-5 minutes per batch.
            Returns {"samples": [{"id", "code", "think_tail"}], "budget_left": int}."""
            _ident("", ctx, "sample_farm")
            n = max(1, min(int(n), 10))
            with lock:
                if budget["left"] <= 0:
                    return {"samples": [], "budget_left": 0,
                            "detail": "farm budget exhausted -- submit your best result."}
                n = min(n, budget["left"])
                budget["left"] -= n
                first = budget["i"]
                budget["i"] += n
            prompt = base_prompt + (("\n\n## Additional instruction from the orchestrator\n"
                                     + instruction) if instruction.strip() else "")
            out = []
            async with httpx.AsyncClient(timeout=1800) as cli:
                r = await cli.post(
                    f"{args.farm_url.rstrip('/')}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {args.farm_key}"},
                    json={"model": args.farm_model, "n": n,
                          "temperature": args.farm_temperature,
                          "max_tokens": args.farm_max_tokens,
                          "messages": [{"role": "user", "content": prompt}]})
                r.raise_for_status()
                data = r.json()
            for j, ch in enumerate(data.get("choices", [])):
                msg = ch.get("message", {})
                think = msg.get("reasoning_content") or ""
                text = msg.get("content") or ""
                sid = f"farm_{first + j:03d}"
                (farm_raw / f"{sid}.json").write_text(json.dumps(
                    {"id": sid, "instruction": instruction, "think": think, "text": text}))
                code = _strip_fence(text)
                out.append({"id": sid, "code": code,
                            "think_tail": (think or text)[-1500:]})
            _log(farmf, {"instruction": instruction[:200], "n": n,
                         "budget_left": budget["left"], "ts": round(time.time(), 1)})
            return {"samples": out, "budget_left": budget["left"]}

    srv.run(transport="streamable-http")


if __name__ == "__main__":
    main()
