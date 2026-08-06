# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""RQ1 T3: one sol-xhigh orchestrator invocation with a 100-draw budget. Two arms:

  --arm E  (Meta devserver)  multi_agent_v2: the orchestrator spawns up to --budget subagent
           calls (model pinned via spawn_agent model override), <= --concurrency concurrent.
           Subagents test locally (full shell, no grader) and each finishes with its own
           submit() on the capture MCP.
  --arm F  (neuronic)        no subagents: the capture MCP additionally exposes
           sample_farm(instruction, n) which draws completions from the OSS farm (best T2
           config), out of the same total budget. The orchestrator steers draws, verifies
           candidates locally, and submits the best itself.

E vs F is the matched head-to-head: identical orchestrator model/effort/prompt skeleton and
draw budget; only the draw type differs (agent rollouts vs OSS completions).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import preflight
from collect_t1 import base_env, free_port, load_pack, start_capture

HERE = Path(__file__).resolve().parent

# CDC-style aggressive orchestration (lifted native fc46 +47% in exp-1), grader references
# replaced by local verification. Kept structurally identical between arms E and F.
ORCH_COMMON = """Manage the search with these heuristics:
- Begin with a genuinely diverse portfolio: substantially different formulations, algorithms, \
heuristics, representations, decompositions, relaxations, and computational sanity checks.
- Do not reveal the currently favored approach when tasking new draws -- preserve independence \
early so the search does not converge on one attractive-but-mediocre idea.
- Maintain an explicit registry of approach families (grouped by underlying idea, not wording). \
If many draws converge to one family, redirect toward underexplored ones.
- Keep several incompatible approaches alive; cross-pollinate only after independent draws have \
developed them far enough to expose real strengths and gaps.
- Verify adversarially and LOCALLY: there is no grading service. Compile and run candidates \
yourself in your workspace (you have full shell), measure them against the seed score stated in \
the problem, and reject untested claims, status reports, and vague optimism.
- Do not stop after a first wave fails. Keep searching until your budget or time is nearly \
exhausted, then make sure the single strongest LOCALLY-VALIDATED program has been submitted."""

ORCH_E = """You have a large team of agents available through multiagent v2. Use them
aggressively and dynamically. Your budget is {n} SUBAGENT CALLS IN TOTAL over the whole session
(counting every role -- generators, verifiers, adversarial checkers, refiners); at most {conc}
may run CONCURRENTLY. Spend close to the full budget: an unused agent is wasted search.
Do NOT use a fixed assignment like "N agents for approach X."

{common}

Spawn subagents with spawn_agent(task_name, message, model, reasoning_effort) using
model="{sub_model}". Subagents inherit your tools. EVERY subagent that produces a candidate
program must finish by calling the `capture` MCP tool submit(program, approach, insight,
session=<its task_name>) exactly once -- an unsubmitted candidate is lost. You may also submit
your own final synthesis (once)."""

ORCH_F = """You direct a farm of open-source reasoning models through the `capture` MCP tool
sample_farm(instruction, n): each draw is one independent completion (long thinking + one code
block) from the farm model, which receives the full problem prompt plus your instruction. Your
budget is {n} DRAWS IN TOTAL; batches of up to 10. Spend close to the full budget: an unused
draw is wasted search. Use `instruction` to steer -- an algorithm family to explore, a concrete
program of yours to improve, constraints to respect; empty means the plain problem.

{common}

You are the only agent: triage the returned code yourself, test promising candidates locally,
iterate with steered draws, and finish by calling submit(program, approach, insight,
session="orchestrator") exactly once with the best LOCALLY-VALIDATED program (yours or a
refined farm draw)."""

WRAP = """You are orchestrating a search for a better solution to the problem below.

{orchestration}

You have ~{wall_min} minutes of wall clock in total.

## Problem (the seed program and its production score are stated inside)
{pack}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["E", "F"])
    ap.add_argument("--problem", required=True)
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=25)
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--effort", default="xhigh")
    ap.add_argument("--subagent-model", default="gpt-5.6-sol")
    ap.add_argument("--wall", type=int, default=14400)
    ap.add_argument("--site", choices=["default", "neuronic", "auto"], default="auto")
    ap.add_argument("--out", required=True)
    # ---- arm F ----
    ap.add_argument("--farm-url", default=None)
    ap.add_argument("--farm-key", default="EMPTY")
    ap.add_argument("--farm-model", default=None)
    ap.add_argument("--farm-max-tokens", type=int, default=28000)
    args = ap.parse_args()
    if args.arm == "F" and not (args.farm_url and args.farm_model):
        raise SystemExit("--arm F requires --farm-url and --farm-model")

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    pack, meta = load_pack(args.problem)
    prof = preflight.resolve_site(args.site, out / "preflight")
    port = free_port()
    ch = preflight.write_codex_home(
        out, landlock=prof["landlock"], long_provider=prof["long_provider"],
        multi_agent=(args.concurrency if args.arm == "E" else 0),
        mcp_url=f"http://127.0.0.1:{port}/mcp")

    extra = []
    if args.arm == "F":
        pf = out / "farm_prompt.md"
        pf.write_text((HERE / "data" / args.problem / "prompt_completion.md").read_text())
        extra = ["--farm-url", args.farm_url, "--farm-key", args.farm_key,
                 "--farm-model", args.farm_model, "--farm-budget", str(args.budget),
                 "--farm-prompt-file", str(pf),
                 "--farm-max-tokens", str(args.farm_max_tokens)]
    cap = start_capture(out, port, extra=extra)

    orch = (ORCH_E.format(n=args.budget, conc=args.concurrency, common=ORCH_COMMON,
                          sub_model=args.subagent_model)
            if args.arm == "E" else
            ORCH_F.format(n=args.budget, common=ORCH_COMMON))
    prompt = WRAP.format(orchestration=orch, wall_min=args.wall // 60, pack=pack)
    wd = out / "orch"
    wd.mkdir(exist_ok=True)
    (wd / "prompt.txt").write_text(prompt)
    (out / "manifest.json").write_text(json.dumps({
        "cell": args.arm, "problem": args.problem, "budget": args.budget,
        "concurrency": args.concurrency, "model": args.model, "effort": args.effort,
        "subagent_model": args.subagent_model if args.arm == "E" else None,
        "farm_model": args.farm_model, "wall": args.wall, "site_profile": prof,
        "seed_score": meta.get("seed_score"), "started": time.strftime("%F %T")}, indent=2))

    cmd = ["codex", "exec", "--strict-config", "-m", args.model,
           "-c", f"model_reasoning_effort={args.effort}",
           "-s", "workspace-write", "-c", "approval_policy=never",
           "--json", "-C", str(wd), "-o", str(wd / "final.txt")]
    print(f"[t3] arm {args.arm} on {args.problem}: budget={args.budget}, wall={args.wall}s",
          flush=True)
    t0 = time.time()
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=open(wd / "events.jsonl", "w"),
                         stderr=subprocess.STDOUT, env=base_env(ch), cwd=str(wd), text=True)
    p.stdin.write(prompt)
    p.stdin.close()
    try:
        p.wait(timeout=args.wall + 300)
    except subprocess.TimeoutExpired:
        p.kill()
        print("[t3] wall-killed", flush=True)
    finally:
        cap.terminate()
    subf = out / "submissions.jsonl"
    n = sum(1 for l in subf.read_text().splitlines() if l.strip()) if subf.exists() else 0
    print(f"[t3] DONE in {int(time.time()-t0)}s: {n} submissions -> {out}", flush=True)


if __name__ == "__main__":
    main()
