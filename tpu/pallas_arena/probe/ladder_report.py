"""Ladder report: which rung first produced WORKING code WITH ROOM TO IMPROVE.

The trap this file exists to avoid is the one the arena already fell into:
`tailored|flce` passed 16/16 with a within-group reward spread of 0.0042
against judge noise floors of 0.0069/0.0158. By pass rate it was the best cell
ever measured; as an RL environment it was worthless, because every candidate
converged and the differences it would have trained on were timing jitter.

So a cell here is ranked on a verdict, not a rate:

  NO CODE        nothing passed the judge -- the rung did not produce a
                 working kernel at all.
  NO GRADIENT    candidates PASS, but no complete group's reward spread clears
                 the judge's own measured noise floor for that task. Working
                 code, frozen answer. This is a FAILURE for our purposes and is
                 reported as one.
  SIGNAL         at least one complete group PASSes and spreads further than
                 the noise floor. This is the only outcome that is a win.

Two spreads are reported, because they say different things:
  * `spread`      max-min over the WHOLE group, which a 6/16-pass group makes
                  large for free (some zeros, some ones);
  * `pass spread` max-min over the PASSING candidates only, which is the
                  question "once it works, is there still anything to learn?".
A cell with 32/32 PASS and a pass-spread under the floor is the `tailored`
trap wearing a new prompt, and it is called out by name.

Usage:
  python -m pallas_arena.probe.ladder_report --jsonl x.jsonl --boot-report b.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

RUNGS = ("p1", "p3", "p4")


def _judged(r: dict) -> bool:
    return r.get("stage") in ("pregate", "judge") and r.get("gate") not in (
        None, "unjudged", "pregate-only",
    )


def _reward(r: dict) -> float:
    v = r.get("reward")
    return float(v) if isinstance(v, (int, float)) else 0.0


def _f(x, n=4):
    return "-" if x is None else f"{x:.{n}f}"


def cell_stats(recs: list[dict], floor: float | None, group_size: int) -> dict:
    judged = [r for r in recs if _judged(r)]
    passed = [r for r in judged if r.get("gate") == "all"]
    exported = [r for r in judged if r.get("pregate_passed")]
    rewards = [_reward(r) for r in judged]
    prewards = [_reward(r) for r in passed]

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in judged:
        groups[r.get("group", "?")].append(r)
    complete = [g for g in groups.values() if len(g) >= group_size]
    spreads, sig, pass_spreads = [], 0, []
    for g in complete:
        vals = [_reward(r) for r in g]
        sp = max(vals) - min(vals)
        spreads.append(sp)
        if floor is not None and sp > floor:
            sig += 1
        pv = [_reward(r) for r in g if r.get("gate") == "all"]
        if len(pv) > 1:
            pass_spreads.append(max(pv) - min(pv))

    max_spread = max(spreads) if spreads else 0.0
    pass_spread = max(pass_spreads) if pass_spreads else None
    if not passed:
        verdict = "NO CODE"
    elif sig > 0:
        verdict = "SIGNAL"
    else:
        verdict = "NO GRADIENT"
    saturated = bool(
        judged and len(passed) == len(judged) and pass_spread is not None
        and floor is not None and pass_spread <= floor
    )
    return {
        "n_generated": len(recs),
        "n_judged": len(judged),
        "n_export": len(exported),
        "n_pass": len(passed),
        "export_rate": (len(exported) / len(judged)) if judged else None,
        "pass_rate": (len(passed) / len(judged)) if judged else None,
        "best": max(rewards) if rewards else None,
        "mean": statistics.fmean(rewards) if rewards else None,
        "best_pass": max(prewards) if prewards else None,
        "min_pass": min(prewards) if prewards else None,
        "complete_groups": len(complete),
        "signal_groups": sig,
        "max_spread": max_spread,
        "mean_spread": statistics.fmean(spreads) if spreads else 0.0,
        "pass_spread": pass_spread,
        "floor": floor,
        "spread_over_floor": (max_spread / floor) if floor else None,
        "pass_spread_over_floor": (pass_spread / floor) if (floor and pass_spread) else None,
        "verdict": verdict,
        "saturated_no_gradient": saturated,
        "gate_histogram": dict(Counter(r.get("gate") for r in judged).most_common()),
        "finish_reason": dict(Counter(r.get("finish_reason") for r in recs).most_common()),
        "median_code_chars": statistics.median([r.get("code_chars", 0) for r in recs]) if recs else 0,
        "rewards_sorted": sorted(rewards, reverse=True),
    }


def build(recs: list[dict], floors: dict, group_size: int) -> dict:
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in recs:
        cells[(r.get("task", "?"), r.get("variant", "?"))].append(r)
    return {
        f"{t}|{v}": cell_stats(rs, floors.get(t), group_size) for (t, v), rs in sorted(cells.items())
    }


def ladder_table(stats: dict) -> str:
    rows = [
        "| kernel | rung | judged | export | **PASS** | best | group spread | floor | spread/floor "
        "| PASS-only spread | signal groups | **verdict** |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, c in stats.items():
        task, rung = key.split("|")
        sof = c["spread_over_floor"]
        note = " (SATURATED)" if c["saturated_no_gradient"] else ""
        rows.append(
            f"| {task} | {rung} | {c['n_judged']} | {c['n_export']} | **{c['n_pass']}** | "
            f"{_f(c['best'])} | {_f(c['max_spread'])} | {_f(c['floor'])} | "
            f"{'-' if sof is None else f'{sof:.1f}x'} | {_f(c['pass_spread'])} | "
            f"{c['signal_groups']}/{c['complete_groups']} | **{c['verdict']}{note}** |"
        )
    return "\n".join(rows)


def gate_table(stats: dict) -> str:
    gates = sorted({g for c in stats.values() for g in c["gate_histogram"]}, key=str)
    out = ["| cell | " + " | ".join(f"`{g}`" for g in gates) + " |", "|---" * (len(gates) + 1) + "|"]
    for key, c in stats.items():
        h = c["gate_histogram"]
        out.append(f"| {key} | " + " | ".join(str(h.get(g, 0)) for g in gates) + " |")
    return "\n".join(out)


def reward_table(stats: dict) -> str:
    out = ["| cell | rewards (descending, non-zero only) | zeros |", "|---|---|---|"]
    for key, c in stats.items():
        nz = [f"{v:.4f}" for v in c["rewards_sorted"] if v > 0]
        z = sum(1 for v in c["rewards_sorted"] if v <= 0)
        out.append(f"| {key} | {', '.join(nz) if nz else '-'} | {z} |")
    return "\n".join(out)


def first_working_rung(stats: dict) -> dict:
    per_task: dict[str, dict] = {}
    for key, c in stats.items():
        task, rung = key.split("|")
        per_task.setdefault(task, {})[rung] = c
    out = {}
    for task, rungs in sorted(per_task.items()):
        first_pass = next((r for r in RUNGS if rungs.get(r, {}).get("n_pass", 0) > 0), None)
        first_signal = next((r for r in RUNGS if rungs.get(r, {}).get("signal_groups", 0) > 0), None)
        out[task] = {
            "first_rung_with_working_code": first_pass,
            "first_rung_with_signal": first_signal,
            "best_score_any_rung": max(
                (rungs[r].get("best") or 0.0) for r in rungs
            ),
            "verdicts": {r: rungs[r]["verdict"] for r in RUNGS if r in rungs},
        }
    return out


def best_programs(recs: list[dict], limit_chars: int = 4000) -> str:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        if r.get("gate") == "all":
            by[f"{r.get('task')}|{r.get('variant')}"].append(r)
    out = []
    for key, rs in sorted(by.items()):
        best = max(rs, key=lambda r: _reward(r))
        out.append(f"\n### {key} -- reward {_reward(best):.4f}\n")
        out.append("```\n" + (best.get("observation") or "(no observation)")[:1200] + "\n```\n")
        out.append("```python\n" + (best.get("code") or "")[:limit_chars] + "\n```\n")
    return "\n".join(out) if out else "\n_(no cell produced a passing kernel)_\n"


def modal_failures(recs: list[dict]) -> str:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        if _judged(r) and r.get("gate") != "all":
            by[f"{r.get('task')}|{r.get('variant')}"].append(r)
    out = ["| cell | modal gate | n | observation |", "|---|---|---|---|"]
    for key, rs in sorted(by.items()):
        c = Counter(r.get("gate") for r in rs)
        gate, n = c.most_common(1)[0]
        rep = next(r for r in rs if r.get("gate") == gate)
        obs = (rep.get("observation") or "").replace("\n", " | ").replace("|", "\\|")[:200]
        out.append(f"| {key} | `{gate}` | {n}/{len(rs)} | {obs} |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--boot-report", default=None)
    ap.add_argument("--group-size", type=int, default=16)
    ap.add_argument("--out", default="-")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    recs = [json.loads(ln) for ln in Path(args.jsonl).read_text().splitlines() if ln.strip()]
    floors: dict = {}
    if args.boot_report and Path(args.boot_report).exists():
        rep = json.loads(Path(args.boot_report).read_text())
        floors = {t: r.get("noise_floor") for t, r in rep.items() if isinstance(r, dict)}

    stats = build(recs, floors, args.group_size)
    verdicts = first_working_rung(stats)
    md = "\n\n".join([
        "## The ladder\n\n" + ladder_table(stats),
        "## First rung that worked, per kernel\n\n```json\n"
        + json.dumps(verdicts, indent=1) + "\n```",
        "## Gate histogram\n\n" + gate_table(stats),
        "## Reward distribution\n\n" + reward_table(stats),
        "## Modal failure per cell\n\n" + modal_failures(recs),
        "## Best passing program per cell\n" + best_programs(recs),
    ])
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"cells": stats, "verdicts": verdicts,
                                                   "noise_floors": floors}, indent=1, default=str))
    if args.out == "-":
        print(md)
    else:
        Path(args.out).write_text(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
