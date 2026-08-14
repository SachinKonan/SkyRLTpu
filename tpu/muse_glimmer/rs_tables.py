#!/usr/bin/env python
"""Render the markdown tables for REASONING-STRENGTH.md from a report JSON.

Kept separate from ``rs_analyze.py`` so the numbers in the writeup are
transcribed by a script rather than by hand.
"""

from __future__ import annotations

import argparse
import json


def f(x, nd=4, pct=False):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x*100:.1f}%" if pct else f"{x:.{nd}f}"
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+", help="problem=path/to/report.json")
    args = ap.parse_args()

    for spec in args.reports:
        prob, path = spec.split("=", 1)
        r = json.load(open(path))
        A = r["arms"]
        print(f"\n### {prob}\n")
        print(f"`maximize={r['maximize']}` · phase-1 cap **{r['phase1_max_tokens']}** "
              f"(prompt + reasoning) · `eval_timeout={r['eval_timeout']}` · "
              f"{r['n_graded']} rollouts graded\n")

        for kind, title in (("parent", "Improvement — expanding an existing pool state"),
                            ("start", "Start state — generating from scratch, no parent program")):
            keys = [f"{kind}/high", f"{kind}/xhigh"]
            if not any(k in A for k in keys):
                continue
            print(f"**{title}**\n")
            print("| metric | high | xhigh |")
            print("|---|---|---|")
            rowspec = [
                ("rollouts", "n_rollouts", False),
                ("format rate (code block present)", "format_rate", True),
                ("validity rate (program ran + verified)", "valid_rate", True),
                ("validity excl. missing-module failures", "valid_rate_excl_missing_module", True),
                ("programs importing `numba` (outside the prompt's library list)", "numba_use_rate", True),
                ("validity, STRICT contract (numba users scored 0)", "valid_rate_strict_contract", True),
                ("improvement rate, STRICT contract", "improvement_rate_strict_contract", True),
                ("reasoning channel `to=self` present", "reasoning_channel_rate", True),
                ("answer channel `to=user` present", "answer_channel_rate", True),
                ("**improvement rate** (of all rollouts)", "improvement_rate_all", True),
                ("improvement rate (of valid rollouts)", "improvement_rate_valid", True),
                ("tie rate (of all rollouts)", "tie_rate_all", True),
                ("regression rate (of all rollouts)", "regression_rate_all", True),
                ("**mean % change** (valid rollouts)", "mean_pct_change_valid", False),
                ("mean % change over ALL rollouts (invalid = 0)", "mean_pct_change_all", False),
                ("mean absolute delta vs parent (valid)", "mean_delta_valid", False),
                ("mean absolute delta over ALL rollouts (invalid = 0)", "mean_delta_all", False),
                ("median absolute delta vs parent (valid)", "median_delta_valid", False),
                ("parents whose recorded value is exactly 0", "parent_value_zero_rate", True),
                ("median % change (valid rollouts)", "median_pct_change_valid", False),
                ("mean % change (improving rollouts only)", "mean_pct_change_improving", False),
                ("best single % change", "best_pct_change", False),
                ("**phase-1 truncation rate**", "phase1_truncation_rate", True),
                ("phase-2 rescue used", "phase2_used_rate", True),
                ("**grading-timeout rate**", "grading_timeout_rate", True),
                ("p90 grading seconds", "p90_grade_s", False),
            ]
            for label, key, aspct in rowspec:
                a = A.get(keys[0], {}).get(key)
                b = A.get(keys[1], {}).get(key)
                print(f"| {label} | {f(a, pct=aspct)} | {f(b, pct=aspct)} |")
            for label, key in (("reasoning tokens", "reasoning_tokens"),
                               ("total generated tokens", "total_tokens"),
                               ("phase-1 headroom left (tokens)", "phase1_headroom")):
                a = A.get(keys[0], {}).get(key) or {}
                b = A.get(keys[1], {}).get(key) or {}
                fmt = lambda q: ("—" if not q else
                                 f"med {q['median']} · p90 {q['p90']} · max {q['max']} · mean {q['mean']}")
                print(f"| {label} | {fmt(a)} | {fmt(b)} |")
            print()

        hv = r.get("high_vs_xhigh", {})
        if hv:
            print("**Controlled high vs xhigh (two-proportion z)**\n")
            print("| set | metric | high | xhigh | z |")
            print("|---|---|---|---|---|")
            for kind, d in hv.items():
                for metric, v in d.items():
                    print(f"| {kind} | {metric} | {f(v['high'], pct=True)} | "
                          f"{f(v['xhigh'], pct=True)} | {v['z']} |")
            print()

        bk = r.get("best_of_k_vs_qwen", {})
        if bk:
            print("**Best-of-k per state (the only shape in which the recorded Qwen "
                  "pool can be compared: it kept the top-2 children per parent)**\n")
            print("| source | states with data | mean best % change | median | states beating parent |")
            print("|---|---|---|---|---|")
            names = {"qwen": "Qwen3.5-27B (recorded, top-2 of an unknown N)",
                     "high": "Muse-Glimmer `high` (top-1 of ≤32)",
                     "xhigh": "Muse-Glimmer `xhigh` (top-1 of ≤32)",
                     "high_g16": "Muse-Glimmer `high` (top-1 of first 16)",
                     "xhigh_g16": "Muse-Glimmer `xhigh` (top-1 of first 16)"}
            for k in ("qwen", "high", "xhigh", "high_g16", "xhigh_g16"):
                v = bk.get(k) or {}
                print(f"| {names[k]} | {v.get('n_states','—')} | "
                      f"{f(v.get('mean_best_pct'))} | {f(v.get('median_best_pct'))} | "
                      f"{v.get('states_beating_parent','—')} |")
            print()


if __name__ == "__main__":
    main()
