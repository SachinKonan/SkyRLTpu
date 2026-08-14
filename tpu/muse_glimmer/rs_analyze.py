#!/usr/bin/env python
"""Turn graded Muse-Glimmer rollouts into the REASONING-STRENGTH tables.

Two comparisons, and they are NOT the same kind of claim:

  high vs xhigh  -- CONTROLLED. Same model, same 16 parent states, same
                    prompts; the only difference is one system-prompt line.
  Muse vs Qwen   -- OBSERVATIONAL. Different model, different training
                    history, and (importantly) the pool only kept the top-2
                    children per parent, so Qwen's recorded children are an
                    elite-filtered sample with an unknown denominator. Its
                    "improvement rate" is therefore NOT computable; only a
                    best-of-k value comparison is honest, and Muse's side of
                    that is restricted to its own top-2 per state.

Pool values are stored signed so that higher is always better, and this script
keeps that convention: ``pct = (v_child - v_parent) / |v_parent| * 100``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from collections import Counter, defaultdict

TIE_EPS = 1e-9


def pct_change(v, vp):
    if vp is None or v is None:
        return None
    if abs(vp) < 1e-12:
        return None  # undefined against a zero baseline; report deltas instead
    return (v - vp) / abs(vp) * 100.0


def quantiles(xs):
    if not xs:
        return {}
    xs = sorted(xs)
    def q(p):
        i = min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))
        return xs[i]
    return {
        "n": len(xs), "min": xs[0], "p25": q(0.25), "median": q(0.5),
        "p75": q(0.75), "p90": q(0.9), "max": xs[-1],
        "mean": round(sum(xs) / len(xs), 2),
    }


def two_prop_z(k1, n1, k2, n2):
    """Unpooled-free two-proportion z; None when either arm is empty."""
    if not n1 or not n2:
        return None
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0
    return (p1 - p2) / se


def load(path):
    out = []
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--grades", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase1-max-tokens", type=int, default=13824)
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    gen = {r["item_id"]: r for r in load(args.gen) if "item_id" in r}
    grades = {r["item_id"]: r for r in load(args.grades)}
    maximize = man["maximize"]

    def failure_bucket(g):
        """Coarse taxonomy of why a rollout produced no value.

        ``missing_module`` is called out separately because it is a
        *resource-contract* failure, not a reasoning failure: the erdos prompt
        lists scipy/numpy/cvxpy/math and nothing else, so a program that
        imports numba fails by construction. Whether that should count against
        the model is a judgement call, so the report carries both totals.
        """
        if not g.get("has_code"):
            return "no_code_block"
        if g.get("correctness", 0) > 0:
            return "ok"
        msg = (g.get("msg") or "")
        if "No module named" in msg:
            return "missing_module"
        if "timed out" in msg.lower() or "Timeout" in msg:
            return "timeout"
        if "Evaluation failed" in msg or "execution failed" in msg:
            return "runtime_error"
        return "other"

    rows = []
    for iid, g in grades.items():
        gg = gen.get(iid, {})
        v, vp = g.get("value"), g.get("parent_value")
        rows.append({
            "item_id": iid, "arm": g["arm"], "kind": g["kind"],
            "state_id": g["state_id"], "run": g["run"],
            "parent_value": vp, "value": v,
            "has_code": g.get("has_code", False),
            "valid": bool(g.get("correctness", 0)) and v is not None,
            "pct": pct_change(v, vp),
            "delta": (v - vp) if (v is not None and vp is not None) else None,
            "reasoning_tokens": gg.get("reasoning_tokens"),
            "answer_tokens": gg.get("answer_tokens"),
            "total_tokens": gg.get("total_tokens"),
            "phase1_tokens": gg.get("phase1_tokens"),
            "phase1_budget": gg.get("phase1_budget"),
            "phase1_truncated": gg.get("phase1_truncated"),
            "phase2_used": gg.get("phase2_used"),
            "saw_reasoning": gg.get("saw_reasoning_channel"),
            "saw_answer": gg.get("saw_answer_channel"),
            "prompt_len": gg.get("prompt_len"),
            "grade_s": g.get("grade_s"),
            "timed_out": bool(g.get("timed_out")),
            # STRICT-CONTRACT column. The erdos prompt lists "scipy, numpy,
            # cvxpy[...], math" and not numba; numba is nonetheless a declared
            # dependency of the discover package, so grading runs WITH it and
            # the strict reading is recovered here instead of being baked into
            # the environment. Round 1 conflated the two and reported a
            # missing-module environment artefact as a model behaviour.
            "uses_numba": ("import numba" in (gg.get("answer") or "")
                           or "from numba" in (gg.get("answer") or "")),
            "phase1_finish": gg.get("phase1_finish"),
            "phase2_finish": gg.get("phase2_finish"),
            "bucket": failure_bucket(g),
            "msg": (g.get("msg") or "")[:120],
        })

    report = {"problem": man["problem"], "maximize": maximize,
              "phase1_max_tokens": args.phase1_max_tokens,
              "eval_timeout": man["eval_timeout"],
              "n_graded": len(rows), "arms": {}}

    for kind in ("parent", "start"):
        for arm in ("high", "xhigh"):
            sel = [r for r in rows if r["arm"] == arm and r["kind"] == kind]
            if not sel:
                continue
            n = len(sel)
            valid = [r for r in sel if r["valid"]]
            imp = [r for r in valid if r["value"] > r["parent_value"] + TIE_EPS]
            tie = [r for r in valid if abs(r["value"] - r["parent_value"]) <= TIE_EPS]
            reg = [r for r in valid if r["value"] < r["parent_value"] - TIE_EPS]
            pv = [r["pct"] for r in valid if r["pct"] is not None]
            pi = [r["pct"] for r in imp if r["pct"] is not None]
            dv = [r["delta"] for r in valid if r["delta"] is not None]
            rt = [r["reasoning_tokens"] for r in sel if r["reasoning_tokens"] is not None]
            tt = [r["total_tokens"] for r in sel if r["total_tokens"] is not None]
            trunc = [r for r in sel if r["phase1_truncated"]]
            headroom = [
                r["phase1_budget"] - r["phase1_tokens"]
                for r in sel
                if r["phase1_budget"] is not None and r["phase1_tokens"] is not None
            ]
            report["arms"][f"{kind}/{arm}"] = {
                "n_rollouts": n,
                "format_rate": round(sum(r["has_code"] for r in sel) / n, 4),
                "valid_rate": round(len(valid) / n, 4),
                "reasoning_channel_rate": round(sum(bool(r["saw_reasoning"]) for r in sel) / n, 4),
                "answer_channel_rate": round(sum(bool(r["saw_answer"]) for r in sel) / n, 4),
                "improvement_rate_all": round(len(imp) / n, 4),
                "improvement_rate_valid": round(len(imp) / len(valid), 4) if valid else None,
                "tie_rate_all": round(len(tie) / n, 4),
                "tie_rate_valid": round(len(tie) / len(valid), 4) if valid else None,
                "regression_rate_all": round(len(reg) / n, 4),
                "regression_rate_valid": round(len(reg) / len(valid), 4) if valid else None,
                "mean_pct_change_valid": round(st.mean(pv), 4) if pv else None,
                "median_pct_change_valid": round(st.median(pv), 4) if pv else None,
                # Over ALL rollouts, an invalid one is not "missing data", it
                # is a rollout that produced no improvement -- so it enters at
                # 0. This is the expected yield per rollout, which is what a
                # discovery loop actually spends tokens on.
                "mean_pct_change_all": (
                    round(sum(r["pct"] or 0.0 for r in sel if r["valid"]) / n, 4)
                    if pv else None),
                "mean_delta_all": round(
                    sum(r["delta"] or 0.0 for r in sel if r["valid"]) / n, 6),
                "mean_pct_change_improving": round(st.mean(pi), 4) if pi else None,
                "best_pct_change": round(max(pv), 4) if pv else None,
                "value_achieved": quantiles([r["value"] for r in valid]),
                # JSSP's early pool stores value 0.0 for ~98% of states, so a
                # PERCENT change is undefined there and only the absolute delta
                # carries information. Both are always reported.
                "mean_delta_valid": round(st.mean(dv), 6) if dv else None,
                "median_delta_valid": round(st.median(dv), 6) if dv else None,
                "parent_value_zero_rate": round(
                    sum(1 for r in sel if r["parent_value"] in (0, 0.0)) / n, 4),
                # Grading timeouts fall on SLOW programs. If they land
                # asymmetrically on one arm, that arm looks worse for a reason
                # unrelated to solution quality.
                "grading_timeout_rate": round(
                    sum(1 for r in sel if r["timed_out"]) / n, 4),
                "p90_grade_s": (quantiles([r["grade_s"] for r in sel
                                           if r["grade_s"] is not None]) or {}).get("p90"),
                "reasoning_tokens": quantiles(rt),
                "total_tokens": quantiles(tt),
                "phase1_truncation_rate": round(len(trunc) / n, 4),
                "phase2_used_rate": round(sum(bool(r["phase2_used"]) for r in sel) / n, 4),
                "phase1_headroom": quantiles(headroom),
                "mean_grade_s": round(st.mean([r["grade_s"] for r in sel if r["grade_s"]]), 1)
                if any(r["grade_s"] for r in sel) else None,
                "failure_buckets": dict(Counter(r["bucket"] for r in sel).most_common()),
                # Validity if a missing-module failure were not counted against
                # the model.
                "valid_rate_excl_missing_module": round(
                    len(valid) / max(1, n - sum(1 for r in sel if r["bucket"] == "missing_module")), 4
                ),
                # STRICT reading of the prompt's library list: a program that
                # reaches for numba has broken the stated contract, so it is
                # scored 0 whether or not the interpreter could import it.
                "numba_use_rate": round(sum(1 for r in sel if r["uses_numba"]) / n, 4),
                "valid_rate_strict_contract": round(
                    sum(1 for r in valid if not r["uses_numba"]) / n, 4),
                "improvement_rate_strict_contract": round(
                    sum(1 for r in imp if not r["uses_numba"]) / n, 4),
            }

    # Controlled high-vs-xhigh significance on the two headline rates.
    report["high_vs_xhigh"] = {}
    for kind in ("parent", "start"):
        a = [r for r in rows if r["arm"] == "high" and r["kind"] == kind]
        b = [r for r in rows if r["arm"] == "xhigh" and r["kind"] == kind]
        if not a or not b:
            continue
        ia = sum(1 for r in a if r["valid"] and r["value"] > r["parent_value"] + TIE_EPS)
        ib = sum(1 for r in b if r["valid"] and r["value"] > r["parent_value"] + TIE_EPS)
        va = sum(1 for r in a if r["valid"])
        vb = sum(1 for r in b if r["valid"])
        report["high_vs_xhigh"][kind] = {
            "improvement_rate": {"high": round(ia / len(a), 4), "xhigh": round(ib / len(b), 4),
                                 "z": round(two_prop_z(ia, len(a), ib, len(b)), 3)},
            "valid_rate": {"high": round(va / len(a), 4), "xhigh": round(vb / len(b), 4),
                           "z": round(two_prop_z(va, len(a), vb, len(b)), 3)},
        }

    # ---- Muse vs Qwen, matched on best-of-k -------------------------------
    # The pool kept only the top-2 children per parent, so the only honest
    # comparison is best-child-per-state. Muse is cut to its own top-2, and
    # also to top-2 of its FIRST 16 rollouts (the sweep's group_size).
    qwen = man.get("qwen_children", {})
    best = {"qwen": [], "high": [], "xhigh": [], "high_g16": [], "xhigh_g16": []}
    per_state = []
    by_state = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["kind"] != "parent" or not r["valid"]:
            continue
        by_state[r["state_id"]][r["arm"]].append(r)
    for sid, arms in sorted(by_state.items()):
        vp = man["prompts"][f"parent|{sid}|high"]["parent_value"]
        qk = sorted((c["value"] for c in qwen.get(sid, []) if c.get("value") is not None),
                    reverse=True)[:2]
        entry = {"state_id": sid, "parent_value": vp,
                 "qwen_n_recorded": len(qwen.get(sid, [])),
                 "qwen_best_pct": pct_change(qk[0], vp) if qk else None}
        for arm in ("high", "xhigh"):
            rs = arms.get(arm, [])
            vs = sorted((r["value"] for r in rs), reverse=True)
            g16 = sorted((r["value"] for r in rs
                          if int(r["item_id"].rsplit("r", 1)[1]) < 16), reverse=True)
            entry[f"{arm}_n_valid"] = len(rs)
            entry[f"{arm}_best_pct"] = pct_change(vs[0], vp) if vs else None
            entry[f"{arm}_g16_best_pct"] = pct_change(g16[0], vp) if g16 else None
            if vs:
                best[arm].append(pct_change(vs[0], vp))
            if g16:
                best[f"{arm}_g16"].append(pct_change(g16[0], vp))
        if qk:
            best["qwen"].append(pct_change(qk[0], vp))
        per_state.append(entry)

    def summ(key):
        xs = [x for x in best[key] if x is not None]
        return {"n_states": len(xs),
                "mean_best_pct": round(st.mean(xs), 4) if xs else None,
                "median_best_pct": round(st.median(xs), 4) if xs else None,
                "states_beating_parent": sum(1 for x in xs if x > 0)}

    report["best_of_k_vs_qwen"] = {k: summ(k) for k in best}
    report["per_state"] = per_state

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "per_state"}, indent=2))


if __name__ == "__main__":
    main()
