"""Collector for per-test grading: merge one-case verdicts into ONE candidate
verdict with the same reward semantics as the monolithic judge.

The dispatch unit is (candidate, shape case): each Ray task grades exactly
one case -- forward correctness + 20-pair timing + that case's backward --
and returns the single-case result the PersistentWorker already produces.
This module folds those into the candidate verdict the queue client expects.

RULES (unchanged from the monolith, only relocated):
  * CORRECT-EVERYWHERE: a candidate-fault at ANY case (wrong output, failed
    compile, runtime halt, pregate) zeroes the candidate. Which faults are
    the candidate's is an explicit list -- everything else is the judge's.
  * JUDGE-FAULT EXCLUSION: a case the judge could not measure (tp control
    failed, timer disagreement, task process died without a halt signature)
    contributes NO factor. Judge trouble is never the candidate's penalty.
  * REWARD: geomean over the scored cases' forward ratios; each case is
    gated by ITS OWN measured noise floor (per-test floors are more honest
    than one global -- each test calibrated on its own chip).
  * BACKWARD FOLD: one factor per scored case; wrong/absent backward floors
    that factor (never below max(floor, GRAD_ABSENT_FLOOR)); a correct
    backward is clamped UP to the floor (absent must never beat slow-correct)
    -- exactly timing.fold_grad_reward, applied per case.
  * HOLDOUT: scored only in general mode (both arena problems set
    general_mode=True); otherwise logged-unscored.

All pure python (timing.py is jax-free); unit-tested on synthetic verdicts.
"""

from __future__ import annotations

import math
from typing import Any

from pallas_arena.judge.timing import GRAD_ABSENT_FLOOR, gate_reward

# Gates that mean THE CANDIDATE is wrong. Anything not listed here (and not
# an explicit judge-fault) is treated as the candidate's fault too when the
# per-case worker says passed=False -- unknown failure modes must not leak
# reward -- but these are additionally CANDIDATE-FATAL: they zero the whole
# candidate immediately and cancel its remaining tests.
CANDIDATE_FATAL_GATES = {
    "pregate", "export", "candidate_compile", "fixtures", "correctness",
    "timed_output_correctness", "determinism", "adversarial", "runtime_halt",
}
JUDGE_FAULT_GATES = {"judge_fault"}

# Ray-task death classification: these substrings in the raised error mean
# the CANDIDATE killed the process/device (fatal), not the judge.
FATAL_ERROR_SIGNATURES = (
    "halted unexpectedly", "CoreHalt", "continuator has halted",
    "RuntimeUnexpectedCoreHalt",
)


def case_width(name: str) -> int:
    """Chips a case needs, by the enforced naming convention."""
    if name.startswith("tp8-"):
        return 8
    if name.startswith("tp4-"):
        return 4
    if name.startswith("tp2-"):
        return 2
    return 1


def is_holdout(name: str) -> bool:
    return "holdout" in name


def _geomean(xs: list[float]) -> float:
    return math.exp(sum(math.log(max(x, 1e-30)) for x in xs) / len(xs))


def classify_task_error(err_text: str) -> str:
    """'fatal' when the candidate killed its task/device, else 'judge'."""
    return "fatal" if any(sig in err_text for sig in FATAL_ERROR_SIGNATURES) else "judge"


def merge_case_results(
    problem: str,
    entries: dict[str, dict[str, Any]],
    *,
    general_mode: bool = True,
    has_bwd: bool = True,
    default_floor: float = 0.05,
) -> dict:
    """Fold per-case verdicts into one candidate verdict.

    ``entries`` maps case name -> one of:
      {"result": <single-case worker result>}
      {"judge_fault": "<reason>"}            (excluded from reward)
      {"fatal": ("<gate>", "<reason>")}      (candidate-fatal)
      {"skipped": "<reason>"}                (e.g. width > host chips)
    """
    merged: dict[str, Any] = {
        "ok": True, "problem": problem, "dispatch": "percase",
        "latencies": {}, "speed_of_light_fracs": {}, "mxu_fracs": {},
        "grad_scores": {}, "tp_control": {}, "tp_timer_ratios": {},
        "tp_baseline_impls": {}, "case_noise_floors": {},
        "case_boot_s": {}, "excluded_cases": {}, "skipped_cases": {},
        "per_case": {}, "holdout": {},
    }
    violations: list[str] = []
    fatal: tuple[str, str] | None = None

    fwd: list[float] = []          # gated forward factors (scored cases)
    bwd: list[float] = []          # clamped backward factors (scored cases)
    floors: list[float] = []

    for case, entry in sorted(entries.items()):
        if "skipped" in entry:
            merged["skipped_cases"][case] = entry["skipped"]
            continue
        if "judge_fault" in entry:
            merged["excluded_cases"][case] = entry["judge_fault"]
            continue
        if "fatal" in entry:
            fatal = fatal or entry["fatal"]
            violations.append(f"{case}: {entry['fatal'][1]}")
            continue

        r = entry["result"]
        # merge the per-case diagnostics wholesale
        for k in ("latencies", "speed_of_light_fracs", "mxu_fracs",
                  "grad_scores", "tp_control", "tp_timer_ratios",
                  "tp_baseline_impls"):
            merged[k].update(r.get(k) or {})
        floor = r.get("task_noise_floor") or default_floor
        merged["case_noise_floors"][case] = round(floor, 4)
        if floor > 0.5:
            # A noise floor above 50% means the ref-vs-ref control could not
            # tell the same function from itself -- the chip was thrashing
            # (measured during the device-busy incident: floors so large that
            # garbage 2.8x ratios gated to 1.0). Nothing timed there is
            # meaningful; exclude rather than launder it through the geomean.
            merged["excluded_cases"][case] = f"noise floor {floor:.2f} -- timing environment untrustworthy"
            continue
        if r.get("task_boot_s") is not None:
            merged["case_boot_s"][case] = r["task_boot_s"]

        if not r.get("passed"):
            gate = r.get("gate", "?")
            why = str((r.get("violations") or ["?"])[0])[:300]
            if gate in JUDGE_FAULT_GATES or case in (r.get("skipped_tp") or {}):
                merged["excluded_cases"][case] = f"{gate}: {why}"
            else:
                # candidate fault -- fatal by the correct-everywhere rule
                fatal = fatal or (gate if gate in CANDIDATE_FATAL_GATES else gate, why)
                violations.append(f"{case}: [{gate}] {why}")
            continue

        # skipped-tp inside a passing result (e.g. control failed) = excluded
        for sk_case, sk_why in (r.get("skipped_tp") or {}).items():
            merged["excluded_cases"][sk_case] = sk_why

        raw = r.get("score")
        if raw is None:
            merged["excluded_cases"][case] = "no score in passing result"
            continue
        scored = general_mode or not is_holdout(case)
        (merged["holdout"] if is_holdout(case) else merged["per_case"])[case] = raw
        if not scored:
            continue
        floors.append(floor)
        fwd.append(gate_reward(raw, floor))
        if has_bwd:
            g_floor = max(floor, GRAD_ABSENT_FLOOR)
            if r.get("grad_ok") and case in (r.get("grad_scores") or {}):
                bwd.append(max(r["grad_scores"][case], g_floor))
            elif r.get("grad_ok"):
                pass                       # correct bwd, judge could not time: exclude
            else:
                bwd.append(g_floor)        # wrong/absent: floored

    if fatal is not None:
        merged.update(passed=False, gate=fatal[0], violations=violations,
                      reward=0.0, reward_with_bwd=0.0, score=0.0)
        return merged
    if not fwd:
        merged.update(passed=False, gate="judge_fault",
                      violations=["no case produced a scoreable factor"]
                      + [f"{c}: {w}" for c, w in merged["excluded_cases"].items()],
                      reward=0.0, reward_with_bwd=0.0, score=0.0)
        return merged

    agg_floor = max(floors) if floors else default_floor
    score = _geomean(fwd)
    reward = gate_reward(score, agg_floor)
    if bwd:
        total = (math.prod(fwd) * math.prod(bwd)) ** (1.0 / (len(fwd) + len(bwd)))
        reward_with_bwd = gate_reward(total, agg_floor)
    else:
        reward_with_bwd = reward
    merged.update(
        passed=True, gate="all", violations=[],
        score=score, reward=reward, reward_with_bwd=reward_with_bwd,
        n_scored_cases=len(fwd), n_bwd_factors=len(bwd),
        reward_kind="general" if general_mode else "ours",
    )
    return merged
