"""Probe metrics — including the one that decides whether RL is worth building.

`nonuniform_group_frac` is the headline. ttt-discover normalizes advantage
within a group of rollouts; a group whose rewards are all identical (the
usual case being all-zero) produces a zero advantage vector and therefore no
gradient, however impressive the mean looks. A configuration is trainable to
the extent that its groups DIFFER internally.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict

# gates the client-side pre-gate can reach without a chip
PREGATE_GATES = ("no_code", "ast", "exec", "poison_stub", "aot_export", "timeout", "rlimit", "harness")
# gates that mean "the TPU backend compile itself failed"
COMPILE_GATES = ("compile_budget", "candidate_compile", "artifact_load")
# gates that mean "it compiled and ran, but the answer was wrong"
WRONG_GATES = ("correctness", "timed_output_correctness", "gradient", "determinism", "fixtures")


def _reward(r: dict) -> float:
    v = r.get("reward")
    return float(v) if isinstance(v, (int, float)) else 0.0


def _judged(r: dict) -> bool:
    """Did this candidate get a terminal verdict at all? Anything still in
    flight when the wall clock stopped is excluded from RATES (counting it as
    a failure would bias every rate toward zero) but reported separately."""
    return r.get("stage") in ("pregate", "judge") and r.get("gate") not in (None, "unjudged", "pregate-only")


def _stats(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    s = sorted(vals)
    return {
        "n": len(s),
        "mean": statistics.fmean(s),
        "median": statistics.median(s),
        "max": s[-1],
        "min": s[0],
        "p25": s[len(s) // 4],
        "p75": s[(3 * len(s)) // 4],
        "std": statistics.pstdev(s) if len(s) > 1 else 0.0,
    }


def _histogram(vals: list[float]) -> dict:
    """Score distribution, not just its mean. Zero gets its own bucket
    because 'all zeros' is the failure mode this probe exists to detect."""
    edges = [0.0, 1e-9, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 1.25, 1e9]
    labels = ["=0", "(0,0.1)", "[0.1,0.25)", "[0.25,0.5)", "[0.5,0.75)", "[0.75,0.9)", "[0.9,1.0)", "[1.0,1.25)", ">=1.25"]
    out = dict.fromkeys(labels, 0)
    for v in vals:
        for i in range(len(labels)):
            if edges[i] <= v < edges[i + 1]:
                out[labels[i]] += 1
                break
    return out


def group_uniformity(recs: list[dict], group_size: int) -> dict:
    """Per-group reward spread. Only COMPLETE groups count: a partial group
    is not what the trainer would see."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        if _judged(r):
            groups[r.get("group", "?")].append(r)
    complete, nonuniform, spreads = 0, 0, []
    detail = []
    for g, rs in sorted(groups.items()):
        if len(rs) < group_size:
            continue
        complete += 1
        vals = [_reward(r) for r in rs]
        spread = max(vals) - min(vals)
        spreads.append(spread)
        nz = sum(1 for v in vals if v > 0)
        if spread > 0:
            nonuniform += 1
        detail.append({"group": g, "spread": spread, "nonzero": nz, "max": max(vals)})
    return {
        "complete_groups": complete,
        "nonuniform_groups": nonuniform,
        "nonuniform_group_frac": (nonuniform / complete) if complete else None,
        "mean_within_group_spread": statistics.fmean(spreads) if spreads else 0.0,
        "groups": detail,
    }


def summarize_cell(recs: list[dict], group_size: int) -> dict:
    judged = [r for r in recs if _judged(r)]
    gates = Counter(r.get("gate") for r in judged)
    n = len(judged)
    passed = [r for r in judged if r.get("gate") == "all"]
    exported = [r for r in judged if r.get("pregate_passed")]
    compiled = [r for r in exported if r.get("gate") not in COMPILE_GATES]
    rewards = [_reward(r) for r in judged]
    return {
        "n_generated": len(recs),
        "n_judged": n,
        "n_unjudged": len(recs) - n,
        "export_rate": len(exported) / n if n else None,
        "compile_rate": len(compiled) / n if n else None,
        "correctness_rate": len(passed) / n if n else None,
        "nonzero_reward_rate": sum(1 for v in rewards if v > 0) / n if n else None,
        "score_all": _stats(rewards),
        "score_passing": _stats([_reward(r) for r in passed]),
        "score_histogram": _histogram(rewards),
        "score_values": sorted(rewards, reverse=True)[:64],
        "gate_histogram": dict(gates.most_common()),
        "extraction": dict(Counter(r.get("extraction") for r in recs).most_common()),
        "finish_reason": dict(Counter(r.get("finish_reason") for r in recs).most_common()),
        "median_code_chars": statistics.median([r.get("code_chars", 0) for r in recs]) if recs else 0,
        **{k: v for k, v in group_uniformity(recs, group_size).items() if k != "groups"},
    }


def summarize(recs: list[dict], group_size: int = 16) -> dict:
    cells: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        cells[r.get("config", "?")].append(r)

    per_config = {name: summarize_cell(rs, group_size) for name, rs in sorted(cells.items())}

    def roll(key_fn) -> dict:
        buckets: dict[str, list[dict]] = defaultdict(list)
        for r in recs:
            buckets[key_fn(r)].append(r)
        return {k: summarize_cell(v, group_size) for k, v in sorted(buckets.items())}

    all_judged = [r for r in recs if _judged(r)]
    total_groups = sum(c["complete_groups"] for c in per_config.values())
    total_nonuniform = sum(c["nonuniform_groups"] for c in per_config.values())
    best = max(
        ((name, c["score_all"].get("max", 0.0)) for name, c in per_config.items()),
        key=lambda x: x[1] or 0.0,
        default=(None, 0.0),
    )
    return {
        "n_candidates": len(recs),
        "n_judged": len(all_judged),
        "headline": {
            "overall_nonuniform_groups": f"{total_nonuniform}/{total_groups}",
            "overall_nonuniform_group_frac": (total_nonuniform / total_groups) if total_groups else None,
            "configs_with_any_nonuniform_group": sorted(
                name for name, c in per_config.items() if c["nonuniform_groups"] > 0
            ),
            "best_config_by_max_score": best[0],
            "best_score": best[1],
            "any_candidate_passed": any(r.get("gate") == "all" for r in all_judged),
        },
        "per_config": per_config,
        "by_variant": roll(lambda r: r.get("variant", "?")),
        "by_model": roll(lambda r: r.get("model", "?")),
        "by_task": roll(lambda r: r.get("task", "?")),
        "gate_histogram_overall": dict(Counter(r.get("gate") for r in all_judged).most_common()),
    }
