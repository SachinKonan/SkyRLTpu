"""Probe metrics — including the one that decides whether RL is worth building.

ttt-discover normalizes advantage within a group of rollouts; a group whose
rewards are all identical (the usual case being all-zero) produces a zero
advantage vector and therefore no gradient, however impressive the mean looks.
So `nonuniform_group_frac` was the cold-start probe's headline.

**It is not enough, and the previous run proved it.** `tailored|flce` passed
16/16 with rewards in [0.6966, 0.7008] — a spread of 0.0042 — against judge
noise floors of 0.0069 and 0.0158 for those very shapes. That group is
nominally "non-uniform" and would have been counted, but the differences it
would train on are timing jitter, not kernels. A prompt can be too good.

The headline here is therefore `signal_group_frac`: the fraction of COMPLETE
groups whose within-group reward spread exceeds the JUDGE'S OWN MEASURED NOISE
FLOOR for that task, taken from the worker's boot report. Everything below the
floor is counted separately and reported as noise, never as trainability.
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


def group_uniformity(recs: list[dict], group_size: int, noise_floors: dict | None = None) -> dict:
    """Per-group reward spread, against the judge's own noise floor.

    Only COMPLETE groups count: a partial group is not what the trainer would
    see. `noise_floors` maps task -> the p95 |ref/ref - 1| the judge measured
    at boot; a group whose spread does not clear it is counted as `noise`, not
    as signal, because its within-group differences are timing jitter.
    """
    noise_floors = noise_floors or {}
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        if _judged(r):
            groups[r.get("group", "?")].append(r)
    complete, nonuniform, signal, spreads = 0, 0, 0, []
    detail = []
    for g, rs in sorted(groups.items()):
        if len(rs) < group_size:
            continue
        complete += 1
        vals = [_reward(r) for r in rs]
        spread = max(vals) - min(vals)
        spreads.append(spread)
        nz = sum(1 for v in vals if v > 0)
        floor = noise_floors.get(rs[0].get("task"))
        above = spread > floor if floor is not None else None
        if spread > 0:
            nonuniform += 1
        if above:
            signal += 1
        detail.append(
            {
                "group": g,
                "spread": spread,
                "nonzero": nz,
                "max": max(vals),
                "noise_floor": floor,
                "above_noise_floor": above,
                "spread_over_floor": (spread / floor) if floor else None,
            }
        )
    return {
        "complete_groups": complete,
        "nonuniform_groups": nonuniform,
        "nonuniform_group_frac": (nonuniform / complete) if complete else None,
        # THE HEADLINE: groups whose spread clears the judge's own noise floor
        "signal_groups": signal,
        "signal_group_frac": (signal / complete) if complete else None,
        "mean_within_group_spread": statistics.fmean(spreads) if spreads else 0.0,
        "max_within_group_spread": max(spreads) if spreads else 0.0,
        "groups": detail,
    }


def summarize_cell(recs: list[dict], group_size: int, noise_floors: dict | None = None) -> dict:
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
        "median_fill_chars": (
            statistics.median([r.get("fill_chars", r.get("code_chars", 0)) for r in recs]) if recs else 0
        ),
        "seam_missing_name_rate": (
            sum(1 for r in recs if r.get("seam_missing_names")) / len(recs) if recs else None
        ),
        **{k: v for k, v in group_uniformity(recs, group_size, noise_floors).items() if k != "groups"},
    }


def summarize(recs: list[dict], group_size: int = 16, noise_floors: dict | None = None) -> dict:
    noise_floors = noise_floors or {}
    cells: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        cells[r.get("config", "?")].append(r)

    per_config = {name: summarize_cell(rs, group_size, noise_floors) for name, rs in sorted(cells.items())}

    def roll(key_fn) -> dict:
        buckets: dict[str, list[dict]] = defaultdict(list)
        for r in recs:
            buckets[key_fn(r)].append(r)
        return {k: summarize_cell(v, group_size, noise_floors) for k, v in sorted(buckets.items())}

    all_judged = [r for r in recs if _judged(r)]
    total_groups = sum(c["complete_groups"] for c in per_config.values())
    total_nonuniform = sum(c["nonuniform_groups"] for c in per_config.values())
    total_signal = sum(c["signal_groups"] for c in per_config.values())
    best = max(
        ((name, c["score_all"].get("max", 0.0)) for name, c in per_config.items()),
        key=lambda x: x[1] or 0.0,
        default=(None, 0.0),
    )
    return {
        "n_candidates": len(recs),
        "n_judged": len(all_judged),
        "noise_floors": noise_floors,
        "headline": {
            # THE number: groups whose within-group spread clears the judge's
            # own measured noise floor. `nonuniform` alone counted a group
            # whose entire spread was timing jitter as trainable.
            "overall_signal_groups": f"{total_signal}/{total_groups}",
            "overall_signal_group_frac": (total_signal / total_groups) if total_groups else None,
            "configs_with_signal": sorted(name for name, c in per_config.items() if c["signal_groups"] > 0),
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
        "by_task_variant": roll(lambda r: f"{r.get('task','?')}|{r.get('variant','?')}"),
        "gate_histogram_overall": dict(Counter(r.get("gate") for r in all_judged).most_common()),
    }
