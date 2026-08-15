"""Re-grade the sd-run winners against the CORRECTED denominators.

Every reward in `sd-results-3687904.jsonl` was scored against a baseline this
week's audit showed was crippled: splash ran JAX's placeholder all-128
BlockSizes (~10x slow), megablox ran the default (128,128,128) tiling (~38x
slow), RPA never bound a real kernel at all, and rg_lru's 1.1941 -- the
project's only above-1.0 result -- beat `lax.associative_scan`, not DeepMind's
scan. All four baselines are now bound and configured (aa073941, 95567ad2,
d98f1836), so this regrades the 42 surviving winners' code, unchanged, on a
judge whose denominator is finally real.

The number this produces is the first trustworthy answer to "how close are
model-written kernels to properly configured production kernels" -- and it
decides the NEXT experiment: rewards near 1.0 mean cold-start generation
against real bars is winnable; rewards near 0.1 mean the honest framing is
improve-a-seed, not beat-from-scratch.

Same grading conditions as the original run (probe case sets, adversarial OFF)
so the only variable that moved is the denominator. Cache is OFF: these hashes
already carry verdicts from the crippled-baseline era, and serving them would
regrade nothing.
"""

from __future__ import annotations

import argparse
import json

# The exact case sets the sd run graded on (seam_dialect_probe.sbatch
# PROBLEM_ARG), holdouts included.
# GENERAL-mode sweep sets: single source of truth, so the regrade cannot drift
# from what the judge and prompts declare.
from pallas_arena.probe.configs import TASK_CASES as CASES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", required=True, help="json {task: {name: source}}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--timing-pairs", type=int, default=20)
    args = ap.parse_args()

    from pallas_arena.judge.worker import PersistentWorker

    codes = json.load(open(args.codes))
    report = {"tasks": {}}
    for task, entries in codes.items():
        if task not in CASES:
            report["tasks"][task] = {"error": f"no case set for {task}"}
            continue
        print(f"\n{'=' * 70}\n{task}: boot ({len(entries)} winners)", flush=True)
        w = PersistentWorker(
            task,
            cases=list(CASES[task]),
            use_adversarial=False,  # match the original run's conditions
            timing_pairs=args.timing_pairs,
            worker_id="regrade",
        )
        boot = w.boot()
        trow = {
            "boot_ok": boot.get("ok"),
            "baseline_impl": getattr(type(w.problem), "baseline_impl", "?"),
            "noise_floor": boot.get("noise_floor"),
            "results": {},
        }
        print(f"  baseline_impl={trow['baseline_impl']} noise_floor={trow['noise_floor']}", flush=True)
        if not boot.get("ok"):
            trow["boot_error"] = str(boot.get("error"))[:300]
            report["tasks"][task] = trow
            print(f"  BOOT FAILED: {trow['boot_error']}", flush=True)
            continue

        for name, code in entries.items():
            old_reward = float(name.rsplit("-r", 1)[1]) if "-r" in name else None
            try:
                r = w.grade_code(code, tag=f"regrade:{name}")
                trow["results"][name] = {
                    "old_reward": old_reward,
                    "new_reward": r.get("reward"),
                    "reward_kind": r.get("reward_kind"),
                    "n_scored_cases": r.get("n_scored_cases"),
                    "holdout_scored": r.get("holdout_scored"),
                    "per_case": r.get("per_case"),
                    "holdout": r.get("holdout"),
                    "baseline_impl_per_case": r.get("baseline_impl_per_case"),
                    "timer": r.get("timer"),
                    "new_score": r.get("score"),
                    "passed": r.get("passed"),
                    "gate": r.get("gate"),
                    "violations": (r.get("violations") or [])[:1],
                }
                print(
                    f"  {name[:58]:58s} old={old_reward} -> new={r.get('reward')} "
                    f"gate={r.get('gate')}",
                    flush=True,
                )
            except Exception as e:  # noqa: BLE001
                trow["results"][name] = {"old_reward": old_reward, "error": f"{type(e).__name__}: {str(e)[:200]}"}
                print(f"  {name[:58]:58s} EXCEPTION {type(e).__name__}", flush=True)
        report["tasks"][task] = trow

    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)

    print("\n=== SUMMARY: winners vs REAL denominators ===")
    for task, trow in report["tasks"].items():
        rs = trow.get("results", {})
        graded = [v for v in rs.values() if "new_reward" in v]
        still = [v for v in graded if v.get("passed")]
        beat = [v for v in graded if (v.get("new_reward") or 0) > 1.0]
        best = max((v.get("new_reward") or 0) for v in graded) if graded else None
        print(
            f"  {task:26s} impl={trow.get('baseline_impl', '?'):38.38s} "
            f"graded={len(graded)}/{len(rs)} still-pass={len(still)} above-1.0={len(beat)} best={best}"
        )


if __name__ == "__main__":
    main()
