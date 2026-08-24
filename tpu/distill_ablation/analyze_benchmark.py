"""Assemble the CDC benchmark table: best grade_full of the deep gpt-5.6-sol run vs the 120b
base-prompt best vs reference/SOTA, per problem, + the agent's grade-call volume and best-over-time.
Reads runs/benchmark_cdc/<problem>/result.json (+ grade_log.jsonl) and corpora/initial_*.json.
"""
from __future__ import annotations

import json
import os

REPO = "/n/fs/vision-mix/sk7524/SkyRLTpu"
AB = f"{REPO}/tpu/distill_ablation"
BENCH = f"{REPO}/runs/benchmark_cdc"

# problem -> (initial_json, maximize, sota/reference, sota_label)
META = {
    "erdos": ("initial_erdos.json", False, 0.380927, "Haugland SOTA"),
    "ac1":   ("initial_ac1.json", False, 1.50317, "AlphaEvolve SOTA"),
    "ac2":   ("initial_ac2.json", True, 0.9610, "AlphaEvolve SOTA"),
    "fc46":  ("initial_fcalgo46.json", True, 0.0665, "reference"),
    "fc302": ("initial_fcalgo302.json", True, None, "-"),
}


def baseline_120b(initial_json, maximize):
    p = f"{AB}/corpora/{initial_json}"
    if not os.path.exists(p):
        return None
    scores = [r["score"] for r in json.load(open(p))["records"] if r.get("score") is not None]
    if not scores:
        return None
    return max(scores) if maximize else min(scores)


def trajectory(problem, smoke):
    logf = f"{BENCH}/{problem}{'_smoke' if smoke else ''}/grade_log.jsonl"
    if not os.path.exists(logf):
        return None
    maximize = META[problem][1]
    best, curve = None, []
    for line in open(logf):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r["tool"] == "grade_full" and r.get("valid") and r.get("score") is not None:
            if best is None or (r["score"] > best if maximize else r["score"] < best):
                best = r["score"]
            curve.append(best)
    return curve


def main(smoke=False):
    print(f"\n================ CDC benchmark (gpt-5.6-sol, {'SMOKE' if smoke else 'full'}) ================")
    print(f"{'problem':<8}{'agent best_full':>16}{'120b best':>13}{'ref/SOTA':>12}  "
          f"{'dir':>4}{'fast':>6}{'full':>6}{'secs':>7}")
    print("-" * 82)
    for prob, (ijson, maximize, sota, _lab) in META.items():
        tag = prob + ("_smoke" if smoke else "")
        rp = f"{BENCH}/{tag}/result.json"
        if not os.path.exists(rp):
            print(f"{prob:<8}{'(not run)':>16}")
            continue
        r = json.load(open(rp))
        b120 = baseline_120b(ijson, maximize)
        af = r.get("best_full")
        fmt = lambda x: f"{x:.6f}" if isinstance(x, (int, float)) else "—"
        print(f"{prob:<8}{fmt(af):>16}{fmt(b120):>13}{fmt(sota):>12}  "
              f"{'max' if maximize else 'min':>4}{r.get('n_grade_fast',0):>6}{r.get('n_grade_full',0):>6}"
              f"{r.get('secs',0):>7}")
    print("\nagent best_full = best VALID grade_full the deep-agent run achieved; 120b best = single-shot "
          "base-prompt probe; dir = optimize direction. Higher-vs-baseline is the headline per row.")


if __name__ == "__main__":
    import sys
    main(smoke="--smoke" in sys.argv)
