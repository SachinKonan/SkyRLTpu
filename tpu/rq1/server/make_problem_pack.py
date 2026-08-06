"""Build the portable per-problem data packs consumed by client/ collectors. NEURONIC-ONLY:
imports the discover trees + distill_ablation corpora from the MAIN checkout by absolute path.

Run with the discover venv, on a compute node (it grades each seed once at PRODUCTION budget
to stamp the true seed_score into the prompts -- the exp-1 regression was agents never being
told the number to beat):

  srun ... $PY make_problem_pack.py --problems fc46 erdos ac1 ud

Pack contents (client/data/<problem>/):
  prompt_agent.md        statement + seed + seed PRODUCTION score + direction (T1/T3)
  prompt_completion.md   same content as a pure completion prompt (T2/farm)
  seed.<py|cpp>          the seed program
  seed_construction.json (erdos only) the base construction, preloaded as initial_h_values
  meta.json              {problem, lang, fence, maximize, problem_type, seed_score, ...}

Build asserts the numeric seed score AND the direction sentence appear in both prompts.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO = "/n/fs/vision-mix/sk7524/SkyRLTpu"
AB = f"{REPO}/tpu/distill_ablation"          # untracked, disk-only -- referenced in place
MAIN = f"{REPO}/third_party/discover"
FRONTIER = "/n/fs/vision-mix/sk7524/SkyRLTpu-frontiercs/third_party/discover"
sys.path.insert(0, AB)                        # grading_mcp._grade
sys.path.insert(0, f"{AB}/portfolio")         # prompts.TASK

from grading_mcp import _grade  # noqa: E402
from prompts import TASK  # noqa: E402

CLIENT_DATA = Path(__file__).resolve().parent.parent / "client" / "data"

# problem -> (discover_root, env_module, env_class, problem_type, lang, maximize)
PROBLEMS = {
    "erdos": (MAIN, "examples.erdos_min_overlap.env", "ErdosMinOverlapEnv", "", "python", False),
    "ac1":   (MAIN, "examples.ac_inequalities.env", "AutoCorrInequalityEnv", "ac1", "python", False),
    "fc46":  (FRONTIER, "examples.frontier_algo.env", "FrontierAlgoEnv", "46", "cpp", True),
    "ud":    (MAIN, "examples.frontier_erdos_ud.env", "FrontierErdosUDEnv", "65536", "python", True),
}
FENCE = {"python": "python", "cpp": "cpp"}

UD_TASK = """Planar unit-distance problem: place exactly N = 65536 DISTINCT points in the
Euclidean plane so that the number of unordered pairs of points at Euclidean distance exactly 1
is as large as possible.

## Scoring
- A pair counts if |d^2 - 1| <= 1e-10 (squared-distance tolerance). Make distances exact to full
  float64 precision; do not rely on the tolerance.
- Validity: exactly 65536 points, all coordinates finite, and every two points at least 1e-3
  apart. An invalid construction scores 0.
- Your score is unit_pairs / N (the regular N-gon baseline scores 1.0; a triangular lattice patch
  scores about 3.0; the goal is to go as far beyond that as possible). HIGHER is better.
- If your construction naturally uses a different common distance, scale all coordinates so that
  the repeated distance is exactly 1 before returning.

## Rules
- Return one ```python block defining `run_construction()` (NO arguments) that returns a list of
  65536 (x, y) tuples.
- The production run gives your function 1000 seconds under a hard 1100s wall; return the best
  construction found within that budget.
- You may use numpy, scipy, math. All helper functions top level, no closures or lambdas.
- No filesystem or network IO.

## Background
Known strong constructions come from sections of scaled integer lattices (points with many
representations of a radius as sums of two squares), triangular-lattice patches, and Minkowski
sums / rotated unions of smaller unit-distance graphs (e.g. Moser spindles). The count of the
best known constructions grows superlinearly, n^(1 + c/log log n)."""

# Hand-written UD seed: 256x256 unit square lattice -> pairs = 2*256*255, score ~1.992.
# Deliberately simple with large headroom to the ~3.0 triangular-lattice reference.
UD_SEED = '''"""Seed: axis-aligned 256x256 unit square lattice (each interior point pairs with
its right and up neighbor at distance exactly 1)."""


def run_construction():
    pts = []
    for i in range(256):
        for j in range(256):
            pts.append((float(i), float(j)))
    return pts
'''


def load_seed(problem, idx=None):
    """(program_text, construction|None, recorded_score) -- copied from portfolio/sweep.py
    (same corpora, same median-of-nonzero-records selection)."""
    corpora = Path(AB) / "corpora"
    if problem == "erdos":
        ws = json.loads((corpora / "worse_set.json").read_text())["worse"]
        e = ws[idx if idx is not None else 2]
        return e["code"], list(e["construction"]), -e["value"]
    if problem == "ud":
        return UD_SEED, None, None
    f = {"fc46": "initial_fcalgo46.json", "ac1": "initial_ac1.json"}[problem]
    recs = json.loads((corpora / f).read_text())["records"]
    ok = [r for r in recs if r.get("code") and r.get("score") is not None]
    maximize = PROBLEMS[problem][5]
    if maximize:  # score==0 on the frontier problems means FAILURE, not a weak solution
        ok = [r for r in ok if r["score"] > 0] or ok
    ok.sort(key=lambda r: r["score"], reverse=not maximize)
    e = ok[idx if idx is not None else len(ok) // 2]
    return e["code"], None, e["score"]


def strip_fence(t):
    m = re.search(r"```(?:python|cpp|c\+\+)?\s*([\s\S]*?)\s*```", t or "")
    return m.group(1) if m else (t or "")


def task_text(problem):
    if problem == "ud":
        return UD_TASK
    if problem in TASK:  # erdos, ac1 -- hand-written minimal statements
        return TASK[problem]
    cache = Path(AB) / "corpora" / f"question_{problem}.txt"
    if not (cache.exists() and cache.read_text().strip()):
        raise SystemExit(f"missing cached statement {cache}; generate it via the exp-1 harness")
    return cache.read_text()


def build_one(problem, no_grade=False):
    root, mod, cls, ptype, lang, maximize = PROBLEMS[problem]
    fence = FENCE[lang]
    seed_prog, seed_constr, recorded = load_seed(problem)
    seed_prog = strip_fence(seed_prog)
    d = CLIENT_DATA / problem
    d.mkdir(parents=True, exist_ok=True)

    # ---- grade the seed once at PRODUCTION budget: the number every prompt must state ----
    if no_grade:
        score = recorded
        assert score is not None, f"{problem}: --no-grade needs a recorded seed score"
    else:
        print(f"[pack] {problem}: production-grading seed...", flush=True)
        t0 = time.time()
        payload = f"```{fence}\n{seed_prog}\n```" if lang == "python" else seed_prog
        g = _grade(root, mod, cls, ptype, lang, seed_constr, payload, False, str(d))
        print(f"[pack] {problem}: seed grade={g} in {int(time.time()-t0)}s", flush=True)
        if not g["valid"]:
            raise SystemExit(f"{problem}: seed FAILED production grading: {g['detail']}")
        score = g["score"]
        if recorded is not None and abs(score - recorded) > max(1e-6, abs(recorded) * 0.05):
            print(f"[pack] WARNING {problem}: graded {score} vs recorded {recorded}", flush=True)

    score_str = f"{score:.6g}"
    direction = ("HIGHER scores are BETTER (maximize)" if maximize
                 else "LOWER scores are BETTER (minimize)")
    better = "higher" if maximize else "lower"

    seed_block = [
        "## Starting program to improve on",
        f"This program's PRODUCTION score is {score_str}. {direction}. Your goal is a program "
        f"that scores meaningfully {better} than {score_str} at the same production budget.",
        "", f"```{fence}", seed_prog, "```"]
    if problem == "erdos":
        (d / "seed_construction.json").write_text(json.dumps(seed_constr))
        seed_block += [
            "", "Note: at grading time, `initial_h_values` (the seed's construction, the list "
            "below) and `evaluate_erdos_solution()` are pre-imported for your program. For local "
            "testing, use this same list:",
            "```json", json.dumps([round(x, 12) for x in seed_constr]), "```"]

    body = task_text(problem).rstrip() + "\n\n" + "\n".join(seed_block) + "\n"
    agent = body  # T1/T3 append their own tool/workflow contract at collect time
    completion = (body + "\nThink the problem through, then output your single best COMPLETE "
                  f"program as ONE ```{fence} code block (the last such block in your reply is "
                  "taken as your answer). The program must compute its result within the stated "
                  "budget; embedding a precomputed answer as data (literal arrays, base64) is "
                  "invalid and will be rejected.\n")
    for name, text in [("prompt_agent.md", agent), ("prompt_completion.md", completion)]:
        assert score_str in text, f"{problem}/{name}: seed score missing"
        assert direction in text, f"{problem}/{name}: direction missing"
        (d / name).write_text(text)
    (d / f"seed.{'py' if lang == 'python' else 'cpp'}").write_text(seed_prog)
    (d / "meta.json").write_text(json.dumps({
        "problem": problem, "lang": lang, "fence": fence, "maximize": maximize,
        "problem_type": ptype, "seed_score": score, "seed_recorded_score": recorded,
        "built": time.strftime("%F %T")}, indent=2))
    print(f"[pack] {problem}: OK seed_score={score_str} -> {d}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", nargs="+", default=list(PROBLEMS))
    ap.add_argument("--no-grade", action="store_true",
                    help="use recorded seed scores instead of re-grading (NOT for real runs)")
    args = ap.parse_args()
    for p in args.problems:
        build_one(p, args.no_grade)


if __name__ == "__main__":
    main()
