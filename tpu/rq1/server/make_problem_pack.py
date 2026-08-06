"""Build the portable per-problem data packs consumed by client/ collectors. NEURONIC-ONLY:
imports the discover trees + distill_ablation corpora from the MAIN checkout by absolute path.

PROMPTS ARE ENV-FAITHFUL: the statement is each env's own `get_question()` -- the exact prompt
shape our RL runs use (formal problem + rules + seed program + its score + target/gap +
"improve meaningfully" closer) -- with OUR seed injected as the initial state and its
PRODUCTION-graded score as the state value. Nothing hand-written. This keeps RQ1 cells
comparable to the RL runs and to each other.

Run with the discover venv, on a compute node (seed grading = real 1000s production runs):

  srun ... $PY make_problem_pack.py --problems fc46 erdos ac1 ud [--reuse-scores]

Pack contents (client/data/<problem>/):
  prompt_agent.md        get_question(seed-state) [+ erdos local-testing construction note]
  prompt_completion.md   same + single-code-block closer (T2/farm)
  seed.<py|cpp>          the seed program
  seed_construction.json (erdos) base construction, preloaded as initial_h_values at grading
  meta.json              {problem, lang, fence, maximize, problem_type, seed_score, seed_sha, ...}

Build asserts the graded score AND the correct direction phrase appear in both prompts.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
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

from grading_mcp import _grade  # noqa: E402

CLIENT_DATA = Path(__file__).resolve().parent.parent / "client" / "data"

# problem -> (discover_root, env_module, env_class, problem_type, lang, maximize)
PROBLEMS = {
    "erdos": (MAIN, "examples.erdos_min_overlap.env", "ErdosMinOverlapEnv", "", "python", False),
    "ac1":   (MAIN, "examples.ac_inequalities.env", "AutoCorrInequalityEnv", "ac1", "python", False),
    "fc46":  (FRONTIER, "examples.frontier_algo.env", "FrontierAlgoEnv", "46", "cpp", True),
    "ud":    (MAIN, "examples.frontier_erdos_ud.env", "FrontierErdosUDEnv", "65536", "python", True),
}
FENCE = {"python": "python", "cpp": "cpp"}

# Hand-written UD seed: 256x256 unit square lattice, score 1.9921875 (each interior point
# pairs right+up at distance exactly 1). Simple, valid, big headroom to the ~3.0 lattice ref.
UD_SEED = '''"""Seed: axis-aligned 256x256 unit square lattice (each interior point pairs with
its right and up neighbor at distance exactly 1)."""


def run_construction():
    pts = []
    for i in range(256):
        for j in range(256):
            pts.append((float(i), float(j)))
    return pts
'''


def _load_env_class(root, mod, cls):
    """Both discover trees ship an `examples` package; python caches whichever loaded first.
    Purge and re-point sys.path per problem so each env comes from ITS root."""
    for k in [k for k in sys.modules if k == "examples" or k.startswith("examples.")]:
        del sys.modules[k]
    while root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    return getattr(importlib.import_module(mod), cls)


def sha(s):
    return hashlib.sha1((s or "").encode()).hexdigest()[:12]


def strip_fence(t):
    m = re.search(r"```(?:python|cpp|c\+\+)?\s*([\s\S]*?)\s*```", t or "")
    return m.group(1) if m else (t or "")


def load_seed(problem, idx=None):
    """(program_text, construction|None, parent_values|None, observation, recorded_score)."""
    corpora = Path(AB) / "corpora"
    if problem == "erdos":
        ws = json.loads((corpora / "worse_set.json").read_text())["worse"]
        e = ws[idx if idx is not None else 2]
        return (e["code"], list(e["construction"]), e.get("parent_values"),
                e.get("observation") or "", -e["value"])
    if problem == "ud":
        return UD_SEED, None, None, "", 1.9921875
    f = {"fc46": "initial_fcalgo46.json", "ac1": "initial_ac1.json"}[problem]
    recs = json.loads((corpora / f).read_text())["records"]
    ok = [r for r in recs if r.get("code") and r.get("score") is not None]
    maximize = PROBLEMS[problem][5]
    if maximize:  # score==0 on the frontier problems means FAILURE, not a weak solution
        ok = [r for r in ok if r["score"] > 0] or ok
        # ASCENDING sort, median index -- matches exp-1's sweep.load_seed exactly, keeping the
        # fc46 seed at the 0.073942 program (a descending sort here silently swaps the seed).
        ok.sort(key=lambda r: r["score"])
        e = ok[idx if idx is not None else len(ok) // 2]   # mid-range: room to improve
    else:
        # ac1: 8/22 records sit on the trivial 2.0 plateau (constant-function value); the
        # "median" seed landed there. Take the WORST score that is still genuinely below the
        # plateau -- a real mid-range program with headroom to the 1.503 target.
        genuine = [r for r in ok if r["score"] < 1.999] or ok
        e = max(genuine, key=lambda r: r["score"]) if idx is None else \
            sorted(ok, key=lambda r: r["score"])[idx]
    return e["code"], None, None, "", e["score"]


def build_question(problem, seed_prog, score, constr, parents, observation):
    """env.get_question() with OUR seed as the initial state. get_question implementations
    only touch initial_state / problem_type / eval_timeout (+ os.environ for fc46), so an
    attribute-stub env suffices -- no renderer/sampler/config needed."""
    root, mod, cls, ptype, lang, maximize = PROBLEMS[problem]
    EnvClass = _load_env_class(root, mod, cls)
    state = EnvClass.create_initial_state(ptype)
    state.code = seed_prog
    state.value = score if maximize else -score   # State stores negated values for minimize
    if constr is not None:
        state.construction = constr
    state.parent_values = parents                 # erdos: real before->after like the RL runs
    state.observation = observation
    env = EnvClass.__new__(EnvClass)
    env.initial_state = state
    env.problem_type = ptype
    env.eval_timeout = 1000                        # question text shows the production budget
    q = env.get_question()
    if not maximize:
        # State.to_prompt hardcodes "(higher is better)" in its no-parent branch; our RL runs
        # never hit it (states always have parents). Fix the direction for minimize problems.
        q = q.replace("(higher is better):", "(lower is better):")
    return q


def build_one(problem, reuse_scores=False):
    root, mod, cls, ptype, lang, maximize = PROBLEMS[problem]
    fence = FENCE[lang]
    seed_prog, constr, parents, observation, recorded = load_seed(problem)
    seed_prog = strip_fence(seed_prog)
    seed_sha = sha(seed_prog)
    d = CLIENT_DATA / problem
    d.mkdir(parents=True, exist_ok=True)

    # ---- the number every prompt must state: ONE production grade of the seed ----
    score = None
    mf = d / "meta.json"
    if reuse_scores and mf.exists():
        old = json.loads(mf.read_text())
        if old.get("seed_sha") == seed_sha and old.get("seed_score") is not None:
            score = old["seed_score"]
            print(f"[pack] {problem}: reusing graded seed_score={score}", flush=True)
    if score is None:
        print(f"[pack] {problem}: production-grading seed...", flush=True)
        t0 = time.time()
        payload = f"```{fence}\n{seed_prog}\n```" if lang == "python" else seed_prog
        g = _grade(root, mod, cls, ptype, lang, constr, payload, False, str(d))
        print(f"[pack] {problem}: seed grade={g} in {int(time.time()-t0)}s", flush=True)
        if not g["valid"]:
            raise SystemExit(f"{problem}: seed FAILED production grading: {g['detail']}")
        score = g["score"]
        if recorded is not None and abs(score - recorded) > max(1e-6, abs(recorded) * 0.05):
            print(f"[pack] WARNING {problem}: graded {score} vs recorded {recorded}", flush=True)

    question = build_question(problem, seed_prog, score, constr, parents, observation)

    agent = question
    if problem == "erdos":
        (d / "seed_construction.json").write_text(json.dumps(constr))
        agent += ("\n\n## Local testing note\n"
                  "At grading time `initial_h_values` (the current construction) and "
                  "`evaluate_erdos_solution()` are pre-imported for your program. For local "
                  "testing, `initial_h_values` is exactly this list:\n```json\n"
                  + json.dumps([round(x, 12) for x in constr]) + "\n```")
    completion = (question + "\n\nThink the problem through, then output your single best "
                  f"COMPLETE program as ONE ```{fence} code block (the last such block in "
                  "your reply is taken as your answer). The program must compute its result "
                  "within the stated budget; embedding a precomputed answer as data (literal "
                  "arrays, base64) is invalid and will be rejected.\n")

    s6 = f"{score:.6f}"
    direction = "higher is better" if maximize else "lower is better"
    for name, text in [("prompt_agent.md", agent), ("prompt_completion.md", completion)]:
        assert s6 in text, f"{problem}/{name}: graded seed score {s6} missing from prompt"
        assert direction in text, f"{problem}/{name}: direction phrase '{direction}' missing"
        (d / name).write_text(text)
    (d / f"seed.{'py' if lang == 'python' else 'cpp'}").write_text(seed_prog)
    (d / "meta.json").write_text(json.dumps({
        "problem": problem, "lang": lang, "fence": fence, "maximize": maximize,
        "problem_type": ptype, "seed_score": score, "seed_recorded_score": recorded,
        "seed_sha": seed_sha, "prompt_style": "env_get_question",
        "built": time.strftime("%F %T")}, indent=2))
    print(f"[pack] {problem}: OK seed_score={s6} prompt_chars={len(agent)} -> {d}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", nargs="+", default=list(PROBLEMS))
    ap.add_argument("--reuse-scores", action="store_true",
                    help="reuse an existing meta.json seed_score when the seed is unchanged")
    args = ap.parse_args()
    for p in args.problems:
        build_one(p, args.reuse_scores)


if __name__ == "__main__":
    main()
