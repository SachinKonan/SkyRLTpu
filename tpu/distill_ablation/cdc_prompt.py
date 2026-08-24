"""Build the per-problem CDC-style deep-agent prompt for the benchmark.

Run as a per-problem SUBPROCESS (`--problem fc46 --out prompt.txt`) so the import path is clean
(frontier lives in a different discover tree). The prompt =
  SCAFFOLDING  (OpenAI CDC orchestration, problem-agnostic)
  + env.get_question(base_state)   (exact problem statement, same base the 120b probe used)
  + ADVERSARIAL_CHECKS[problem]     (the problem's real validity conditions — the CDC-audit swap)
  + TOOL_GUIDANCE                   (grade_fast / grade_full — optimize against the real reward)
  + OUTPUT_CONTRACT                 (return one code block in the right language)
"""
from __future__ import annotations

import argparse
import importlib
import sys

REPO = "/n/fs/vision-mix/sk7524/SkyRLTpu"
AB = f"{REPO}/tpu/distill_ablation"

SCAFFOLDING = """You have a large team of agents available through multiagent v2. Use them \
aggressively and dynamically — up to 32 concurrent agents. Do NOT use a fixed assignment like \
"N agents for approach X." Manage the search with these heuristics:
- Begin with a genuinely diverse portfolio: substantially different formulations, algorithms, \
heuristics, representations, decompositions, relaxations, and computational sanity checks.
- Do not tell most agents the currently favored approach — preserve independence in early rounds so \
the team does not all converge on one attractive-but-mediocre idea.
- Maintain an explicit registry of approach families (grouped by the underlying idea, not wording). \
If many agents converge to one family, redirect some toward underexplored ones.
- Keep several incompatible approaches alive across rounds; cross-pollinate only after independent \
agents have developed them far enough to expose their real strengths and gaps.
- Use adversarial agents throughout: every candidate must be checked against the validity conditions \
below and its score confirmed before you trust it.
- Require agents to return concrete programs/constructions and their measured scores — reject status \
reports, vague optimism, and untested claims.
- The root agent should repeatedly synthesize, challenge, redirect, and launch new rounds. Do not \
stop after the first wave fails. Keep searching for a better solution until your time budget is \
nearly exhausted; then return the single strongest VALIDATED solution you found."""

ADVERSARIAL_CHECKS = {
    "erdos": """## Validity & scoring (audit every candidate)
An invalid construction scores nothing. The returned h MUST satisfy 0 ≤ h[i] ≤ 1 for all i and \
sum(h)*dx == 1 exactly (equivalently sum(h) == n_points/2). The score is C5 = \
max(np.correlate(h, 1-h, mode="full"))*dx and **LOWER is better** (you are pushing the upper bound \
on the Erdős constant down). Reject any candidate that violates the box/sum constraints or does not \
actually lower C5 below the current record.""",
    "ac1": """## Validity & scoring (audit every candidate)
The sequence must be nonnegative, its sum must not be near zero, and its autoconvolution must have \
no inf/nan. The objective is C1 = ||f*f||_inf / ||f||_1^2 and **LOWER is better**. Reject degenerate \
sequences.""",
    "ac2": """## Validity & scoring (audit every candidate)
The sequence must be nonnegative with sum ≥ 0.01 and no inf/nan/blow-up in f*f. The objective is \
C2 = ||f*f||_2^2 / (||f*f||_1 * ||f*f||_inf) and **HIGHER is better**. WARNING: pushing toward a \
spike or a near-zero sum to inflate the ratio makes the construction INVALID (it scores nothing) — \
check every candidate stays strictly feasible.""",
    "fc46": """## Validity & scoring (audit every candidate)
Output a legal NON-PREEMPTIVE schedule: each job is processed once on each machine in its route \
order, and each machine runs at most one operation at a time. Your C++17 program must compile and \
each test case must finish within its CPU time limit. The score rewards a LOWER makespan (**higher \
ratio is better**). Reject illegal schedules and time-limit violations.""",
    "fc302": """## Validity & scoring (audit every candidate)
The output string must use EXACTLY the required per-letter usage counts over the given alphabet and \
finish within the time limit. The score rewards a LOWER interference penalty (**higher ratio is \
better**). Reject wrong letter counts and time-limit violations.""",
}

TOOL_GUIDANCE = """## Reward tools (optimize against the REAL evaluator)
You have two MCP grading tools that run the actual scorer:
- `grade_fast(solution)` — reduced budget (~60s), cheap and approximate. Use it to screen many \
candidate ideas quickly.
- `grade_full(solution)` — full budget (~1000s / native limits), slow but AUTHORITATIVE. This is the \
metric that counts; use it to confirm a candidate you already screened with `grade_fast`.
Pass the full solution source (a code block or raw source). Your goal: **maximize the `grade_full` \
score in the direction the problem requires** (see "Validity & scoring" above for whether lower or \
higher is better). Generate diverse candidates, screen with `grade_fast`, and confirm the best with \
`grade_full`. Call the tools liberally — that is how you optimize."""

OUTPUT_CONTRACT = """## Output
When your time budget is nearly exhausted, return your SINGLE best solution — the one with the best \
`grade_full` score — as ONE ```{fence} code block, with nothing after it. It must pass `grade_full` \
(be valid)."""

# problem -> (discover_root, env_module, env_class, problem_type, lang)
PROBLEMS = {
    "erdos": (f"{REPO}/third_party/discover", "examples.erdos_min_overlap.env", "ErdosMinOverlapEnv", "", "python"),
    "ac1":   (f"{REPO}/third_party/discover", "examples.ac_inequalities.env", "AutoCorrInequalityEnv", "ac1", "python"),
    "ac2":   (f"{REPO}/third_party/discover", "examples.ac_inequalities.env", "AutoCorrInequalityEnv", "ac2", "python"),
    "fc46":  ("/n/fs/vision-mix/sk7524/SkyRLTpu-frontiercs/third_party/discover", "examples.frontier_algo.env", "FrontierAlgoEnv", "46", "cpp"),
    "fc302": ("/n/fs/vision-mix/sk7524/SkyRLTpu-frontiercs/third_party/discover", "examples.frontier_algo.env", "FrontierAlgoEnv", "302", "cpp"),
}
GRPO_SNAPSHOT = f"{REPO}/runs/ttd_obj_grpo_16x32/tinker_log/erdos-20b-16x32-grpo/puct_sampler_step_000019.json"


class _DummySampler:
    pass


def get_question(problem):
    root, mod, cls, pt, lang = PROBLEMS[problem]
    sys.path.insert(0, root)
    sys.path.insert(0, AB)
    import gen_initial  # reuse env_bits_generic (renderer/tok/cfg for any env)
    EnvClass = getattr(importlib.import_module(mod), cls)
    renderer, tok, cfg = gen_initial.env_bits_generic(EnvClass, pt, f"/tmp/cdcprompt_{problem}", 1100)
    if problem == "erdos":
        import common
        sampler = common.load_pool_sampler(GRPO_SNAPSHOT, f"/tmp/cdcprompt_{problem}/pool")
        with sampler._lock:
            base = max((s for s in sampler._states if s.value is not None and s.code and s.code.strip()),
                       key=lambda s: s.value)
    else:
        base = EnvClass.create_initial_state(pt)
    env = EnvClass(renderer, initial_state=base, sampler=_DummySampler(), config=cfg)
    return env.get_question(), lang


def build_cdc_prompt(problem):
    q, lang = get_question(problem)
    fence = "python" if lang == "python" else "cpp"
    return "\n\n".join([
        SCAFFOLDING, q, ADVERSARIAL_CHECKS[problem], TOOL_GUIDANCE,
        OUTPUT_CONTRACT.format(fence=fence),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True, choices=list(PROBLEMS))
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    prompt = build_cdc_prompt(args.problem)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(prompt)
        print(f"[cdc_prompt] {args.problem}: wrote {len(prompt)} chars -> {args.out}")
    else:
        sys.stdout.write(prompt)


if __name__ == "__main__":
    main()
