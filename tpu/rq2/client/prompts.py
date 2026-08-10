"""The three prompts RQ2 uses. Held IDENTICAL across compositions, scales and problems -- only
the state differs -- so any measured difference is attributable to the treatment.

Design note on strategy guidance. Exp-1 measured a directive CDC prompt moving orchestrated
quality +47% at identical agent count, so strategy text is powerful enough to swamp the effects
under study. The resolution is not to avoid it but to hold it CONSTANT: every orchestrated cell
gets the same CDC scaffolding, which makes it the definition of the treatment rather than a
variable. The compaction prompt gets the opposite treatment -- for PDR the workspace content is
precisely what is under test, so telling the compactor what to look for would be us testing our
own priors.
"""
from __future__ import annotations

# ----------------------------------------------------------------- generator (on-prem models)
ROLLOUT_TAIL = """
Think the problem through, then output your single best COMPLETE program as ONE ```{fence} code
block (the last such block in your reply is taken as your answer). The program must compute its
result within the stated budget; embedding a precomputed answer as data (literal arrays, base64)
is invalid and will be rejected.
"""

# Reflection is a SECOND call that resumes the same conversation once the grade is known, so the
# model reflects on a measured outcome rather than on its intentions. Two sentences keeps it
# cheap (short decode over a cached prefix) and forces a point rather than a summary.
REFLECT = """{outcome}

In exactly two sentences: what did this approach actually do, and what should the next attempt
change? Be concrete and technical -- name the mechanism, not the intention."""


def reflect_prompt(score, maximize, baseline):
    if score is None:
        return REFLECT.format(
            outcome="Your program did NOT produce a usable result -- it failed to run, timed "
                    "out, or returned something invalid.")
    direction = "higher is better" if maximize else "lower is better"
    if baseline is None:
        verdict = ""
    elif (score > baseline) == maximize and score != baseline:
        verdict = f" That beats the {baseline} starting point."
    else:
        verdict = f" That does NOT beat the {baseline} starting point."
    return REFLECT.format(outcome=f"Your program scored {score} ({direction}).{verdict}")


# ------------------------------------------------------------------- orchestrator (sol, xhigh)
# CDC scaffolding from exp-1, minus its budget-nagging (submit_plan enforces the budget
# structurally) and minus its grader-verification clauses (there is no grader in this loop).
ORCHESTRATOR = """You are allocating a budget of {n} sub-agent calls for one step of a search for
a better program.

Each sub-agent is an independent language model that will receive the problem plus the
instruction you assign its group, and will return one program. You decide how many distinct
instructions there are and how many sub-agents get each.

Manage the search with these heuristics:
- Begin from a genuinely diverse portfolio: substantially different formulations, algorithms,
  heuristics, representations, decompositions and relaxations.
- Do not give every group the currently favoured approach -- preserve independence so the step
  does not converge on one attractive-but-mediocre idea.
- Maintain an explicit registry of approach families, grouped by underlying idea rather than
  wording, and steer budget toward underexplored ones.
- Keep several incompatible approaches alive; cross-pollinate only once independent groups have
  developed them far enough to expose their real strengths and gaps.

WHAT YOU HAVE: {state_desc}
{state_where}

WHAT YOU MUST DO: write a JSON file, then call submit_plan(path) with its absolute path.
  [{{"instruction": "<what this group should try>", "n": <int>}}, ...]
The group sizes must sum to EXACTLY {n}. A plan that totals anything else is rejected with the
reason; fix it and resubmit. You have {turns} turns; spend them reading the state before you
commit, then submit once.
"""


def orchestrator_prompt(n, turns, state_desc, state_where):
    return ORCHESTRATOR.format(n=n, turns=turns, state_desc=state_desc, state_where=state_where)


# ------------------------------------------------------- compaction (gpt-5.3-codex-spark, PDR)
# Deliberately says nothing about WHAT to keep: the content of the workspace is the treatment.
COMPACTOR = """One step of a program-search just finished. Every program it produced, with its
score, is in this file:

  {round_md}

The current shared workspace is here (it may be empty on the first step):

  {workspace}

Write the NEW shared workspace to {out_path}.

That workspace is the only thing carried into the next step: every sub-agent in the next round
will be given it alongside the problem, and nothing else from this round survives. You decide
entirely what it should contain and how long it should be.

The round file may be large -- read it however you like (grep, head, sed). You have {turns}
turns. Write the file, then stop.
"""


def compactor_prompt(round_md, workspace, out_path, turns):
    return COMPACTOR.format(round_md=round_md, workspace=workspace,
                            out_path=out_path, turns=turns)
