# RQ1 collector — operator runbook (Meta devserver agent)

You are operating the **Meta-side half** of a two-site experiment. You run the full-agent
collection cells (B and E) on this devserver; everything else (open-source model farm, all
grading, analysis) happens on the owner's cluster. Your job is to produce clean run
directories, nothing more.

## What this experiment is (context)

RQ1: given one code-optimization problem, what is the best way to collect N candidate
solutions? Cell B = 200 *independent* codex rollouts. Cell E = one orchestrator that spends the
same call budget on subagents it manages dynamically. Rollouts get the problem + a seed program
+ the seed's production score, full local shell to test with, and **no grader** — they finish by
calling `submit()` on a local capture MCP exactly once. Scores are computed later, offline, by
the owner. Comparability is the whole point: do not tweak prompts, budgets, or models.

## One-time setup

1. Needs: Linux, `uv` (https://astral.sh/uv), the `codex` CLI on PATH and logged in
   (`codex login` if `~/.codex/auth.json` is missing), `g++` (C++ problems), `python3`.
2. Get this directory (`tpu/rq1/client/` of branch `rq1-collection`) onto the box — git clone
   or the tarball you were given. Everything runs from inside `client/`.
3. Preflight (2 short codex probes, ~2 min):

       uv run preflight.py

   Read its recommendation. On a normal devserver expect `--site default`; only if it reports
   a broken sandbox or stream drops use the profile it suggests. If codex/auth is broken it
   says so — fix that before anything else.

## Cell B (run this first)

Smoke, then the real thing. `--out` dirs live under `runs/` next to `client/`.

    # smoke: 3 rollouts, ~25 min. PASS = >=2 rows in runs/fc46_B_smoke/submissions.jsonl
    uv run collect_t1.py --problem fc46 --n 3 --concurrency 3 \
        --model gpt-5.6-sol --effort xhigh --site auto --out ../../runs/fc46_B_smoke

    # real cell B: 200 rollouts, 25 concurrent, ~3-4 h wall
    uv run collect_t1.py --problem fc46 --n 200 --concurrency 25 \
        --model gpt-5.6-sol --effort xhigh --site auto --out ../../runs/fc46_B --resume

- `--resume` is safe to re-run after any interruption; finished rollouts are skipped.
- Healthy signs: steady `[t1] rNNN: ok` lines; `submissions.jsonl` growing; occasional
  `+salvaged` is fine. Bad sign: many `+no-program` → open one `runs/fc46_B/rNNN/events.jsonl`
  and look for `bwrap`/`stream disconnected`/auth errors, re-run preflight, fix, resume.
- Success bar: >= 180/200 submissions.

## Cell E (after B finishes)

    # one orchestrator, 100-subagent budget, <=25 concurrent, ~4 h wall
    uv run orchestrate_t3.py --arm E --problem fc46 --budget 100 --concurrency 25 \
        --model gpt-5.6-sol --effort xhigh --subagent-model gpt-5.6-sol \
        --site auto --out ../../runs/fc46_E

  Only `gpt-5.6-sol` is available here — do not substitute another model anywhere.
  Success bar: `submissions.jsonl` is non-empty (subagents each submit; count varies by the
  orchestrator's choices — that variance is data, not a bug; do NOT re-run a completed E
  because the count "looks low").

## Other problems

Same two commands with `--problem erdos | ac1 | ud` **once those packs exist in `client/data/`
and the owner says go**. fc46 is the shakedown; do not start the others on your own.

## Hand results back

After each cell completes:

    tar czf rq1_runs_$(date +%Y%m%d_%H%M).tgz -C ../../ runs

and deliver the tarball the way the owner asked (or `git add -f runs && git push` to the
`rq1-collection` branch if you have push access). A run dir is complete when it has
`manifest.json`, `submissions.jsonl`, `solutions/*.txt`, and per-rollout `r*/events.jsonl`.

## Hard rules

- NEVER grade, score, or filter solutions; never edit anything in `data/` or the prompts in
  the collector scripts. If something seems wrong with a pack, stop and report it.
- One cell at a time (B, then E). Never exceed `--n 200` / `--budget 100` / 25 concurrent.
- Don't delete or rewrite an existing run dir — `--resume` or a NEW `--out` suffix (`_r2`).
- Keep the raw event logs; they are part of the deliverable.
- If >20% of rollouts fail with infrastructure errors, stop and report rather than burning
  the rest of the budget.
