"""The two state representations under test in RQ2, behind one interface.

The loop is identical for both; the treatment is entirely what `render` hands the sub-agents and
what `update` keeps. Weights never move, so every difference in outcome is attributable to this
file.

  PuctState       a tree of programs, each with its score and its own grade-aware reflection.
                  render() selects up to 5 states per prompt (SimpleTES mixing) via RPUCG, so
                  every sub-agent in a step sees a DIFFERENT lineage. Nothing is ever discarded,
                  so a mediocre program can be revisited once something makes it look promising.

  WorkspaceState  one unbounded markdown document that a compaction agent rewrites each step.
                  render() hands every sub-agent the SAME text, so the round explores from one
                  consensus view, and the step's n programs collapse into whatever the compactor
                  chose to write down.

Selection reuses the RPUCG store built for RQ1 (percentile-based value propagation, one-hop
exclusion) rather than reimplementing it -- see portfolio/store.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/n/fs/vision-mix/sk7524/SkyRLTpu/tpu/distill_ablation/portfolio")
from store import Store  # noqa: E402

MAX_MIX = 5          # matches SimpleTES's num_inspirations default of 5
MAX_PROGRAM_CHARS = 24000
# SimpleTES's generation prompt carries THREE context channels, not one (verified against the
# repo: templates/generation.py + templates/elite_context.py). We had only the first:
#   [SAMPLED INSPIRATIONS] ~5 full solutions WITH code+reflections   -- depth
#   [ELITE POOL OVERVIEW]  N solutions, scores+insights, NO code     -- breadth
#   [FAILURE PATTERNS]     aggregated common errors                  -- negative constraints
# The latter two are assembled in CODE, not by an LLM, so they cost nothing and add no
# aggregator to this arm.
N_ELITE = 12
N_FAILURE_PATTERNS = 4


class State:
    """Common interface. `render` returns exactly n prompt bodies; `update` folds a graded
    round in; `best` is what the cell finally reports."""

    def render(self, n: int, task: str, fence: str) -> list[str]:
        raise NotImplementedError

    def update(self, results: list[dict], step: int, **kw) -> None:
        raise NotImplementedError

    def best(self) -> tuple[str | None, float | None]:
        raise NotImplementedError


# --------------------------------------------------------------------------- PUCT / SimpleTES
class PuctState(State):
    def __init__(self, root: Path, maximize: bool, seed_program: str, seed_score: float | None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.maximize = maximize
        gpath = self.root / "graph.json"
        if gpath.exists():
            self.store = Store.load(gpath)
        else:
            self.store = Store(maximize, gpath)
            self.store.add(seed_program, seed_score, [], feedback="seed", rnd=0, summary="seed")
            self.store.backup_U()
            self.store.save()

    def render(self, n, task, fence):
        """n prompts, each conditioned on its own group of up to MAX_MIX states.

        RPUCG's one-hop exclusion caps how many genuinely distinct groups exist, which is much
        smaller than n at these scales (n=500 against a tree that starts with one node). Rather
        than force spurious distinctness, cycle the selected groups -- repeats are real parallel
        attempts on the same lineage, which is what a wide step IS."""
        groups = self.store.rpucg_select(max(1, n), MAX_MIX) or [[self.store.best()]]
        out = []
        for i in range(n):
            grp = groups[i % len(groups)]
            metas = [self.store.meta(g, with_program=True) for g in grp]
            parts = [task, ""]
            if len(metas) == 1:
                parts.append("## Improve substantially on this program")
            else:
                parts.append(f"## These {len(metas)} programs take DIFFERENT approaches. "
                             "Combine their complementary strengths into one better program")
            for m in metas:
                parts.append(f"\n### node {m['id']} (score {m['r']})")
                if m.get("summary"):
                    parts.append(f"Reflection from its author: {m['summary']}")
                parts.append(f"```{fence}\n{(m.get('program') or '')[:MAX_PROGRAM_CHARS]}\n```")
            elite = self._elite_overview()
            if elite:
                parts += ["", elite]
            fails = self._failure_patterns()
            if fails:
                parts += ["", fails]
            out.append("\n".join(parts))
        return out

    def _elite_overview(self):
        """Breadth channel: what directions have already been tried, scores and reflections only
        -- deliberately NO code, so it stays cheap enough to show many more than the 5 mixed
        inspirations."""
        scored = [n for n in self.store.g.nodes if self.store.g.nodes[n].get("r") is not None]
        if len(scored) < 2:
            return ""
        scored.sort(key=lambda n: self.store.g.nodes[n]["r"], reverse=self.maximize)
        lines = [f"[ELITE POOL OVERVIEW] ({min(N_ELITE, len(scored))} of {len(scored)} solutions "
                 f"so far; scores and insights only)",
                 "Use this to see which directions are already explored, avoid duplicating them,",
                 "and identify gaps worth a genuinely different approach.", ""]
        for i, n in enumerate(scored[:N_ELITE], 1):
            nd = self.store.g.nodes[n]
            refl = (nd.get("summary") or "").strip().replace("\n", " ")
            lines.append(f"#{i} [score {nd['r']}] {refl[:240]}")
        return "\n".join(lines)

    def _failure_patterns(self):
        """Negative constraints: the round's most common failure reasons, counted in code."""
        from collections import Counter
        import re as _re
        c = Counter()
        for n in self.store.g.nodes:
            nd = self.store.g.nodes[n]
            if nd.get("r") is not None:
                continue
            d = (nd.get("feedback") or "").strip()
            if not d:
                continue
            c[_re.sub(r"0x[0-9a-f]+|\d{3,}", "#", d)[:110]] += 1
        if not c:
            return ""
        return ("[FAILURE PATTERNS] (common errors to avoid)\n"
                + "\n".join(f"- ({k}x) {m}" for m, k in c.most_common(N_FAILURE_PATTERNS)))

    def update(self, results, step, **kw):
        """Append every graded program with its own reflection. Nothing is discarded -- the
        population IS the state."""
        for r in results:
            if not r.get("program"):
                continue
            self.store.add(r["program"], r.get("score"), r.get("parents") or [],
                           feedback=(r.get("detail") or "")[:200], rnd=step,
                           summary=(r.get("reflection") or "")[:400])
        self.store.backup_U()
        self.store.save()

    def best(self):
        b = self.store.best()
        if b is None:
            return None, None
        nd = self.store.g.nodes[b]
        return nd.get("program"), nd.get("r")


# ------------------------------------------------------------------------------- PDR workspace
class WorkspaceState(State):
    def __init__(self, root: Path, maximize: bool, seed_program: str, seed_score: float | None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.maximize = maximize
        self.seed_program = seed_program
        self.seed_score = seed_score
        self.wpath = self.root / "workspace.md"
        self.bpath = self.root / "best.json"
        if not self.wpath.exists():
            self.wpath.write_text("")          # starts EMPTY, by design
        if not self.bpath.exists():
            self.bpath.write_text(json.dumps({"program": seed_program, "score": seed_score}))

    @property
    def workspace(self):
        return self.wpath.read_text()

    def render(self, n, task, fence):
        """n IDENTICAL prompts: the whole point of the treatment is that every sub-agent in a
        step reads the same accumulated text."""
        parts = [task, ""]
        ws = self.workspace.strip()
        if ws:
            parts += ["## Accumulated findings so far (shared workspace)", ws, ""]
        parts += ["## Starting program",
                  f"```{fence}\n{self.seed_program[:MAX_PROGRAM_CHARS]}\n```"]
        return ["\n".join(parts)] * n

    def round_md(self, results, step):
        """The round dumped to markdown for the compaction agent to grep. Deliberately the FULL
        round -- PDR's premise is that the compactor sees every solution, and a file it can
        search sidesteps the context limit that inlining 500 programs would hit."""
        p = self.root / f"round_{step:02d}.md"
        ok = [r for r in results if r.get("score") is not None]
        ok.sort(key=lambda r: r["score"], reverse=self.maximize)
        L = [f"# Step {step}: {len(results)} programs, {len(ok)} valid",
             f"direction: {'higher is better' if self.maximize else 'lower is better'}", "",
             "## Scores", "", "| id | score | status |", "|---|---|---|"]
        for r in results:
            L.append(f"| {r.get('sid')} | {r.get('score')} | "
                     f"{'ok' if r.get('score') is not None else (r.get('detail') or 'invalid')[:60]} |")
        L += ["", "## Programs (best first)"]
        for r in ok:
            L += [f"\n### {r['sid']}  score={r['score']}", "```",
                  (r.get("program") or "")[:MAX_PROGRAM_CHARS], "```"]
        p.write_text("\n".join(L))
        return p

    def update(self, results, step, compact_fn=None, **kw):
        """compact_fn(round_md_path, current_workspace_path) -> new workspace text.
        Supplied by the loop so this class stays free of agent plumbing."""
        md = self.round_md(results, step)
        cur = json.loads(self.bpath.read_text())
        for r in results:
            if r.get("score") is None:
                continue
            better = (cur["score"] is None
                      or (r["score"] > cur["score"] if self.maximize else r["score"] < cur["score"]))
            if better:
                cur = {"program": r["program"], "score": r["score"]}
        self.bpath.write_text(json.dumps(cur))
        if compact_fn is not None:
            new = compact_fn(md, self.wpath)
            if new and new.strip():
                self.wpath.write_text(new)
                (self.root / f"workspace_{step:02d}.md").write_text(new)

    def best(self):
        c = json.loads(self.bpath.read_text())
        return c.get("program"), c.get("score")


def make_state(kind, root, maximize, seed_program, seed_score):
    if kind == "puct":
        return PuctState(root, maximize, seed_program, seed_score)
    if kind == "workspace":
        return WorkspaceState(root, maximize, seed_program, seed_score)
    raise ValueError(f"unknown state kind: {kind}")
