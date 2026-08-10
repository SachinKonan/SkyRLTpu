"""State-reuse strategies for the RQ2 discovery loop, as a base class + two subclasses.

The unit of sampling is a BUNDLE, matching SimpleTES's batch semantics exactly
(scheduler.py:133-138 of the reference repo): one bundle = one inspiration set + one prompt,
served to K candidates dispatched as independent k=1 generation jobs; at finalize the local
best of the K commits to its chain and receives the reflection. PDR is the degenerate case:
one bundle for everything, context = the shared workspace, no inspirations, no reflections.

SimpleTesState implements the reference scaffold faithfully:
  * C independent chains (default 4): selection draws ONLY from a chain's committed members,
    with per-chain visit counts and expansion totals, so chains cannot collapse onto one
    another -- the diversity-preservation mechanism the single-pool version lacked;
  * RPUCG scoring: V(s)=max(g(s), GAMMA*max V(child)) bottom-up over the DAG, Q and P as
    GLOBAL percentile ranks, score = Q + c*P*sqrt(1+expansions_c)/(1+visits_c[node]),
    greedy selection with 1-hop anti-inbreeding (self+parents+children);
  * num_inspirations=5 per bundle; elite-pool overview (scores+insights, NO code) and the
    auto-accumulated failure histogram ride along in every bundle;
  * only the batch WINNER is committed to the chain and reflected on. Losers stay in the DB,
    where they still shape Q/P percentiles and the elite overview.

One documented deviation: V propagates over direction-aware PERCENTILES, not raw scores.
The reference runs V on raw scores and its bundled evaluators keep every metric
higher-is-better (their erdos evaluator returns 1/(1e-8+C5)); our erdos/ac1 report raw
minimize metrics, where GAMMA*(-0.38) = -0.30 > -0.38 inflates every parent regardless of its
children. Percentiles are positive and higher-is-better by construction for both directions.

The DB is portfolio/store.py's Store, used strictly through its public API -- that file also
serves the ttt-discover PUCT scaffolding and is left untouched.
"""
from __future__ import annotations

import json
import math
import sys
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, "/n/fs/vision-mix/sk7524/SkyRLTpu/tpu/distill_ablation/portfolio")
from store import Store  # noqa: E402

NUM_INSPIRATIONS = 5     # SimpleTES num_inspirations default
NUM_CHAINS = 4           # SimpleTES num_chains default
RPUCG_C = 1.0
N_ELITE = 12
N_FAILURE_PATTERNS = 4
MAX_PROGRAM_CHARS = 24000


@dataclass
class Bundle:
    """One batch's worth of context. Serves `k` candidates that share the same prompt."""
    batch_id: int
    k: int
    inspirations: list[dict] = field(default_factory=list)   # node meta incl. program
    context: str = ""                                        # elite overview | workspace
    failures: str = ""
    chain_idx: int | None = None


class StateReuse(ABC):
    """The loop talks to state ONLY through this interface."""

    @abstractmethod
    def sample(self, n: int, k: int) -> list[Bundle]:
        """Bundles covering n candidates: ceil(n/k) bundles, the last possibly smaller."""

    @abstractmethod
    def reflection_targets(self, grouped: list[tuple[Bundle, list[dict]]]) -> list[dict]:
        """Which results deserve the (paid) reflection call."""

    @abstractmethod
    def update(self, grouped: list[tuple[Bundle, list[dict]]], step: int, **kw) -> None:
        ...

    @abstractmethod
    def best(self) -> tuple[str | None, float | None]:
        ...

    def render_bundle(self, bundle: Bundle, task: str, fence: str) -> str:
        """Default prompt assembly shared by both arms; subclasses shape it via the bundle."""
        parts = [task, ""]
        if bundle.context:
            parts += [bundle.context, ""]
        if bundle.inspirations:
            if len(bundle.inspirations) == 1:
                parts.append("## Improve substantially on this program")
            else:
                parts.append(f"[SAMPLED INSPIRATIONS] ({len(bundle.inspirations)} solutions "
                             "sampled for detailed reference)\n"
                             "Learn from these specific implementations -- study their "
                             "patterns and techniques.")
            for m in bundle.inspirations:
                parts.append(f"\n--- Inspiration [node {m['id']}] Score: {m['r']} ---")
                if m.get("summary") and m["summary"] != "seed":
                    parts.append(f"Reflection:\n{m['summary']}")
                parts.append(f"```{fence}\n{(m.get('program') or '')[:MAX_PROGRAM_CHARS]}\n```")
        if bundle.failures:
            parts += ["", bundle.failures]
        return "\n".join(parts)


# ------------------------------------------------------------------------- SimpleTES scaffold
class SimpleTesState(StateReuse):
    def __init__(self, root: Path, maximize: bool, seed_program: str, seed_score: float | None,
                 num_chains: int = NUM_CHAINS):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.maximize = maximize
        gpath = self.root / "graph.json"
        cpath = self.root / "chains.json"
        if gpath.exists():
            self.store = Store.load(gpath)
        else:
            self.store = Store(maximize, gpath)
            self.store.add(seed_program, seed_score, [], feedback="seed", rnd=0, summary="seed")
            self.store.backup_U()
            self.store.save()
        if cpath.exists():
            c = json.loads(cpath.read_text())
            self.chains = {int(i): v for i, v in c["chains"].items()}
            self.visits = {int(i): {int(k2): v2 for k2, v2 in v.items()}
                           for i, v in c["visits"].items()}
            self.expansions = {int(i): v for i, v in c["expansions"].items()}
        else:
            # every chain starts from the seed node, so selection is non-empty from step 1
            seed_id = next(iter(self.store.g.nodes))
            self.chains = {i: [seed_id] for i in range(num_chains)}
            self.visits = {i: {} for i in range(num_chains)}
            self.expansions = {i: 0 for i in range(num_chains)}
            self._save_chains()
        self.cpath = self.root / "chains.json"

    def _save_chains(self):
        (self.root / "chains.json").write_text(json.dumps(
            {"chains": self.chains, "visits": self.visits, "expansions": self.expansions}))

    # ---- RPUCG selection, per chain over global percentiles ----
    def _q_p(self):
        """Global percentile ranks of propagated value (Q) and own score (P). Store.backup_U
        has already computed U (percentile-propagated V) and p per node."""
        import bisect
        u = {n: self.store.g.nodes[n].get("U") for n in self.store.g.nodes
             if self.store.g.nodes[n].get("U") is not None}
        srt = sorted(u.values())
        k = len(srt)
        q = {n: bisect.bisect_left(srt, v) / k for n, v in u.items()} if k else {}
        p = {n: self.store.g.nodes[n].get("p") or 0.0 for n in self.store.g.nodes}
        return q, p

    def _select_from_chain(self, chain_idx: int, n: int) -> list[int]:
        members = [m for m in self.chains[chain_idx]
                   if self.store.g.nodes[m].get("r") is not None]
        if not members:
            return []
        q, p = self._q_p()
        vis = self.visits[chain_idx]
        te = self.expansions[chain_idx]
        scored = sorted(
            ((m, q.get(m, 0.0) + RPUCG_C * p.get(m, 0.0)
              * math.sqrt(1 + te) / (1 + vis.get(m, 0))) for m in members),
            key=lambda x: x[1], reverse=True)
        picked, excluded = [], set()
        for nid, _ in scored:
            if nid in excluded:
                continue
            picked.append(nid)
            if len(picked) >= n:
                break
            excluded.add(nid)
            excluded.update(self.store.g.predecessors(nid))
            excluded.update(self.store.g.successors(nid))
        return picked

    def _elite_overview(self):
        scored = [n for n in self.store.g.nodes if self.store.g.nodes[n].get("r") is not None]
        if len(scored) < 2:
            return ""
        scored.sort(key=lambda n: self.store.g.nodes[n]["r"], reverse=self.maximize)
        lines = [f"[ELITE POOL OVERVIEW] ({min(N_ELITE, len(scored))} of {len(scored)} "
                 "solutions so far; scores and insights only)",
                 "Use this to see which directions are already explored, avoid duplicating "
                 "them, and identify gaps worth a genuinely different approach.", ""]
        for i, n in enumerate(scored[:N_ELITE], 1):
            refl = (self.store.g.nodes[n].get("summary") or "").strip().replace("\n", " ")
            lines.append(f"#{i} [score {self.store.g.nodes[n]['r']}] {refl[:240]}")
        return "\n".join(lines)

    def _failure_patterns(self):
        import re
        c = Counter()
        for n in self.store.g.nodes:
            nd = self.store.g.nodes[n]
            if nd.get("r") is None and (nd.get("feedback") or "").strip():
                c[re.sub(r"0x[0-9a-f]+|\d{3,}", "#", nd["feedback"].strip())[:110]] += 1
        if not c:
            return ""
        return ("[FAILURE PATTERNS] (common errors to avoid)\n"
                + "\n".join(f"- ({k}x) {m}" for m, k in c.most_common(N_FAILURE_PATTERNS)))

    def sample(self, n, k):
        bundles = []
        elite = self._elite_overview()
        fails = self._failure_patterns()
        remaining, bid = n, 0
        while remaining > 0:
            kk = min(k, remaining)
            chain = bid % len(self.chains)
            insp = self._select_from_chain(chain, NUM_INSPIRATIONS)
            metas = [self.store.meta(i, with_program=True) for i in insp]
            bundles.append(Bundle(batch_id=bid, k=kk, inspirations=metas,
                                  context=elite, failures=fails, chain_idx=chain))
            remaining -= kk
            bid += 1
        return bundles

    def _winner(self, results):
        ok = [r for r in results if r.get("score") is not None]
        if not ok:
            return None
        return (max if self.maximize else min)(ok, key=lambda r: r["score"])

    def reflection_targets(self, grouped):
        """SimpleTES reflects on the batch winner only (base.py:477 reflect_on_winner)."""
        out = []
        for bundle, results in grouped:
            w = self._winner(results)
            if w is not None:
                out.append(w)
        return out

    def update(self, grouped, step, **kw):
        for bundle, results in grouped:
            chain = bundle.chain_idx
            parents = [m["id"] for m in bundle.inspirations]
            winner = self._winner(results)
            for r in results:
                if not r.get("program"):
                    continue
                nid = self.store.add(r["program"], r.get("score"), parents,
                                     feedback=(r.get("detail") or "")[:200], rnd=step,
                                     summary=(r.get("reflection") or "")[:400])
                if winner is not None and r is winner:
                    self.chains[chain].append(nid)      # ONLY the winner joins the chain
            for pid in parents:
                self.visits[chain][pid] = self.visits[chain].get(pid, 0) + 1
            self.expansions[chain] += 1
        self.store.backup_U()
        self.store.save()
        self._save_chains()

    def best(self):
        b = self.store.best()
        if b is None:
            return None, None
        nd = self.store.g.nodes[b]
        return nd.get("program"), nd.get("r")


# ------------------------------------------------------------------------------ PDR workspace
class WorkspaceState(StateReuse):
    def __init__(self, root: Path, maximize: bool, seed_program: str, seed_score: float | None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.maximize = maximize
        self.seed_program = seed_program
        self.wpath = self.root / "workspace.md"
        self.bpath = self.root / "best.json"
        if not self.wpath.exists():
            self.wpath.write_text("")
        if not self.bpath.exists():
            self.bpath.write_text(json.dumps({"program": seed_program, "score": seed_score}))

    def sample(self, n, k):
        ws = self.wpath.read_text().strip()
        ctx = f"## Accumulated findings so far (shared workspace)\n{ws}" if ws else ""
        seedblk = (f"## Starting program\n```\n{self.seed_program[:MAX_PROGRAM_CHARS]}\n```")
        return [Bundle(batch_id=0, k=n, inspirations=[],
                       context=(ctx + "\n\n" + seedblk).strip(), failures="", chain_idx=None)]

    def reflection_targets(self, grouped):
        return []                                   # PDR carries no per-node reflections

    def round_md(self, results, step):
        p = self.root / f"round_{step:02d}.md"
        ok = [r for r in results if r.get("score") is not None]
        ok.sort(key=lambda r: r["score"], reverse=self.maximize)
        L = [f"# Step {step}: {len(results)} programs, {len(ok)} valid",
             f"direction: {'higher is better' if self.maximize else 'lower is better'}",
             "", "| id | score | status |", "|---|---|---|"]
        for r in results:
            L.append(f"| {r.get('sid')} | {r.get('score')} | "
                     f"{'ok' if r.get('score') is not None else (r.get('detail') or 'invalid')[:60]} |")
        L += ["", "## Programs (best first)"]
        for r in ok:
            L += [f"\n### {r['sid']}  score={r['score']}", "```",
                  (r.get("program") or "")[:MAX_PROGRAM_CHARS], "```"]
        p.write_text("\n".join(L))
        return p

    def update(self, grouped, step, compact_fn=None, **kw):
        results = [r for _, rs in grouped for r in rs]
        md = self.round_md(results, step)
        cur = json.loads(self.bpath.read_text())
        for r in results:
            if r.get("score") is None:
                continue
            if (cur["score"] is None
                    or (r["score"] > cur["score"] if self.maximize else r["score"] < cur["score"])):
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


def make_state_reuse(kind, root, maximize, seed_program, seed_score, num_chains=NUM_CHAINS):
    if kind in ("puct", "simpletes"):        # grid name "puct" maps to the SimpleTES scaffold
        return SimpleTesState(root, maximize, seed_program, seed_score, num_chains)
    if kind == "workspace":
        return WorkspaceState(root, maximize, seed_program, seed_score)
    raise ValueError(f"unknown state kind: {kind}")
