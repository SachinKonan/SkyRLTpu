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
import random
import sys
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, "/n/fs/vision-mix/sk7524/SkyRLTpu/tpu/distill_ablation/portfolio")
from store import Store  # noqa: E402

NUM_INSPIRATIONS = 5     # SimpleTES num_inspirations default
SAMPLE_GAMMA = 0.75      # geometric rank weight for inspiration sampling (rank r -> 0.75^r)
# Verbatim from the reference templates/generation.py -- part of the PUCTS treatment definition,
# applied to every SimpleTES bundle and never to PDR (whose workspace content is what is under
# test).
GENERATION_STRATEGY = """=== GENERATION STRATEGY ===
- Prioritize NOVEL approaches not yet seen in the elite pool
- Only refine existing approaches if you identify clear improvement potential
- Combine insights from multiple solutions when beneficial
- Avoid the listed failure patterns"""
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
    context: str = ""                                        # elite overview | workspace | memory
    failures: str = ""
    strategy: str = ""                                       # SimpleTES GENERATION STRATEGY
    chain_idx: int | None = None
    agent_idx: int = 0                                       # S4/S5 team arms; 0 otherwise


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
        if bundle.strategy:
            parts += ["", bundle.strategy]
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

    def _select_from_chain(self, chain_idx: int, n: int, rng: random.Random) -> list[int]:
        """Rank RPUCG-style, then SAMPLE the n inspirations with probability skewed toward the
        top (geometric rank weights SAMPLE_GAMMA**rank, without replacement, 1-hop
        anti-inbreeding as in the reference). Deliberate deviation from the reference's pure
        greedy pick (rpucg.py:206): with C < B, several bundles draw from one chain in the same
        step against identical visit counts, and greedy selection would hand them IDENTICAL
        inspiration sets (effective B collapses to C). The async reference never hits this --
        visits update between a chain's consecutive batches; here within-step diversity comes
        from sampling noise instead. rng is seeded deterministically by the caller so resume
        replays the same bundles."""
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
        weight = {nid: SAMPLE_GAMMA ** rank for rank, (nid, _) in enumerate(scored)}
        avail = [nid for nid, _ in scored]
        picked, excluded = [], set()
        while avail and len(picked) < n:
            choice = rng.choices(avail, weights=[weight[m] for m in avail], k=1)[0]
            picked.append(choice)
            excluded.add(choice)
            excluded.update(self.store.g.predecessors(choice))
            excluded.update(self.store.g.successors(choice))
            avail = [m for m in avail if m not in excluded]
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
        pop = len(self.store.g.nodes)   # deterministic per step: store only grows in update()
        while remaining > 0:
            kk = min(k, remaining)
            chain = bid % len(self.chains)
            insp = self._select_from_chain(chain, NUM_INSPIRATIONS,
                                           random.Random(f"{pop}:{chain}:{bid}"))
            metas = [self.store.meta(i, with_program=True) for i in insp]
            bundles.append(Bundle(batch_id=bid, k=kk, inspirations=metas,
                                  context=elite, failures=fails,
                                  strategy=GENERATION_STRATEGY, chain_idx=chain))
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


# ---------------------------------------------------- S3-S5: discovery-PUCT + perennial memory
MEMORY_MAX_CHARS = 18_000    # mechanical backstop for the 5k-token MEMORY.md contract
PUCT_C = 1.0
TOPK_CHILDREN = 2            # discovery sampler default: top-k results of a group join the buffer


class PerennialState(StateReuse):
    """S3/S4/S5: ttt-discover-style PUCT buffer(s) + spark-edited perennial MEMORY.md file(s).

    Lineage follows the RL sampler's semantics (tinker_utils/sampler.py PUCTSampler), NOT the
    SimpleTES RPUCG: no chains -- one population per buffer; Q(i) = best child value if i has
    children else own value (one-hop, not deep backup); P = rank-based prior; exploration scaled
    by the value range; FULL-lineage exclusion between same-step picks (which is also what keeps
    B same-step bundles distinct without stochastic draws -- each pick removes its ancestors and
    descendants from the pool; the pool cycles if exhausted). Each bundle carries exactly ONE
    state (the program to improve), as in RL. Top-2 valid results of a bundle commit as children.

    Memory replaces the SimpleTES elite-overview + failure-pattern + strategy channels AND the
    per-winner reflections: the ONLY language channel is MEMORY.md, appended to every prompt and
    edited (not rebuilt) each step by a spark exec the loop provides via update(memory_fn=...).

    n_agents=1, shared -> S3. n_agents=2, shared buffer -> S4 (memories private, merit-gated
    crossing: agent i's editor also sees teammate results strictly better than i's best-so-far).
    n_agents=2, split -> S5 (per-agent buffers; crossing ONLY through the memory channel).
    """

    def __init__(self, root: Path, maximize: bool, seed_program: str, seed_score: float | None,
                 n_agents: int = 1, shared_buffer: bool = True):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.maximize = maximize
        self.n_agents = n_agents
        self.shared = shared_buffer
        n_stores = 1 if shared_buffer else n_agents
        self.stores = []
        for s in range(n_stores):
            gp = self.root / (f"graph_{s}.json" if n_stores > 1 else "graph.json")
            if gp.exists():
                self.stores.append(Store.load(gp))
            else:
                st = Store(maximize, gp)
                st.add(seed_program, seed_score, [], feedback="seed", rnd=0, summary="seed")
                st.save()
                self.stores.append(st)
        vp = self.root / "visits.json"
        self.visits = json.loads(vp.read_text()) if vp.exists() else \
            {str(s): {} for s in range(n_stores)}
        # Per-agent lifetime best of the agent's OWN rollouts. The merit gate must compare
        # against this, never against the buffer: with a shared buffer (S4) the store's best
        # already includes teammate commits, so a store-derived bar can never be beaten and
        # nothing would ever cross (caught by unit test).
        ap_ = self.root / "agent_best.json"
        self.abest = json.loads(ap_.read_text()) if ap_.exists() else \
            {str(a): seed_score for a in range(n_agents)}
        for a in range(n_agents):
            mp = self._mem_path(a)
            if not mp.exists():
                mp.write_text("")

    def _mem_path(self, agent):
        return self.root / (f"memory_{agent}.md" if self.n_agents > 1 else "MEMORY.md")

    def _store_of(self, agent):
        return self.stores[0] if self.shared else self.stores[agent]

    def _vkey(self, agent):
        return "0" if self.shared else str(agent)

    def _select(self, store, vis, n_pick, rng):
        """Discovery-PUCT pick of n_pick states with full-lineage exclusion. Deterministic
        greedy like the RL sampler; rng only breaks exact score ties stably."""
        g = store.g
        pop = [nid for nid in g.nodes if g.nodes[nid].get("r") is not None]
        if not pop:
            return []
        v = {nid: (g.nodes[nid]["r"] if self.maximize else -g.nodes[nid]["r"]) for nid in pop}
        scale = (max(v.values()) - min(v.values())) or 1.0
        ranked = sorted(pop, key=lambda nid: v[nid])
        prior = {nid: (i + 1) / len(ranked) for i, nid in enumerate(ranked)}
        T = sum(vis.values())

        def q(nid):
            kids = [c for c in g.successors(nid) if c in v]
            return max((v[c] for c in kids), default=v[nid])

        def score(nid):
            return q(nid) + PUCT_C * scale * prior[nid] * math.sqrt(1 + T) / (1 + vis.get(nid, 0))

        def lineage(nid):
            import networkx as nx
            return {nid} | nx.ancestors(g, nid) | nx.descendants(g, nid)

        picked, excluded = [], set()
        order = sorted(pop, key=lambda nid: (-score(nid), str(nid)))
        while len(picked) < n_pick:
            avail = [nid for nid in order if nid not in excluded]
            if not avail:                       # pool exhausted (early steps): cycle
                excluded = set()
                avail = order
            choice = avail[0]
            picked.append(choice)
            excluded |= lineage(choice)
        return picked

    def sample(self, n, k):
        bundles, remaining, bid = [], n, 0
        n_bundles = (n + k - 1) // k
        picks = {}                              # per-store selections, computed once
        for skey in range(len(self.stores)):
            agents_here = [a for a in range(self.n_agents)
                           if (0 if self.shared else a) == skey]
            n_here = sum(1 for b in range(n_bundles)
                         if (b % self.n_agents) in agents_here)
            picks[skey] = self._select(self.stores[skey],
                                       self.visits[str(skey)], n_here, None)
        cursor = {skey: 0 for skey in picks}
        while remaining > 0:
            kk = min(k, remaining)
            agent = bid % self.n_agents
            skey = 0 if self.shared else agent
            nid = picks[skey][cursor[skey]]
            cursor[skey] += 1
            store = self.stores[skey]
            meta = store.meta(nid, with_program=True)
            mem = self._mem_path(agent).read_text().strip()
            ctx = f"## Long-term memory (MEMORY.md -- accumulated across all rounds)\n{mem}" \
                if mem else ""
            bundles.append(Bundle(batch_id=bid, k=kk, inspirations=[meta], context=ctx,
                                  failures="", strategy="", chain_idx=None, agent_idx=agent))
            remaining -= kk
            bid += 1
        return bundles

    def reflection_targets(self, grouped):
        return []                               # memory is the only language channel

    def _round_md(self, agent, own, crossed, step):
        p = self.root / f"round_{step:02d}_agent{agent}.md"
        L = [f"# Step {step} -- agent {agent}: {len(own)} own results"
             + (f", {len(crossed)} teammate results that beat your best" if crossed else ""),
             f"direction: {'higher' if self.maximize else 'lower'} is better", ""]
        def block(rs, title):
            if not rs:
                return
            L.append(f"## {title}")
            ok = sorted([r for r in rs if r.get("score") is not None],
                        key=lambda r: r["score"], reverse=self.maximize)
            bad = [r for r in rs if r.get("score") is None]
            L.append("| id | score | status |"); L.append("|---|---|---|")
            for r in ok + bad:
                L.append(f"| {r.get('sid')} | {r.get('score')} | "
                         f"{'ok' if r.get('score') is not None else (r.get('detail') or 'invalid')[:60]} |")
            for r in ok:
                L.extend([f"\n### {r['sid']}  score={r['score']}", "```",
                          (r.get("program") or "")[:MAX_PROGRAM_CHARS], "```"])
        block(own, "Your rollouts")
        block(crossed, "Teammate rollouts that outperformed your best (merit-gated share)")
        p.write_text("\n".join(L))
        return p

    def update(self, grouped, step, memory_fn=None, **kw):
        by_agent = {a: [] for a in range(self.n_agents)}
        # commit: top-2 valid results per bundle join the owning buffer as children
        for bundle, results in grouped:
            agent = bundle.agent_idx
            store = self._store_of(agent)
            parent = bundle.inspirations[0]["id"] if bundle.inspirations else None
            ok = sorted([r for r in results if r.get("score") is not None],
                        key=lambda r: r["score"], reverse=self.maximize)
            for r in ok[:TOPK_CHILDREN]:
                store.add(r["program"], r["score"], [parent] if parent is not None else [],
                          feedback=(r.get("detail") or "")[:200], rnd=step, summary="")
            if parent is not None:
                vk = self._vkey(agent)
                self.visits[vk][str(parent)] = self.visits[vk].get(str(parent), 0) + 1
            by_agent[agent] += results
        for st in self.stores:
            st.save()
        (self.root / "visits.json").write_text(json.dumps(self.visits))
        # per-agent lifetime bests (own rollouts only), including this step's
        for a in range(self.n_agents):
            own = [r["score"] for r in by_agent[a] if r.get("score") is not None]
            if own:
                cand = max(own) if self.maximize else min(own)
                cur = self.abest.get(str(a))
                if cur is None or ((cand > cur) if self.maximize else (cand < cur)):
                    self.abest[str(a)] = cand
        (self.root / "agent_best.json").write_text(json.dumps(self.abest))
        # memory edits: one spark exec per agent, merit-gated teammate visibility
        if memory_fn is not None:
            for a in range(self.n_agents):
                bar = self.abest.get(str(a))
                crossed = []
                for other in range(self.n_agents):
                    if other == a:
                        continue
                    for r in by_agent[other]:
                        s = r.get("score")
                        if s is None or bar is None:
                            continue
                        if (s > bar) if self.maximize else (s < bar):
                            crossed.append(r)
                md = self._round_md(a, by_agent[a], crossed, step)
                new = memory_fn(a, md, self._mem_path(a))
                if new and new.strip():
                    self._mem_path(a).write_text(new)
                    (self.root / f"memory_{a}_step{step:02d}.md").write_text(new)

    def best(self):
        top, prog = None, None
        for store in self.stores:
            b = store.best()
            if b is None:
                continue
            nd = store.g.nodes[b]
            r = nd.get("r")
            if top is None or ((r > top) if self.maximize else (r < top)):
                top, prog = r, nd.get("program")
        return prog, top


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
    if kind == "perennial":                  # S3: discovery-PUCT buffer + one MEMORY.md
        return PerennialState(root, maximize, seed_program, seed_score,
                              n_agents=1, shared_buffer=True)
    if kind == "team":                       # S4: shared buffer, N=2 private memories
        return PerennialState(root, maximize, seed_program, seed_score,
                              n_agents=2, shared_buffer=True)
    if kind == "team-split":                 # S5: per-agent buffers, memory-only crossing
        return PerennialState(root, maximize, seed_program, seed_score,
                              n_agents=2, shared_buffer=False)
    raise ValueError(f"unknown state kind: {kind}")
