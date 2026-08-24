"""Program-DAG state store for the portfolio search.

Nodes carry {program, score r, U (gamma-backed-up), feedback, round, n_used, valid}.
Edges parent -> child ("child was generated with parent in its seed group").
The aggregator agent reads this via store_mcp tools; the DRIVER is the only writer
(executors are untrusted -- nothing enters the store without an independent grade).
"""
from __future__ import annotations

import bisect
import json
import math
from pathlib import Path

import networkx as nx

GAMMA = 0.8


class Store:
    def __init__(self, maximize: bool, path: Path):
        self.g = nx.DiGraph()
        self.maximize = maximize
        self.path = Path(path)
        self._next = 0

    # ---------- persistence ----------
    def save(self):
        data = {
            "maximize": self.maximize,
            "next": self._next,
            "nodes": [{"id": n, **self.g.nodes[n]} for n in self.g.nodes],
            "edges": list(self.g.edges),
        }
        self.path.write_text(json.dumps(data, indent=1))

    @classmethod
    def load(cls, path: Path):
        data = json.loads(Path(path).read_text())
        s = cls(data["maximize"], path)
        s._next = data["next"]
        for nd in data["nodes"]:
            nid = nd.pop("id")
            s.g.add_node(nid, **nd)
        s.g.add_edges_from(data["edges"])
        return s

    # ---------- writes (driver only) ----------
    def add(self, program: str, score: float | None, parents: list[int],
            feedback: str = "", rnd: int = 0, summary: str = "") -> int:
        nid = self._next
        self._next += 1
        self.g.add_node(nid, program=program, r=score, U=score, feedback=feedback[:400],
                        round=rnd, n_used=0, summary=summary[:300], valid=score is not None)
        for p in parents:
            self.g.add_edge(p, nid)
        return nid

    def mark_used(self, ids):
        for i in ids:
            self.g.nodes[i]["n_used"] += 1

    def _percentiles(self):
        """Direction-aware percentile rank in [0,1) of each node's own score; higher = better.
        Scale-free, which is what lets one selection formula work across problems whose scores
        span 0.08 to 1e8 (SimpleTES: 'both Q and P are percentile ranks ... no scale factor')."""
        vals = {n: self.g.nodes[n]["r"] for n in self.g.nodes if self.g.nodes[n]["r"] is not None}
        if not vals:
            return {}
        srt = sorted(v if self.maximize else -v for v in vals.values())
        k = len(srt)
        return {n: bisect.bisect_left(srt, v if self.maximize else -v) / k for n, v in vals.items()}

    def backup_U(self):
        """SimpleTES RPUCG value propagation:  V(s) = max(g(s), GAMMA * max_child V(c)),
        bottom-up over the DAG, so a mediocre node inherits credit from strong descendants.

        NB: the reference implementation runs this on the raw score, which silently breaks for
        MINIMISE problems (discover stores value=-c5, and GAMMA*(-0.38) = -0.30 > -0.38, so every
        node's V inflates toward zero regardless of its children). We therefore propagate over the
        direction-aware percentile g in [0,1) -- positive and higher-is-better by construction --
        which is well behaved for both directions and keeps the intended semantics.

        Stores: p = own-score percentile (the P term), U = propagated value (the Q term's input).
        """
        g = self._percentiles()
        for n in self.g.nodes:
            self.g.nodes[n]["p"] = g.get(n)
        for n in reversed(list(nx.topological_sort(self.g))):
            base = g.get(n)
            best = base
            for c in self.g.successors(n):
                cu = self.g.nodes[c].get("U")
                if cu is None:
                    continue
                cand = GAMMA * cu
                best = cand if best is None else max(best, cand)
            self.g.nodes[n]["U"] = best

    def rpucg_scores(self, total_expansions=None, c=1.0):
        """RPUCG(i) = Q_i + c * P_i * sqrt(1+total_expansions) / (1+n_used_i),
        with Q = percentile rank of the propagated value U, P = own-score percentile."""
        u = {n: self.g.nodes[n].get("U") for n in self.g.nodes
             if self.g.nodes[n].get("U") is not None}
        if not u:
            return {}
        srt = sorted(u.values())
        k = len(srt)
        q = {n: bisect.bisect_left(srt, v) / k for n, v in u.items()}
        te = sum(self.g.nodes[n]["n_used"] for n in self.g.nodes) if total_expansions is None \
            else total_expansions
        out = {}
        for n in u:
            p = self.g.nodes[n].get("p") or 0.0
            nc = self.g.nodes[n]["n_used"]
            out[n] = q[n] + c * p * math.sqrt(1 + te) / (1 + nc)
        return out

    def rpucg_select(self, n_groups, max_group=1):
        """Formula-only selector (no LLM): greedy by RPUCG score, excluding the ONE-HOP
        neighbourhood (self + parents + children) of everything already picked."""
        sc = sorted(self.rpucg_scores().items(), key=lambda kv: kv[1], reverse=True)
        picked, excluded = [], set()
        for nid, _ in sc:
            if nid in excluded or self.g.nodes[nid]["r"] is None:
                continue
            picked.append([nid])
            excluded.add(nid)
            excluded.update(self.g.predecessors(nid))
            excluded.update(self.g.successors(nid))
            if len(picked) >= n_groups:
                break
        return picked

    # ---------- reads ----------
    def best(self):
        valid = [n for n in self.g.nodes if self.g.nodes[n]["r"] is not None]
        if not valid:
            return None
        key = lambda n: self.g.nodes[n]["r"]
        return (max if self.maximize else min)(valid, key=key)

    def meta(self, n, with_program=False):
        d = dict(self.g.nodes[n])
        if not with_program:
            d.pop("program", None)
        d["id"] = n
        d["parents"] = list(self.g.predecessors(n))
        d["children"] = list(self.g.successors(n))
        return d

    # ---------- seed-group constraints (hard, code-enforced) ----------
    def check_groups(self, groups: list[list[int]], n_groups: int, max_group: int = 3,
                     n_used_cap: int = 4) -> str | None:
        """Return None if OK, else a rejection reason the agent can act on."""
        if len(groups) != n_groups:
            return f"need exactly {n_groups} groups, got {len(groups)}"
        flat = [i for g in groups for i in g]
        if len(set(flat)) != len(flat):
            return "groups must be pairwise disjoint (a node appears twice)"
        for g in groups:
            if not (1 <= len(g) <= max_group):
                return f"group size must be 1..{max_group}: {g}"
            for i in g:
                if i not in self.g.nodes:
                    return f"unknown node id {i}"
                if self.g.nodes[i]["r"] is None:
                    return f"node {i} is invalid (no score); not seedable"
                if self.g.nodes[i]["n_used"] >= n_used_cap:
                    return f"node {i} already used {n_used_cap}x; pick something fresher"
            # Anti-redundancy rule for MERGE groups, matching SimpleTES's Phi: exclude ONE-HOP
            # NEIGHBOURS (a node together with its direct parent or direct child). Deliberately not
            # stricter: banning any shared non-root ancestor deadlocks once merges start (a merged
            # node inherits both parents' ancestor sets, so overlaps quickly become universal and
            # no legal merge group exists).
            if len(g) > 1:
                for a in g:
                    for b in g:
                        if a >= b:
                            continue
                        if self.g.has_edge(a, b) or self.g.has_edge(b, a):
                            return (f"nodes {a},{b} are one-hop neighbours (direct parent/child); "
                                    f"pick less directly related programs to merge")
        best = self.best()
        if best is not None and best not in flat:
            return f"monotonic override: current best node {best} must appear in some group"
        return None
