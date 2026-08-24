"""Store MCP: read-only graph browsing + constrained seed-group submission,
for the AGGREGATOR codex session. streamable-HTTP, same pattern as grading_mcp.

The driver starts this per round with --graph <graph.json> --out <groups.json>;
submit_seed_groups() validates against Store.check_groups and, on acceptance,
writes groups.json and returns OK (the driver reads the file after the session).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import uvicorn
from mcp.server.fastmcp import FastMCP

from store import Store

ap = argparse.ArgumentParser()
ap.add_argument("--graph", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--n-groups", type=int, required=True)
ap.add_argument("--max-group", type=int, default=3)
args = ap.parse_args()

S = Store.load(Path(args.graph))
mcp = FastMCP("store", stateless_http=True)


@mcp.tool()
def overview() -> str:
    """Store summary: node count, best node, per-lineage sizes and best scores."""
    roots = [n for n in S.g.nodes if S.g.in_degree(n) == 0]
    lin = []
    for r in roots:
        desc = {r} | nx.descendants(S.g, r)
        scored = [S.g.nodes[n]["r"] for n in desc if S.g.nodes[n]["r"] is not None]
        agg = (max if S.maximize else min)(scored) if scored else None
        lin.append({"root": r, "size": len(desc), "best_r": agg})
    return json.dumps({"n_nodes": S.g.number_of_nodes(), "maximize": S.maximize,
                       "best_node": S.best(), "lineages": lin})


@mcp.tool()
def top_k(by: str = "rpucg", k: int = 10) -> str:
    """Top-k nodes ranked by 'rpucg' (default), 'U', 'p', or 'r'.
      r     = own score (raw, problem units)
      p     = own-score percentile in [0,1), higher = better regardless of direction
      U     = value propagated up the DAG (gamma=0.8) -- U >> p means the node spawned
              winners, i.e. a fertile stepping-stone even if its own score is mediocre
      rpucg = the selection score Q + c*P*sqrt(1+expansions)/(1+n_used): exploit + explore."""
    if by == "rpucg":
        sc = S.rpucg_scores()
        order = sorted(sc, key=lambda n: sc[n], reverse=True)[:k]
        return json.dumps([{**S.meta(n), "rpucg": round(sc[n], 4)} for n in order])
    valid = [n for n in S.g.nodes if S.g.nodes[n].get(by) is not None]
    valid.sort(key=lambda n: S.g.nodes[n][by], reverse=(S.maximize if by == "r" else True))
    return json.dumps([S.meta(n) for n in valid[:k]])


@mcp.tool()
def node(node_id: int, with_program: bool = True) -> str:
    """Full metadata (+program text) for one node."""
    if node_id not in S.g.nodes:
        return json.dumps({"error": f"no node {node_id}"})
    return json.dumps(S.meta(node_id, with_program=with_program))


@mcp.tool()
def relatives(node_id: int) -> str:
    """Ancestors and descendants of a node (ids + scores) -- lineage inspection."""
    if node_id not in S.g.nodes:
        return json.dumps({"error": f"no node {node_id}"})
    anc = [{"id": a, "r": S.g.nodes[a]["r"]} for a in nx.ancestors(S.g, node_id)]
    dec = [{"id": d, "r": S.g.nodes[d]["r"]} for d in nx.descendants(S.g, node_id)]
    return json.dumps({"ancestors": anc, "descendants": dec})


@mcp.tool()
def submit_seed_groups(groups_json: str) -> str:
    """Submit your seed groups as JSON [[id,...],...]. A 1-node group = refine it;
    a multi-node group = merge/cross-pollinate them. Hard constraints are enforced:
    exactly the required group count, disjoint, size<=max, no same-lineage merges,
    overuse cap, and the current best node must appear somewhere. On REJECTED, fix
    and resubmit."""
    try:
        groups = json.loads(groups_json)
        groups = [[int(i) for i in g] for g in groups]
    except Exception as e:
        return f"REJECTED: bad JSON ({e})"
    err = S.check_groups(groups, args.n_groups, args.max_group)
    if err:
        return f"REJECTED: {err}"
    Path(args.out).write_text(json.dumps(groups))
    return f"ACCEPTED: {groups}. You are done; end the session."


app = mcp.streamable_http_app()
uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
