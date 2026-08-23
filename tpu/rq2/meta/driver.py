"""Meta-search over models, above a complete pucts+workspace run.

One EXPANSION is an ordinary `perennial` cell -- 16x16 x 10 rounds = 2,560 rollouts, one agent,
one model, its own MEMORY.md and PUCT buffer. One META-ROUND picks a node per model and expands
it with each, in parallel, giving 2 children (5,120 rollouts = one whole RQ2 campaign cell).
Depth D of those makes a meta-cell.

The meta-layer needs no new StateReuse class: PerennialState.__init__ warm-starts from whatever
is in its state dir (graph.json / visits.json / agent_best.json / MEMORY.md), and loop.py derives
done_steps from trace.jsonl -- so "copy the parent's state dir, omit trace.jsonl, run loop.py" is
a complete expansion primitive. Every meta-node is therefore a normal cell that regrade_topk.py
and the existing trace tooling already read.

TREATMENTS
  continuation  a  tree, visits RESET   (graph only)      -- "exhausted" is a fact about the model
                b  tree + visits        (graph + visits)  -- "exhausted" is a fact about the node
                c  top-k only           (k best programs) -- shed the tail, keep the frontier
  selection     greedy    both models expand the current best node
                mcts      per-model UCB; the two models may be sent to different nodes

Round 1 is identical for all six treatments (only the root exists, and it is the bare seed), so
it is run once per problem by --bootstrap and shared.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

MAIN = Path("/n/fs/vision-mix/sk7524/SkyRLTpu")
sys.path.insert(0, str(MAIN / "tpu/distill_ablation/portfolio"))
from store import Store  # noqa: E402

PY = MAIN / "third_party/discover/.venv-ttd-discover/bin/python"
LOOP = MAIN / "tpu/rq2/client/loop.py"
META_ROOT = MAIN / "runs/rq2/meta"

MODELS = ("qwen", "gemma")               # loop.py --composition values (single-model cells)
INNER = {"B": 16, "G": 16, "steps": 10, "concurrency": 256}
ROLLOUTS_PER_EXPANSION = INNER["B"] * INNER["G"] * INNER["steps"]      # 2,560
TOPK_C = 16                              # rule (c) keeps this many; matches B so each is picked once

# problem -> (maximize, fast_budget, grade_concurrency, reference score for headroom)
# reference = best known at planning time; used ONLY to normalise Q, never reported.
PROBLEMS = {
    "erdos": dict(maximize=False, fast_budget=10, grade_conc=16, reference=0.3808753),
    "fc46":  dict(maximize=True,  fast_budget=10, grade_conc=8,  reference=0.234594),
    "ac1":   dict(maximize=False, fast_budget=60, grade_conc=16, reference=1.50286),
}
UCB_C = 1.0


# ----------------------------------------------------------------- meta bookkeeping
def load_meta(path):
    return json.loads(path.read_text()) if path.exists() else {"nodes": [], "rounds_done": 0}


def save_meta(path, meta):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=1))
    tmp.replace(path)


def node_by_id(meta, nid):
    return next(n for n in meta["nodes"] if n["id"] == nid)


def better(a, b, maximize):
    """Is a strictly better than b? None loses to everything."""
    if a is None:
        return False
    if b is None:
        return True
    return a > b if maximize else a < b


# ----------------------------------------------------------------- value model
def gain(parent_best, child_best, cfg):
    """Headroom-normalised improvement: what fraction of the remaining gap was closed.

    Raw deltas are useless across depths -- erdos moves in the 6th decimal by round 10 while fc46
    moves in the 3rd -- so a raw improvement would make early expansions dominate and the tree
    would always backtrack to the root.
    """
    if parent_best is None or child_best is None:
        return 0.0
    ref, mx = cfg["reference"], cfg["maximize"]
    improvement = (child_best - parent_best) if mx else (parent_best - child_best)
    headroom = (ref - parent_best) if mx else (parent_best - ref)
    if headroom <= 1e-12:                 # already at/past the reference: any gain is huge
        return 1.0 if improvement > 0 else 0.0
    return max(-1.0, min(1.0, improvement / headroom))


def q_value(meta, node, model, cfg):
    """Mean normalised gain this model has achieved from this node; falls back to the node's own
    standing so an unexpanded good node still looks attractive."""
    gains = [c["gain"] for c in meta["nodes"]
             if c.get("parent") == node["id"] and c.get("model") == model
             and c.get("gain") is not None]
    if gains:
        return sum(gains) / len(gains)
    best_overall = max((n["best"] for n in meta["nodes"] if n["best"] is not None),
                       key=lambda v: v if cfg["maximize"] else -v, default=None)
    if node["best"] is None or best_overall is None:
        return 0.0
    return 1.0 if node["best"] == best_overall else 0.0


def select_node(meta, model, policy, cfg):
    """One node per model per round."""
    live = [n for n in meta["nodes"] if n["best"] is not None]
    if not live:
        return None
    if policy == "greedy":
        return max(live, key=lambda n: n["best"] if cfg["maximize"] else -n["best"])
    n_m = sum(n["expansions"].get(model, 0) for n in live) or 1
    def ucb(n):
        return q_value(meta, n, model, cfg) + UCB_C * math.sqrt(
            math.log(n_m + 1) / (1 + n["expansions"].get(model, 0)))
    return max(live, key=ucb)


# ----------------------------------------------------------------- continuation rules
def apply_continuation(parent_state: Path, child_state: Path, rule: str, maximize: bool):
    """Build the child's state/ dir from the parent's. Memory crosses in every rule.

    trace.jsonl is never copied -- its absence is what makes the child run a fresh 10 rounds.
    """
    child_state.mkdir(parents=True, exist_ok=True)
    for name in ("MEMORY.md", "agent_best.json"):
        src = parent_state / name
        if src.exists():
            shutil.copy2(src, child_state / name)

    if rule in ("a", "b"):
        shutil.copy2(parent_state / "graph.json", child_state / "graph.json")
        if rule == "b" and (parent_state / "visits.json").exists():
            shutil.copy2(parent_state / "visits.json", child_state / "visits.json")
        return

    if rule != "c":
        raise ValueError(f"unknown continuation rule {rule!r}")

    # (c) fresh graph holding only the k best programs, as roots (no lineage, no visits)
    src = Store.load(parent_state / "graph.json")
    scored = [(src.g.nodes[n]["r"], n) for n in src.g.nodes if src.g.nodes[n].get("r") is not None]
    scored.sort(key=lambda t: t[0], reverse=maximize)
    fresh = Store(maximize, child_state / "graph.json")
    for r, n in scored[:TOPK_C]:
        nd = src.g.nodes[n]
        fresh.add(nd["program"], r, [], feedback=nd.get("feedback", ""),
                  rnd=0, summary=nd.get("summary", ""))
    fresh.save()


# ----------------------------------------------------------------- expansion
def expansion_done(node_dir: Path):
    """A finished cell has result.json and a full trace."""
    res = node_dir / "result.json"
    trace = node_dir / "trace.jsonl"
    if not res.exists() or not trace.exists():
        return False
    return sum(1 for ln in trace.read_text().splitlines() if ln.strip()) >= INNER["steps"]


def read_best(node_dir: Path):
    try:
        return float(json.loads((node_dir / "result.json").read_text())["best_fast_score"])
    except Exception:
        return None


def launch(problem, model, node_dir: Path, cfg, log: Path):
    cmd = [str(PY), str(LOOP),
           "--problem", problem, "--state", "perennial", "--execution", "simple",
           "--composition", model,
           "--B", str(INNER["B"]), "--G", str(INNER["G"]), "--steps", str(INNER["steps"]),
           "--concurrency", str(INNER["concurrency"]),
           "--fast-budget", str(cfg["fast_budget"]),
           "--grade-concurrency", str(cfg["grade_conc"]),
           "--out", str(node_dir)]
    env = os.environ.copy()
    env.update(TTD_EVAL_BACKEND="local", TTD_DISCOVER_SYNC="0")
    return subprocess.Popen(cmd, stdout=open(log, "a"), stderr=subprocess.STDOUT,
                            env=env, cwd=str(MAIN / "tpu/rq2/client"))


def run_pair(jobs):
    """jobs: list of (node_dir, popen). Wait for all; report which produced a usable cell."""
    for _, p in jobs:
        p.wait()
    return [(d, expansion_done(d)) for d, _ in jobs]


def mem_health(node_dir: Path):
    """Fraction of steps whose memory editor actually wrote. Dead memory is the failure that
    silently hollowed out 10 cells of the original campaign."""
    t = node_dir / "trace.jsonl"
    if not t.exists():
        return None
    oks = []
    for ln in t.read_text().splitlines():
        if not ln.strip():
            continue
        try:
            rec = json.loads(ln)
        except Exception:
            continue
        oks += rec.get("mem_ok") or []
    return (sum(1 for o in oks if o) / len(oks)) if oks else 0.0


# ----------------------------------------------------------------- driver
def bootstrap(problem, cfg):
    """Meta-round 1, shared by all six treatments: expand the bare seed with each model."""
    rd = META_ROOT / problem / "round1"
    rd.mkdir(parents=True, exist_ok=True)
    jobs, nodes = [], []
    for model in MODELS:
        nd = rd / f"n001_{model}"
        nodes.append((model, nd))
        if expansion_done(nd):
            print(f"[meta] bootstrap {problem}/{model}: already done", flush=True)
            continue
        print(f"[meta] bootstrap {problem}/{model}: launching", flush=True)
        jobs.append((nd, launch(problem, model, nd, cfg, rd / f"n001_{model}.log")))
    if jobs:
        run_pair(jobs)
    out = []
    for model, nd in nodes:
        b = read_best(nd)
        print(f"[meta] bootstrap {problem}/{model}: best={b} mem_ok={mem_health(nd)}", flush=True)
        out.append({"id": f"n001_{model}", "path": str(nd), "parent": None, "model": model,
                    "round": 1, "best": b, "gain": None, "expansions": {m: 0 for m in MODELS}})
    return out


def run_treatment(problem, rule, policy, depth, cfg):
    cell = META_ROOT / problem / f"{rule}_{policy}"
    cell.mkdir(parents=True, exist_ok=True)
    mpath = cell / "meta.json"
    meta = load_meta(mpath)

    if not meta["nodes"]:                       # seed from the shared round 1
        meta["nodes"] = bootstrap(problem, cfg)
        meta["rounds_done"] = 1
        save_meta(mpath, meta)

    while meta["rounds_done"] < depth:
        rnd = meta["rounds_done"] + 1
        jobs, pending = [], []
        for model in MODELS:
            parent = select_node(meta, model, policy, cfg)
            if parent is None:
                print(f"[meta] {cell.name} round {rnd}: no live parent, stopping", flush=True)
                return
            nid = f"n{rnd:03d}_{model}"
            nd = cell / nid
            if not expansion_done(nd):
                apply_continuation(Path(parent["path"]) / "state", nd / "state",
                                   rule, cfg["maximize"])
                jobs.append((nd, launch(problem, model, nd, cfg, cell / f"{nid}.log")))
            print(f"[meta] {cell.name} round {rnd}: {model} <- {parent['id']} "
                  f"(best={parent['best']})", flush=True)
            pending.append((nid, nd, parent, model))
        if jobs:
            run_pair(jobs)

        for nid, nd, parent, model in pending:
            b = read_best(nd)
            mh = mem_health(nd)
            if mh is not None and mh < 0.5:
                print(f"[meta] WARNING {nid}: memory editor wrote on only {mh:.0%} of steps",
                      flush=True)
            node_by_id(meta, parent["id"])["expansions"][model] = \
                node_by_id(meta, parent["id"])["expansions"].get(model, 0) + 1
            meta["nodes"].append({
                "id": nid, "path": str(nd), "parent": parent["id"], "model": model,
                "round": rnd, "best": b, "gain": gain(parent["best"], b, cfg),
                "mem_ok_frac": mh, "expansions": {m: 0 for m in MODELS}})
            print(f"[meta] {cell.name} {nid}: best={b} gain={gain(parent['best'], b, cfg):+.4f}",
                  flush=True)
        meta["rounds_done"] = rnd
        save_meta(mpath, meta)

    live = [n for n in meta["nodes"] if n["best"] is not None]
    if live:
        best = max(live, key=lambda n: n["best"] if cfg["maximize"] else -n["best"])
        print(f"[meta] {cell.name} DONE depth={depth} best={best['best']} via {best['id']} "
              f"({len(meta['nodes'])} nodes, {len(meta['nodes'])*ROLLOUTS_PER_EXPANSION} rollouts)",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True, choices=sorted(PROBLEMS))
    ap.add_argument("--rule", choices=["a", "b", "c"])
    ap.add_argument("--policy", choices=["greedy", "mcts"])
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--bootstrap", action="store_true",
                    help="run only the shared meta-round 1 for this problem")
    # inner-run overrides, for smoke tests only -- production uses the module defaults
    ap.add_argument("--B", type=int)
    ap.add_argument("--G", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--concurrency", type=int)
    ap.add_argument("--root", help="override META_ROOT (smoke tests write elsewhere)")
    a = ap.parse_args()
    for k in ("B", "G", "steps", "concurrency"):
        if getattr(a, k) is not None:
            INNER[k] = getattr(a, k)
    if a.root:
        globals()["META_ROOT"] = Path(a.root)
    cfg = PROBLEMS[a.problem]
    if a.bootstrap:
        bootstrap(a.problem, cfg)
        return
    if not (a.rule and a.policy):
        ap.error("--rule and --policy are required unless --bootstrap")
    run_treatment(a.problem, a.rule, a.policy, a.depth, cfg)


if __name__ == "__main__":
    main()
