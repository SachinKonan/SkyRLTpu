#!/usr/bin/env python3
"""Build a generation-0 PUCT seed from N finished member trees.

Seed operators (the axis that fuses member-selection and state memory):
  winner-top16 : argmax member by VALIDATED best C5, then top-k within its tree
  mix-top16    : top-k across the union of all member trees (no member selection)

Both use the elite-slot rule at k: rank by validated value, at most one state
per lineage family (greedy: take best, block its ancestors+descendants, next).
Seeded states get `parents` STRIPPED (each becomes an independent root -- the
sampler's per-batch lineage blocking would otherwise return short batches, and
elite slots would collapse to the number of distinct source families) and
`initial_states` set to the seeds themselves (pins them against buffer
eviction; they carry real code so the elite-slot initial-state filter is a
no-op). PUCT statistics start empty: puct_n={}, puct_m={}, puct_T=0.

Values are ALWAYS rewritten to the independently recomputed C5 (never the
stored value): selection and the seed's Q/prior must rank on truth. States
whose construction fails validation are dropped, not repaired.

Usage:
  build_meta_seed.py --op winner-top16 --k 16 --out seed.json \
      --tree qwen=path/to/qwen_final.json --tree gemma=... --tree muse=...
Writes seed to --out and a selection report to <out>.report.json.
"""
import argparse, json, sys
import numpy as np


def true_c5(h_list):
    """Erdos min-overlap C5, renormalized -- mirrors verify_erdos.py exactly."""
    h = np.array(h_list, dtype=float)
    n = len(h)
    if n == 0 or not np.all(np.isfinite(h)):
        return None
    if np.any(h < 0) or np.any(h > 1):
        return None
    t = n / 2.0
    dx = 2.0 / n
    if h.sum() != t:
        if h.sum() <= 0:
            return None
        h = h * (t / h.sum())
        if np.any(h < 0) or np.any(h > 1):
            return None
    return float((np.correlate(h, 1.0 - h, mode="full") * dx).max())


def load_tree(path):
    d = json.load(open(path))
    return d.get("states") or []


def family_ids(state, children_map):
    """state.id + embedded ancestor ids + reachable descendant ids (same rule
    as PUCTSampler._get_full_lineage, computed offline)."""
    fam = {state["id"]}
    for p in (state.get("parents") or []):
        if p.get("id"):
            fam.add(str(p["id"]))
    queue, seen = [state["id"]], {state["id"]}
    while queue:
        sid = queue.pop(0)
        for cid in children_map.get(sid, ()):  # descendants
            if cid not in seen:
                seen.add(cid)
                fam.add(cid)
                queue.append(cid)
    return fam


def children_map_of(states):
    cm = {}
    for s in states:
        for p in (s.get("parents") or []):
            pid = p.get("id")
            if pid:
                cm.setdefault(str(pid), set()).add(s["id"])
    return cm


def validated_pool(states, tag):
    """(true_value, state) for every state with a valid construction; drops
    invalid ones. Tags provenance on the state dict."""
    out = []
    for s in states:
        c = s.get("construction")
        if not c or not (s.get("code") or "").strip():
            continue
        t = true_c5(c)
        if t is None:
            continue
        s = dict(s)
        s["meta_member"] = tag
        out.append((t, s))
    return out


def lineage_topk(pool, children_map, k):
    """Greedy top-k by validated value (lower C5 = better), one per family,
    cross-tree duplicate constructions deduped."""
    picked, blocked, seen_constructions = [], set(), set()
    for t, s in sorted(pool, key=lambda e: e[0]):
        if s["id"] in blocked:
            continue
        key = tuple(round(float(x), 12) for x in s["construction"])
        if key in seen_constructions:
            continue
        picked.append((t, s))
        seen_constructions.add(key)
        blocked |= family_ids(s, children_map)
        if len(picked) >= k:
            break
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True, choices=["winner-top16", "winner-tree", "mix-top16"])
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tree", action="append", required=True,
                    help="tag=path, repeatable")
    args = ap.parse_args()

    trees = {}
    for spec in args.tree:
        tag, path = spec.split("=", 1)
        trees[tag] = load_tree(path)

    pools = {tag: validated_pool(sts, tag) for tag, sts in trees.items()}
    bests = {tag: (min(p)[0] if p else None) for tag, p in pools.items()}
    eligible = {t: b for t, b in bests.items() if b is not None}
    if not eligible:
        sys.exit("no member has any validated state -- refusing to build a seed")
    winner = min(eligible, key=eligible.get)

    if args.op == "mix-top16":
        pool = [e for p in pools.values() for e in p]
        cmap = {}
        for sts in trees.values():
            cmap.update(children_map_of(sts))
        picked = lineage_topk(pool, cmap, args.k)
    elif args.op == "winner-top16":
        picked = lineage_topk(pools[winner], children_map_of(trees[winner]), args.k)
    else:  # winner-tree: full winner tree, values validated, stats cleared
        picked = sorted(pools[winner], key=lambda e: e[0])

    if not picked:
        sys.exit("seed would be empty -- refusing")

    seeds = []
    for t, s in picked:
        s = dict(s)
        s["value"] = -t                    # stored value is -C5, from truth
        if args.op != "winner-tree":
            s["parents"] = []              # independent roots
            s["parent_values"] = []
        seeds.append(s)

    store = {
        "step": 0,
        "states": seeds,
        "initial_states": seeds if args.op != "winner-tree" else [],
        "puct_n": {},
        "puct_m": {},
        "puct_T": 0,
    }
    json.dump(store, open(args.out, "w"))

    contrib = {}
    for _, s in picked:
        contrib[s["meta_member"]] = contrib.get(s["meta_member"], 0) + 1
    report = {
        "op": args.op,
        "winner": winner,
        "member_best_validated_c5": bests,
        "seed_size": len(seeds),
        "seed_best_c5": picked[0][0],
        "contribution": contrib,
        "short": len(seeds) < args.k and args.op != "winner-tree",
    }
    json.dump(report, open(args.out + ".report.json", "w"), indent=1)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
