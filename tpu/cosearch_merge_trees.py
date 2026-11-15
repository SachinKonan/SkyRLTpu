#!/usr/bin/env python3
"""Merge two (or more) PUCT tree snapshots into ONE co-search super-tree.

Each model ran its own siloed warm-up (e.g. 12 steps) with its own PUCT tree.
Co-search continues from the union: one global PUCTSampler selects parents over
both trees (elite slots and lineage blocking unchanged).

    python tpu/cosearch_merge_trees.py \\
        --tree gemma=.../puct_sampler_step_000012.json \\
        --tree qwen=.../puct_sampler_step_000012.json \\
        --step 12 --out <client log dir>/puct_sampler_step_000012.json

Merge rules (puct_sampler_step_*.json as PUCTSampler._save writes it):
  states          union, in --tree order. State ids are UUIDs: an id present in
                  two trees must be the SAME state (byte-identical dict, e.g. a
                  shared bootstrap root) and is kept once; any other collision
                  fails the merge.
  initial_states  union by the same rule.
  puct_n          per id: sum over trees (visits are counts).
  puct_m          per id: max over trees (best reachable value).
  puct_T          sum over trees.
  step            --step. Resume loads puct_sampler_step_<start batch>.json and
                  picks the snapshot with the MOST states, so the merged file
                  must be named for the members' checkpoint batch and should be
                  the only snapshot in the client log dir.
Parent links are copied as they are; references to states that are no longer
in either tree (pruned by earlier flushes) are counted, not repaired, exactly
as in a single-model tree.

Without --out it only prints the report.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

MAX_BUFFER_SIZE = 1000  # PUCTSampler.max_buffer_size: the next flush keeps the best this many by value


class MergeError(ValueError):
    pass


def _canon(d: dict) -> str:
    return json.dumps(d, sort_keys=True)


def _best(states: list[dict]):
    vals = [s["value"] for s in states if s.get("value") is not None]
    return max(vals) if vals else None


def _union(field: str, trees: list[tuple[str, dict]], report: dict) -> list[dict]:
    out: list[dict] = []
    seen: dict[str, tuple[str, str]] = {}
    for tag, tree in trees:
        for s in tree.get(field, []) or []:
            sid = s.get("id")
            if not sid:
                raise MergeError(f"{tag}: a {field} entry has no id")
            if sid in seen:
                other_tag, canon = seen[sid]
                if canon != _canon(s):
                    raise MergeError(f"{field} id {sid} appears in {other_tag} and {tag} with DIFFERENT content")
                report["shared_identical"][field] += 1
                continue
            seen[sid] = (tag, _canon(s))
            out.append(s)
    return out


def merge_snapshots(trees: list[tuple[str, dict]], step: int, max_buffer: int = MAX_BUFFER_SIZE) -> tuple[dict, dict]:
    """(merged snapshot, report). ``trees`` = [(source tag, snapshot dict), ...]."""
    if len(trees) < 2:
        raise MergeError("need at least two trees to merge")
    tags = [t for t, _ in trees]
    if len(set(tags)) != len(tags):
        raise MergeError(f"duplicate source tags {tags}")
    report: dict = {"step": step, "sources": {}, "shared_identical": {"states": 0, "initial_states": 0}}
    for tag, tree in trees:
        if not isinstance(tree.get("states"), list):
            raise MergeError(f"{tag}: not a PUCT snapshot (no states list)")
        origins: dict[str, int] = {}
        for s in tree["states"]:
            o = s.get("origin") or "seed"
            origins[o] = origins.get(o, 0) + 1
        report["sources"][tag] = dict(
            source_step=tree.get("step"), states=len(tree["states"]),
            initial_states=len(tree.get("initial_states") or []), puct_T=int(tree.get("puct_T") or 0),
            best_value=_best(tree["states"]), origins=origins)
    states = _union("states", trees, report)
    initial = _union("initial_states", trees, report)
    n: dict[str, int] = {}
    m: dict[str, float] = {}
    for _, tree in trees:
        for k, v in (tree.get("puct_n") or {}).items():
            n[k] = n.get(k, 0) + int(v)
        for k, v in (tree.get("puct_m") or {}).items():
            m[k] = max(m[k], v) if k in m else v
    ids = {s["id"] for s in states}
    # Parents with children in more than one source tree (shared roots): the first flush keeps only the top
    # TTD_TOPK_CHILDREN children of each parent, so those parents lose children from one of the trees.
    parent_sources: dict[str, set] = {}
    for tag, tree in trees:
        for s in tree["states"]:
            for p in (s.get("parents") or [])[:1]:
                if p.get("id"):
                    parent_sources.setdefault(str(p["id"]), set()).add(tag)
    shared_parents = sorted(pid for pid, tags_ in parent_sources.items() if len(tags_) > 1)
    dangling = sum(1 for s in states for p in (s.get("parents") or []) if p.get("id") and str(p["id"]) not in ids)
    merged = {
        "step": step,
        "states": states,
        "initial_states": initial,
        "puct_n": n,
        "puct_m": m,
        "puct_T": sum(int(t.get("puct_T") or 0) for _, t in trees),
    }
    best_by_source = {tag: info["best_value"] for tag, info in report["sources"].items()}
    report.update(
        merged_states=len(states), merged_initial_states=len(initial), merged_puct_T=merged["puct_T"],
        merged_best_value=_best(states), dangling_parent_refs=dangling, shared_parent_ids=shared_parents,
        best_source=max((t for t in best_by_source if best_by_source[t] is not None),
                        key=lambda t: best_by_source[t], default=None),
        over_buffer=len(states) > max_buffer)
    return merged, report


def write_atomic(path: str, data: dict) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".merge-", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f)
    os.chmod(tmp, 0o664)  # mkstemp creates 0600; the run dir is shared
    os.replace(tmp, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", action="append", required=True, metavar="TAG=PATH",
                    help="source snapshot (repeat; order = state order in the merged tree)")
    ap.add_argument("--step", type=int, required=True, help="step field of the merged snapshot (members' batch)")
    ap.add_argument("--out", help="write the merged snapshot here (default: report only)")
    ap.add_argument("--force", action="store_true", help="overwrite --out if it exists")
    ap.add_argument("--max-buffer", type=int, default=MAX_BUFFER_SIZE,
                    help="the run's PUCT tree cap (TTD_PUCT_MAX_BUFFER), for the over-cap warning")
    x = ap.parse_args(argv)
    trees = []
    for spec in x.tree:
        tag, sep, path = spec.partition("=")
        if not sep or not tag or not path:
            ap.error(f"--tree wants TAG=PATH, got {spec!r}")
        with open(path) as f:
            trees.append((tag, json.load(f)))
    try:
        merged, report = merge_snapshots(trees, x.step, x.max_buffer)
    except MergeError as exc:
        print(f"MERGE REFUSED: {exc}", file=sys.stderr)
        return 2
    for tag, info in report["sources"].items():
        print(f"{tag:>10}: step={info['source_step']} states={info['states']} initial={info['initial_states']} "
              f"T={info['puct_T']} best={info['best_value']} origins={info['origins']}")
    print(f"{'merged':>10}: step={x.step} states={report['merged_states']} initial={report['merged_initial_states']} "
          f"T={report['merged_puct_T']} best={report['merged_best_value']} (from {report['best_source']}) "
          f"shared-identical={report['shared_identical']} dangling-parent-refs={report['dangling_parent_refs']}")
    if report["over_buffer"]:
        print(f"WARNING: {report['merged_states']} states > PUCT max_buffer_size {x.max_buffer}: the first flush "
              "keeps only the best by value", file=sys.stderr)
    if report["shared_parent_ids"]:
        print(f"WARNING: {len(report['shared_parent_ids'])} parent id(s) have children in more than one tree; the "
              f"first flush keeps only the top TTD_TOPK_CHILDREN children of each: {report['shared_parent_ids'][:5]}",
              file=sys.stderr)
    if x.out:
        if os.path.basename(x.out) != f"puct_sampler_step_{x.step:06d}.json":
            print(f"WARNING: resume loads puct_sampler_step_{x.step:06d}.json; --out is named "
                  f"{os.path.basename(x.out)}", file=sys.stderr)
        if os.path.exists(x.out) and not x.force:
            print(f"refusing to overwrite {x.out} (pass --force)", file=sys.stderr)
            return 2
        write_atomic(x.out, merged)
        print(f"wrote {x.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
