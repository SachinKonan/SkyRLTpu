"""Grade a gens jsonl on REAL silicon through the arena queue.

The CPU grade_smoke path validates semantics only (interpret mode) -- the
whole banked-"correct" corpus passed it and then failed Mosaic. With a
judge in the cell, arm outputs are graded where it counts: every extracted
program is submitted to the queue, the per-test Ray pool grades it
(fwd+bwd, all cases incl. tp4), and this script collects the verdicts into
a graded summary shaped like grade_smoke's output (cells -> rows).

Runs where the queue is reachable (the cell's w0, or any compute node with
a route). Usage:
    python3 grade_gens_via_queue.py --gens <gens.jsonl> --queue http://W1:8791 \\
        --out <graded.json> [--max-wait-s 14400]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # .../tpu

from pallas_arena.probe.gen_smoke import extract_completion  # noqa: E402


def _post(base, path, payload):
    req = urllib.request.Request(f"{base}{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=30))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", required=True)
    ap.add_argument("--queue", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-wait-s", type=float, default=14400.0)
    args = ap.parse_args()
    base = args.queue.rstrip("/")

    rows = [json.loads(l) for l in open(args.gens) if l.strip()]
    wids: dict[str, tuple[str, str, int]] = {}   # wid -> (task, variant, idx)
    cells: dict[str, dict] = defaultdict(lambda: {
        "n": 0, "gen_errors": 0, "no_program": 0, "rows": []})

    for r in rows:
        cell = f"{r.get('task')}:{r.get('variant')}"
        cells[cell]["n"] += 1
        if "text" not in r:
            cells[cell]["gen_errors"] += 1
            cells[cell]["rows"].append({"idx": r.get("idx"),
                                        "outcome": f"gen_error: {str(r.get('error'))[:120]}"})
            continue
        program = extract_completion(r["text"])
        if not program:
            cells[cell]["no_program"] += 1
            cells[cell]["rows"].append({"idx": r.get("idx"), "outcome": "no_program"})
            continue
        wid = _post(base, "/submit", {
            "problem": r["task"], "code": program,
            "tag": f"arm-{r.get('variant')}-{r.get('idx')}"})["work_id"]
        wids[wid] = (r["task"], r["variant"], r.get("idx"))
    print(f"submitted {len(wids)} programs from {len(rows)} gens", flush=True)

    deadline = time.time() + args.max_wait_s
    pending = dict(wids)
    while pending and time.time() < deadline:
        got = _post(base, "/results", {"work_ids": list(pending)})["results"]
        for wid in list(pending):
            rec = got.get(wid) or {}
            if not rec.get("done"):
                continue
            res = rec.get("result") or rec
            task, variant, idx = pending.pop(wid)
            cell = f"{task}:{variant}"
            rw = res.get("reward_with_bwd") or res.get("reward") or 0.0
            if res.get("passed"):
                outcome = f"passed reward={rw:.4f}"
            else:
                v0 = str((res.get("violations") or ["?"])[0])[:200]
                outcome = f"[{res.get('gate')}] {v0}"
            cells[cell]["rows"].append({
                "idx": idx, "outcome": outcome, "passed": bool(res.get("passed")),
                "reward_with_bwd": rw,
                "observation": str(res.get("observation") or "")[:1500],
                "case_boot_s": res.get("case_boot_s"),
                "excluded_cases": res.get("excluded_cases"),
            })
            print(f"[{cell} idx={idx}] {outcome}", flush=True)
        if pending:
            time.sleep(20)
    for wid, (task, variant, idx) in pending.items():
        cells[f"{task}:{variant}"]["rows"].append(
            {"idx": idx, "outcome": "no verdict before deadline"})

    for cell, d in cells.items():
        graded = [r for r in d["rows"] if "passed" in r]
        d["passed"] = sum(1 for r in graded if r["passed"])
        d["graded"] = len(graded)
        rws = sorted(r["reward_with_bwd"] for r in graded if r["passed"])
        d["best_reward"] = rws[-1] if rws else None
        print(f"{cell}: n={d['n']} graded={d['graded']} passed={d['passed']} "
              f"best_reward={d['best_reward']}", flush=True)
    Path(args.out).write_text(json.dumps(cells, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
