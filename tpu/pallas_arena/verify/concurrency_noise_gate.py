"""ACCEPTANCE GATE for the Ray grading pool: does concurrency move the floor?

The reward is an interleaved median latency RATIO, and scores inside
[1-floor, 1+floor] collapse to exactly 1.0 -- so the noise floor decides
whether a real few-percent improvement is visible at all. It was measured at
~4.8% with ONE grade running on the host. Chips have private HBM and we time
with hermetic xprof (active device time, not wallclock), so co-tenancy
SHOULD be invisible; this measures it instead of assuming it.

Method: submit the SAME reference-equivalent candidate N times at once, so
every actor is busy simultaneously, and compare each verdict's boot-reported
ref-vs-ref floor and the per-case scores against the single-tenant numbers.
A ratio far from 1.0, or a floor materially above the single-tenant value,
means co-tenancy perturbs the measurement and the pool should shrink.

Run on the grading host (or anywhere that can reach the queue):
    python3 -m pallas_arena.verify.concurrency_noise_gate \\
      --queue http://127.0.0.1:8791 --problem rg_lru --n 8
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request


def _post(base: str, path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"{base}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--problem", default="rg_lru")
    ap.add_argument("--candidate", required=True, help="path to a known-correct program")
    ap.add_argument("--n", type=int, default=8, help="simultaneous submissions (>= actors)")
    ap.add_argument("--timeout-s", type=float, default=3600)
    ap.add_argument("--out", default="/n/fs/vision-mix/sk7524/SkyRLTpu/runs/pallas_arena/concurrency-noise-gate.json")
    args = ap.parse_args()

    base = args.queue.rstrip("/")
    code = open(args.candidate).read()
    # Distinct tags defeat the reward cache: identical code would otherwise
    # return one silicon measurement N times, measuring nothing.
    wids = [_post(base, "/submit", {"problem": args.problem, "code": code,
                                    "tag": f"noise-gate-{i}"})["work_id"]
            for i in range(args.n)]
    print(f"submitted {len(wids)} simultaneous grades")

    results, deadline = {}, time.time() + args.timeout_s
    pending = dict.fromkeys(wids)
    while pending and time.time() < deadline:
        got = _post(base, "/results", {"work_ids": list(pending)})["results"]
        for wid in list(pending):
            rec = got.get(wid) or {}
            if rec.get("done"):
                results[wid] = rec.get("result") or rec
                del pending[wid]
        if pending:
            time.sleep(10)

    floors, scores, walls = [], {}, []
    for wid, r in results.items():
        boot = r.get("worker_boot") or {}
        if boot.get("noise_floor") is not None:
            floors.append(float(boot["noise_floor"]))
        walls.append(float(r.get("item_wall_s") or 0))
        for case, d in (r.get("latencies") or {}).items():
            ref, cand = d.get("ref_median_s"), d.get("cand_median_s")
            if ref and cand:
                scores.setdefault(case, []).append(ref / cand)

    print(f"\n{len(results)}/{args.n} verdicts; wall median {statistics.median(walls) if walls else 0:.1f}s")
    if floors:
        print(f"boot noise floor under concurrency: median {statistics.median(floors):.4f} "
              f"max {max(floors):.4f}  (single-tenant reference: ~0.048)")
    verdict_ok = True
    for case, ss in sorted(scores.items()):
        spread = (max(ss) - min(ss)) if len(ss) > 1 else 0.0
        print(f"  {case}: n={len(ss)} median {statistics.median(ss):.4f} spread {spread:.4f}")
        if spread > 0.10:
            verdict_ok = False
    if floors and max(floors) > 0.08:
        verdict_ok = False
    print("\nGATE:", "PASS -- concurrency does not move the measurement"
          if verdict_ok else "FAIL -- shrink the pool or serialize timing")
    json.dump({"results": results, "floors": floors, "scores": scores, "pass": verdict_ok},
              open(args.out, "w"), indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
