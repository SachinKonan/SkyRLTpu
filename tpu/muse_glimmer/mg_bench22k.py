#!/usr/bin/env python3
"""Throughput benchmark for Muse-Glimmer at 22k context: 1xTP=4 vs 2xTP=2.

Runs inside the vLLM venv on the TPU host.  Stdlib only (urllib + threads).
Prompts are REAL erdos rollout prompts (env-built token ids from
muse-rs2/manifests/erdos.json), submitted as explicit ``list[int]``.

Measurement discipline -- the whole point of this harness:

* **Every shape is warmed before anything is timed.**  The prior 6.687x
  saturated-throughput figure is suspected to have paid XLA compilation
  inside the timed window (SKIP_JAX_PRECOMPILE=1).  Here, for every
  concurrency bucket, an untimed warm pass runs the *same* prompts at the
  *same* offered concurrency (compiling the prefill buckets and the decode
  batch shape) before the timed passes.  Prefill probes are warmed twice
  before three timed repeats.
* **Steady-state decode rate by differencing.**  Each bucket is timed twice,
  identical except max_tokens (short vs long).  ``steady = c * (long - short)
  / (wall_long - wall_short)`` cancels prefill time, ramp-up, and client
  overhead; the raw long-run throughput is reported alongside.
* **Multi-engine = one offered load, split.**  With two ports, requests are
  round-robined across engines and both are driven CONCURRENTLY -- the
  deployment shape -- and the reported concurrency is the total offered load.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List


def post(url: str, payload: dict, timeout: int = 3600) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def complete(base: str, model: str, prompt_ids: List[int],
             max_tokens: int) -> dict:
    t0 = time.time()
    out = post(
        f"{base}/v1/completions", {
            "model": model,
            "prompt": prompt_ids,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "ignore_eos": True,
        })
    out["_elapsed"] = time.time() - t0
    return out


def fire(bases: List[str], model: str, prompts: List[List[int]],
         max_tokens: int) -> Dict[str, Any]:
    """Fire len(prompts) requests, round-robined across engines, all live at
    once.  Returns wall time and completion-token accounting from the server's
    own usage blocks (never assumed equal to max_tokens)."""
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, len(prompts))) as ex:
        futs = [
            ex.submit(complete, bases[i % len(bases)], model, p, max_tokens)
            for i, p in enumerate(prompts)
        ]
        outs = [f.result() for f in futs]
    wall = time.time() - t0
    comp = sum(o.get("usage", {}).get("completion_tokens", 0) for o in outs)
    per_port = [0] * len(bases)
    for i, o in enumerate(outs):
        per_port[i % len(bases)] += o.get("usage", {}).get(
            "completion_tokens", 0)
    return {"wall": wall, "completion_tokens": comp, "per_port": per_port}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", required=True,
                    help="comma-separated, e.g. 8001 or 8001,8002")
    ap.add_argument("--prompts", required=True, help="bench22k_prompts.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True, help="e.g. 1xTP4 or 2xTP2")
    ap.add_argument("--model", default="muse-glimmer-30b")
    ap.add_argument("--concurrencies", default="1,8,32,64,96,128")
    ap.add_argument("--gen-short", type=int, default=256)
    ap.add_argument("--gen-long", type=int, default=768)
    ap.add_argument("--warm-tokens", type=int, default=64)
    args = ap.parse_args()

    bases = [f"http://127.0.0.1:{p}" for p in args.ports.split(",")]
    pack = json.load(open(args.prompts))
    pool: List[List[int]] = [
        pack["prompts"][k] for k in sorted(pack["prompts"])
    ]
    res: Dict[str, Any] = {
        "label": args.label,
        "ports": args.ports,
        "gen_short": args.gen_short,
        "gen_long": args.gen_long,
        "warmed_before_timing": True,
        "prompt_lens": sorted(len(p) for p in pool),
        "buckets": [],
        "prefill": {},
    }

    print(f"=== bench [{args.label}] engines={bases} ===", flush=True)
    for b in bases:
        r = complete(b, args.model, pool[0][:16], 2)
        print(f"  liveness {b}: {r['_elapsed']:.1f}s", flush=True)

    # ------------------------------------------------ prefill probes --------
    for name, ids in pack.get("prefill_probes", {}).items():
        for _ in range(2):                       # warm: compile this bucket
            complete(bases[0], args.model, ids, 1)
        walls = [
            complete(bases[0], args.model, ids, 1)["_elapsed"]
            for _ in range(3)
        ]
        med = statistics.median(walls)
        res["prefill"][name] = {
            "prompt_tokens": len(ids),
            "walls": walls,
            "median": med
        }
        print(
            f"  PREFILL {name}: {len(ids)} tok -> median {med:.2f}s "
            f"(3 timed runs, 2 warm runs discarded)",
            flush=True)

    # ------------------------------------------------ throughput ladder ----
    for c in [int(x) for x in args.concurrencies.split(",") if x.strip()]:
        prompts = [pool[i % len(pool)] for i in range(c)]
        t0 = time.time()
        fire(bases, args.model, prompts, args.warm_tokens)   # WARM, untimed
        warm_wall = time.time() - t0
        short = fire(bases, args.model, prompts, args.gen_short)
        long_ = fire(bases, args.model, prompts, args.gen_long)
        dtok = long_["completion_tokens"] - short["completion_tokens"]
        dwall = long_["wall"] - short["wall"]
        steady = dtok / dwall if dwall > 0 else float("nan")
        raw = long_["completion_tokens"] / long_["wall"]
        row = {
            "concurrency": c,
            "warm_wall_untimed": warm_wall,
            "short": short,
            "long": long_,
            "steady_decode_tok_s": steady,
            "raw_long_tok_s": raw,
        }
        res["buckets"].append(row)
        print(
            f"  BENCH c={c:4d} steady={steady:8.1f} tok/s  "
            f"raw={raw:8.1f} tok/s  walls short/long "
            f"{short['wall']:.1f}/{long_['wall']:.1f}s  per-port "
            f"{long_['per_port']}",
            flush=True)

    json.dump(res, open(args.out, "w"), indent=1)
    print(f"WROTE {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
