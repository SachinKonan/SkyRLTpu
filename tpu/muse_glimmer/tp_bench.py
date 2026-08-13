#!/usr/bin/env python3
"""Throughput / capacity bench for the TP head-to-head (one engine vs two).

Muse-Glimmer has ``num_key_value_heads = 2``, and
``tpu_inference/utils.py::get_padded_num_heads`` pads the KV head count *up* to
the shard count whenever heads < shards:

    get_padded_num_heads(2, 2) -> 2      (no waste)
    get_padded_num_heads(2, 4) -> 4      (2x waste)
    get_padded_num_heads(2, 8) -> 8      (4x waste)

so per-token KV bytes scale with TP, not with the model's real KV width.  The
counter-pressure is that N independent engines hold N copies of the weights.
This script measures the serving-side half of that trade (latency and
throughput); the capacity half (KV pool, max concurrency, HBM) is read out of
the engine's own boot log by the driver.

Prompts go in as explicit ``list[int]`` so no tokenizer is needed on the host,
and ``ignore_eos`` + ``temperature=0`` fix the generated length exactly, which
is what makes tok/s comparable between configurations.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

# Ids well inside the 202048 vocab; content is irrelevant to throughput, only
# length is.  Deterministic so every configuration sees the identical work.
def make_prompt(n: int) -> List[int]:
    return [1000 + (i * 7919) % 50000 for i in range(n)]


def post(url: str, payload: dict, timeout: int = 1800) -> dict:
    req = urllib.request.Request(url,
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


class Engine:

    def __init__(self, port: int, model: str):
        self.port = port
        self.base = f"http://127.0.0.1:{port}"
        self.model = model

    def complete(self, prompt_ids: List[int], max_tokens: int,
                 timeout: int = 1800) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "prompt": prompt_ids,
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "return_token_ids": True,
        }
        t0 = time.time()
        r = post(f"{self.base}/v1/completions", payload, timeout=timeout)
        dt = time.time() - t0
        ch = r["choices"][0]
        ids = ch.get("token_ids") or ch.get("generated_token_ids") or []
        usage = r.get("usage") or {}
        n_out = usage.get("completion_tokens") or len(ids)
        return {"wall": dt, "n_out": int(n_out), "n_ids": len(ids)}

    def alive(self) -> bool:
        try:
            self.complete(make_prompt(8), 2, timeout=120)
            return True
        except Exception:
            return False


def bench_batch1(engines: List[Engine], prompt_len: int, gen: int,
                 repeats: int) -> Dict[str, Any]:
    """Single sequence, nothing else in flight: pure per-token decode latency.

    This is the arm where a wider TP is expected to WIN -- decode is bandwidth
    bound on the weights, and TP=4 reads half the bytes per chip per step that
    two independent TP=2 engines each do.
    """
    out: Dict[str, Any] = {}
    for eng in engines:
        eng.complete(make_prompt(prompt_len), 8)  # warm the compiled shapes
        samples = []
        for _ in range(repeats):
            r = eng.complete(make_prompt(prompt_len), gen)
            if r["n_out"]:
                samples.append(r["n_out"] / r["wall"])
        out[f"port_{eng.port}"] = {
            "tok_s": round(statistics.median(samples), 2) if samples else None,
            "samples": [round(s, 2) for s in samples],
        }
    vals = [v["tok_s"] for v in out.values() if v["tok_s"]]
    out["median_over_engines_tok_s"] = round(statistics.median(vals),
                                             2) if vals else None
    return out


def bench_saturation(engines: List[Engine], prompt_len: int, gen: int,
                     concurrency: int) -> Dict[str, Any]:
    """`concurrency` sequences in flight, spread evenly over the engines.

    Aggregate tok/s is the number that decides the configuration: two engines
    only have to beat one if the total work per host is what you care about.
    """
    n_eng = len(engines)
    per = max(1, concurrency // n_eng)
    total = per * n_eng
    prompt = make_prompt(prompt_len)

    def one(i: int) -> Dict[str, Any]:
        return engines[i % n_eng].complete(prompt, gen)

    # Warm every engine at this shape first; the first batched call pays XLA
    # compilation for the new batch dimension and would otherwise be counted.
    with ThreadPoolExecutor(max_workers=total) as ex:
        list(ex.map(one, range(total)))

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=total) as ex:
        res = list(ex.map(one, range(total)))
    wall = time.time() - t0

    toks = sum(r["n_out"] for r in res)
    lat = sorted(r["wall"] for r in res)
    return {
        "concurrency_requested": concurrency,
        "concurrency_actual": total,
        "per_engine": per,
        "gen_tokens_each": gen,
        "total_output_tokens": toks,
        "wall_s": round(wall, 3),
        "aggregate_tok_s": round(toks / wall, 2) if wall > 0 else None,
        "latency_p50_s": round(lat[len(lat) // 2], 3) if lat else None,
        "latency_max_s": round(lat[-1], 3) if lat else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", required=True,
                    help="comma-separated engine ports (1 = TP=N, 2 = 2xTP=N/2)")
    ap.add_argument("--model", default="muse-glimmer-30b")
    ap.add_argument("--label", default="")
    ap.add_argument("--tp", type=int, default=0, help="TP of EACH engine")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=64)
    args = ap.parse_args()

    ports = [int(p) for p in args.ports.split(",") if p.strip()]
    engines = [Engine(p, args.model) for p in ports]

    res: Dict[str, Any] = {
        "label": args.label,
        "ports": ports,
        "n_engines": len(engines),
        "tp_per_engine": args.tp,
        "prompt_len": args.prompt_len,
        "gen": args.gen,
    }

    # Never report numbers for an engine that is not actually answering.
    live = [e for e in engines if e.alive()]
    res["engines_alive"] = [e.port for e in live]
    if len(live) != len(engines):
        res["error"] = (f"only {len(live)}/{len(engines)} engines answered; "
                        "numbers below are NOT a valid comparison")
        json.dump(res, open(args.out, "w"), indent=2)
        print(json.dumps(res, indent=2))
        return

    # Analytic per-token KV: 2 (K and V) * layers * padded_kv_heads * head_dim
    # * 2 bytes(bf16).  padded_kv_heads = max(num_kv_heads, TP) for this model.
    if args.tp:
        padded = max(2, args.tp)
        per_tok = 2 * 52 * padded * 128 * 2
        res["analytic"] = {
            "padded_kv_heads": padded,
            "per_token_kv_bytes_per_engine": per_tok,
            "per_token_kv_bytes_per_engine_KiB": round(per_tok / 1024, 1),
            "note": "num_key_value_heads=2 padded up to TP by get_padded_num_heads",
        }

    try:
        res["batch1"] = bench_batch1(live, args.prompt_len, args.gen,
                                     args.repeats)
    except Exception as e:  # a failed arm must not cost the other one
        res["batch1"] = {"error": repr(e)}
    try:
        res["saturation"] = bench_saturation(live, args.prompt_len, args.gen,
                                             args.concurrency)
    except Exception as e:
        res["saturation"] = {"error": repr(e)}

    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
