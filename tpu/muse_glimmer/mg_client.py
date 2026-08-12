#!/usr/bin/env python3
"""Greedy token-for-token comparison against the HF reference, on the TPU host.

Runs inside the vLLM venv on the TPU VM.  Stdlib only (urllib + threads) so it
has no dependency on whatever happens to be installed alongside vLLM.

Token ids go in and token ids come out -- prompts are submitted as explicit
``list[int]`` and the generated ids are read from vLLM's ``return_token_ids``
extension.  Nothing in this comparison ever round-trips through text, so a
tokenizer or BOS discrepancy cannot masquerade as a model mismatch.

Phases:
  1. one real request (proves the engine answers before anything is measured)
  2. sequential greedy per prompt        -> token-for-token vs HF
  3. all prompts fired CONCURRENTLY      -> same ids as phase 2? (paged KV
                                            cache correctness with >1 live
                                            sequence, not just batch-1)
  4. a throughput batch                  -> tokens/sec, and again identical ids
  5. optional teacher-forced prompt_logprobs on the shortest prompt
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional


def post(url: str, payload: dict, timeout: int = 1800) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def get(url: str, timeout: int = 60) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


class Engine:

    def __init__(self, base: str, model: str):
        self.base = base.rstrip("/")
        self.model = model

    def complete(self,
                 prompt_ids: List[int],
                 max_tokens: int,
                 logprobs: Optional[int] = 5,
                 prompt_logprobs: Optional[int] = None) -> dict:
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt_ids,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 0,
            "return_token_ids": True,
            "return_tokens_as_token_ids": True,
            # Never stop early: we are comparing a fixed number of steps
            # against HF, and an EOS-triggered short output would look like a
            # divergence that it is not.
            "ignore_eos": True,
        }
        if logprobs is not None:
            payload["logprobs"] = logprobs
        if prompt_logprobs is not None:
            payload["prompt_logprobs"] = prompt_logprobs
        t0 = time.time()
        out = post(f"{self.base}/v1/completions", payload)
        out["_elapsed"] = time.time() - t0
        return out


def gen_ids(resp: dict) -> List[int]:
    ch = resp["choices"][0]
    if ch.get("token_ids"):
        return list(ch["token_ids"])
    raise RuntimeError(
        "engine did not return token_ids (needs vLLM's return_token_ids "
        f"support); choice keys = {sorted(ch)}")


def top_logprobs_at(resp: dict, step: int) -> List[tuple]:
    """[(token_id, logprob), ...] sorted desc, at generation step `step`."""
    lp = resp["choices"][0].get("logprobs") or {}
    tops = lp.get("top_logprobs") or []
    if step >= len(tops) or not tops[step]:
        return []
    items = []
    for k, v in tops[step].items():
        # return_tokens_as_token_ids renders keys as "token_id:NNN"
        tid = int(k.split(":")[-1]) if ":" in str(k) else None
        items.append((tid, float(v), str(k)))
    items.sort(key=lambda x: -x[1])
    return items


def compare(name: str, ref: List[int], got: List[int]) -> Dict[str, Any]:
    n = min(len(ref), len(got))
    first = None
    for i in range(n):
        if ref[i] != got[i]:
            first = i
            break
    return {
        "n_ref": len(ref),
        "n_got": len(got),
        "n_compared": n,
        "exact_match": first is None and len(ref) == len(got),
        "first_divergence": first,
        "n_matching_prefix": n if first is None else first,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="hf_greedy.json")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--model", default="muse-glimmer-30b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--throughput-batch", type=int, default=8)
    ap.add_argument("--throughput-tokens", type=int, default=64)
    ap.add_argument("--try-prompt-logprobs", action="store_true")
    args = ap.parse_args()

    eng = Engine(f"http://127.0.0.1:{args.port}", args.model)
    ref = json.load(open(args.ref))
    names = list(ref.keys())
    results: Dict[str, Any] = {"prompts": {}, "phases": {}}

    # ---------------------------------------------------------- phase 1 ----
    print("=== phase 1: engine liveness (a REAL request) ===", flush=True)
    served = get(f"{eng.base}/v1/models")
    print("served models:", [m["id"] for m in served.get("data", [])],
          flush=True)
    probe_ids = ref[names[0]]["prompt_ids"][:8]
    probe = eng.complete(probe_ids, 4)
    print(f"probe ok in {probe['_elapsed']:.1f}s -> ids {gen_ids(probe)} "
          f"text {probe['choices'][0]['text']!r}", flush=True)
    results["phases"]["liveness"] = {
        "served_models": [m["id"] for m in served.get("data", [])],
        "probe_ids": gen_ids(probe),
        "probe_seconds": probe["_elapsed"],
    }

    # ---------------------------------------------------------- phase 2 ----
    print("\n=== phase 2: sequential greedy vs HF ===", flush=True)
    seq: Dict[str, List[int]] = {}
    for name in names:
        entry = ref[name]
        pids = entry["prompt_ids"]
        want = entry["greedy_ids"]
        resp = eng.complete(pids, len(want))
        got = gen_ids(resp)
        seq[name] = got
        cmp_ = compare(name, want, got)
        cmp_["prompt_tokens"] = len(pids)
        cmp_["seconds"] = resp["_elapsed"]
        cmp_["exceeds_sliding_window"] = len(pids) > 2048
        rp = resp["choices"][0].get("prompt_token_ids")
        cmp_["prompt_ids_roundtrip_ok"] = (rp is None or list(rp) == list(pids))
        cmp_["text"] = resp["choices"][0]["text"]

        if cmp_["first_divergence"] is not None:
            k = cmp_["first_divergence"]
            tops = top_logprobs_at(resp, k)
            cmp_["divergence_detail"] = {
                "step": k,
                "hf_token": want[k],
                "tpu_token": got[k],
                "tpu_top5": [{
                    "token_id": t,
                    "logprob": v,
                    "key": s
                } for t, v, s in tops[:5]],
                "tpu_top1_top2_gap":
                (tops[0][1] - tops[1][1]) if len(tops) > 1 else None,
                "hf_token_rank_on_tpu":
                next((i for i, (t, _, _) in enumerate(tops) if t == want[k]),
                     None),
                "hf_token_logprob_on_tpu":
                next((v for t, v, _ in tops if t == want[k]), None),
            }
        results["prompts"][name] = cmp_
        status = "EXACT" if cmp_["exact_match"] else (
            f"diverges at {cmp_['first_divergence']}")
        print(f"  {name:20s} prompt={len(pids):5d} gen={len(want):3d} "
              f"{status}  ({resp['_elapsed']:.1f}s)", flush=True)
        if not cmp_["exact_match"]:
            print(f"      detail: "
                  f"{json.dumps(cmp_.get('divergence_detail'), indent=8)}",
                  flush=True)

    # ---------------------------------------------------------- phase 3 ----
    print("\n=== phase 3: all prompts CONCURRENT (paged KV, >1 sequence) ===",
          flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(names)) as ex:
        futs = {
            name: ex.submit(eng.complete, ref[name]["prompt_ids"],
                            len(ref[name]["greedy_ids"]), 1)
            for name in names
        }
        conc = {name: gen_ids(f.result()) for name, f in futs.items()}
    wall = time.time() - t0
    same = {n: conc[n] == seq[n] for n in names}
    total_new = sum(len(v) for v in conc.values())
    print(f"  {sum(same.values())}/{len(names)} identical to the sequential "
          f"run; {total_new} tokens in {wall:.1f}s", flush=True)
    for n in names:
        if not same[n]:
            d = compare(n, seq[n], conc[n])
            print(f"    MISMATCH {n}: first divergence at "
                  f"{d['first_divergence']}", flush=True)
    results["phases"]["concurrent"] = {
        "identical_to_sequential": same,
        "all_identical": all(same.values()),
        "wall_seconds": wall,
        "total_new_tokens": total_new,
        "concurrent_requests": len(names),
    }

    # ---------------------------------------------------------- phase 4 ----
    print(f"\n=== phase 4: throughput ({args.throughput_batch} concurrent x "
          f"{args.throughput_tokens} tokens) ===", flush=True)
    base_name = names[0]
    pids = ref[base_name]["prompt_ids"]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.throughput_batch) as ex:
        futs = [
            ex.submit(eng.complete, pids, args.throughput_tokens, None)
            for _ in range(args.throughput_batch)
        ]
        outs = [gen_ids(f.result()) for f in futs]
    wall = time.time() - t0
    gen_tok = sum(len(o) for o in outs)
    identical = all(o == outs[0] for o in outs)
    # Same prompt + greedy => every concurrent sequence must produce the SAME
    # ids. A difference here is a paged-KV cross-contamination bug.
    matches_seq = outs[0][:len(seq[base_name])] == seq[base_name][:len(outs[0])]
    print(f"  {gen_tok} generated tokens in {wall:.1f}s = "
          f"{gen_tok / wall:.1f} tok/s (batch {args.throughput_batch})",
          flush=True)
    print(f"  all {len(outs)} concurrent outputs identical: {identical}; "
          f"consistent with the batch-1 run: {matches_seq}", flush=True)
    results["phases"]["throughput"] = {
        "concurrent_requests": args.throughput_batch,
        "tokens_per_request": args.throughput_tokens,
        "generated_tokens": gen_tok,
        "wall_seconds": wall,
        "output_tok_per_s": gen_tok / wall,
        "all_outputs_identical": identical,
        "consistent_with_batch1": matches_seq,
    }

    # ---------------------------------------------------------- phase 5 ----
    if args.try_prompt_logprobs:
        print("\n=== phase 5: teacher-forced prompt_logprobs (shortest "
              "prompt only; known to be able to kill EngineCore) ===",
              flush=True)
        try:
            short = min(names, key=lambda n: len(ref[n]["prompt_ids"]))
            r = eng.complete(ref[short]["prompt_ids"], 1, 1, prompt_logprobs=1)
            pl = r["choices"][0].get("prompt_logprobs")
            results["phases"]["prompt_logprobs"] = {
                "prompt": short,
                "n_positions": len(pl) if pl else 0,
                "ok": bool(pl),
            }
            print(f"  ok: {len(pl) if pl else 0} positions", flush=True)
        except Exception as exc:  # noqa: BLE001
            results["phases"]["prompt_logprobs"] = {
                "ok": False,
                "error": str(exc)
            }
            print(f"  prompt_logprobs failed (non-fatal): {exc}", flush=True)

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {args.out}", flush=True)

    print("\n================ GREEDY SUMMARY ================")
    n_exact = 0
    for name, r in results["prompts"].items():
        flag = "EXACT" if r["exact_match"] else \
            f"DIVERGE@{r['first_divergence']}/{r['n_ref']}"
        n_exact += bool(r["exact_match"])
        print(f"  {name:20s} prompt={r['prompt_tokens']:5d} "
              f"gen={r['n_ref']:3d} {flag}"
              f"{'  [>2048 window]' if r['exceeds_sliding_window'] else ''}")
    print(f"  {n_exact}/{len(results['prompts'])} prompts token-exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
