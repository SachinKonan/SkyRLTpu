#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""On-slice LoRA smoke for the torch/vLLM Muse-Glimmer model.

The question is narrow and the answer must be unambiguous: **did the adapter
actually attach?**  A server that accepts `/v1/load_lora_adapter`, lists the
adapter as a model, and then generates the base model's tokens has silently
injected zero adapters -- which is precisely the failure mode the MaxText side
was bitten by (a name mismatch yields zero adapters and a plausible loss
curve).  So the pass condition is:

    the adapter loads   AND   greedy output with it differs from base
                        AND   unloading it restores the base output.

This mirrors what SkyRL's RL loop does every weight-sync step: upload an
adapter directory, generate against it, replace it next step.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from typing import Any

PROMPTS = [
    "The capital of France is",
    "def rms_norm(x, eps):",
    "In one sentence, what is a queued resource?",
]


def _req(url: str, payload: dict | None, timeout: int = 900,
         method: str = "POST") -> tuple[int, Any]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode()
            try:
                return resp.status, json.loads(body)
            except json.JSONDecodeError:
                return resp.status, body
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:2000]


def greedy(base: str, model: str, prompt: str, n: int = 24) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": n,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "return_token_ids": True,
        "ignore_eos": True,
    }
    t0 = time.time()
    code, out = _req(f"{base}/v1/completions", payload)
    if code != 200 or not isinstance(out, dict):
        return {"error": out, "code": code}
    ch = out["choices"][0]
    return {
        "ids": list(ch.get("token_ids") or []),
        "text": ch.get("text", ""),
        "elapsed": time.time() - t0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--base-model", default="muse-glimmer-30b")
    ap.add_argument("--lora-name", default="rl-smoke")
    ap.add_argument("--lora-path", required=True)
    ap.add_argument("--second-lora-path", default="",
                    help="a DIFFERENT adapter; loading it after unloading the "
                         "first is the RL weight-sync round trip")
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    res: dict[str, Any] = {"lora_name": args.lora_name,
                           "lora_path": args.lora_path}

    print("== 1. baseline greedy (no adapter) ==")
    before = {p: greedy(base, args.base_model, p, args.tokens) for p in PROMPTS}
    for p, r in before.items():
        print(f"  {p[:40]!r:44s} -> {r.get('ids', [])[:8]}")
    res["before"] = before

    print("\n== 2. load the adapter ==")
    code, out = _req(f"{base}/v1/load_lora_adapter", {
        "lora_name": args.lora_name,
        "lora_path": args.lora_path,
    })
    print(f"  load_lora_adapter -> HTTP {code}: {str(out)[:400]}")
    res["load_status"] = code
    res["load_body"] = str(out)[:2000]
    if code != 200:
        res["verdict"] = "FAIL: adapter did not load"
        print("\n*** LORA-LOAD-FAILED ***")
        _write(args.out, res)
        return 1

    code, models = _req(f"{base}/v1/models", None, method="GET")
    listed = [m.get("id") for m in (models or {}).get("data", [])] \
        if isinstance(models, dict) else []
    print(f"  /v1/models -> {listed}")
    res["models_after_load"] = listed
    if args.lora_name not in listed:
        res["verdict"] = "FAIL: adapter loaded but not served as a model"
        print("\n*** LORA-NOT-LISTED ***")
        _write(args.out, res)
        return 1

    print("\n== 3. greedy WITH the adapter ==")
    after = {p: greedy(base, args.lora_name, p, args.tokens) for p in PROMPTS}
    for p, r in after.items():
        print(f"  {p[:40]!r:44s} -> {r.get('ids', [])[:8]}")
    res["after"] = after

    changed = 0
    for p in PROMPTS:
        a = before[p].get("ids") or []
        b = after[p].get("ids") or []
        same = a == b
        print(f"  {p[:40]!r:44s} identical to base: {same}")
        if not same:
            changed += 1
    res["n_prompts"] = len(PROMPTS)
    res["n_changed"] = changed

    # An RL weight-sync step is exactly this: SkyRL's
    # /skyrl/v1/upload_lora_adapter unloads the previous adapter and loads the
    # new one under a new name (tpu/vllm_tpu_server.py:138-145). Doing that
    # once, with a genuinely different adapter, is the round trip.
    n_swapped = -1
    if args.second_lora_path:
        print("\n== 4. RL weight-sync round trip: unload step N, load step N+1 ==")
        code, out = _req(f"{base}/v1/unload_lora_adapter",
                         {"lora_name": args.lora_name})
        print(f"  unload({args.lora_name}) -> HTTP {code}")
        second = f"{args.lora_name}-step2"
        code, out = _req(f"{base}/v1/load_lora_adapter", {
            "lora_name": second,
            "lora_path": args.second_lora_path,
        })
        print(f"  load({second}) -> HTTP {code}: {str(out)[:200]}")
        res["swap_load_status"] = code
        if code == 200:
            swapped = {p: greedy(base, second, p, args.tokens)
                       for p in PROMPTS}
            res["swapped"] = swapped
            n_swapped = sum(
                1 for p in PROMPTS
                if (swapped[p].get("ids") or []) != (after[p].get("ids") or [])
                and (swapped[p].get("ids") or []) != (before[p].get("ids") or []))
            print(f"  step-2 adapter differs from BOTH step-1 and base on "
                  f"{n_swapped}/{len(PROMPTS)} prompts")
            _req(f"{base}/v1/unload_lora_adapter", {"lora_name": second})
        res["n_swapped"] = n_swapped

    print("\n== 5. unload and confirm the base output returns ==")
    code, out = _req(f"{base}/v1/unload_lora_adapter",
                     {"lora_name": args.lora_name})
    print(f"  unload_lora_adapter -> HTTP {code}")
    res["unload_status"] = code
    restored = {p: greedy(base, args.base_model, p, args.tokens)
                for p in PROMPTS}
    n_restored = sum(1 for p in PROMPTS
                     if (restored[p].get("ids") or []) == (before[p].get("ids")
                                                           or []))
    print(f"  base output restored on {n_restored}/{len(PROMPTS)} prompts")
    res["n_restored"] = n_restored

    ok = (changed == len(PROMPTS)) and (n_restored == len(PROMPTS))
    if args.second_lora_path and n_swapped != len(PROMPTS):
        ok = False
    res["verdict"] = "PASS" if ok else "FAIL"
    print()
    if ok:
        # The string the driver greps for.
        print("LORA-CHANGED-OUTPUT: adapter attached, all "
              f"{changed}/{len(PROMPTS)} prompts changed, all restored on "
              "unload")
        if n_swapped >= 0:
            print(f"LORA-WEIGHT-SYNC-ROUNDTRIP: step-2 adapter swapped in "
                  f"cleanly, {n_swapped}/{len(PROMPTS)} prompts distinct from "
                  "both step-1 and base")
    else:
        print(f"*** LORA-SMOKE-FAIL: changed {changed}/{len(PROMPTS)}, "
              f"restored {n_restored}/{len(PROMPTS)}, "
              f"swapped {n_swapped}/{len(PROMPTS)} ***")
        if changed == 0:
            print("*** ZERO prompts changed: the adapter loaded but injected "
                  "no live weights -- this is the silent-zero-adapters "
                  "failure mode ***")
    _write(args.out, res)
    return 0 if ok else 1


def _write(path: str, res: dict) -> None:
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    print(f"wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
